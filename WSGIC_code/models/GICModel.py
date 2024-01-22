from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AttModel import pack_wrapper, AttModel, Attention
from .TransformerModel import LayerNorm, attention, clones, SublayerConnection, PositionwiseFeedForward
from einops import rearrange, reduce, repeat

class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_glu=0, norm_q=0,
                 dropout_glu=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        self.d_k = d_model * scale // h
        self.h = h

        self.project_k_v = project_k_v

        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        self.output_layer = nn.Linear(d_model * scale, d_model)

        self.use_glu = do_glu
        if self.use_glu:
            self.glu_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())

            if dropout_glu > 0:
                self.dropout_glu = nn.Dropout(p=dropout_glu)
            else:
                self.dropout_glu = lambda x: x

        if self.use_glu or not use_output_layer:
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(512, 2048, 0.1)
        self.sublayer = SublayerConnection(512, dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:  # this project_k_v = 1
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query_, key_, value_, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.use_glu:
            x = self.glu_layer(self.dropout_glu(torch.cat([x, query], -1)))

        if single_query:
            x = self.sublayer(x, self.feed_forward)
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x, self.attn

class Verb_MHA(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_glu=0, norm_q=0,
                 dropout_glu=0.3):
        super(Verb_MHA, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        self.project_k_v = project_k_v

        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        self.output_layer = nn.Linear(d_model * scale, d_model)

        self.use_glu = do_glu
        if self.use_glu:
            self.glu_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())

            if dropout_glu > 0:
                self.dropout_glu = nn.Dropout(p=dropout_glu)
            else:
                self.dropout_glu = lambda x: x

        if self.use_glu or not use_output_layer:
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_model*4, 0.1)
        self.sublayer = SublayerConnection(d_model, dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)


        query = self.norm(query)
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query_, key_, value_, mask=mask,  # eval time - 5, 8, 36, 64
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.use_glu:
            x = self.glu_layer(self.dropout_glu(torch.cat([x, query], -1)))

        if single_query:
            x = self.sublayer(x, self.feed_forward)
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x, self.attn

class deit_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(deit_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, mask):
        x_temp, verb_attn = self.self_attn(x, x, x, mask)
        x = self.norm(x + self.dropout(x_temp))
        return self.norm(self.sublayer(x, self.feed_forward)), verb_attn

class Deit_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(Deit_Refiner_Core, self).__init__()
        attn = Verb_MHA(opt.cv_num_heads, opt.cv_size, project_k_v=0, scale=1, do_glu=0, norm_q=0, dropout_glu=0.3)
        layer = deit_Refiner_Layer(opt.cv_size, attn, PositionwiseFeedForward(opt.cv_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 3)

    def forward(self, x, mask):
        attn_weights = []
        for layer in self.layers:
            x, verb_attn = layer(x, mask)
            attn_weights.append(verb_attn)
        return x, attn_weights



class GIC_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(GIC_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.multi_head_scale = 1
        self.use_ctx_drop = 1
        self.out_res = getattr(opt, 'out_res', 0)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)
        self.decoder_type = 'GIC'
        self.use_multi_head = 2
        self.att_supervise = opt.att_supervise
        self.fc = nn.Linear(576, 512)
        self.fc1 = nn.Linear(768, 1024)

        if self.decoder_type == 'GIC':
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * self.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            self.att2ctx = nn.LSTMCell(self.d_model * self.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * self.multi_head_scale + opt.rnn_size, opt.rnn_size), nn.ReLU())

        if self.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
                                                             scale=self.multi_head_scale, use_output_layer=0, do_glu=1,
                                                             norm_q=1)
        else:
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)        
        else:
            self.ctx_drop = lambda x :x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, vit_feats, re_VIT_feats, mean_vit, pre_attn, state, att_masks=None): #p_att_feats 50, 34, 1024

        vit_feats = self.fc1(re_VIT_feats)

        pre_att = self.fc(pre_attn[0][:,2:-1])

        h_att, c_att = self.att_lstm(torch.cat([xt, mean_vit + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        h_att_attn = h_att + pre_att

        if self.use_multi_head == 2:

            att, weight = self.attention(h_att_attn, vit_feats.narrow(2, 0, self.multi_head_scale * self.d_model), vit_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)

            weight_att = weight.sum(1).squeeze(1)

        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))
            pre_attn = torch.stack((weight_att, pre_attn[0]))
        if self.out_res:
            output = output + h_att

        output = self.out_drop(output)
        if self.att_supervise:
            return output, state, pre_attn, weight
        else:
            return output, state, pre_attn

class GICModel(AttModel):
    def __init__(self, opt):
        super(GICModel, self).__init__(opt)
        self.num_layers = 2
        self.use_mean_feats = 1
        self.refine = 1
        self.multi_head_scale = 1

        self.use_multi_head = 2
        self.fc2 = nn.Linear(768, opt.rnn_size)

        if self.use_multi_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * self.multi_head_scale * opt.rnn_size)

        if self.use_mean_feats:
            del self.fc_embed
        if self.refine:
            self.refiner = Deit_Refiner_Core(opt)
        else:
            self.refiner = lambda x,y : x
        self.core = GIC_Decoder_Core(opt)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))

        self.ffn = nn.Sequential(nn.Linear(768, 2048),
                                 nn.Sigmoid(),
                                 nn.Linear(2048, 512)
                                 )

        self.verb_projection = nn.Linear(512, 72)

        self.positions = nn.Parameter(torch.randn((384 // 16) ** 2 + 1, 768))

        self.cv_MSA = Deit_Refiner_Core(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks, VIT_feats):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        b, _, _ = VIT_feats.shape
        verb_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        verb_repre = torch.cat([VIT_feats[:, 2:], verb_tokens], dim=1)
        verb_repre += self.positions

        verb_feats, verb_attn = self.cv_MSA(verb_repre, att_masks)
        verb_mlp = self.ffn(verb_feats)

        verb_out = torch.mean(self.verb_projection(verb_mlp), dim=1)

        re_VIT_feats = torch.cat([VIT_feats, verb_feats[:, -1:]], dim=1)
        mean_vit = torch.mean(self.fc2(re_VIT_feats), dim=1)

        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = torch.mean(att_feats, dim=1)
            else:
                mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, mean_vit, att_masks, re_VIT_feats, verb_out, verb_attn
