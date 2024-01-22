# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954

# RT: Code from original author's repo: https://github.com/husthuaan/AoANet/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AttModel import pack_wrapper, AttModel, Attention
from .TransformerModel import LayerNorm, attention, clones, SublayerConnection, PositionwiseFeedForward


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0,
                 dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(512, 2048, 0.1)
        self.sublayer = SublayerConnection(512, dropout)

    def forward(self, query, value, key, mask=None):
        # print(mask.size())
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        # print(query.size())  # 50, 1, 512 language Q

        query = self.norm(query)
        # print(query.size()) #50, 1, 512 language Q
        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:  # this project_k_v = 1
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query_, key_, value_, mask=mask,  # eval time - 5, 8, 36, 64
                                 dropout=self.dropout)
        # attn_view = self.attn #5, 196

        # print(x.size()) #145, 8, 36, 64 #vit - [50, 8, 1, 64]
        # print("att:", self.attn.size()) #145, 8, 36, 36
        # weight = self.attn
        # x_attn = x_attn.view(nbatches, -1, 36)
        # x_attn = torch.mean(x_attn, dim=1)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # training time vl - 100, 1, 512
        # print('x:',x.size()) #145, 36, 512
        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        #x = (self.output_layer(x)) + query  # training time vl - 100, 1, 512
        x = self.sublayer(x, self.feed_forward)
        #x = query + x

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x, self.attn


class Sec_MultiHeadedDotAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, scale=1, project_k_v=1, use_output_layer=1, do_aoa=0, norm_q=0,
                 dropout_aoa=0.3):
        super(Sec_MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
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
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query_, key_, value_, mask=mask,
                                 dropout=self.dropout)
        #print(x.size())  #eval - 5, 8, 1, 196
        #print("att:", self.attn.size())  #vit- 18, 8, 1, 196
        # weight = self.attn
        #x_attn = self.attn.view(nbatches, -1, 196)
        #x_attn = torch.mean(x_attn, dim=1)
        #print('x_attn', x_attn.size()) #145, 36 # 1, 196

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        #print('x:',x.size()) #
        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        #x = (self.output_layer(x)) + query
        x = self.sublayer(x, self.feed_forward)
        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x, self.attn #x_attn

class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1+self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x

class AoA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=1, scale=1, do_aoa=1, norm_q=0, dropout_aoa=0.3)
        layer = AoA_Refiner_Layer(opt.rnn_size, attn, PositionwiseFeedForward(opt.rnn_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 4)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class deit_tr_Core(nn.Module):
    def __init__(self, opt):
        super(deit_tr_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads_deit, opt.deit_size, project_k_v=1, scale=1, do_aoa=0, norm_q=0,
                                       dropout_aoa=0.3)
        layer = AoA_Refiner_Layer(opt.deit_size, attn,
                                  PositionwiseFeedForward(opt.deit_size, 2048, 0.1) if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 4)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class AoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        #self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = 1
        self.use_ctx_drop = 1 #getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        #self.decoder_type = getattr(opt, 'decoder_type', 'AoA')
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        #self.att_lstm = nn.LSTMCell(768 + opt.rnn_size, opt.rnn_size)  # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)
        self.decoder_type = 'AoA'
        self.use_multi_head = 2
        self.att_supervise = opt.att_supervise
        self.fc = nn.Linear(576, 512)
        self.fc1 = nn.Linear(768, 1024)

        if self.decoder_type == 'AoA':
            # AoA layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * self.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            # LSTM layer
            self.att2ctx = nn.LSTMCell(self.d_model * self.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            # Base linear layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * self.multi_head_scale + opt.rnn_size, opt.rnn_size), nn.ReLU())

        # if opt.use_multi_head == 1: # TODO, not implemented for now           
        #     self.attention = MultiHeadedAddAttention(opt.num_heads, opt.d_model, scale=opt.multi_head_scale)
        if self.use_multi_head == 2:
            if self.att_supervise:
                self.attention = Sec_MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0, scale=self.multi_head_scale, use_output_layer=0, do_aoa=1, norm_q=1)
            else:
                self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
                                                             scale=self.multi_head_scale, use_output_layer=0, do_aoa=1,
                                                             norm_q=1)
        else:
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)        
        else:
            self.ctx_drop = lambda x :x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, vit_feats, mean_vit, pre_attn, state, att_masks=None): #p_att_feats 50, 34, 1024
        #print(image_vit.size()) 50, 198, 768
        #print(p_att_feats.size()) 50, 36, 1024
        # state[0][1] is the context vector at the last step
        #print(vit_features.size())
        #vit_features_mean = self.fc1(torch.mean(vit_features, dim=1)) # 50, 768 -> 50, 512
        vit_feats = self.fc1(vit_feats) #1024

        pre_att = self.fc(pre_attn[0])

        #print(p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model).size()) #50, 36, 512
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_vit + self.ctx_drop(state[0][1])], 1), (state[0][0], state[1][0]))
        h_att_attn = h_att + pre_att

        # print(self.d_model) 512
        if self.use_multi_head == 2:
            if self.att_supervise:
                att, weight = self.attention(h_att_attn, vit_feats.narrow(2, 0, self.multi_head_scale * self.d_model), vit_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)


                weight_att = weight.sum(1).squeeze(1)

            else:
                att, weight = self.attention(h_att_attn, vit_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
                                     vit_feats.narrow(2, self.multi_head_scale * self.d_model, self.multi_head_scale * self.d_model), att_masks)


                weight_att = weight.sum(1).squeeze(1) #50, 576
                #attn_feats = self.fc(torch.cat([att, weight_att], 1))
                #print('language attention')
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            #print('output:', output.size()) #145, 512
            # save the context vector to state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))
            pre_attn = torch.stack((weight_att, pre_attn[0]))
        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        if self.att_supervise:
            return output, state, pre_attn, weight
        else:
            return output, state, pre_attn

class AoAModel(AttModel):
    def __init__(self, opt):
        super(AoAModel, self).__init__(opt)
        self.num_layers = 2
        # mean pooling
        self.use_mean_feats = 1 #getattr(opt, 'mean_feats', 1)
        self.refine = 1
        self.multi_head_scale = 1
        #mean_feats: 1
        #ctx_drop: 1
        #dropout_aoa: 0.3
        self.use_multi_head = 2
        self.fc2 = nn.Linear(768, 512)

        if self.use_multi_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * self.multi_head_scale * opt.rnn_size)

        if self.use_mean_feats:
            del self.fc_embed
        if self.refine:
            self.refiner = AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x,y : x
        self.core = AoA_Decoder_Core(opt)

        self.deit_refiner = deit_tr_Core(opt)


    def _prepare_feature(self, fc_feats, att_feats, att_masks, VIT_feats):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed att feats
        #VIT_feats = self.fc2(VIT_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        #att_feats = self.refiner(att_feats, att_masks)

        mean_vit = torch.mean(self.fc2(VIT_feats), dim=1)

        #image_vit_mean = torch.mean(VIT_feats, dim=1)
        #100, 36, 512
        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = torch.mean(att_feats, dim=1) #this
                #100, 512
            else:
                mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        #p_att_feats_a = self.fc(p_att_feats)
        #print(p_att_feats_a.size())

        return mean_feats, att_feats, p_att_feats, mean_vit, att_masks
