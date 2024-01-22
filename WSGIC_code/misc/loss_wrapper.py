import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward
import pdb
import torch.nn as nn
import torch.nn.functional  as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class LossWrapper(torch.nn.Module):
	def __init__(self, model, opt):
		super(LossWrapper, self).__init__()
		self.opt = opt
		self.model = model
		if opt.label_smoothing > 0:
			self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
		else:
			self.crit = utils.LanguageModelCriterion()
		self.rl_crit = utils.RewardCriterion()
		self.crit_cls = nn.CrossEntropyLoss().cuda()
		self.crit_MLS = torch.nn.MultiLabelSoftMarginLoss().cuda()

		if opt.att_supervise:
			if opt.att_sup_crit == 'KL':
				self.kl_crit=nn.KLDivLoss(reduction='batchmean')
			elif opt.att_sup_crit == 'NLL':
				self.nll = nn.NLLLoss()
			elif opt.att_sup_crit == 'ExtendNLL':
				self.extendnll = utils.ExtendNLLCrit()
			else:
				raise NotImplementedError
		self.min_value=1e-8

	def forward(self, fc_feats, att_feats, labels, masks, att_masks, images, targets, gts, gt_indices,
				sc_flag, box_inds):
		out = {}
		if not sc_flag:
			outputs_all = self.model(fc_feats, att_feats, labels, images, att_masks)[0:2]
			outputs, verb_outs = outputs_all[0], outputs_all[1]

			targets[targets == 0] = 1
			targets[targets == -1] = 0

			targets = Variable(targets).float()
			cls_loss = self.crit_MLS(verb_outs, targets)
			lan_loss = self.crit(outputs, labels[:, 1:], masks[:, 1:])
			loss = lan_loss +  cls_loss
		else:
			if self.opt.att_supervise:
				gen_result, sample_logprobs, attn_weights, verb_feats = self.model(fc_feats, att_feats, images, att_masks, opt={'sample_max':0}, mode='sample')
			else:
				gen_result, sample_logprobs = self.model(fc_feats, att_feats, images, att_masks, opt={'sample_max':0}, mode='sample')
			gts = [gts[_] for _ in gt_indices.tolist()]

			if self.opt.att_supervise:
				reward = get_self_critical_reward(self.model, fc_feats, att_feats, images,
																		   att_masks, gts, gen_result, vars(self.opt))
			else:
				reward = get_self_critical_reward(self.model, fc_feats, att_feats, images, att_masks, gts, gen_result, vars(self.opt))
			reward = torch.from_numpy(reward).float().to(gen_result.device)
			loss = self.rl_crit(sample_logprobs, gen_result.data, reward)

			out['reward'] = reward[:,0].mean()
		out['loss'] = loss
		return out
