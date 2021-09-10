import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from decoder import *


# create gru init state
class FcLayer(nn.Module):
    def __init__(self, nin, nout):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        out = torch.tanh(self.fc(x))
        return out


# Embedding
class Embedding(nn.Module):
    def __init__(self, params):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(params['K'], params['m'])

    def forward(self, params, y):
        emb = self.embedding(y)
        return emb




class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()

        self.encoder1 = DenseNet(growthRate=params['growthRate'], reduction=params['reduction'], bottleneck=params['bottleneck'], use_dropout=params['use_dropout'])

        if params['L2R'] == 1: 
            self.init_GRU_model = FcLayer(params['D'], params['n'])
            self.emb_model = Embedding(params)
            self.gru_model = Gru_cond_layer_aam(params)
            self.gru_prob_model = Gru_prob(params)

        if params['R2L'] == 1:
            params['dim_attention'] = 512
            self.init_GRU_model2 = FcLayer(params['D'], params['n'])
            self.emb_model2 = Embedding(params)
            self.gru_model2 = Gru_cond_layer_aam(params)
            self.gru_prob_model2 = Gru_prob(params)


    def forward(self, params, x, x_mask, y, y_mask, y_reverse,y_mask_reverse,one_step=False):
        # recover permute
        y = y.permute(1, 0)
        y_mask = y_mask.permute(1, 0)
        y_reverse = y_reverse.permute(1, 0)
        y_mask_reverse = y_mask_reverse.permute(1, 0)

        out_mask = x_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        x_mask = out_mask[:, 0::2, 0::2]
        ctx_mask = x_mask
        ctx1 = self.encoder1(x)

        if params['L2R'] == 1:        
            ctx_mean1 = (ctx1 * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]
            init_state1 = self.init_GRU_model(ctx_mean1)

            # two GRU layers
            emb1 = self.emb_model(params, y)
            h2ts1, cts1, alphas1, _alpha_pasts = self.gru_model(params, emb1, y_mask, ctx1, ctx_mask, one_step, init_state1, alpha_past=None)
            scores1 = self.gru_prob_model(cts1, h2ts1, emb1, use_dropout=params['use_dropout'])
            # permute for multi-GPU training
            alphas1 = alphas1.permute(1, 0, 2, 3)
            scores1 = scores1.permute(1, 0, 2)

        if params['R2L'] == 1:
            ctx_mean2 = (ctx1 * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]
            init_state2 = self.init_GRU_model2(ctx_mean2)  
            # # two GRU layers
            emb2 = self.emb_model2(params, y_reverse)
            h2ts2, cts2, alphas2, _alpha_pasts = self.gru_model2(params, emb2, y_mask_reverse, ctx1, ctx_mask, one_step, init_state2, alpha_past=None)
            scores2 = self.gru_prob_model2(cts2, h2ts2, emb2, use_dropout=params['use_dropout'])
            # permute for multi-GPU training
            alphas2 = alphas2.permute(1, 0, 2, 3)
            scores2 = scores2.permute(1, 0, 2)

        if params['L2R'] == 1 and params['R2L'] == 1:
            return scores1, alphas1,scores2, alphas2
        if params['L2R'] == 1 and params['R2L'] == 0:
            return scores1, alphas1, None, None
        if params['L2R'] == 0 and params['R2L'] == 1:
            return None, None, scores2, alphas2 


    # decoding: encoder part
    def f_init(self, x, x_mask=None,idx_decoder=1):
        if x_mask is None:
            shape = x.shape
            x_mask = torch.ones(shape).cuda()

        out_mask = x_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        x_mask = out_mask[:, 0::2, 0::2]
        ctx_mask = x_mask
        
        ctx1= self.encoder1(x) 
        ctx_mean1 = ctx1.mean(dim=3).mean(dim=2)
        if idx_decoder==1:
            init_state1 = self.init_GRU_model(ctx_mean1)
        elif idx_decoder==2:
            init_state1 = self.init_GRU_model2(ctx_mean1)

        return init_state1,ctx1


    # decoding: decoder part
    def f_next(self, params, y, y_mask, ctx, ctx_mask, init_state, alpha_past, one_step,idx_decoder=1):

        if idx_decoder == 1:

            emb_beam = self.emb_model(params, y)

            # one step of two gru layers
            next_state, cts, _alpha, next_alpha_past = self.gru_model(params, emb_beam, y_mask, ctx, ctx_mask, one_step, init_state, alpha_past)
            # reshape to suit GRU step code
            next_state_ = next_state.view(1, next_state.shape[0], next_state.shape[1])
            cts = cts.view(1, cts.shape[0], cts.shape[1])
            emb_beam = emb_beam.view(1, emb_beam.shape[0], emb_beam.shape[1])
            # calculate probabilities
            scores = self.gru_prob_model(cts, next_state_, emb_beam, use_dropout=params['use_dropout'])
            scores = scores.view(-1, scores.shape[2])
            next_probs = F.softmax(scores, dim=1)

        elif idx_decoder ==2:
            emb_beam = self.emb_model2(params, y)

            # one step of two gru layers
            next_state, cts, _alpha, next_alpha_past = self.gru_model2(params, emb_beam, y_mask, ctx, ctx_mask, one_step, init_state, alpha_past)
            # reshape to suit GRU step code
            next_state_ = next_state.view(1, next_state.shape[0], next_state.shape[1])
            cts = cts.view(1, cts.shape[0], cts.shape[1])
            emb_beam = emb_beam.view(1, emb_beam.shape[0], emb_beam.shape[1])

            # calculate probabilities
            scores = self.gru_prob_model2(cts, next_state_, emb_beam, use_dropout=params['use_dropout'])
            scores = scores.view(-1, scores.shape[2])
            next_probs = F.softmax(scores, dim=1)

        return next_probs, next_state, next_alpha_past,_alpha

