import numpy as np
import copy
import sys
import pickle as pkl
import torch
from torch import nn
import os
import cv2
from numpy import mean
import numpy
import torch.nn.functional as F
from typing import List
from torch import FloatTensor, LongTensor
from einops import rearrange, repeat
from collections import OrderedDict



# load data
def dataIterator(feature_file, label_file, dictionary, batch_size, batch_Imagesize, maxlen, maxImagesize):
    # offline-train.pkl
    fp = open(feature_file, 'rb')
    features = pkl.load(fp)
    fp.close()

    # train_caption.txt
    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()

    targets = {}
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if dictionary.__contains__(w):
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                sys.exit()
        targets[uid] = w_list

    imageSize = {}
    for uid, fea in features.items():
        imageSize[uid] = fea.shape[1] * fea.shape[2]
    # sorted by sentence length, return a list with each triple element
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = []
    biggest_image_size = 0

    i = 0
    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        lab = targets[uid]
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        elif size > maxImagesize:
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print('total ', len(feature_total), 'batch data loaded')
    return list(zip(feature_total, label_total)), uidList






# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total Latex class: ', len(lexicon))
    return lexicon




def prepare_data_bidecoder(options, images_x, seqs_y):
    """
    """


    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1


    #L2R  y_in: <sos> y1, y2, ..., yn
    #L2R  y_out: y1, y2, ..., yn, <eos>
    x = np.zeros((n_samples, options['input_channels'], max_height_x, max_width_x)).astype(np.float32)
    y_in  = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <sos> must be 0 in the dict
    y_out = np.ones((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 1 in the dict

    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)


    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y_in[1:(lengths_y[idx]+1), idx] = s_y
        y_out[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.


    #R2L: y_in:  <eos> yn, yn-1, ..., y3, y2, y1
    #R2L: y_out: yn, yn-1, ..., y2, y1, <sos>

    y_reverse_in   = np.ones((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    y_reverse_out  = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    y_reverse_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)

    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        y_reverse_in[1:(lengths_y[idx]+1), idx] = s_y[::-1]
        y_reverse_out[:lengths_y[idx], idx] = s_y[::-1]
        y_reverse_mask[:lengths_y[idx] + 1, idx] = 1.


    return x, x_mask, y_in, y_out, y_mask, y_reverse_in, y_reverse_out, y_reverse_mask




def gen_sample_bidirection(model, x, params, gpu_flag, k=1, maxlen=30, idx_decoder=1):
    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)

    if gpu_flag:
        next_state, ctx0 = model.module.f_init(x,idx_decoder=idx_decoder)
    else:
        next_state, ctx0 = model.f_init(x,idx_decoder=idx_decoder)


    if idx_decoder ==1:  #'1 denote L2R'
        next_w = np.zeros((1,)).astype(np.int64)
        end =1
    else:
        next_w = np.ones((1,)).astype(np.int64)
        end =0
    next_w = torch.from_numpy(next_w).cuda()
    next_alpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    ctx0 = ctx0.cpu().numpy()

    
    alpha_sum = []
    next_alpha_sum = []

    for ii in range(maxlen):
        ctx = np.tile(ctx0, [live_k, 1, 1, 1])
        ctx = torch.from_numpy(ctx).cuda()
        if gpu_flag:
            next_p, next_state, next_alpha_past,alpha = model.module.f_next(params, next_w, None, ctx, None, next_state,
                                                                      next_alpha_past, True,idx_decoder=idx_decoder)
        else:
            next_p, next_state, next_alpha_past,alpha = model.f_next(params, next_w, None, ctx, None, next_state,
                                                               next_alpha_past, True,idx_decoder=idx_decoder)

        next_alpha_sum.append(next_alpha_past)


        next_p = next_p.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_alpha_past = next_alpha_past.cpu().numpy()

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()

        ranks_flat = cand_flat.argsort()[:(k - dead_k)]
        voc_size = next_p.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        alpha_sum.append(alpha)
        

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k - dead_k).astype(np.float32)
        new_hyp_states = []
        new_hyp_alpha_past = []
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        hyp_alpha_past = []



        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == end:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_alpha_past.append(new_hyp_alpha_past[idx])
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)
        next_alpha_past = np.array(hyp_alpha_past)
        next_w = torch.from_numpy(next_w).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_alpha_past = torch.from_numpy(next_alpha_past).cuda()

        
    return sample, sample_score,alpha_sum,next_alpha_sum




# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass



def weight_init_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass


def load_checkpoint_part_weight(model, checkpoint_path):
    print("loadding model ...")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoints    = torch.load(checkpoint_path, map_location='cpu')
        model_params   = model.state_dict()
        new_state_dict = OrderedDict()
        checkpoint = checkpoints['state_dict'] if 'state_dict' in checkpoints.keys() else checkpoints

        for k, v in model_params.items():
            if k in checkpoint.keys():
                v_checkpoint = checkpoint[k]
                if v_checkpoint.shape == v.shape:
                    new_state_dict[k] = checkpoint[k]
                else:
                    print(k)
                    new_state_dict[k] = v
            else:
                print("intialize params: ", k)
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)


