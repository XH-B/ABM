import argparse
import numpy as np
import os
import re
import torch
from utils import *
from encoder_decoder import * 


def main(beam_k,model_path, dictionary_target, fea, latex, saveto, output, idx_decoder):

    # model architecture
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 113
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 1

    params['L2R'] = 0 
    params['R2L'] = 0

    if idx_decoder ==1:
        params['L2R'] = 1
        end = 1
    if idx_decoder ==2:
        params['R2L'] = 1
        end = 0
    
    # load model
    model = Encoder_Decoder(params)
    load_checkpoint_part_weight(model, model_path)
    model.cuda()
    model.eval()

    # load dictionary
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    # load data
    test, test_uid_list = dataIterator(fea, latex, worddicts, batch_size=8, batch_Imagesize=500000, maxlen=20000, maxImagesize=500000)

    with torch.no_grad():
        fpp_sample = open(saveto, 'w')
        test_count_idx = 0

        for x, y in test:
            for xx in x:
                xx_pad = xx.astype(np.float32) / 255.
                xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                #direction
                sample, score ,attn_weights,next_alpha_sum = gen_sample_bidirection(model, xx_pad, params, False, k=beam_k, maxlen=1000, idx_decoder=int(idx_decoder))
                score = score / np.array([len(s) for s in sample])
                if len(score)==0:continue
                ss = sample[score.argmin()]
                # write decoding results
                fpp_sample.write(test_uid_list[test_count_idx])
                
                prd_strs=''
                for vv in ss:
                    if vv == end:  # <eos>   # 'L2R' 1
                        break
                    prd_strs +=worddicts_r[vv] +' '
                    fpp_sample.write(' ' + worddicts_r[vv])
                fpp_sample.write('\n')

                print(test_count_idx, prd_strs)
                test_count_idx+=1

    fpp_sample.close()
    print('test set decode done')
    os.system('python compute-wer.py ' + saveto + ' ' + latex + ' ' + output+ ' '+str(idx_decoder))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-dictionary_target', type=str)
    parser.add_argument('-test_dataset', type=str)
    parser.add_argument('-label', type=str)
    parser.add_argument('-saveto', type=str)
    parser.add_argument('-output', type=str)
    parser.add_argument('-idx_decoder', type=int)
    

    args = parser.parse_args()
    main(args.k, args.model_path, args.dictionary_target, args.test_dataset, args.label, args.saveto, args.output, args.idx_decoder)
