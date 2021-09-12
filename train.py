import time
import os
import re
import numpy as np
import random
import torch
from torch import optim, nn
from utils import *
from encoder_decoder import *
import sys
import torch.nn.functional as F
from datetime import datetime


# whether init params
init_param_flag = True
use_pretrianed_model = False
pretrained_model_path = ""
gpu_number = torch.cuda.device_count()
if gpu_number>1:
    multi_gpu_flag = True
else:
    multi_gpu_flag = False
device_ids = [i for i in range(gpu_number)]

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)


result_path = sys.argv[1]
direction   = sys.argv[2]

assert direction in ['L2R', 'R2L', 'L2R-R2L'], 'Currently, the alg supports L2R and L2RR2L options.'

params = {}
params['L2R'] = 0
params['R2L'] = 0

all_directions = direction.split('-')
if len(all_directions) == 1:
    params[all_directions[0]] = 1
if len(all_directions) == 2:
    params[all_directions[0]] = 1
    params[all_directions[1]] = 1


if not os.path.exists(result_path):
    os.mkdir(result_path)


dictionaries = ['./data/dictionary.txt']
datasets = ['./data/offline-train.pkl', r'./data/train_caption.txt']
valid_datasets = ['./data/offline-2014-test.pkl', './data/test-caption-2014.txt']
valid_output = [result_path+ '/decode_result/']


if not os.path.exists(valid_output[0]):
    os.mkdir(valid_output[0])
valid_result = [result_path+ 'valid.wer']
saveto = result_path


# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
finish_after = 10000000


patience = 15
lr_decrease_rate = 2
halfLrFlag_set = 10
lrate = 1



# training settings
if multi_gpu_flag:
    batch_Imagesize = 500000
    maxImagesize = 500000
    valid_batch_Imagesize = 500000
    batch_size = 32
    valid_batch_size = 32
else:
    batch_Imagesize = 320000
    maxImagesize = 320000
    valid_batch_Imagesize = 320000
    batch_size =16
    valid_batch_size = 16



print('patence:',patience)
print('strat_lr:',lrate)
print('lr_decrease_rate:',lr_decrease_rate)
print('halfFlag:',halfLrFlag_set)
print('batch_size:',batch_size)
print('batch_size_valid:',valid_batch_size)
print('multi_gpu:',multi_gpu_flag)
print('os.environ:',os.environ['CUDA_VISIBLE_DEVICES'])
print('valid_out:',valid_output)
print('pretrained_use:',use_pretrianed_model)
print('pretrianed_model_path:',pretrained_model_path)


maxlen = 200
max_epochs = 5000

my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.

# model architecture
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


print(params)


# load dictionary
worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

# load data
train, train_uid_list = dataIterator(datasets[0], datasets[1], worddicts, batch_size=batch_size,
                                     batch_Imagesize=batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)

valid, valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1], worddicts, batch_size=valid_batch_size,
                                     batch_Imagesize=valid_batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)

# display
uidx = 0  # count batch
loss_s = 0. 
loss_s2=0
loss_kl = 0
ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
if validFreq == -1:
    validFreq = len(train)
if saveFreq == -1:
    saveFreq = len(train)
if sampleFreq == -1:
    sampleFreq = len(train)


beta = 0.5
model = Encoder_Decoder(params)
print(model)


if init_param_flag:
    model.apply(weight_init_kaiming_uniform)
    if params['R2L'] == 1:
        model.init_GRU_model2.apply(weight_init)
        model.emb_model2.apply(weight_init)
        model.gru_model2.apply(weight_init)


if multi_gpu_flag:
    model = nn.DataParallel(model, device_ids=device_ids)
model.cuda()


start_epoch = 0 
if use_pretrianed_model:
    print('***load_pretrained_model****')
    start_epoch = int(pretrained_model_path.split('/')[-1][5:7])
    model.load_state_dict(torch.load(pretrained_model_path, map_location=lambda storage, loc: storage))



encoder_params = 0
all_params = 0
for k, p in model.named_parameters():
    if k.startswith('encoder1'):
        encoder_params+=p.numel()
    all_params += p.numel()
print('encoder_params',encoder_params/1024/1024,'all_parmas',all_params/1024/1024)


# loss function
criterion = torch.nn.CrossEntropyLoss(reduce=False)
KL_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()

# optimizer
optimizer = optim.Adadelta(model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)

print('Optimization')

# statistics
history_errs = []
history_loss = []
min_Valid_WER=200
max_ExpRate=0


for eidx in range(max_epochs):
    print('epoch',eidx)
    eidx = eidx +start_epoch
    n_samples = 0
    ud_epoch = time.time()
    random.shuffle(train)


    uidx =0 

    for x, y in train:
        model.train()
        ud_start = time.time()
        n_samples += len(x)
        uidx += 1

        x, x_mask, y,y_out, y_mask, y_reverse,y_reverse_out, y_reverse_mask = prepare_data_bidecoder(params, x, y)

        x = torch.from_numpy(x).cuda()
        x_mask = torch.from_numpy(x_mask).cuda()

        #L2R
        y = torch.from_numpy(y).cuda()
        y_out = torch.from_numpy(y_out).cuda()
        y_mask = torch.from_numpy(y_mask).cuda()
        y = y.permute(1, 0)
        y_mask = y_mask.permute(1, 0)


        #R2L
        y_reverse = torch.from_numpy(y_reverse).cuda()
        y_reverse_out = torch.from_numpy(y_reverse_out).cuda()
        y_reverse_mask = torch.from_numpy(y_reverse_mask).cuda()
        y_reverse = y_reverse.permute(1, 0)
        y_reverse_mask = y_reverse_mask.permute(1, 0)


        scores, alphas,scores2, alphas2 = model(params, x, x_mask, y, y_mask,y_reverse,y_reverse_mask)

        if scores != None:
            #L2R
            alphas = alphas.permute(1, 0, 2, 3) # 
            scores = scores.permute(1, 0, 2) 
            y_mask = y_mask.permute(1,0) 

            #for kl
            new_scores = scores[1:] 
            new_y_mask = y_mask[1:] 
            new_scores = new_scores.contiguous()
            new_scores = new_scores.view(-1, new_scores.shape[2]) 
            new_y_mask = new_y_mask.view(-1) 
            new_y_mask_nonzero = torch.nonzero(new_y_mask)   
            new_scores = torch.index_select(new_scores, 0, new_y_mask_nonzero.view(-1)) 

            scores = scores.contiguous()
            scores = scores.view(-1, scores.shape[2])
            loss = criterion(scores, y_out.view(-1))
            loss = loss.view(y_out.shape[0], y_out.shape[1])
            loss = (loss * y_mask).sum(0) / y_mask.sum(0)
            loss = loss.mean()
            loss_s += loss.item()


        if scores2 != None:
            #R2L
            alphas2 = alphas2.permute(1, 0, 2, 3)
            scores2 = scores2.permute(1, 0, 2)
            y_reverse_mask = y_reverse_mask.permute(1, 0)

            #for kl
            new_scores2 = scores2[1:] 
            new_y_mask2 = y_reverse_mask[1:] 
            new_scores2 = torch.flip(new_scores2,[0]) 
            new_y_mask2 = torch.flip(new_y_mask2,[0]) 
            new_scores2 = new_scores2.contiguous()
            new_scores2 = new_scores2.view(-1, new_scores2.shape[2])  
            new_y_mask2 = new_y_mask2.view(-1)
            new_y_mask_nonzero2 = torch.nonzero(new_y_mask2)  
            new_scores2 = torch.index_select(new_scores2, 0, new_y_mask_nonzero2.view(-1))

            scores2 = scores2.contiguous()
            scores2 = scores2.view(-1, scores2.shape[2])
            loss2 = criterion(scores2, y_reverse_out.view(-1))
            loss2 = loss2.view(y_reverse_out.shape[0], y_reverse_out.shape[1])
            loss2 = (loss2 * y_reverse_mask).sum(0) / y_reverse_mask.sum(0)
            loss2 = loss2.mean()
            loss_s2 += loss2.item()
            loss += loss2

            scores2 = torch.flip(scores2, dims=[0])


            T = 4
            pred_kl_loss = KL_loss(F.log_softmax(new_scores2/T,dim=-1) , F.softmax(new_scores/T,dim=-1)) *T*T
            loss += pred_kl_loss*beta
            loss_kl += pred_kl_loss.item()*beta


        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_c)

        # update
        optimizer.step()
        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            loss_s2/=dispFreq
            loss_kl/=dispFreq
            print('Data',datetime.now().strftime('%Y-%m-%d %H-%M-%S'),'Epoch',eidx, 'Update/total: ',str(uidx)+'/'+str(len(train)), 'Cost_L2R: ', loss_s,'Cost_R2L: ', loss_s2,'KL: ', loss_kl, 'time ', ud_s, 'lrate ', lrate, 'eps', my_eps, 'bad_counter', bad_counter)
            ud_s = 0
            loss_s = 0.
            loss_s2=0
            loss_kl=0

        # valid_loss
        if True:
            if np.mod(uidx, validFreq) == 0: 
                valid_loss_start = time.time()
                loss_valid = 0.0
                valid_count_batch = 0 
                model.eval()
                with torch.no_grad():
                    for x, y in valid:
                        x, x_mask, y,y_out, y_mask, y_reverse,y_reverse_out, y_reverse_mask = prepare_data_bidecoder(params, x, y)
                        x = torch.from_numpy(x).cuda()
                        x_mask = torch.from_numpy(x_mask).cuda()

                        #L2R
                        y = torch.from_numpy(y).cuda()
                        y_out = torch.from_numpy(y_out).cuda()
                        y_mask = torch.from_numpy(y_mask).cuda()
                        y = y.permute(1, 0)
                        y_mask = y_mask.permute(1, 0)

                        y_reverse = torch.from_numpy(y_reverse).cuda()
                        y_reverse_out = torch.from_numpy(y_reverse_out).cuda()
                        y_reverse_mask = torch.from_numpy(y_reverse_mask).cuda()
                        y_reverse = y_reverse.permute(1, 0)
                        y_reverse_mask = y_reverse_mask.permute(1, 0)

                        scores, alphas,scores2, alphas2 = model(params, x, x_mask, y, y_mask,y_reverse,y_reverse_mask)

                        #L2R
                        alphas = alphas.permute(1, 0, 2, 3)
                        scores = scores.permute(1, 0, 2)
                        scores = scores.contiguous()
                        scores = scores.view(-1, scores.shape[2])
                        y_mask = y_mask.permute(1, 0)

                        loss = criterion(scores, y_out.view(-1))
                        loss = loss.view(y_out.shape[0], y_out.shape[1])
                        loss = (loss * y_mask).sum(0) / y_mask.sum(0)
                        loss = loss.mean()

                        loss_valid += loss.item()
                        valid_count_batch += 1
                        if valid_count_batch% 1000 ==0:
                            print('valid_id:',valid_count_batch, time.time()-valid_loss_start)
                            valid_loss_start = time.time()
                        

                loss_valid /= len(valid)
                history_loss.append(loss_valid)
                print('validing__','id',uidx,'loss',loss_valid,'min_loss',np.array(history_loss).min())
                print('valid_loss_time',time.time()-valid_loss_start)


                save_model_path= os.path.join(saveto,'models')
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)

                if multi_gpu_flag:
                    torch.save(model.module.state_dict(), save_model_path+'/epoch'+str(eidx)+'.pkl')
                else:
                    torch.save(model.state_dict(), save_model_path+'/epoch'+str(eidx)+'.pkl')

                #save_model
                if  loss_valid <= np.array(history_loss).min():
                    bad_counter = 0
                    print('Saving model params ... ')

                if loss_valid > np.array(history_loss).min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if halfLrFlag == halfLrFlag_set:
                            print('Early Stop!')
                            estop = True
                            break
                        else:
                            print('Lr decay and retrain!')
                            bad_counter = 0
                            lrate = lrate / lr_decrease_rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lrate
                            halfLrFlag += 1

        valid_stop = False
        valid_sample_time = time.time()
        if np.mod(uidx, validFreq) == 0 :
            model.eval()
            
            with torch.no_grad():
                fpp_sample = open(valid_output[0]+str(eidx)+'.txt', 'w')
                valid_count_idx = 0
                for x, y in valid:
                    for xx in x:
                        xx_pad = xx.astype(np.float32) / 255.
                        xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda() 
                        sample, score, _, _ = gen_sample_bidirection(model, xx_pad, params, multi_gpu_flag, k=10, maxlen=1000)
                        if len(score) == 0:
                            print('valid decode error happens')
                            valid_stop = True
                            break
                        score = score / np.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                        # write decoding results
                        fpp_sample.write(valid_uid_list[valid_count_idx])
                        valid_count_idx = valid_count_idx + 1
                        # symbols (without <eos>)

                        for vv in ss:
                            if vv == 1:  # <eos>  sos 0
                                break
                            fpp_sample.write(' ' + worddicts_r[vv])
                        fpp_sample.write('\n')
                    if valid_stop:
                        break
            fpp_sample.close()
            print('valid set decode done','valid_number:',valid_count_idx)
            


        # calculate wer and expRate
        if np.mod(uidx, validFreq) == 0 and valid_stop == False :
            os.system('python compute-wer.py ' + valid_output[0]+str(eidx)+'.txt' + ' ' + valid_datasets[
                1] + ' ' + valid_result[0]+' 1')

            fpp = open(valid_result[0])
            stuff = fpp.readlines()
            fpp.close()
            m = re.search('WER (.*)\n', stuff[0])
            valid_err = 100. * float(m.group(1))
            m = re.search('ExpRate (.*)\n', stuff[1])
            valid_sacc = 100. * float(m.group(1))
            history_errs.append(valid_err)

        
            if valid_err<min_Valid_WER:
                min_Valid_WER = valid_err
            if valid_sacc>max_ExpRate:
                max_ExpRate = valid_sacc
                print('Saving best model params ... ')

                save_model_path= os.path.join(saveto,'models')
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)

                if multi_gpu_flag:
                    torch.save(model.module.state_dict(), save_model_path+'/ABM_params'+str(eidx)+'_lr'+str(lrate)+'_'+str(valid_sacc)+'.pkl')
                else:
                    torch.save(model.state_dict(), save_model_path+'/ABM_params'+str(eidx)+'_lr'+str(lrate)+'_'+str(valid_sacc)+'.pkl')


            print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_err, valid_sacc),'min Valid WER: %.2f%%, max ExpRate: %.2f%%' % (min_Valid_WER, max_ExpRate))
            print('valid_sample_time',time.time()-valid_sample_time)
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('epoch_toal_cost time ... minute', ud_epoch)

        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
