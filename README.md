# ABM
# Handwritten Mathematical Expression Recognition via AttentionAggregation based Bi-directional Mutual Learning (AAAI2022)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)




## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Data
We use public 
The Offline Handwritten Formula Recognition dataset CROHME 2014, 2016 and 2019 used in the paper can be download in the [TC-11 Online Resources](http://tc11.cvc.uab.es/datasets/ICDAR2019-CROHME-TDF_1) or in offical [ICDAR 2019 CROHME + TFD Competition ](https://www.cs.rit.edu/~crohme2019/dataANDtools.html).
The required data files should be put into `data/` folder. A demo slice of the data is illustrated in the following figure. 

<p align="center">
<img src="./img/data.png" height = "168" alt="" align=center />
<br><br>
<b>Figure 3.</b> An example of the ETT data.
</p>


## Usage
<span id="colablink">Colab Examples:</span> We provide google colabs to help reproducing and customing our repo, which includes `experiments(train and test)`, `prediction`, `visualization` and `custom data`.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC)

Note that we train our model on one train set CROHME 2014 and test the model on three test sets CROHME 2014, CROHME 2016, and CROHME2019. Commands for training and testing the model with single decoder branch or two decoder branches respectively:

```bash
# for training on CROHME 2014  with one L2R branch (baseline model)
sh train.sh -L2R

# for training on CROHME 2014  with two branches (L2R and R2L) (our model, ABM)
sh train.sh -L2RR2L


# for testing on CROHME 2014, 2016, 2019 with L2R branch
sh test.sh -2014  L2R


# for testing on CROHME 2014, 2016, 2019 with R2L branch
sh test.sh -2014  R2L

```

# 



We provide a more detailed and complete command description for training and testing the model:

```python
python -u main_informer.py --model <model> --data <data>
--root_path <root_path> --data_path <data_path> --features <features>
--target <target> --freq <freq> --checkpoints <checkpoints>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --c_out <c_out> --d_model <d_model>
--n_heads <n_heads> --e_layers <e_layers> --d_layers <d_layers>
--s_layers <s_layers> --d_ff <d_ff> --factor <factor> --padding <padding>
--distil --dropout <dropout> --attn <attn> --embed <embed> --activation <activation>
--output_attention --do_predict --mix --cols <cols> --itr <itr>
--num_workers <num_workers> --train_epochs <train_epochs>
--batch_size <batch_size> --patience <patience> --des <des>
--learning_rate <learning_rate> --loss <loss> --lradj <lradj>
--use_amp --inverse --use_gpu <use_gpu> --gpu <gpu> --use_multi_gpu --devices <devices>
```

The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
| model | The model of experiment. This can be set to `informer`, `informerstack`, `informerlight(TBD)` |
| dictionaries           |        dictionaries for 113 class symbols      (defaults to `./data/dictionary_bid.txt`)                                |
| train_datasets      | The root path of the data file (defaults to `./data/offline-train.pkl`)  and label file  (defaults to `./data/train_caption.txt`)  |
| valid_datasets      | The root path of the data file (defaults to `./data/offline-test.pkl`)  and label file  (defaults to `./data/test_caption..txt`)  |
| valid_output       | The path to store the decoding LaTeX result of each epoch|
| saveto         | The path to store the model of each epoch          |
| n           | GRU1 dimension  (defaults to 256)   |
| m    |  GRU2 dimension  (defaults to 256)  |
| dim_attention | attention dimension (defaults to 512)  |
| D | The channel of feature map generated encoder (DenseNet)  (defaults to 628)  |
| growthRate | The growthrate of DenseNet (defaults to 24) |
| reduction | The reduction of DenseNet (defaults to 0.5)|
| bottleneck | The bottleneck of DenseNet (defaults to True) |
| use_dropout | The use_dropout of DenseNet (defaults to True) |
| input_channels | The input_channels of DenseNet (defaults to 1) |
| batch_Imagesize | The max batch_Imagesize for training (defaults to 320000)|
| maxImagesize | The max batch_Imagesize for training(defaults to 320000) |
| valid_batch_Imagesize | The max batch_Imagesize for testing (defaults to 320000) |
| batch_size | The batch size of training input data (defaults to 16) |
| valid_batch_size | The batch size of testing input data (defaults to 16) |
| batch_Imagesize | The max batch_Imagesize for training (defaults to 320000)|
| maxlen | The max length for one sample (defaults to 200) |
| max_epochs | The max train epochs (defaults to 5000) |
| dispFreq | The batch size of testing input data (defaults to 100) |
| patience | The patience of changing learning rate  (defaults to 15) |
| learning_rate | The learning rate of optimizer  (defaults to 1) |
| lr_decrease_rate | The decline rate of learning rate (defaults to 2) |
| halfLrFlag_set | The flag of early stop (defaults to 10) |
| finish_after | The max iteration of early stop (defaults to 10000000) |
| my_eps | The eps of optimizer (defaults to 1e-6) |
| decay_c | The weight_decay of optimizer (defaults to 1e-4) |
| clip_c | clip_grad_norm (defaults to 100.) |
| patience | changing learning rate patience (defaults to 3) |
| device_ids | Device ids of gpu (defaults to `0`) |


## <span id="resultslink">Results</span>

We have updated the experiment results of all methods due to the change in data scaling. We are lucky that Informer gets performance improvement. Thank you @lk1983823 for reminding the data scaling in [issue 41](https://github.com/zhouhaoyi/Informer2020/issues/41).

Besides, the experiment parameters of each data set are formated in the `.sh` files in the directory `./scripts/`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse and mae results or draw better prediction figures.

<p align="center">
<img src="./img/result_univariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 4.</b> Univariate forecasting results.
</p>

<p align="center">
<img src="./img/result_multivariate.png" height = "500" alt="" align=center />
<br><br>
<b>Figure 5.</b> Multivariate forecasting results.
</p>
