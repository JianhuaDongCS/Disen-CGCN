# Disentangled Cascaded Graph Convolution Networks for Multi-Behavior Recommendation

This is our implementation for the paper:

Zhiyong Cheng*, Jianhua Dong, Fan Liu, Lei Zhu, Xun Yang, Meng Wang. (“*”= Corresponding author)

**Please cite our paper if you use our codes or datasets. Thanks!**

## environment:

Python 3.6,
TensorFlow 1.14.

## Reproducibility

Train and evaluate our model:

Dataset:Tmall

```
python Disen-CGCN.py --data_path Tmall/
                     --dataset 'buy'
                     --pretrain 1
                     --epoch 1000
                     --embed_size 64
                     --meta_embed_size 64
                     --layer_size2 [64,64,64]
                     --layer_size3 [64,64,64,64]
                     --layer_size4 [64,64]
                     --batch_size 2048
                     --attention_enlarge_num 4
                     --n_factors 2
                     --cor_flag3 1
                     --regs [1e-4]
                     --reg_c [1e0]
                     --reg_W_l2 [1e-4]
                     --reg_W_PFT_l2 [1e-4]
                     --lr 0.001
```

The Tmall dataset and its pre-trained files can be downloaded from [Disen-CGCN_data](https://drive.google.com/drive/folders/1ptnnEnlhNVgNvtGJ8svWAXrfvzY1SDt8?usp=drive_link).

Last Update Date: JAN. 10, 2024


