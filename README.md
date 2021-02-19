# WaNet - Imperceptible Warping-based Backdoor Attack 

<img src="Teaser.png" width="800px"/>

This is an official implementation of the ICLR 2021 Paper "WaNet - Imperceptible Warping-based Backdoor Attack" in Pytorch. This repository includes:
- Training and evaluation code.
- Defense experiments used in the paper.
- Pretrained checkpoints used in the paper. 

## Requirements
- Install required python packages:
```
$ python -m pip install -r requirements.py
```

- Download and re-organize GTSRB dataset from its official website:
```
$ bash gtsrb_download.sh
```

## Pretrained models
We also provide pretrained checkpoints used in the original paper. The checkpoints could be found at[here](https://drive.google.com/file/d/1yuinSv5Ny_gZ2rU4-fjwAofvG0x_o1wk/view?usp=sharing). Just download and decompress it in this project's repo for evaluating. 

## Evaluation 

## Reference 
If you find this repo useful for your research, please consider citing our paper
```
@inproceedings{
nguyen2021wanet,
title={WaNet - Imperceptible Warping-based Backdoor Attack},
author={Tuan Anh Nguyen and Anh Tuan Tran},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=eEn8KTtJOx}
}
```

## Contacts

If you have any questions, drop an email to _v.anhtt152@vinai.io_ , _v.anhnt479@vinai.io_  or leave a message below with GitHub (log-in is needed).