# CROPUP
<h3 align="center">CROPUP: Historical products are all you need? An end-to-end cross-year crop map updating framework without the need for in situ samples </h3>

<h5 align="right">by <a href="https://ll0912.github.io/">Lei Lei</a>,  <a href="https://jszy.whu.edu.cn/WangXinyu/zh_CN/index.htm">Xinyu Wang </a>, <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a>, Xin Hu and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

![introduction](img/CROPUP.jpg)

This is an official implementation of CROPUP in our RSE 2021 paper <a>CROPUP: Historical products are all you need? An end-to-end cross-year crop map updating framework without the need for in situ samples </a>.

## Getting started
### Prepare environment

### Prepare dataset
1. [CDL download](data/download_tile_gee.js) from <a href="https://code.earthengine.google.com/">GEE</a> platform

2. Training and test dataset preparation
```bash
python data/cdldataset_rr.py
```
3. prepare congif file

## Training and evaluation 
```bash
python training_unite_cropup.py
```

## Inferring for crop map
```bash
python inferring_unite.py
```
