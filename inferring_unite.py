# -*- ecoding: utf-8 -*-
"""
@enviroment: pytorch 1.6.0 CUDA 9.0
@Date:2022/04/07/15:11
@File:inferring_transformer.py
@Author: LeiLei leilei912@whu.edu.cn
"""
import sys
sys.path.append("..")
import os
import numpy as np
import yaml
import torch
from utils.io_func import read_ENVI,write_ENVI,make_parent_dir_list
from utils.img_tool import img_norm_2
from utils.img_tool import stretch_data
from utils.helper_unite import predict_batch
from models.UNITE import CTA
import time
from utils.time import format_timedelta
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# model_yaml_path="./config/model/transformer.yaml"
data_yaml_path="/home/ll22/code/two_stage_Cropup/config/17TKF_S2_2020.yaml"



# with open(model_yaml_path) as f:
#     cfg_m = yaml.safe_load(f)
with open(data_yaml_path) as f:
    cfg_d=yaml.safe_load(f)

make_parent_dir_list(cfg_d["File"]["save_score_file"] + "/", cfg_d["File"]["save_pred_file"] + "/")


img_list=os.listdir(cfg_d["File"]["image_path"])
print(len(img_list))
doy=np.asarray(cfg_d["Data"]["doy"]).reshape(1,-1)

#define the models, lose function, optimizer,scheduler
model=CTA(input_dim=cfg_d["Model"]["num_feature"],num_classes=cfg_d["Model"]["num_classes"])
model.load_state_dict(torch.load(cfg_d["File"]["best_model_path"]))
model.cuda()
num_pred=len(img_list)
count=len(os.listdir(cfg_d["File"]["save_pred_file"]))
start=time.time()
for file in tqdm(img_list):
  if os.path.exists(os.path.join(cfg_d["File"]["save_pred_file"], file)):
    print("pass")
    continue
  else:
#    print("start")
    image,trans_2,proj_2=read_ENVI(os.path.join(cfg_d["File"]["image_path"], file))

    # image=image[:,:,0:34]
    height,width,bands=image.shape
    image_norm=img_norm_2(image, np.asarray(cfg_d["Data"]["mean"]), np.asarray(cfg_d["Data"]["std"]))
      #strench
    image_norm=image_norm.reshape((-1,bands))
    data=stretch_data(image_norm, cfg_d["Model"]["num_seq"], cfg_d["Model"]["num_feature"])

      #predict
    result=predict_batch(model, data, doy, cfg_d["Model"]["batch_size"], cfg_d["Model"]["num_classes"])
    pred = result["label_map"]
    pred=pred.reshape((height,width))


    score=result["score_map"]
    score=score.reshape((height, width, cfg_d["Model"]["num_classes"]))

    write_ENVI(os.path.join(cfg_d["File"]["save_score_file"], file), score, trans_2, proj_2)
    write_ENVI(os.path.join(cfg_d["File"]["save_pred_file"], file), pred, trans_2, proj_2)

    count += 1
#    print("%d image proceed finished!" % (count))
    del image,data,result,pred,score

end=time.time()
format_timedelta(start,end)
print("Successfully prediction！")


