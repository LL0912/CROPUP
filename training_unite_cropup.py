# -*- ecoding: utf-8 -*-
"""
@enviroment: pytorch 1.6.0 CUDA 9.0
@Date:2021/10/19/22:39
@File:training_transformer.py
@Author: LeiLei leilei912@whu.edu.cn
"""
import numpy as np
import torch,os
import torch.nn as nn
import sys
sys.path.append("..")
import yaml
from torch.utils.tensorboard import SummaryWriter
from models.UNITE import unite
from utils.helper_unite_cropup import *
from utils.io_func import save_dict_to_excel
from utils.io_func import make_parent_dir_list
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.sgd import SGD

os.environ['CUDA_VISIBLE_DEVICES']='2'
# device_ids = [ 0,1,2,3]
import sys
module_path = os.path.abspath(os.path.join(".."))

if module_path not in sys.path:
    sys.path.append(module_path)
with open("./config/model/unite_cropup.yaml") as f:
    cfg_m=yaml.safe_load(f)

with open("./config/dataset/14UPU_S2_2020.yaml") as f:
    cfg_d=yaml.safe_load(f)

with open("./config/dataset/14UPU_S2_2021.yaml") as f:
    cfg_t=yaml.safe_load(f)
#constant variable

EPOCH=cfg_d["Model"]["epoch"]
batch_size=cfg_d["Model"]["batch_size"]
steps_per_epoch=cfg_d["Model"]["steps_per_epoch"]

device=torch.device("cuda:0")
seed=233
#models save path
model_start_path=cfg_d["File"]["model_start_path"]
summary_path=os.path.join(cfg_d["File"]["summary_path"],cfg_d["File"]["file_name"])
model_save_path=os.path.join(cfg_d["File"]["model_save_path"],cfg_d["File"]["file_name"])
train_excel_path=os.path.join(cfg_d["File"]["train_excel_path"],"train_"+cfg_d["File"]["file_name"]+".xlsx")
val_excel_path=os.path.join(cfg_d["File"]["val_excel_path"],"val_"+cfg_d["File"]["file_name"]+".xlsx")
test_excel_path=os.path.join(cfg_d["File"]["test_excel_path"],"test_"+cfg_d["File"]["file_name"]+".xlsx")

make_parent_dir_list(summary_path+"/",model_save_path+"/",os.path.dirname(train_excel_path),os.path.dirname(val_excel_path),os.path.dirname(test_excel_path))

summary=SummaryWriter(summary_path)
# category=["paddy rice","soybean","maize","peanut","water","impervious","leisure land","other vegetable"]
category=cfg_d["Class_name"]
header={"loss":["loss"],"overall":["overall"],"kappa":["kappa"],"recall":category,"precision":category,"f1":category,"Confusion matrix":category,"iou":category,"miou":["miou"]}
header_train={"loss":["loss"],"accuracy_source":["accuracy_source"],"accuracy_target":["accuracy_target"]}
#import the training data

train_loader_source=make_dataloader(cfg_d["File"]["train_npy_data_source"],cfg_d["File"]["train_npy_label_source"],cfg_d["Model"]["num_seq"],cfg_d["Model"]["num_feature"],cfg_d["Data"]["doy"],batch_size)
train_loader_target=make_dataloader(cfg_d["File"]["train_npy_data_target"],cfg_d["File"]["train_npy_label_target"],cfg_t["Model"]["num_seq"],cfg_t["Model"]["num_feature"],cfg_t["Data"]["doy"],batch_size)
test_loader_target=make_dataloader(cfg_d["File"]["test_npy_data_target"],cfg_d["File"]["test_npy_label_target"],cfg_t["Model"]["num_seq"],cfg_t["Model"]["num_feature"],cfg_t["Data"]["doy"],batch_size)
label_source=np.load(cfg_d["File"]["train_npy_label_source"])

#define the models, lose function, optimizer,scheduler
model=unite(input_dim=cfg_d["Model"]["num_feature"],num_classes=cfg_d["Model"]["num_classes"])

model.to(device)
criterion=nn.CrossEntropyLoss()
classifier_params = [
                {"params": model.channel_attention.parameters(), "lr": 0.1 * cfg_m["optimizer"]["base_lr"]},
                {"params": model.time_attention.parameters(), "lr": 0.1 * cfg_m["optimizer"]["base_lr"]},
                {"params": model.decoder.parameters(), "lr": 1.0 * cfg_m["optimizer"]["base_lr"]}]
optimizer = SGD(
    classifier_params,
    cfg_m["optimizer"]["lr"],
    momentum=0.9,
    weight_decay=cfg_m["optimizer"]["weight_decay"],
    nesterov=True,
)
lr_scheduler = LambdaLR(
    optimizer, lambda x: cfg_m["optimizer"]["lr"] * (1.0 + cfg_m["optimizer"]["lr_gamma"] * float(x)) ** (-cfg_m["optimizer"]["lr_decay"])
)

train_metric_all,test_metric_all=train_model(seed,cfg_m,model,model_start_path,EPOCH,optimizer,lr_scheduler,criterion,device,
            train_loader_source,train_loader_target,test_loader_target,label_source,steps_per_epoch,
            batch_size,summary,model_save_path,category)

save_dict_to_excel(test_metric_all,test_excel_path,header)
save_dict_to_excel(train_metric_all,train_excel_path,header_train)

