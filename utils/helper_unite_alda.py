# -*- ecoding: utf-8 -*-
# @enviroment: pytorch 1.6.0 CUDA 9.0
# @Author: LeiLei leilei912@whu.edu.cn
import torch,os
import torch.nn.functional as F
from torch.utils.data import  DataLoader,Dataset
from utils.metrics import over_kappa_pro_user_f1_con
from utils.io_func import save_to_pth
from tqdm import tqdm
import time
from utils.time import format_timedelta
import torch.nn as nn
import numpy as np
import math
from utils.img_tool import Patch,padWithZeros
import loss.alda_loss as loss
#prepare the dataloader



class pixeldataset_singletile(Dataset):
    def __init__(self,data_path,label_path,seq_len,num_fea,doy):
        super(pixeldataset_singletile,self).__init__()
        self.train_x,self.train_y=self.load_data(data_path,label_path,seq_len,num_fea)
        self.doy=np.asarray(doy)
    def load_data(self,data_path,label_path,seq_len,num_fea,):
        train_x = np.load(data_path)
        train_y =np.load(label_path)
        train_x = train_x.squeeze()
        train_x = train_x.reshape((-1, seq_len, num_fea))
        train_y = train_y.reshape((-1, 1)).astype(np.int32)
        train_y = self.process_label(train_y)
        return train_x,train_y

    def __getitem__(self, index):
        return torch.from_numpy(self.train_x[index]).float(),torch.from_numpy(self.train_y[index]).long(),torch.from_numpy(self.doy)

    def __len__(self):
        return self.train_x.shape[0]

    def process_label(self,label):
        num_classes=np.unique(label)
        if num_classes[0]!=0:
            label_new=label-1
        else:
            label_new=label
        return label_new

def make_dataloader(data_path,label_path,seq_len,num_fea,doy,batch_size):
    dataset=pixeldataset_singletile(data_path,label_path,seq_len,num_fea,doy)
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    return data_loader
def cycle(iterable):  # Don't use itertools.cycle, as it repeats the same shuffle
    while True:
        for x in iterable:
            yield x

@torch.no_grad()
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return preds.eq(targets).float().mean().item()

#validate
def _eval_epoch(model,criterion,data_loader,batch_size,device,mode,category):
    label_category = np.arange(len(category))
    model.eval()
    with torch.no_grad():
        # attn_batch_list = []
        y_pred_batch = torch.zeros(len(data_loader.dataset)).to(device)
        y_true_batch = torch.zeros(len(data_loader.dataset)).to(device)
        losses = 0
        # correct = 0
        for batch_id, data in enumerate(data_loader):
            x, y, doy = data
            x = x.to(device)
            doy = doy.to(device)
            y_true = y.to(device)

            outputs = model(x,doy)
            y_true.squeeze_()

            loss = criterion(outputs, y_true)
            losses += loss.item()
            y_pred = torch.max(outputs, dim=1)[1]
            # correct += (y_pred == y_true).sum().item()
            # attn_batch_list.append(attn_batch)
            y_pred_batch[batch_id * batch_size:min((batch_id + 1) * batch_size, len(data_loader.dataset))] = y_pred
            y_true_batch[batch_id * batch_size:min((batch_id + 1) * batch_size, len(data_loader.dataset))] = y_true
        running_loss = losses / len(data_loader.dataset)
        # acc = correct / len(data_loader.dataset)
        # attn = torch.cat(attn_batch_list, dim=0).cpu().numpy()

        real = y_true_batch.detach().to("cpu").numpy()
        result = y_pred_batch.detach().to("cpu").numpy()
        o, k, recall, precision, f1, C=over_kappa_pro_user_f1_con(real,result,label_category)
        metric={"loss":running_loss,"overall":o,"kappa":k,"recall":recall,"precision":precision,"f1":f1,"Confusion matrix":C}
        print("eval_loss on "+mode,running_loss)
    return metric

#train the models
def train_model(seed, cfg_m,model,ad_net, model_start_path, epochs, optimizer, lr_scheduler, criterion, device, train_loader_source,
                train_loader_target, test_loader_target, steps_per_epoch,batch_size, Summary, model_path, category):
    # train the models on train dataset, evaluate it on the validate dataset
    # save the best models
    torch.manual_seed(seed)
    best_epoch = 0


    if len(model_start_path) > 0:
        model.load_state_dict(torch.load(model_start_path), strict=False)

    start = time.time()
    model.to(device)
    model.train()

    source_iter, target_iter = iter(cycle(train_loader_source)), iter(cycle(train_loader_target))


    for i in tqdm(range(1, 1 + epochs)):

        losses = AverageMeter()
        class_accs = AverageMeter()

        progress_bar = tqdm(range(steps_per_epoch), desc=f'cta_cropup Epoch {i}/{epochs}')
        for _ in progress_bar:
            data_source, data_target = next(source_iter), next(target_iter)

            x_source, y_source, doy_source= data_source
            x_target, y_target, doy_target= data_target

            if torch.cuda.is_available():
                x_source, y_source, doy_source = x_source.to(device), y_source.to(device), doy_source.to(device)
                x_target, y_target, doy_target = x_target.to(device), y_target.to(device), doy_target.to(device)
                y_source.squeeze_()
                y_target.squeeze_()

            y_s, f_s = model(x_source, doy_source, return_feats=True)
            y_t, f_t = model(x_target, doy_target, return_feats=True)

            features = torch.cat((f_s, f_t), dim=0)
            outputs = torch.cat((y_s, y_t), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            ad_out = ad_net(features)
            adv_loss, reg_loss, correct_loss = loss.ALDA_loss(ad_out, y_source,
                                                              softmax_out, threshold=cfg_m["transfer_loss"]["pseudo_threshold"])
            adv_weight = cfg_m["transfer_loss"]["trade_off"]
            trade_off = calc_coeff(i, high=1.0)

            transfer_loss = adv_weight * adv_loss + adv_weight * trade_off * correct_loss
            for param in model.parameters():
                param.requires_grad = False
            reg_loss.backward(retain_graph=True)
            for param in model.parameters():
                param.requires_grad = True

            cls_loss = criterion(y_s, y_source)

            total_loss = cls_loss + transfer_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()


            losses.update(total_loss.item(), batch_size)
            class_accs.update(accuracy(y_s, y_source), batch_size)

        progress_bar.close()

        Summary.add_scalar("train_loss", losses.avg, i)
        Summary.add_scalar("train_acc", class_accs.avg, i)

        if i == epochs:
            save_to_pth(model, path=os.path.join(model_path, "epoch_" + str(i) + ".pth"))
            best_epoch = i

    best_model_path = os.path.join(model_path, "epoch_" + str(best_epoch) + ".pth")
    model.load_state_dict(torch.load(best_model_path), strict=False)
    model.to(device)
    test_metric = _eval_epoch(model, criterion, test_loader_target, batch_size, device, "test", category)
    end = time.time()
    format_timedelta(start, end)
    print(test_metric)
    return test_metric


def test_model(model, best_model_path, device, criterion, test_loader, batch_size,category):
    model.load_state_dict(torch.load(best_model_path), strict=False)
    # torch.nn.DataParallel(models, device_ids=device_ids)
    model.to(device)
    test_metric = _eval_epoch(model, criterion, test_loader, batch_size, device, "test",category)
    print(test_metric)
    return test_metric


#predict
def predict_batch(model,data,batch_size,num_classes):
    bach_len=math.ceil(data.shape[0]//batch_size)
    data_lenth=data.shape[0]
    prob=np.zeros((data.shape[0],int(num_classes)))
    # prob = np.zeros(data.shape[0])
    classes = np.zeros(data.shape[0])
    result={}
    with torch.no_grad():
        for i in range(bach_len):
            begin=i*batch_size
            end=min(begin+batch_size,data_lenth)
            mini_batch=torch.from_numpy(data[begin:end,:,:]).float().cuda()
            out=model(mini_batch)
            out_class = torch.max(F.softmax(out,dim=1), dim=1)[1]
            # out_prob= torch.max(F.softmax(out,dim=1), dim=1)[0]
            prob[begin:end,:]=F.softmax(out,dim=1).cpu().detach().numpy()
            classes[begin:end]=out_class.cpu().detach().numpy()
    result["label_map"]=classes
    result["score_map"] = prob
    return result

#predict
def predict_patch(model,img,patchsize,mode,num_class):
    height=img.shape[0]
    width=img.shape[1]
    bands = img.shape[2]
    outputs = np.zeros((height, width))
    score = np.zeros((height, width,num_class))
    img = padWithZeros(img, patchsize// 2)
    num_pre=0
    model.eval()
    if mode=='3D':
        s=[1,1,bands,5,5]
    if mode=='2D':
        s=[1,bands,5,5]
    with torch.no_grad():
        for i in range(height):
            for j in range(width):
                if num_pre % 10000 == 0:
                    print(num_pre, (height * width), (num_pre / (height * width)))
                if img[i,j,0]==0:
                    num_pre += 1
                else:
                    num_pre += 1
                    image_patch = Patch(img, i, j,patchsize)
                    image_patch = image_patch.transpose((2, 0, 1))
                    image_patch = torch.from_numpy(image_patch)
                    out_prob = model(image_patch.view(s).float().cuda())
                    out_prob = F.softmax(out_prob,dim=1)
                    pre = torch.max(out_prob, dim=1)[1]
                    outputs[i, j] = pre.cpu().numpy()
                    score[i,j,:]=out_prob.cpu().numpy()
    outputs=outputs+1
    return outputs,score

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



