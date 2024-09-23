# -*- ecoding: utf-8 -*-
# @enviroment: pytorch 1.6.0 CUDA 9.0
# @Author: LeiLei leilei912@whu.edu.cn
import torch,os
import torch.nn.functional as F
from torch.utils.data import  DataLoader,Dataset
from utils.metrics import over_kappa_pro_user_f1_con,add_dict
from utils.io_func import write_dict_to_summary,save_to_pth
from tqdm import tqdm
import time
from utils.time import format_timedelta
import numpy as np
import math
from utils.img_tool import  Patch,padWithZeros
#prepare the dataloader



class pixeldataset_singletile(Dataset):
    def __init__(self,data_path,label_path,seq_len,num_fea,doy,start_doy=0):
        super(pixeldataset_singletile,self).__init__()
        self.train_x,self.train_y=self.load_data(data_path,label_path,seq_len,num_fea)
        self.doy=self.doy_to_positions(start_doy,np.asarray(doy))
    def load_data(self,data_path,label_path,seq_len,num_fea,):
        train_x = np.load(data_path)
        train_y =np.load(label_path)
        train_x = train_x.squeeze()
        train_x = train_x.reshape((-1, seq_len, num_fea))
        train_y = train_y.reshape((-1, 1)).astype(np.int32)
        train_y = process_label(train_y)
        return train_x,train_y

    def __getitem__(self, index):
        return torch.from_numpy(self.train_x[index]).float(),torch.from_numpy(self.train_y[index]).long(),torch.from_numpy(self.doy)

    def __len__(self):
        return self.train_x.shape[0]

    def doy_to_positions(self,start_doy,doy_npy):
        new_doy=doy_npy-start_doy

        return new_doy
def make_dataloader(data_path,label_path,seq_len,num_fea,doy,batch_size,start_doy=0):
    dataset=pixeldataset_singletile(data_path,label_path,seq_len,num_fea,doy,start_doy)
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    return data_loader


#epoch train
def _train_epoch(model,optimizer,criterion,data_loader,device):
    model.train()
    losses=0
    correct=0

    for i,data in enumerate(data_loader):
        x,y,doy=data
        if torch.cuda.is_available():
            x = x.to(device)
            doy = doy.to(device)
            y = y.to(device)
        outputs = model(x,doy)
        y.squeeze_()

        loss = criterion(outputs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses+=loss.item()
        y_pred = torch.max(outputs, dim=1)[1]
        correct += (y_pred == y).sum().item()

    loss_epoch= losses / len(data_loader)
    acc_epoch = correct / len(data_loader.dataset)
    print(loss_epoch)
    return loss_epoch,acc_epoch

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
        running_loss = losses / len(data_loader)
        # acc = correct / len(data_loader.dataset)
        # attn = torch.cat(attn_batch_list, dim=0).cpu().numpy()

        real = y_true_batch.detach().to("cpu").numpy()
        result = y_pred_batch.detach().to("cpu").numpy()
        o, k, recall, precision, f1, C=over_kappa_pro_user_f1_con(real,result,label_category)
        metric={"loss":running_loss,"overall":o,"kappa":k,"recall":recall,"precision":precision,"f1":f1,"Confusion matrix":C}
        print("eval_loss on "+mode,running_loss)
    return metric

#train the models
def train_model(seed, model, model_start_path, epochs, optimizer, scheduler, criterion, device, train_loader,
                val_loader, test_loader, batch_size, Summary, model_path, category):
    # train the models on train dataset, evaluate it on the validate dataset
    # save the best models
    torch.manual_seed(seed)
    train_metric_all = {}
    val_metric_all = {}
    best_epoch = 0
    best_kappa = 0
    least_epoch = 2
    min_threshold = 0.02
    if len(model_start_path) > 0:
        model.load_state_dict(torch.load(model_start_path), strict=False)

    start = time.time()
    # models = torch.nn.DataParallel(models, device_ids=device_ids)
    model.to(device)
    for i in tqdm(range(1, 1 + epochs)):

        train_loss, train_acc = _train_epoch(model, optimizer, criterion, train_loader, device)
        scheduler.step(train_loss)

        train_metric = _eval_epoch(model, criterion, train_loader, batch_size, device, "train", category)
        train_metric_all = add_dict(i, train_metric_all, train_metric)
        Summary.add_scalar("train_loss", train_loss, i)
        Summary.add_scalar("train_acc", train_acc, i)
        #
        if val_loader != None:
            val_metric = _eval_epoch(model, criterion, val_loader, batch_size, device, "val", category)
            print(val_metric)
            write_dict_to_summary(Summary, val_metric, i)
            val_metric_all = add_dict(i, val_metric_all, val_metric)
            #

            if ((i >= least_epoch)
                    # and (np.array(val_metric_all["loss"][i - least_epoch:i - 1]).ptp() < min_threshold)
                    and (val_metric["kappa"] > best_kappa)):
                # if val_metric["kappa"]>best_kappa (or i == epochs):
                best_kappa = val_metric["f1"][0][1]
                save_to_pth(model, path=os.path.join(model_path, "epoch_" + str(i) + ".pth"))
                best_epoch = i
        else:
            if i == epochs:
                save_to_pth(model, path=os.path.join(model_path, "epoch_" + str(i) + ".pth"))
                best_epoch = i

    best_model_path = os.path.join(model_path, "epoch_" + str(best_epoch) + ".pth")
    model.load_state_dict(torch.load(best_model_path), strict=False)
    # torch.nn.DataParallel(models, device_ids=device_ids)
    model.to(device)
    test_metric = _eval_epoch(model, criterion, test_loader, batch_size, device, "test", category)

    end = time.time()
    format_timedelta(start, end)
    print(test_metric)
    if val_loader != None:
        return train_metric_all, val_metric_all, test_metric
    else:
        return train_metric_all, test_metric


def test_model(model, best_model_path, device, criterion, test_loader, batch_size,category):
    model.load_state_dict(torch.load(best_model_path), strict=False)
    # torch.nn.DataParallel(models, device_ids=device_ids)
    model.to(device)
    test_metric = _eval_epoch(model, criterion, test_loader, batch_size, device, "test",category)
    print(test_metric)
    return test_metric


#predict
def predict_batch(model,data,doy,batch_size,num_classes):
    bach_len=math.ceil(data.shape[0]//batch_size)
    data_lenth=data.shape[0]
    prob=np.zeros((data.shape[0],int(num_classes)))
    # prob = np.zeros(data.shape[0])
    classes = np.zeros(data.shape[0])
    result={}
    model.eval()
    with torch.no_grad():
        for i in range(bach_len):
            begin=i*batch_size
            end=min(begin+batch_size,data_lenth)
            mini_batch=torch.from_numpy(data[begin:end,:,:]).float().cuda()
            doy_r=np.repeat(doy,(end-begin),axis=0)
            doy_cuda=torch.from_numpy(doy_r).cuda()
            out=model(mini_batch,doy_cuda)
            out_class = torch.max(F.softmax(out,dim=1), dim=1)[1]
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


def select_high_label(model,data,label,batch_size,believe=0.95):
    bach_len = math.ceil(data.shape[0] // batch_size)
    data_lenth = data.shape[0]
    prob = np.zeros(data_lenth)
    classes = np.zeros(data_lenth)

    for i in range(bach_len):
        begin=i*batch_size
        end=min(begin+batch_size,data_lenth)
        mini_batch=torch.from_numpy(data[begin:end,:,:]).float().cuda()
        out, _=model(mini_batch)
        out_prob = torch.max(F.softmax(out,dim=1), dim=1)[0]
        out_class = torch.max(out, dim=1)[1]
        prob[begin:end] = out_prob.cpu().detach().numpy()
        classes[begin:end] = out_class.cpu().detach().numpy()

    indices=np.argwhere((label.squeeze()==classes) & (prob>believe))
    new_data=data[indices]
    new_label=label[indices]
    print(classes.shape)
    print(label.shape)
    print(prob)
    print(classes)
    print(label)

    print("origin data length:%d"%(label.shape[0]))
    print("refined data length:%d" % (new_label.shape[0]))
    print("retain ratio: %.4f"%(new_label.shape[0]/label.shape[0]))

    return indices,new_data,new_label


def process_label(label):
    num_classes=np.unique(label)
    if num_classes[0]!=0:
        label_new=label-1
    else:
        label_new=label
    return label_new

def sum_pixels(label):
    num_class=np.unique(label).shape[0]
    count=np.zeros(num_class)
    for i in range(0,num_class):
        count[i]=label[np.where(label==i)].shape[0]
    return count

def extract_shuffle_data(data,label,i,need_count):
    idx=np.argwhere(label==i)
    np.random.shuffle(idx)
    idx=idx[0:need_count]
    label_new=label[idx].squeeze()
    data_new=data[idx,:,:].squeeze()
    return data_new,label_new

def extract_balance_sample(data,label):
    label_from_zero=process_label(label)
    count=sum_pixels(label_from_zero)
    min_num=int(np.min(count))

    for i in range(0,count.shape[0]):
        data_i,label_i=extract_shuffle_data(data,label,i,min_num)
        data_new=np.concatenate([data_new,data_i],axis=0) if i!=0 else data_i
        label_new = np.concatenate([label_new, label_i], axis=0) if i != 0 else label_i

    permutation = np.random.permutation(label_new.shape[0])
    data_new = data_new[permutation, :, :]
    label_new = label_new[permutation]
    return  data_new.squeeze(),label_new.squeeze()




