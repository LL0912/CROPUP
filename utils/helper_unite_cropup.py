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
import numpy as np
import math
from utils.img_tool import  Patch,padWithZeros
import torch.nn as nn
import ot
from copy import deepcopy
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
        train_y = process_label(train_y)
        return train_x,train_y

    def __getitem__(self, index):
        return torch.from_numpy(self.train_x[index]).float(),torch.from_numpy(self.train_y[index]).long(),torch.from_numpy(self.doy),index

    def __len__(self):
        return self.train_x.shape[0]




def make_dataloader(data_path,label_path,seq_len,num_fea,doy,batch_size):
    dataset=pixeldataset_singletile(data_path,label_path,seq_len,num_fea,doy)
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    return data_loader
def cycle(iterable):  # Don't use itertools.cycle, as it repeats the same shuffle
    while True:
        for x in iterable:
            yield x



def readjust_label(real,pred,direction):
    real_copy = deepcopy(real)
    pred_copy = deepcopy(pred)
    if direction is not None:
        if direction.split("_")[0] == "IW":
            mask = (real == 3) | (real == 4) | (real == 5) | (real ==6)
            real_copy[mask] = real_copy[mask] - 1

            mask = (pred == 3) | (pred == 4) | (pred == 5) | (pred == 6)
            pred_copy[mask] = pred_copy[mask] - 1

        elif direction.split("_")[0] == "OH":
            mask_1 = (real == 3) | (real == 4)
            mask_2 = (real == 5) | (real == 6) | (real == 7)
            real_copy[mask_1] = 2
            real_copy[mask_2] = real_copy[mask_2] - 2

            mask_1 = (pred == 3) | (pred == 4)
            mask_2 = (pred == 5) | (pred == 6) | (pred == 7)
            pred_copy[mask_1] = 2
            pred_copy[mask_2] = pred_copy[mask_2] - 2

        elif direction.split("_")[0] == "ND":
            mask_1 = (real == 3) | (real == 4) | (real == 5)
            mask_2 = (real == 6) | (real == 7) | (real == 8)
            real_copy[mask_1] = 2
            real_copy[mask_2] = real_copy[mask_2] - 3

            mask_1 = (pred == 3) | (pred == 4)| (real == 5)
            mask_2 = (real == 6) | (real == 7) | (real == 8)
            pred_copy[mask_1] = 2
            pred_copy[mask_2] = pred_copy[mask_2] - 3

    return real_copy,pred_copy

@torch.no_grad()
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return preds.eq(targets).float().mean().item()

#validate
def _eval_epoch(model,criterion,data_loader,batch_size,device,mode,category,direction=None):
    label_category = np.arange(len(category))
    model.eval()
    with torch.no_grad():
        # attn_batch_list = []
        y_pred_batch = torch.zeros(len(data_loader.dataset)).to(device)
        y_true_batch = torch.zeros(len(data_loader.dataset)).to(device)
        losses = 0
        # correct = 0
        for batch_id, data in enumerate(data_loader):
            x, y, doy, _ = data
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

        if direction is not None:
            real,result=readjust_label(real, result, direction)

        o, k, recall, precision, f1, C,iou,miou=over_kappa_pro_user_f1_con(real,result,label_category)
        metric={"loss":running_loss,"overall":o,"kappa":k,"recall":recall,"precision":precision,"f1":f1,"Confusion matrix":C,"iou":iou,"miou":miou}
        print("eval_loss on "+mode,running_loss)
    return metric

#train the models
def train_model(seed, cfg_m,model, model_start_path, epochs, optimizer, lr_scheduler, criterion, device, train_loader_source,
                train_loader_target, test_loader_target,label_source, steps_per_epoch,batch_size, Summary, model_path, category):
    # train the models on train dataset, evaluate it on the validate dataset
    # save the best models
    training_loss_all=[]
    training_acc_all_source=[]
    training_acc_all_target=[]

    torch.manual_seed(seed)
    best_epoch = 0

    if len(model_start_path) > 0:
        model.load_state_dict(torch.load(model_start_path), strict=False)

    start = time.time()
    model.to(device)
    model.train()

    source_iter, target_iter = iter(cycle(train_loader_source)), iter(cycle(train_loader_target))
    y_onehot_s = torch.zeros(label_source.shape[0], len(category)).scatter_(1, torch.from_numpy(label_source).type(
        torch.LongTensor).view(-1, 1), len(category))

    for i in tqdm(range(1, 1 + epochs)):

        losses = AverageMeter()
        class_accs_source = AverageMeter()
        class_accs_target = AverageMeter()

        y = y_onehot_s
        progress_bar = tqdm(range(steps_per_epoch), desc=f'cta_cropup Epoch {i}/{epochs}')
        for _ in progress_bar:
            data_source, data_target = next(source_iter), next(target_iter)

            x_source, y_source, doy_source, index_source = data_source
            x_target, y_target, doy_target, _ = data_target

            if torch.cuda.is_available():
                x_source, y_source, doy_source = x_source.to(device), y_source.to(device), doy_source.to(device)
                x_target, y_target, doy_target = x_target.to(device), y_target.to(device), doy_target.to(device)
                y_source.squeeze_()
                y_target.squeeze_()

            y_s, f_s = model(x_source, doy_source, return_feats=True)
            y_t, f_t = model(x_target, doy_target, return_feats=True)
            pred_x_t = F.softmax(y_t, 1)
            tot_loss = criterion(y_s, y_source)

            logsoftmax = nn.LogSoftmax(dim=1).cuda()
            softmax = nn.Softmax(dim=1).cuda()

            yy = y[index_source, :]
            yy = torch.FloatTensor(yy).cuda()
            yy = torch.autograd.Variable(yy, requires_grad=True)
            last_y_var = F.softmax(yy, dim=1)
            lc = torch.mean(softmax(y_s) * (logsoftmax(y_s) - torch.log((last_y_var))))
            lo = criterion(last_y_var, y_source)
            le = - torch.mean(torch.mul(softmax(y_s), logsoftmax(y_s)))
            noisy_loss = lc + cfg_m["noisy_loss"]["alpha"] * lo + cfg_m["noisy_loss"]["beta"] * le
            y_source_label = torch.max(last_y_var, 1)[1]

            one_hot_labels_s = F.one_hot(y_source_label, num_classes=len(category)).float()
            M_embed = torch.cdist(f_s, f_t) ** 2  # term on embedded data
            M_sce = - torch.mm(one_hot_labels_s, torch.transpose(torch.log(pred_x_t), 0, 1))  # term on labels
            M = cfg_m["transfer_loss"]["eta1"] * M_embed +  cfg_m["transfer_loss"]["eta2"] * M_sce

            a, b = ot.unif(f_s.size()[0]), ot.unif(f_t.size()[0])
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), cfg_m["transfer_loss"]["epsilon"], cfg_m["transfer_loss"]["tau"])
            pi = torch.from_numpy(pi).float().cuda()  # Transport plan between minibatches
            transfer_loss = torch.sum(pi * M)
            tot_loss += transfer_loss
            tot_loss += noisy_loss

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.update(tot_loss.item(), batch_size)
            class_accs_source.update(accuracy(y_s, y_source), batch_size)
            class_accs_target.update(accuracy(y_t, y_target), batch_size)


            lambda1 = cfg_m["noisy_loss"]["lambda1"]
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1 * yy.grad.data)
            y_onehot_s[index_source, :] = yy.data.cpu()

        Summary.add_scalar("train_loss", losses.avg, i)
        Summary.add_scalar("train_acc", class_accs_source.avg, i)


        training_loss_all.append(losses.avg)
        training_acc_all_source.append(class_accs_source.avg)
        training_acc_all_target.append(class_accs_target.avg)


        if i == epochs:
            save_to_pth(model, path=os.path.join(model_path, "epoch_" + str(i) + ".pth"))
            best_epoch = i

    best_model_path = os.path.join(model_path, "epoch_" + str(best_epoch) + ".pth")
    model.load_state_dict(torch.load(best_model_path), strict=False)
    model.to(device)
    test_metric = _eval_epoch(model, criterion, test_loader_target, batch_size, device, "test", category)
    end = time.time()
    format_timedelta(start, end)
    train_metric_all = {"loss": training_loss_all, "accuracy_source": training_acc_all_source,"accuracy_target": training_acc_all_target}
    print(test_metric)
    return train_metric_all,test_metric


def test_model(model, best_model_path, device, criterion, test_loader, batch_size,category,direction=None):
    model.load_state_dict(torch.load(best_model_path), strict=False)
    # torch.nn.DataParallel(models, device_ids=device_ids)
    model.to(device)
    test_metric = _eval_epoch(model, criterion, test_loader, batch_size, device, "test",category,direction=direction)
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


