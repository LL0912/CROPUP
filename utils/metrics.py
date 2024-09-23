
from sklearn.metrics import confusion_matrix
import numpy as np

def kappa(m):
    sum_real=np.sum(m,1)
    sum_pre=np.sum(m,0)
    s=np.sum(sum_real)
    c1=0#
    c2=0
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if i==j:
                c1=c1+m[i,j]
    for i in range(m.shape[0]):
        c2=c2+sum_real[i]*sum_pre[i]
    p0=c1/s
    pe=c2/(s*s)
    kappa=(p0-pe)/(1-pe)
    over = c1 / s
    return kappa,over

def cal_recall_precision(m):
    sum_real=np.sum(m,1)
    sum_pre=np.sum(m,0)
    recall=np.zeros((1,m.shape[0]))
    precision=np.zeros((1,m.shape[0]))
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if i==j:
                recall[0,i]=m[i,j]/sum_real[i]
                precision[0,i]=m[i,j]/sum_pre[i]
    return recall,precision

#producer 召回；user 精确度
def over_kappa_pro_user_f1_con(y, prediction,labels):
    C = confusion_matrix(y, prediction,labels=labels)
    k, o = kappa(C)
    recall,precision = cal_recall_precision(C)
    f1=2/(1/recall+1/precision)
    true_positive = np.diag(C)
    false_positive = np.sum(C, 0) - true_positive
    false_negative = np.sum(C, 1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative + 1e-12)
    miou = np.nanmean(iou)
    return o,k,recall, precision,f1,C,iou,miou



def evaluate_metrics_from_cm(confusion_matrix):
    cn_acc_dict={}
    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / (total + 1e-12)

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc + 1e-12)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    #iou
    true_positive = np.diag(confusion_matrix)
    false_positive = np.sum(confusion_matrix, 0) - true_positive
    false_negative = np.sum(confusion_matrix, 1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative+ 1e-12)
    miou=np.nanmean(iou)

    cn_acc_dict["acc"]=overall_accuracy
    cn_acc_dict["kappa"] = kappa
    cn_acc_dict["precision"] = precision
    cn_acc_dict["recall"] = recall
    cn_acc_dict["f1"] = f1
    cn_acc_dict["iou"] = iou
    cn_acc_dict["miou"] = miou
    cn_acc_dict["cm"] = confusion_matrix
    return cn_acc_dict


def add_dict(epoch,dic_all,dic_i):
    if epoch==1:
        dic_all=dic_i
    else:
        for key in dic_all.keys():
            dic_all[key]=np.vstack((np.asarray(dic_all[key]),np.asarray(dic_i[key])))
    return dic_all

