# -*- ecoding: utf-8 -*-
# @enviroment: pytorch 1.6.0 CUDA 9.0
# @Author: LeiLei
import copy

#include normalization,dropnan,
import numpy as np
import math
import os
import tifffile
from osgeo import gdal
def img_norm_std(img):
    size = img.ndim
    row = 0
    col = 0
    if size == 3:
        row = img.shape[0]
        col = img.shape[1]
        num = row * col
    else:
        num = img.shape[0]
    data = img.reshape((-1, img.shape[-1]))
    newdata = np.zeros((num, img.shape[-1]))
    for i in range(img.shape[-1]):
        newdata[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()
    if size == 3:
        newdata = newdata.reshape((row, col, img.shape[-1]))
    return newdata

def batch_norm_mean_std(img):

    batch_size=img.shape[0]
    bands=img.shape[-1]
    mean=np.zeros(bands)
    std = np.zeros(bands)
    for i in range(batch_size):
        img_batch_i=img[i,:,:,:]
        mean_i=img_batch_i.reshape((-1, bands)).mean(axis=0)
        std_i=img_batch_i.reshape((-1, bands)).std(axis=0)
        mean=mean+mean_i
        std=std+std_i
    mean=mean/batch_size
    std=std/batch_size
    shape = [1] * img.ndim
    shape[-1] = -1
    nor_img=(img- mean.reshape(shape)) / std.reshape(shape)
    return nor_img

#
# def mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
#     """
#     Args:
#         image: 3-D array of shape [height, width, channel]
#         mean:  a list or tuple
#         std: a list or tuple
#     Returns:
#     """
#     if isinstance(image, np.ndarray):
#         return _np_mean_std_normalize(image, mean, std)
#     elif isinstance(image, torch.Tensor):
#         return _th_mean_std_normalize(image, mean, std)
#     else:
#         raise ValueError('The type {} is not support'.format(type(image)))
#
# def _th_mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
#     """ this version faster than torchvision.transforms.functional.normalize
#     Args:
#         image: 3-D or 4-D array of shape [batch (optional) , height, width, channel]
#         mean:  a list or tuple or ndarray
#         std: a list or tuple or ndarray
#     Returns:
#     """
#     shape = [1] * image.dim()
#     shape[-1] = -1
#     mean = torch.tensor(mean, requires_grad=False).reshape(*shape)
#     std = torch.tensor(std, requires_grad=False).reshape(*shape)
#
#     return image.sub(mean).div(std)
#
# def _np_mean_std_normalize(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
#     """
#     Args:
#         image: 3-D array of shape [height, width, channel]
#         mean:  a list or tuple or ndarray
#         std: a list or tuple or ndarray
#     Returns:
#     """
#     if not isinstance(mean, np.ndarray):
#         mean = np.array(mean, np.float32)
#     if not isinstance(std, np.ndarray):
#         std = np.array(std, np.float32)
#     shape = [1] * image.ndim
#     shape[-1] = -1
#     return (image - mean.reshape(shape)) / std.reshape(shape)

def stretch_data(data,num_seq=12,num_feature=2):
    #input data:[num,(num_feature)*num_seq]
    num=data.shape[0]
    new=data.reshape((num,num_seq,num_feature))
    return new

def img_norm_maxmin(img):
    size = img.ndim
    if size == 3:
        row = img.shape[0]
        col = img.shape[1]
        bands=img.shape[2]
        num = row * col
    else:
        num = img.shape[0]
        bands=img.shape[1]
    data = img.reshape((-1, img.shape[-1]))
    newdata = np.zeros((num, bands))
    for i in range(bands):
        min = np.nanmin(data[:, i])
        max = np.nanmax(data[:, i])
        print("i:{},min:{:.2f},max:{:.2f}".format(i, min, max))
        newdata[:, i] = (data[:, i] - min) / (max - min)
    if size == 3:
        newdata = newdata.reshape((row, col, bands))
    return newdata


def get_nan_mask(image):
    row,col,bands = image.shape
    image=image.reshape((-1, bands))
    mask=np.ones(image.shape)
    zero=np.argwhere((image[:,0]==0)&(image[:,1]==0)&(image[:,2]==0)&(image[:,3]==0)&
                     (image[:,4]==0)&(image[:,5]==0)&(image[:,6]==0)&(image[:,7]==0)&
                     (image[:,8]==0)&(image[:,9]==0)&(image[:,10]==0)&(image[:,11]==0)&
                     (image[:,12]==0)&(image[:,13]==0)&(image[:,14]==0)&(image[:,15]==0))
    mask[zero,:]=0 #将全是0的位置变为1
    mask = mask > 0
    return mask,zero


def img_norm_mean_std(img,mean,std):
    img=(img-mean)/std
    return img

def img_norm_2(image,mean,std):
    row, col, bands = image.shape
    image = image.reshape((-1, bands))
    shape = [1] * image.ndim
    shape[-1] = -1
    nor_img = (image - mean.reshape(shape)) / std.reshape(shape)
    nor_img=nor_img.reshape((row,col,bands))
    return nor_img

def get_mean_std_2(image):
    row, col, bands = image.shape
    image = image.reshape((-1, bands))
    mean=image.mean(axis=0)
    std=image.std(axis=0)
    return mean,std

def drop_outliner(img,condition="nan"):
    size=img.ndim
    if size == 3:
        row = img.shape[0]
        col = img.shape[1]
        num = row * col
    img = img.reshape((-1, img.shape[-1]))
    if isinstance(condition,str):
        n_place=np.isnan(img)
        n_place=np.argwhere(n_place==True)
    else:
        n_place=np.argwhere(img==condition)
    for i in range(n_place.shape[0]):
        img[n_place[i][0],n_place[i][1]]=0
    img=img.reshape((row, col, img.shape[-1]))
    return img

# crop the image
def img_crop(img,label,patch_height,patch_width,overlap,savepath_img,savepath_label):
    X_height,X_width,_=img.shape
    # the dimension change with the number of band
    stride_width=math.ceil((1-overlap)*patch_width)
    stride_height=math.ceil((1-overlap)*patch_height)
    num_height=math.ceil((X_height-patch_height)/stride_height+1)
    num_width=math.ceil((X_width-patch_width)/stride_width+1)
    count=1
    for i in range(num_height):
        if i<num_height-1:
            start_height=int(i*stride_height)
        else:
            start_height=int(X_height-patch_height)
        for j in range(num_width):
            if j < num_width - 1:
                start_width = int(i * stride_width)
            else:
                start_width = int(X_width - patch_width)

            subimage=img[start_height:start_height+int(patch_height),
                         start_width:start_width+int(patch_width),:]
            sublabel=label[start_height:start_height+int(patch_height),
                         start_width:start_width+int(patch_width)]

            # np.save(os.path.join(savepath_img,"img_"+str(count)+".npy"),subimage)
            # np.save(os.path.join(savepath_label, "label_" + str(count) + ".npy"), sublabel)

            tifffile.imwrite(os.path.join(savepath_img,"img_"+str(count)+".tif"),subimage)
            tifffile.imwrite(os.path.join(savepath_label, "label_" + str(count) + ".tif"), sublabel)
            count+=1

def get_extent(single_image_path):
    ds=gdal.Open(single_image_path)
    gt=ds.GetGeoTransform()
    # [经度，经度方向的分辨率，0（指北为0）,纬度，纬度方向的分辨率，0（指北为0）]

    return (gt[0],gt[3],gt[0]+gt[1]*ds.RasterXSize,gt[3]+gt[5]*ds.RasterYSize)

def mosaic(image_dir,save_dir):

    image_list=os.listdir(image_dir)
    min_x,max_y,max_x,min_y=get_extent(os.path.join(image_dir,image_list[0]))
    for file in image_list[1:]:
        # print(file)
        try:
            minx,maxy,maxx,miny=get_extent(os.path.join(image_dir,file))
        except:
            print(file)
        min_x=min(min_x,minx)
        max_y=max(max_y,maxy)
        max_x=max(max_x,maxx)
        min_y=min(min_y,miny)

    in_ds=gdal.Open(os.path.join(image_dir,image_list[0]))
    gt=in_ds.GetGeoTransform()
    rows=math.ceil((max_y-min_y)/-gt[5])
    columns=math.ceil((max_x-min_x)/gt[1])


    driver=gdal.GetDriverByName("GTiff")
    out_ds=driver.Create(save_dir, columns, rows)
    out_ds.SetProjection(in_ds.GetProjection()) #设置投影
    out_bands=out_ds.GetRasterBand(1)

    gt=list(in_ds.GetGeoTransform())
    gt[0],gt[3]=min_x,max_y
    out_ds.SetGeoTransform(gt) #设置坐标

    for file in image_list:
        in_ds=gdal.Open(os.path.join(image_dir,file))
        trans=gdal.Transformer(in_ds,out_ds,[])
        success,xyz=trans.TransformPoint(False,0,0)
        x,y,z=map(int,xyz)
        try:
            data = in_ds.GetRasterBand(1).ReadAsArray()
            data_b = out_bands.ReadAsArray(x, y, in_ds.RasterXSize, in_ds.RasterYSize)
            # if data_b.shape==3:
            #     data_b=data_b.transpose((1,2,0))
            #     data_b=data_b[:,:,3]
            mask=data_b==0  #没有值的
            # mask_repeat=data_b!=0
            # data_b[mask_repeat]=int((data_b[mask_repeat]+data[mask_repeat])/2)
            data_b[mask]=data[mask]
            out_bands.WriteArray(data_b,x,y)
        except:
            print("write error")
    del in_ds,out_ds,out_bands

def mosaic_v2(image_dir,save_dir,start_zero=False,reduce_class=0):

    image_list=os.listdir(image_dir)
    min_x,max_y,max_x,min_y=get_extent(os.path.join(image_dir,image_list[0]))
    for file in image_list[1:]:
        # print(file)
        try:
            minx,maxy,maxx,miny=get_extent(os.path.join(image_dir,file))
        except:
            print(file)
        min_x=min(min_x,minx)
        max_y=max(max_y,maxy)
        max_x=max(max_x,maxx)
        min_y=min(min_y,miny)

    in_ds=gdal.Open(os.path.join(image_dir,image_list[0]))
    gt=in_ds.GetGeoTransform()
    rows=math.ceil((max_y-min_y)/-gt[5])
    columns=math.ceil((max_x-min_x)/gt[1])


    driver=gdal.GetDriverByName("GTiff")
    out_ds=driver.Create(save_dir, columns, rows)
    out_ds.SetProjection(in_ds.GetProjection()) #设置投影
    out_bands=out_ds.GetRasterBand(1)

    gt=list(in_ds.GetGeoTransform())
    gt[0],gt[3]=min_x,max_y
    out_ds.SetGeoTransform(gt) #设置坐标

    for file in image_list:
        in_ds=gdal.Open(os.path.join(image_dir,file))
        trans=gdal.Transformer(in_ds,out_ds,[])
        success,xyz=trans.TransformPoint(False,0,0)
        x,y,z=map(int,xyz)
        try:
            data_new = in_ds.GetRasterBand(1).ReadAsArray()

            data_old = out_bands.ReadAsArray(x, y, in_ds.RasterXSize, in_ds.RasterYSize)
            #
            # if data_b.shape==3:
            #     data_b=data_b.transpose((1,2,0))
            #     data_b=data_b[:,:,2]

            # mask_repeat=data_b!=0
            # data_b[mask_repeat]=int((data_b[mask_repeat]+data[mask_repeat])/2)

            # 没有值的
            # mask_h = data_patch == 0
            # data_patch[mask_h]=data_pre[mask_h]

            mask_repeat=(data_old!=0)&(data_old!=reduce_class)&(data_new==reduce_class)
            data_new[mask_repeat]=data_old[mask_repeat]
            if start_zero:
                data_new = data_new+1
            out_bands.WriteArray(data_new,x,y)
        except:
            print("write error")
    del in_ds,out_ds,out_bands



def split_train_val_data(ratio,data,label):
    num=data.shape[0]
    train_split=int(num*ratio)
    train_data=data[:train_split]
    train_label=label[:train_split]

    test_data=data[train_split:]
    test_label=label[train_split:]
    return train_data,train_label,test_data,test_label

def split_train_val_test(ratio_list:list,data,label):
    num=data.shape[0]
    train_split=int((ratio_list[0]/sum(ratio_list))*num)
    val_split=int(((ratio_list[0]+ratio_list[1])/sum(ratio_list))*num)

    train_data=data[:train_split]
    train_label=label[:train_split]
    
    val_data=data[train_split:val_split]
    val_label=label[train_split:val_split]

    test_data=data[val_split:]
    test_label=label[val_split:]

    return train_data,train_label,val_data,val_label,test_data,test_label

def get_need_band(band_index,num_seq):
    for i in range(num_seq):
        select_index_now=band_index+i*12
        select_index=np.concatenate([select_index,select_index_now],axis=0) if i>0 else select_index_now
        select_index=list(select_index)
    return select_index


def Patch(data, height_index, width_index,PATCH_SIZE):
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
# class crop_merge(object):
#     def __init__(self,img,sub_img_path,patch_height,patch_width,overlap=0):
#         super(crop_merge, self).__init__()
#         self.X_height=img.shape[0]
#         self.X_width = img.shape[1]
#         self.img=img
#         self.count = 0
#         self.sub_img_path=sub_img_path
#         self.patch_height=patch_height
#         self.patch_width = patch_width
#         self.stride_width=math.ceil((1-overlap)*patch_width)
#         self.stride_height=math.ceil((1-overlap)*patch_height)
#         self.num_height=math.ceil((self.X_height-patch_height)/self.stride_height+1)
#         self.num_width=math.ceil((self.X_width-patch_width)/self.stride_width+1)
#         self.crop_img=np.zeros(self.num_width*self.num_height)
#     def img_crop(self,img):
#         # the dimension change with the number of band
#         for i in range(self.num_height):
#             if i<self.num_height-1:
#                 start_height=int(i*self.stride_height)
#             else:
#                 start_height=int(self.X_height-self.patch_height)
#             for j in range(self.num_width):
#                 if j < self.num_width - 1:
#                     start_width = int(i * self.stride_width)
#                 else:
#                     start_width = int(self.X_width - self.patch_width)
#
#                 subimage=self.img[start_height:start_height+int(self.patch_height),
#                              start_width:start_width+int(self.patch_width),:]
#                 self.crop_result[self.count]=subimage
#                 # sublabel=label[start_height:start_height+int(patch_height),
#                 #              start_width:start_width+int(patch_width)]
#                 # np.save(os.path.join(savepath_img,"img_"+str(count)+".npy"),subimage)
#                 # np.save(os.path.join(savepath_label, "label_" + str(count) + ".npy"), sublabel)
#                 # tifffile.imwrite(os.path.join(self.sub_img_path,"img_"+str(self.count)+".tif"),subimage)
#                 self.count+=1
#
#     def image_merge(self,sub_img_rootpath,merge_img_path):
#         temp_v=np.array([])
#         temp_h=np.array([])
#         count=int(len(os.listdir(sub_img_rootpath)))
#         print(count)
#         for num in range(1,count+1):
#             img_sub=tifffile.imread(os.path.join(sub_img_rootpath,"img_"+str(num)+".tif"))
#             for i in range(0, self.num_height):
#                 for j in range(1, self.num_width + 1):
#                     if j==1:
#                         temp_h =img_sub
#                     else:
#                         temp_h = np.hstack((temp_h, img_sub))
#                 if i==0:
#                     temp_v=temp_h
#                 else:
#                     temp_v=np.vstack((temp_v,temp_h))
#         tifffile.imwrite(merge_img_path,temp_v)

class crop_merge(object):
    def __init__(self,img_height,img_width,bands,patch_height,patch_width,overlap=0):
        super(crop_merge, self).__init__()
        self.X_height=img_height
        self.X_width = img_width
        self.count = 1
        self.patch_height=patch_height
        self.patch_width = patch_width
        self.bands = bands
        self.stride_width=math.ceil((1-overlap)*patch_width)
        self.stride_height=math.ceil((1-overlap)*patch_height)
        self.num_height=math.ceil((self.X_height-patch_height)/self.stride_height+1)
        self.num_width=math.ceil((self.X_width-patch_width)/self.stride_width+1)

    def img_crop(self,img):
        crop_img = {}
        # the dimension change with the number of band
        for i in range(self.num_height):
            for j in range(self.num_width):

                start_row = int(i * self.stride_width)
                end_row = int(start_row + self.patch_height)
                start_col = int(j * self.stride_height)
                end_col = int(start_col + self.patch_width)
                if self.bands == 1:
                    crop_img[str(self.count)] = img[start_row:min(end_row, self.X_height), start_col:min(end_col, self.X_width)]
                else:
                    crop_img[str(self.count)] = img[start_row:min(end_row, self.X_height), start_col:min(end_col, self.X_width), :]
                self.count += 1
        self.count-=1
        return crop_img

    def image_merge(self,img_dict):
        temp_v = np.array([])
        temp_h = np.array([])
        for i in range(0, self.num_height):
            for j in range(1, self.num_width + 1):
                if j == 1:
                    temp_h = img_dict[str(i *  self.num_width + j)]
                else:
                    temp_h = np.hstack((temp_h,img_dict[str(i *  self.num_width + j)]))
            if i == 0:
                temp_v = temp_h
            else:
                temp_v = np.vstack((temp_v, temp_h))
        return temp_v
        # tifffile.imwrite(merge_img_path,temp_v)

def Process_nan(image,label):
    # nan_mask = np.isnan(image)
    # place = np.argwhere(image[:,:,0] == 0)
    # result=False
    # if place.any():
    #     result=True
    #     temp = np.unique(place[:, 0:2], axis=0)
    #     label[temp] = 0
    mask=image[:,:,0]==0
    label[mask]=0
    return label

def sum_pixels(label):
    num_class=np.unique(label).shape[0]
    count=np.zeros(num_class)
    for i in range(0,num_class):
        count[i]=label[np.where(label==i)].shape[0]
    return count
def sum_pixels_v2(label,num_class):
    print(label.shape)
    if (isinstance(label, np.ndarray)):
        count = np.zeros(num_class)
        for i in range(0, num_class):
            count[i] = np.where(label == i)[0].shape[0]
    if torch.is_tensor(label):
        count = torch.zeros(num_class)
        for i in range(0, num_class):
            count[i] = torch.where(label == i)[0].shape[0]
    return count
