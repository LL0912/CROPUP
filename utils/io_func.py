from osgeo import gdal
import numpy as np
import os,re
from skimage import io
from utils.img_tool import img_norm_maxmin
import pandas as pd
import torch
import tifffile
from shutil import copyfile

def _assert_suffix_match(suffix, path):
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"


def make_parent_dir(filepath):
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))



def read_ENVI(filepath) :
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, cols, rows)
    if len(data.shape) == 3:
        # print(filepath)
        data = data.transpose((1, 2, 0))
        # print(data.shape)
        data[np.isnan(data)] = 0
    return data,transform,projection



def read_tiff(path):
    data=tifffile.imread(path)
    data[np.isnan(data)] = 0
    return data

def write_ENVI(filepath,img,transform,projection):
    # img:[row,col] or [bands,row,col]
    if 'int8' in img.dtype.name:  # 判断栅格数据的数据类型
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")
    if len(img.shape) == 2:
        row, col = img.shape
        dataset = driver.Create(filepath, col, row, 1, datatype)
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(projection)
        band1 = dataset.GetRasterBand(1)
        band1.WriteArray(img)
    else:
        row, col, bands = img.shape
        dataset = driver.Create(filepath, col, row, bands, datatype)
        dataset.SetGeoTransform(transform)
        dataset.SetProjection(projection)
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(img[:,:,i])

def geo_to_tif(geo_file_path,file_path,save_path):
    _,trans,proj=read_ENVI(geo_file_path)
    image,_,_=read_ENVI(file_path)
    write_ENVI(save_path,image,trans,proj)



def tif_write(filename, im_data, im_proj, im_geotrans):

    if 'int8' in im_data.dtype.name: # 判断栅格数据的数据类型
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape # 多维或1.2维
# 创建文件
    driver = gdal.GetDriverByName("GTiff") # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans) # 写入仿射变换参数
    dataset.SetProjection(im_proj) # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data) # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def read_ENVI_ensemble(file_path,mode):
    img=[]
    tran={}
    proj={}
    if os.path.isdir(file_path):
        fileList = os.listdir(file_path)
        for subfile in fileList:
            file_path_i=os.path.join(file_path,subfile)
            img_i=[]
            if mode=="ENVI":
                img_i,tran_i,proj_i=read_ENVI(file_path_i)
                tran[subfile] = tran_i
                proj[subfile] = proj_i
            if mode=="tif":
                img_i = read_tiff(file_path_i)
            img.append(img_i)
    img=np.asarray(img)
    return img


def write_dict_to_summary(Summary,dict,epoch):
    for key,value in dict.items():
        if type(value) is np.ndarray:
            pass
        else:
            Summary.add_scalar(key,value,epoch)
    return Summary


def save_dict_to_excel(mydict, path,header):
    #save dict to multi-sheet
    _assert_suffix_match("xlsx", path)
    make_parent_dir(path)
    write = pd.ExcelWriter(path)
    for key,value in mydict.items():
        if type(value) is np.ndarray and len(value.shape)>2:
            pass
        else:
            if isinstance(value,float):
                value=np.asarray([value]).reshape((-1,1))
            pd_i=pd.DataFrame(value)
            pd_i.to_excel(write,sheet_name=key,header=header[key],index=False)
            write. _save()
        # print(key," finished")
    write._save()


def save_to_csv(data, path, header=None, index=None):
    _assert_suffix_match("csv", path)
    make_parent_dir(path)
    pd.DataFrame(data).to_csv(path, header=header, index=index)
    print("[INFO] Save as csv: '{}'".format(path))

def load_from_csv(path, header=None, index_col=None):
    return pd.read_csv(path, header=header, index_col=index_col)


def save_to_pth(data, path, model=True):
    _assert_suffix_match("pth", path)
    make_parent_dir(path)
    if model:
        if hasattr(data, "module"):
            data = data.module.state_dict()
        else:
            data = data.state_dict()
    torch.save(data, path)
    print("[INFO] Save as pth: '{}'".format(path))

def sava_to_path_DP(data,path):
    _assert_suffix_match("pth", path)
    make_parent_dir(path)
    torch.save(data.module.state_dict(), path)
    print("[INFO] Save as pth: '{}'".format(path))


def load_from_pth(path):
    return torch.load(path)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parent_dir_list(*args):
    for file in args:
        make_parent_dir(file)

def Copyfile(source_path,target_path):
    copyfile(source_path,target_path)
    print("Recording Finished!")

def save_multi_dict_to_excel(mydict, path):
    #save dict to multi-sheet
    _assert_suffix_match("xlsx", path)
    make_parent_dir(path)
    write = pd.ExcelWriter(path)
    for key,value in mydict.items():#key:sheet_name
        sheet_name=key
        s_col=0
        for k,v in value.items():
            if type(v) is np.ndarray and len(v.shape) > 2:
                pass
            else:
                if isinstance(v, float):
                    v = np.asarray([v]).reshape((-1, 1))
                if isinstance(v, np.ndarray):
                    v = v.reshape((-1, 1))
                if isinstance(v, list):
                    v = np.asarray(v).reshape((-1, 1))
                pd_i=pd.DataFrame(v)
                pd_i.to_excel(write,sheet_name=sheet_name,header=[str(k)],index=False,startcol=s_col) ##列名必须是list,想要不覆盖之前的结果必须加上startcol或者startrow
                s_col=s_col+1
                write.save()

        # print(key," finished")
    write.save()

def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files

def read_process(record_file):
    process_list=[]
    with open(record_file, 'r') as f:  # 由于我使用的pycharm已经设置完了路径，因此我直接写了文件名
        line=f.readline()
        while line:
            process_list.append(line.replace("\n",""))
            line = f.readline()
    return process_list
def record_process(list,record_file):
    with open(record_file, 'a', encoding='utf-8') as f:
        for i in list:  # content into txt
            f.writelines(i + '\n')
        f.close()
def save_dict_to_excel_whole(mydict, path):
    #save dict to multi-sheet
    _assert_suffix_match("xlsx", path)
    make_parent_dir(path)
    write = pd.ExcelWriter(path)
    key=np.asarray(mydict.keys()).reshape(-1,1)
    value=np.asarray(mydict.values()).reshape(-1,1)
    all_data=np.concatenate((key,value),axis=1)
    pd_i=pd.DataFrame(all_data)
    pd_i.to_excel(write,index=False)
    write.save()
# if __name__ == "__main__":
#     rootpath="/home/ll20/data/2021hubei_SAR_1/hubei"
#     file=list_all_files(rootpath)
#     print(file)
#     acc=np.array(0.8)
#     kappa = np.array([[0.3], [0.5]])
#     recall=np.array([[0.8,0.9,0.6,0.7],[0.8,0.9,0.6,0.7]])
#     precision = np.array([[0.8, 0.5, 0.6, 0.7], [0.8, 0.8, 0.9, 0.7]])
#     c = np.array([[[110, 10, 6, 7], [8, 118, 9, 7],[8, 5, 126, 7], [8, 8, 9, 217]],[[110, 10, 6, 7], [8, 118, 9, 7],[8, 5, 126, 7], [8, 8, 9, 217]]])
#     category=["class_1","class_2","class_3","class_4"]
#     header={"acc":["acc"],"kappa":["kappa"],"recall":category,"precision":category,"confusion":category}
#     accuracy={"acc":acc,"kappa":kappa,"recall":recall,"precision":precision,"confusion":c}
#     excel_path=r"/mnt/disk2/ll/code/myproject/result/result.xlsx"
#     # write = pd.ExcelWriter(excel_path)
#     # for key, value in accuracy.items():
#     #     pd_i = pd.DataFrame(value)
#     #     pd_i.to_excel(write, sheet_name=key, header=header, index=False)
#     #     write.save()
#     save_dict_to_excel(accuracy, excel_path,header)
