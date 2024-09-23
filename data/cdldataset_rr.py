import sys
sys.path.append("..")
import os
import numpy as np
from utils.io_func import read_ENVI,save_dict_to_excel_whole,write_ENVI,make_parent_dir
from utils.img_tool import sum_pixels_v2,get_mean_std_2,img_norm_2
import argparse
from tqdm import tqdm
import yaml
import pandas as pd
##对整个数据集的影像进行预处理
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
label_map_dict={
    0:[1],
    1:[5],
    2:[176],
    3:[141,195,190,36,28,37,143,12,27,24,142,152,23,59,42,4,21,205,60,58,68,240,53,70,13,6,29,26,31,44,22,69],
    4:[111],
    5:[61,131],
    6:[121,122,123,124]

}

def non_value_band_filter(cfg):
    file_list=os.listdir(cfg.img_root_path)
    print(len(file_list))
    count=0
    for file_name in tqdm(file_list):
        file_path = os.path.join(cfg.img_root_path, file_name)
        image,_,_=read_ENVI(file_path)

        c,h,all_band=image.shape
        band=int(all_band/cfg.num_fea)
        image = image[:, :, ::cfg.num_fea]
        image_b2=image.reshape(-1,band)

        null_mask=np.sum(image_b2!=0,axis=1)
        non_null_image=np.sum(null_mask!=0,axis=0)
        if non_null_image==c*h:
            count=count+1
            with open(cfg.reserve_path, "a") as f:
                f.write(str(file_name)+"\n")
    print(count)


def statistic_class_number(cfg):
    file_list = os.listdir(cfg.label_root_path)
    print(len(file_list))
    num_count_dict = {}
    for file_name in tqdm(file_list):
        file_path = os.path.join(cfg.label_root_path, file_name)
        image, _, _ = read_ENVI(file_path)
        label=image.reshape(-1)
        uniq_id=np.unique(label)
        for id in uniq_id:
            count_i=np.sum(label==id)
            if id in num_count_dict.keys():
                num_count_dict[id]=num_count_dict[id]+count_i
            else:
                num_count_dict[id]=count_i
    save_dict_to_excel_whole(num_count_dict,cfg.statistic_path)

def get_norm(cfg):
    bands=cfg.num_fea*cfg.num_time
    file_list = []
    with open(cfg.reserve_path) as f:
        line = f.readline().strip()
        while line:
            file_list.append(line)
            line = f.readline().strip()

    mean = np.zeros(bands)
    std = np.zeros(bands)

    for subfile in tqdm(file_list):
        file_path_i = os.path.join(cfg.img_root_path, subfile)
        image,_,_ = read_ENVI(file_path_i)
        print(image.shape)
        # image=image*(-1) #for SAR data
        shape = [1] * image.ndim
        shape[-1] = -1

        mean_i,std_i=get_mean_std_2(image)
        mean=mean+mean_i
        std=std+std_i
        del image
    mean=mean/len(file_list)
    std=std/len(file_list)
    print(mean)
    print(std)
    with open(cfg.mean_std_path, "a") as f:
        f.write("mean:" + "\n")
        np.savetxt(f, mean, delimiter=",", fmt="%.8f")
        f.write("std:" + "\n")
        np.savetxt(f, std, delimiter=",", fmt="%.8f")
    return mean,std

def extraxt_samples_product(cfg,use_norm=True,ext_year="h",add_condi="random",plot_label=False):
    with open(cfg.yaml_path) as f:
        cfg_s = yaml.safe_load(f)
        mean=np.asarray(cfg_s["Data"]["mean"])
        std = np.asarray(cfg_s["Data"]["std"])
        label_map_dict=cfg_s["Data"]["label_map_dict"]
        
    os.makedirs(cfg.data_train_path, exist_ok=True)
    os.makedirs(cfg.label_train_path, exist_ok=True)
    os.makedirs(cfg.data_test_path, exist_ok=True)
    os.makedirs(cfg.label_test_path, exist_ok=True)
    os.makedirs(cfg.data_wo_norm_path, exist_ok=True)
    os.makedirs(cfg.label_wo_norm_path, exist_ok=True)
    
    half_window=int(cfg.window_size/2)
    count_all_class_train =np.zeros(len(label_map_dict))
    count_all_class_test = np.zeros(len(label_map_dict))
    file_list = []
    print(cfg.reserve_path)
    with open(cfg.reserve_path) as f:
        line = f.readline().strip()
        while line:
            file_list.append(line)
            line = f.readline().strip()

    
    len_file_list=len(file_list)
    print(file_list)

    random.seed(233)
    random.shuffle(file_list)
    all_test_position= []
    for i,file in tqdm(enumerate(file_list)):
        if ext_year=="t_test" and i<=int(len_file_list/2):
            continue
        else:
            test_samples=[]
            test_labels=[]
            training_samples = []
            training_labels = []
            tile_name=file.split("_")[0]
            year=file.split("-")[0].split("_")[-1]

            img_path=os.path.join(cfg.img_root_path,file)
            label_path=os.path.join(cfg.label_root_path,file.replace("S2","CDL"))
            confi_path = os.path.join(cfg.confi_root_path, file.replace("S2", "confidence"))

            image, _, _ =read_ENVI(img_path)
            label, _, _ = read_ENVI(label_path)
            confi, _, _ = read_ENVI(confi_path)

            h,w,all_bands=image.shape
            if use_norm:
                image=img_norm_2(image,mean,std)
            t=int(all_bands/cfg.num_fea)
            #将label进行映射，
            label_new=np.ones_like(label)*255
            for key in label_map_dict.keys():
                all_ids=label_map_dict[key]
                for v_id in all_ids:
                    mask=label==v_id
                    label_new[mask]=int(key)
            if ext_year=="h":
                for class_i in label_map_dict.keys():
                    sample_count= 0 # 每一类的总数
                    extraxct_count = 0  # 每一类提取的数量是否满足要求
                    idxs=np.argwhere(label_new==int(class_i))
                    shuffle=np.random.permutation(len(idxs))
                    idxs=idxs[shuffle]
                    if len(idxs)<=cfg.count_class:
                        all_idx=idxs
                    else:
                        all_idx=idxs[:cfg.count_class]

                    extracted_sample = [image[p[0], p[1], :] for p in all_idx]
                    extracted_label = [label_new[p[0], p[1]] for p in all_idx]
                    training_samples.extend(extracted_sample)
                    training_labels.extend(extracted_label)

            if ext_year=="t_train":
                if i <= int(len_file_list / 2):
                    for class_i in label_map_dict.keys():
                        idxs = np.argwhere(label_new == int(class_i))
                        shuffle = np.random.permutation(len(idxs))
                        idxs = idxs[shuffle]
                        if len(idxs) <= 2*cfg.count_class:
                            all_idx = idxs
                        else:
                            all_idx = idxs[:2*cfg.count_class]

                        extracted_sample = [image[p[0], p[1], :] for p in all_idx]
                        extracted_label = [label_new[p[0], p[1]] for p in all_idx]
                        training_samples.extend(extracted_sample)
                        training_labels.extend(extracted_label)

            if ext_year=="t_test":
                if add_condi=="random" or add_condi=="confidence":
                    for class_i in label_map_dict.keys():
                        if add_condi=="random":
                            idxs = np.argwhere(label_new == int(class_i))
                        elif add_condi=="confidence":
                            idxs=np.argwhere(label_new == int(class_i)&(confi>=cfg.confi_thr))


                        shuffle = np.random.permutation(len(idxs))
                        idxs = idxs[shuffle]
                        if len(idxs) <= 2*cfg.count_class:
                            all_idx = idxs
                        else:
                            all_idx = idxs[:2*cfg.count_class]

                        extracted_sample = [image[p[0], p[1], :] for p in all_idx]
                        extracted_label = [label_new[p[0], p[1]] for p in all_idx]
                        test_samples.extend(extracted_sample)
                        test_labels.extend(extracted_label)

                        all_test_position.append(all_idx)

                elif add_condi=="window":
                    for class_i in label_map_dict.keys():
                        sample_count = 0  # 每一类的总数
                        extraxct_count = 0  # 每一类提取的数量是否满足要求
                        idxs = np.argwhere(label_new == int(class_i))
                        shuffle = np.random.permutation(len(idxs))
                        idxs = idxs[shuffle]
                        # 随机选择的点的坐标找到，开一个窗口，符合条件就加入进来，直到加到符合条件或者所有的样本都已经检查过了
                        while sample_count < len(idxs):

                            if extraxct_count < 2*cfg.count_class:
                                # 对该位置进行检查，是否在边缘
                                position = idxs[sample_count]
                                row_center, col_center = position[0], position[1]
                                row_s = row_center - half_window
                                col_s = col_center - half_window
                                row_e = row_center + half_window
                                col_e = col_center + half_window

                                if row_s >= 0 & col_s >= 0 & row_e <= h & col_e <= w:
                                    # 对该位置开一个窗口，并检查是否符合条件
                                    win_label = label_new[row_s:row_e, col_s:col_e]
                                    center_label = np.unique(win_label)
                                    confi_center = confi[row_center, col_center]
                                    if len(center_label) == 1 & (confi_center>=cfg.confi_thr):
                                        # 提取样本
                                        extracted_sample = image[row_center, col_center, :]
                                        extracted_label = label_new[row_center, col_center]

                                        test_samples.append(extracted_sample)
                                        test_labels.append(extracted_label)
                                        # 记录下标
                                        extraxct_count += 1
                                        sample_count += 1
                                        all_test_position.append([row_center, col_center])
                                    else:
                                        sample_count += 1

                                else:
                                    sample_count += 1
                            else:
                                break

            if ext_year=="h" or ext_year=="t_train":
                training_samples = np.asarray(training_samples).squeeze().reshape(-1, t, cfg.num_fea)
                training_labels = np.asarray(training_labels).squeeze().reshape(-1)
                count_all_class_train += sum_pixels_v2(training_labels, cfg.num_class)

                shuffle = np.random.permutation(training_samples.shape[0])
                training_samples = training_samples[shuffle]
                training_labels = training_labels[shuffle]
                np.save(os.path.join(cfg.data_train_path,"train_data_"+year+"_S2_"+tile_name+"_"+str(i)+".npy"),training_samples)
                np.save(os.path.join(cfg.label_train_path,"train_label_"+year+"_S2_"+tile_name+"_"+str(i)+".npy"),training_labels)
                print("label information train:", count_all_class_train)

            elif ext_year=="t_test":
                test_samples = np.asarray(test_samples).squeeze().reshape(-1, t, cfg.num_fea)
                test_labels = np.asarray(test_labels).squeeze().reshape(-1)

                shuffle = np.random.permutation(test_samples.shape[0])
                test_samples = test_samples[shuffle]
                test_labels = test_labels[shuffle]
                count_all_class_test += sum_pixels_v2(test_labels, cfg.num_class)

                np.save(os.path.join(cfg.data_test_path, "test_data_" + year + "_S2_" + tile_name + "_" + str(i) + ".npy"),
                        test_samples)
                np.save(os.path.join(cfg.label_test_path, "test_label_" + year + "_S2_" + tile_name + "_" + str(i) + ".npy"),
                        test_labels)
                print("label information test:", count_all_class_test)

        if plot_label and ext_year.split("_")[1]=="test":
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, 512)
            ax.set_ylim(0, 512)
            ax.set_facecolor((1, 1, 1, 0.6))  # 背景设置为白色，透明度60%

            # 绘制每个点及其对应的5x5红色窗口
            for point in all_test_position:
                rect = patches.Rectangle((point[0] - 2.5, point[1] - 2.5), 5, 5, linewidth=1, edgecolor='r',
                                         facecolor='r')
                ax.add_patch(rect)

            rect_border = patches.Rectangle((0, 0), 512, 512, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect_border)

            # 去掉坐标轴
            ax.axis('off')

            # 显示绘图
            plt.gca().invert_yaxis()  # 确保坐标系符合常规显示习惯

            plt.savefig(os.path.join(cfg.point_map_path,file.split(".")[0]+".png"), bbox_inches='tight')
            plt.show()



def output_tsNDVI(cfg):

    for i,file in enumerate(os.listdir(cfg.data_wo_norm_path)):
        train_data_path=os.path.join(cfg.data_wo_norm_path,file)
        train_label_path = os.path.join(cfg.label_wo_norm_path, file.replace("data","label"))
        data_i=np.load(train_data_path)
        label_i=np.load(train_label_path)
        train_data=np.concatenate((train_data,data_i),axis=0) if i>0 else data_i
        train_label=np.concatenate((train_label,label_i),axis=0) if i>0 else label_i
    print(train_data.shape)
    print(train_label.shape)

    for i in range(cfg.num_class):
        mask=train_label==i
        class_data=train_data[mask].reshape(-1, cfg.num_time, cfg.num_fea)
        class_tsNDVI=(class_data[:,:,6]-class_data[:,:,2])/(class_data[:,:,6]+class_data[:,:,2]+1e-10).reshape(-1, cfg.num_time)
        #class_tsNDVI=(class_data[:,:,3]-class_data[:,:,2])/(class_data[:,:,3]+class_data[:,:,2]+1e-10).reshape(-1, cfg.num_time) #LANDSAT-8
        mean_tsNDVI=np.mean(class_tsNDVI,axis=0).reshape(-1, cfg.num_time)
        std_tsNDVI=np.std(class_tsNDVI,axis=0).reshape(-1, cfg.num_time)
        mean_tsNDVI_all=np.concatenate((mean_tsNDVI_all,mean_tsNDVI),axis=0) if i>0 else mean_tsNDVI
        std_tsNDVI_all=np.concatenate((std_tsNDVI_all,std_tsNDVI),axis=0) if i>0 else std_tsNDVI


    write = pd.ExcelWriter(cfg.ndvi_excel_path)
    pd_i = pd.DataFrame(mean_tsNDVI_all)
    pd_i.to_excel(write, sheet_name="mean",index=False)
    write.save()

    pd_i = pd.DataFrame(std_tsNDVI_all)
    pd_i.to_excel(write, sheet_name="std", index=False)
    write.save()
    
def sample_information(cfg):
    train_file_llist=os.listdir(cfg.label_train_path)
    test_file_llist = os.listdir(cfg.label_test_path)
    count_train,count_test=np.zeros(cfg.num_class), np.zeros(cfg.num_class)
    for file in train_file_llist:
        file_path=os.path.join(cfg.label_train_path,file)
        label=np.load(file_path)
        count_train += sum_pixels_v2(label, cfg.num_class)
    for file in test_file_llist:
        file_path=os.path.join(cfg.label_test_path,file)
        label=np.load(file_path)
        count_test += sum_pixels_v2(label, cfg.num_class)
    print("train_information:",count_train)
    print("test_information:", count_test)
    
def simplify_label(cfg):
    with open(cfg.yaml_path) as f:
        cfg_s = yaml.safe_load(f)
        label_map_dict=cfg_s["label_map_dict"]
    make_parent_dir(cfg.label_simply_root_path+"/")
    file_list = os.listdir(cfg.label_root_path)
    print(len(file_list))
    for file_name in tqdm(file_list):
        file_path = os.path.join(cfg.label_root_path, file_name)
        save_file_path = os.path.join(cfg.label_simply_root_path, file_name)
        label, trans, proj = read_ENVI(file_path)
        label_new = np.ones_like(label) * 255
        for key in label_map_dict.keys():
            all_ids = label_map_dict[key]
            for v_id in all_ids:
                mask = label == v_id
                label_new[mask] = int(key)
        write_ENVI(save_file_path,label_new, trans, proj)
        
def concat_sample(cfg):
    train_file_list = os.listdir(cfg.label_train_path)
    test_file_list = os.listdir(cfg.label_test_path)
    if cfg.ext_year=="h" or cfg.ext_year.split("_")[1]=="train":
        for i,file in enumerate(train_file_list):
            file_path_label = os.path.join(cfg.label_train_path, file)
            file_path_data = os.path.join(cfg.data_train_path, file.replace( "label","data"))
            label = np.load(file_path_label)
            data =np.load(file_path_data)

            train_data = np.concatenate((train_data, data), axis=0) if i > 0 else data
            train_label = np.concatenate((train_label, label), axis=0) if i > 0 else label

        np.save(os.path.join(cfg.data_cat_path, "train_data_all.npy"),
                train_data)
        np.save(os.path.join(cfg.label_cat_path, "train_label_all.npy"),
                train_label)
    else:
        for i,file in enumerate(test_file_list):
            file_path_label = os.path.join(cfg.label_test_path, file)
            file_path_data = os.path.join(cfg.data_test_path, file.replace( "label","data"))
            label = np.load(file_path_label)
            data =np.load(file_path_data)

            test_data = np.concatenate((test_data, data), axis=0) if i > 0 else data
            test_label = np.concatenate((test_label, label), axis=0) if i > 0 else label

        np.save(os.path.join(cfg.data_cat_path, "test_data_all.npy"),
                test_data)
        np.save(os.path.join(cfg.label_cat_path, "test_label_all.npy"),
                test_label)

def statistic_sample(cfg):
    label=np.load(cfg.label_cat_path)
    print(np.unique(label))
    num_class=len(np.unique(label))-1
    count=np.bincount(np.int64(label),minlength=num_class)
    print(count)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Setup parameters
    parser.add_argument('--img_root_path', default='/data2/ll22/Upcrop/CDL_OH_L8/Takeout/Cloud/P20R32_L8_2021', type=str,help='Path to datasets root directory')
    parser.add_argument('--label_root_path', default='/data2/ll22/Upcrop/CDL_OH/Takeout/Cloud/OH_2021_CDL', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--confi_root_path', default='/data2/ll22/Upcrop/CDL_OH_L8/Takeout/Cloud/P20R32_Conf_2021', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--reserve_path', default='/data2/ll22/Upcrop/CDL_OH_L8/statistic/reserve_file_2021.txt', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--statistic_path', default='/data2/ll22/Upcrop/CDL_OH_L8/statistic/label_info_2021.xlsx', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--yaml_path', default='/home/ll22/code/Cropup/config/dataset/17TKF_S2_2019.yaml', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--mean_std_path', default='/data2/ll22/Upcrop/CDL_OH_L8/statistic/mean_std_2021.txt', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--ndvi_excel_path', default='/data2/ll22/Upcrop/CDL_OH_L8/statistic/ndvi_2021.xlsx', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--label_simply_root_path', default='/data2/ll22/Upcrop/CDL_OH/Takeout/Cloud/OH_2021_CDL_simply/patch', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--extract_year', default='h', type=str,choices=["h","t_train","t_test"],
                        help='Path to datasets root directory')

    parser.add_argument('--add_condi', default='random', type=str,choices=["random","window","confidence"],
                        help='Path to datasets root directory')
                        
    parser.add_argument('--num_time', default=8, type=int,
                        help='Path to datasets root directory')
    parser.add_argument('--window_size', default=5, type=int,
                        help='Path to datasets root directory')
    parser.add_argument('--count_class', default=400, type=int,
                        help='Path to datasets root directory')
    parser.add_argument('--confi_thr', default=100, type=int,
                        help='Path to datasets root directory')
    parser.add_argument('--num_fea', default=10, type=int,
                        help='Path to datasets root directory')
    parser.add_argument('--num_class', default=7, type=int,
                        help='Path to datasets root directory')
    parser.add_argument('--data_train_path', default='/data2/ll22/Upcrop/CDL_OH/pixel_sample/2021/train_data_npy', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--label_train_path', default='/data2/ll22/Upcrop/CDL_OH/pixel_sample/2021/train_label_npy', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--data_test_path', default='/data2/ll22/Upcrop/CDL_OH/pixel_sample/2021/test_data_npy', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--label_test_path', default='/data2/ll22/Upcrop/CDL_OH/pixel_sample/2021/test_label_npy', type=str,
                        help='Path to datasets root directory')
                        
    parser.add_argument('--data_cat_path', default='/data2/ll22/Upcrop/CDL_OH/pixel_sample/2021/data_npy', type=str,
                        help='Path to datasets root directory')
    parser.add_argument('--label_cat_path', default='/data2/ll22/Upcrop/CDL_OH/pixel_sample/2021/label_npy', type=str,
                        help='Path to datasets root directory')

    parser.add_argument('--point_map_path', default='', type=str,
                        help='Path to datasets root directory')

    cfg=parser.parse_args()
    #1. first step
    non_value_band_filter(cfg)
#    statistic_class_number(cfg)
    get_norm(cfg)
    #2. second step
    extraxt_samples_product(cfg,use_norm=True,ext_year=cfg.extract_year,add_condi=cfg.add_condi)
    concat_sample(cfg)
#    output_tsNDVI(cfg)
#    extraxt_samples_product(cfg,use_norm=True)
#    sample_information(cfg)
#    concat_sample(cfg)
#    simplify_label(cfg)
#    statistic_sample(cfg)
    # date= { "15TVH": [155, 163, 165, 173, 183, 200, 223, 225, 228, 230, 233, 243, 248, 250, 258, 260, 265, 270],
    #      }
    # from psaedataset import PixelSetData
    # datatset=PixelSetData(data_path="/data1/ll20/Upcrop/CDL_15TVH/pixel_sample/2019/confidence_100/train_data_npy/",label_path="/data1/ll20/Upcrop/CDL_15TVH/pixel_sample/2019/confidence_100/train_label_npy/",start_doy=0,
    #              doy_dict=date,
    #              num_seq=8,
    #              num_feature=10,
    #              transform=None,
    #              indices=None,
    #              select_seq=None,
    #              add_noisy=0.0,)
  

#    train_data=np.load("/data1/ll20/Upcrop/CDL_15TVH/pixel_sample/2019/confidence_100/train_data_npy/train_data_2019S215TVH_0.npy")
#    print(train_data.shape)
#    train_label = np.load(
#         "/data1/ll20/Upcrop/CDL_15TVH/pixel_sample/2019/confidence_100/train_label_npy/train_label_2019S215TVH_0.npy")
#    print(train_label.shape)
#    print(np.unique(train_label))
#    print(sum_pixels_v2(train_label,7))
#    path=[cfg.data_train_path,cfg.label_train_path,cfg.data_test_path,cfg.label_test_path]
#    for p in path:
#        count = 0
#        delete = []
#        file_list=os.listdir(p)
#        for file in file_list:
#            if file.split("_")[3]!="15TVH":
#                os.remove(os.path.join(p,file))
#                count+=1
#                delete.append(file)
#        print(count)