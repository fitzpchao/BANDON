
import cv2
import random
import math
import numpy as np
import numbers
import collections
from PIL import Image, ImageOps, ImageFilter
from skimage.segmentation import clear_border
from skimage.morphology   import remove_small_objects
import os
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import sys
import cmath
import math
import skimage.transform
import pdb

from .builder import DATASETS
from .builder import PIPELINES
from .pipelines import Compose
from .pipelines import segtransformsForOffsetField
from .pipelines import geometric_data_v2_seg as geometric_datasets
from .pipelines import geometric_transforms_v2_cls_seg_aug_orient as geometric_transforms

global use_offset_flag
use_offset_flag=True
"""子类继承父类的方法：
1.
super().func()表示执行父类的方法，如果
自定义：继续执行自己的方法
2.未定义该方法，表示继承父类的所有方法
class Apple(Fruit):
    pass
3.覆盖父类的方法
class Apple(Fruit):
    def color(self):
        func()
"""
def offset_conver(offset):
    """
    change offset from (3,H,W) to (2,H,W),change ignore_flage from(+-400),(+-500),(255) to nan
    """
    offset_2=offset[0:2].clone() #(2,H,W)
    offset_2[torch.where(torch.abs(offset)>=400)]=np.nan# change the ignore value to np.nan
    ignore_255=torch.where(torch.abs(offset[2])==255)#get where the channel 3 is 255
    offset_2[0][ignore_255]=np.nan
    offset_2[1][ignore_255]=np.nan
    return offset_2

@DATASETS.register_module()
class BuildData(Dataset):
    CLASSES=('1')
    PALETTE=[[1,1,1]]
    def __init__(self,   pipeline,p_use_offset_flag=False,data_list_roof='/mnt/lustre/menglingxuan/buildingwolf/20200329/buildrooflabel/roof_side_foot_angle_20201016.csv',data_list_offset='/mnt/lustre/menglingxuan/buildingwolf/20200329/offset_20200924.csv',expand=1):
        self.pipeline=Compose(pipeline)
        self.geometric_data=geometric_datasets.Geometric_Data(num_label_per_sample=3, data_root='',data_list=data_list_roof)
        self.offset_data=segtransformsForOffsetField.OffsetFieldData(num_label_per_sample=1,data_root='',data_list=data_list_offset)
        self.data_list=get_data_intersection(self.geometric_data.data_list,self.offset_data.data_list)
        if expand>1:
            self.data_list=self.data_list*expand
        self.geometric_data.data_list=[(_[0],_[1]) for _ in self.data_list]
        self.offset_data.data_list=[(_[0],_[2]) for _ in self.data_list]
        global use_offset_flag
        use_offset_flag=p_use_offset_flag
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample=self.geometric_data.__getitem__(index)
        sample_label_offset=self.offset_data.__getitem__(index)
        sample['label_offset']=sample_label_offset

        sample=self.pipeline(sample)
        global use_offset_flag
        if 'label_offset_b' in sample.keys() and use_offset_flag:
            sample['label_offset_b']=offset_conver(sample['label_offset_b'])
        else:
            sample['label_offset_b']={}
        if 'label_offset' in sample.keys() and use_offset_flag:
            sample['label_offset']=offset_conver(sample['label_offset'])
        else:
            sample['label_offset']={}
        results= {'img':sample['image'][0],'gt_semantic_seg':sample['label_roofside'].unsqueeze(0),'gt_edge_seg':sample['label_edge5cls'].unsqueeze(0),'gt_orient_edge_36_seg':sample['label_edge_orient'].unsqueeze(0),'gt_label_offset_reg':sample['label_offset'],'img_metas':{}}
        # for keys in results:
        #     try:
        #         print(keys,type(results[keys]),results[keys].shape)
        #     except:
        #         continue
        return results



@PIPELINES.register_module()
class B_RandRotate(geometric_transforms.RandRotate):
    def __call__(self,sample):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.RandRotate(sample,angle_degree=sample['angle'])
        return sample
@PIPELINES.register_module()
class B_RandScale(geometric_transforms.RandScale):
    def __call__(self,sample):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.RandScale(sample,scale_factor=sample['scale_factor'])
        return sample
@PIPELINES.register_module()
class B_Crop(geometric_transforms.Crop):
    def __call__(self,sample,crop_coords=None):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.Crop(sample,crop_coords=sample['crop_coords'])
        return sample

@PIPELINES.register_module()
class B_RandomHorizontalFlip(geometric_transforms.RandomHorizontalFlip):
    def __call__(self,sample):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.RandomHorizontalFlip(sample,sample['HorizontalFlip_flag'])
        return sample
@PIPELINES.register_module()
class B_RandomVerticalFlip(geometric_transforms.RandomVerticalFlip):
    def __call__(self,sample):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.RandomVerticalFlip(sample,sample['VerticalFlip_flag'])
        return sample
@PIPELINES.register_module()
class B_RandRotate90n(geometric_transforms.RandRotate90n):
    def __call__(self,sample):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.RandRotate90n(sample,sample['rand_angle'])
        return sample

@PIPELINES.register_module()
class B_ToTensor(geometric_transforms.ToTensor):
    def __call__(self,sample):
        sample=super().__call__(sample)
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.ToTensor(sample)
        return sample

@PIPELINES.register_module()
class B_ColorJitter(geometric_transforms.ColorJitter):
    pass
@PIPELINES.register_module()
class B_RandomGaussianBlur(geometric_transforms.RandomGaussianBlur):
    pass
@PIPELINES.register_module()
class B_RandChannelShiftScale(geometric_transforms.RandChannelShiftScale):
    pass
@PIPELINES.register_module()
class B_RandPixelNoise(geometric_transforms.RandPixelNoise):
    pass
@PIPELINES.register_module()
class B_label_final_edge5cls_to_roof_roofcontour_roofside_edge5cls_edge1cls(geometric_transforms.label_final_edge5cls_to_roof_roofcontour_roofside_edge5cls_edge1cls):
    pass
@PIPELINES.register_module()
class B_label_final_angle_float_to_angle_cls(geometric_transforms.label_final_angle_float_to_angle_cls):
    pass
@PIPELINES.register_module()
class B_label1234_2_label12345678(geometric_transforms.label1234_2_label12345678):
    pass
@PIPELINES.register_module()
class B_label_roof_side_to_edgeorient(geometric_transforms.label_roof_side_to_edgeorient):
    pass
@PIPELINES.register_module()
class B_Normalize(geometric_transforms.Normalize):
    pass

@PIPELINES.register_module()
class B_get_len_angle():
    def __call__(self,sample):
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.get_len_angle(sample)
        return sample



@PIPELINES.register_module()
class B_offset_origin_to_offset_B():
    def __call__(self,sample):
        global use_offset_flag
        if use_offset_flag:
            sample=segtransformsForOffsetField.offset_origin_to_offset_B(sample)
        return sample

# class Compose(object):
#     def __init__(self, geometric_offset_transforms):
#         self.geometric_offset_transforms = geometric_offset_transforms

#     def __call__(self, sample):
#         for t in self.geometric_offset_transforms:
#             sample = t(sample)
#         return sample

def get_data_intersection(data_list1,data_list2):
    """get the intersection of two datasets"""
    data_list1_image=[_[0] for _ in data_list1]#获得输入图像的列表
    data_list2_image=[_[0] for _ in data_list2]#获得输入图像的列表
    data_list_merge=[]
    for _ in data_list2:
        if _[0] in data_list1_image:
            #如果输入图像一致，则将label组合起来
            index=data_list1_image.index(_[0])
            data=(_[0],data_list1[index][1],_[1])
            data_list_merge.append(data)
    return data_list_merge


if __name__ == '__main__':
    value_scale = 255
    mean = [0, 0, 0]
    std = [1, 1, 1]
    std = [item * value_scale for item in std]

    train_dict = {}
    train_dict['max_color_shift'] = 10
    train_dict['min_contrast'] = 0.8
    train_dict['max_contrast'] = 1.2
    train_dict['max_brightness_shift'] = 10
    train_dict['max_jitter'] = 0.3
    train_dict['max_pixel_noise'] = 20
    train_dict['max_pixel_noise_instance_adjust'] = 1


    train_transform1 = Compose([
        RandRotate([-180,180], padding=[0, 0, 0], ignore_label=255),
        RandScale([0.5,2.0]),
        Crop([200,200], crop_type='rand', padding=[0, 0, 0], ignore_label=255),
        ColorJitter(train_dict),
        RandomGaussianBlur(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandRotate90n(),
        RandChannelShiftScale(train_dict),
        RandPixelNoise(train_dict),
        label_final_edge5cls_to_roof_roofcontour_roofside_edge5cls_edge1cls(),
        label_final_angle_float_to_angle_cls(),
        label_roof_side_to_edgeorient(wmap_vec=[1,1,1,1,1]), ### added
        get_len_angle(),
        ToTensor()])

    build_data=BuildData(train_transform1)
    #build_data=BuildData()
    train_loader = torch.utils.data.DataLoader(build_data, batch_size=2, shuffle=True)

    for i, sample in enumerate(train_loader):
        print(i)
        continue
        weights=segtransformsForOffsetField.get_weights_map_from_line56(sample,special_weights=0)
        image=sample['image'][0] #image的shape为[batch,channels,H,W]
        image=image[0]
        image=segtransformsForOffsetField.torch_tensor_to_ImageRGB(image,mean,std)
        labelAll=sample['label_offset']
        tagt_len=sample['label_len_angle']
        label=labelAll[0]
        labeln=label.numpy()
        print(np.max(labeln),np.min(labeln))
        label_rgb=segtransformsForOffsetField.show_flow_hsv(label)
        label_hsv=segtransformsForOffsetField.show_flow_hsv(label,hsv_flag=True)
        label_len=segtransformsForOffsetField.show_depth(tagt_len[0])
        # if not os.path.isdir('offsetDataVisForAug'):
        #     os.makedirs('offsetDataVisForAug/augImg/')
        #     os.makedirs('offsetDataVisForAug/augOffset/')
        #     os.makedirs('offsetDataVisForAug/augLabel/')
        #     os.makedirs('offsetDataVisForAug/augOffset_hsv/')
        #     os.makedirs('offsetDataVisForAug/aug_leng/')

        # cv2.imwrite('offsetDataVisForAug/augImg/'+str(i)+'.jpg',np.asarray(image))
        # cv2.imwrite('offsetDataVisForAug/augOffset/'+str(i)+'.jpg',label_rgb)
        # cv2.imwrite('offsetDataVisForAug/augOffset_hsv/'+str(i)+'.jpg',label_hsv)
        # cv2.imwrite('offsetDataVisForAug/aug_leng/'+str(i)+'.jpg',label_len)
        # label_2=sample['label_seg'][0]
        # label_2=segtransformsForOffsetField.torch_tensor_to_label255(label_2)
        # cv2.imwrite('offsetDataVisForAug/augLabel/'+str(i)+'.jpg',label_2)
        #测试loss
        inpu=torch.zeros(labelAll.shape[0],2,int(labelAll.shape[2]),int(labelAll.shape[3]))
        loss,weights=segtransformsForOffsetField.EPE_loss_with_special_weights(inpu,labelAll,sample,special_weights=0)

        inpu=torch.zeros(labelAll.shape[0],2,int(labelAll.shape[2]/8),int(labelAll.shape[3]/8))
        loss=segtransformsForOffsetField.EPE_loss_with_ignore(inpu,labelAll)
        loss_2=segtransformsForOffsetField.leng_loss_with_ignore(inpu[:,0:1,:,:],tagt_len,upsample_flag=True)
        print('loss is',loss)
        print('loss_2 is',loss_2)
        if i==30:
            break

        print(1)
    print(1)
