import cv2
import random
import math
import numpy as np
import numbers
import collections
from PIL import Image, ImageOps, ImageFilter
from skimage.segmentation import clear_border
from skimage.morphology   import remove_small_objects
import skimage
import os
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn


def offset_origin_to_offset_B(sample):
    label_offset=sample['label_offset']#label_offset的shape为[H,W,3]
    label_foot=sample['label_foot']#label_foot的shape为[H,W],忽略的地方为255
    label_roofside=sample['label_roofside']
    offset_b=np.zeros((label_offset.shape))
    #把底座和被忽略的地方设置为nan
    offset_b[label_foot==1,0]=500#底座区域在x,y通道都设置为nan
    offset_b[label_foot==1,1]=500
    offset_b[label_foot==255,2]=255#被忽略区域在第3通道设置nan
    offset_b[label_offset[:,:,2]==255,2]=255
    #因为屋顶区域的值肯定没有nan,所以通过一个遍历,填充底座;因为底座不会有重复(或者只有很少会重复),所以不必单独处理底座有重复的情况
    for index_i in range(offset_b.shape[0]):
        for index_j in range(offset_b.shape[1]):
            #先判断index_i,index_j是屋顶区域,然后index_i,index_j-offset落在了底座区域,则都不是被ignore的区域,则处理该点,否则,直接跳过
            #label_offset[index_i,index_j]
            if label_roofside[index_i,index_j]==1:
                if abs(label_offset[index_i,index_j,0])<=400 and abs(label_offset[index_i,index_j,1])<=400:#如果该处的offset没有被忽略
                    foot_index_i=index_i-int(label_offset[index_i,index_j,1])
                    foot_index_j=index_j-int(label_offset[index_i,index_j,0])
                    if 0<=foot_index_i<=label_foot.shape[0]-1 and 0<=foot_index_j<=label_foot.shape[1]-1:#and label_foot[index_i,index_j]==1
                        offset_b[foot_index_i,foot_index_j,0]=label_offset[index_i,index_j,0]
                        offset_b[foot_index_i,foot_index_j,1]=label_offset[index_i,index_j,1]
    sample['label_offset_b']=offset_b
    return sample
def get_len_angle(sample):
    def xy2la(label_offset_xy):
        r=np.sqrt(label_offset_xy[:,:,0]**2+label_offset_xy[:,:,1]**2)
        phi=(np.arcsin(label_offset_xy[:,:,1]/(r+1e-10)))+(math.pi/2)
        phi=np.angle(label_offset_xy[:,:,0]-1j*label_offset_xy[:,:,1])-math.pi/2
        return r,phi
    def log_normalize(length,max_depth=560,min_depth=0):
        depth_map_clipped=np.clip(length,min_depth,max_depth)
        depth_map=np.log(depth_map_clipped-min_depth+1)/np.log(max_depth-min_depth+1)
        return depth_map

    label_offset=sample['label_offset']
    ignoreIndex0=np.where(abs(label_offset[:,:,0])==500)
    ignoreIndex1=np.where(abs(label_offset[:,:,1])==500)
    ignoreIndex2=np.where(abs(label_offset[:,:,0])==400)
    ignoreIndex3=np.where(abs(label_offset[:,:,1])==400)

    label_offset_la=np.zeros((label_offset.shape[0],label_offset.shape[1],3))
    length,angle=xy2la(label_offset[:,:,0:2])
    #对长度做log归一化
    length=log_normalize(length)
    label_offset_la_ignore=label_offset[:,:,2]
    label_offset_la[:,:,0],label_offset_la[:,:,1],label_offset_la[:,:,2]=length,angle,label_offset_la_ignore
    #将被忽略的地方设置为忽略
    label_offset_la[:,:,0][ignoreIndex0]=500
    label_offset_la[:,:,1][ignoreIndex1]=500
    label_offset_la[:,:,0][ignoreIndex2]=500
    label_offset_la[:,:,1][ignoreIndex3]=500

    sample['label_len_angle']=label_offset_la
    return sample



def RandRotate(sample,angle_degree):
    #rotate the offset of degree counter clockwise around its centre
    # if debug_flag:
    #     angle_degree=90
    #     print(angle_degree)
    #     image=sample['image']
    #     image[0]=np.asarray(image[0]).astype('int')
    #     cv2.imwrite('offsetDataVisForAug/oriImg/'+'1.jpg',image[0])
    #     img_offset=segtransformsForOffsetField.show_flow_hsv(sample['label_offset'])
    #     cv2.imwrite('offsetDataVisForAug/oriImg/'+'1_offset.jpg',img_offset)
    #     img_offset=segtransformsForOffsetField.show_flow_hsv(sample['label_offset'],hsv_flag=True)
    #     cv2.imwrite('offsetDataVisForAug/oriImg/'+'1_offset_hsv.jpg',img_offset)
    #     label_2=np.asarray(sample['label_baseRoof']).astype('int')
    #     label_2=torch_tensor_to_label255(torch.Tensor(label_2))
    #     cv2.imwrite('offsetDataVisForAug/oriImg/'+'1_baseRoof.jpg',label_2)
    def xy2la(label_offset_xy):
        label_offset_la=np.zeros((label_offset_xy.shape[0],label_offset_xy.shape[1],2))
        r=np.sqrt(label_offset_xy[:,:,0]**2+label_offset_xy[:,:,1]**2)
        phi=(np.arcsin(label_offset_xy[:,:,1]/(r+1e-10)))+(math.pi/2)
        phi=np.angle(label_offset_xy[:,:,0]-1j*label_offset_xy[:,:,1])-math.pi/2
        label_offset_la[:,:,0],label_offset_la[:,:,1]=r,phi
        return label_offset_la
    def la2xy(label_offset_la):
        label_offset_xy=np.zeros((label_offset_la.shape[0],label_offset_la.shape[1],2))
        ignore_index=np.where(abs(label_offset_la[:,:,0])>=400*1.4)
        x=-label_offset_la[:,:,0]*np.sin(label_offset_la[:,:,1])
        y=-label_offset_la[:,:,0]*np.cos(label_offset_la[:,:,1])
        x[ignore_index]=500
        y[ignore_index]=500
        label_offset_xy[:,:,0],label_offset_xy[:,:,1]=x,y
        return label_offset_xy
    angle=math.radians(angle_degree)
    #label_offset=sample['label_offset']
    label_offset = sample
    label_offset_la=xy2la(label_offset[:,:,0:2])
    label_offset_la[:,:,1]=label_offset_la[:,:,1]+angle
    label_offset_la_rotate=skimage.transform.rotate(image=label_offset_la,angle=angle_degree,order=0)#order=0表示最近邻采样
    label_offset_xy_rotate=la2xy(label_offset_la_rotate)
    label_ignore_255_rotate=skimage.transform.rotate(image=label_offset[:,:,2:],angle=angle_degree,order=0)
    label_offset_rotate=np.concatenate((label_offset_xy_rotate,label_ignore_255_rotate),axis=2)
    #sample['label_offset']=label_offset_rotate
    return label_offset_rotate

def RandScale(sample,scale_factor):
    #label_offset = sample['label_offset']
    label_offset = sample
    h, w = label_offset.shape[:2]
    # w,h,(scale_factor_h,scale_factor_w)=w,h,scale_factor
    #  = int(w * scale_factor_w)
    new_h,new_w = scale_factor
    scale_factor_w = float(new_w) / float(w)
    scale_factor_h = float(new_h) / float(h)
    label_offset=skimage.transform.resize(label_offset,(new_h,new_w),order=0,mode='constant',anti_aliasing=False)#order=0表示最近邻采样,注意new_h和new_w的顺醋和Image.resize不同,anti_aliasing表示是否做高斯模糊
    #print('after',len(np.where(label_offset[:,:,1]==500)[0]))
    ignoreIndex1=np.where(abs(label_offset[:,:,1])==500)
    ignoreIndex0=np.where(abs(label_offset[:,:,0])==500)
    ignoreIndex2=np.where(abs(label_offset[:,:,0])==400)
    ignoreIndex3=np.where(abs(label_offset[:,:,1])==400)
    #因为尺度缩放，所以offset值也变化，offset值表示偏移多少像素
    label_offset[:,:,0]=scale_factor_w*label_offset[:,:,0]
    label_offset[:,:,1]=scale_factor_h*label_offset[:,:,1]
    label_offset[:,:,0][ignoreIndex0]=500
    label_offset[:,:,1][ignoreIndex1]=500
    label_offset[:,:,0][ignoreIndex2]=500
    label_offset[:,:,1][ignoreIndex3]=500
    #sample['label_offset'] = label_offset
    return label_offset

def Crop(sample,crop_coords):
    w_off,h_off,w_end,h_end=crop_coords
    label_offset = sample['label_offset']
    assert h_end<=label_offset.shape[0]
    assert w_end<=label_offset.shape[1]
    label_offset = label_offset[h_off:h_end,w_off:w_end]
    sample['label_offset']=label_offset
    return sample

def RandomHorizontalFlip(sample):#左右翻转,对应的Image操作为transpose(Image.FLIP_LEFT_RIGHT),对应的np操作为np.clip(array,axis=1)

    label_offset = sample
    label_offset = np.flip(label_offset, axis=1)
    ignoreIndex = np.where(abs(label_offset[:, :, 0]) == 500)
    label_offset[:, :, 0] = -label_offset[:, :, 0]
    label_offset[:, :, 0][ignoreIndex] = 500
    #sample['label_offset'] = label_offset

    return label_offset

def RandomVerticalFlip(sample):#VerticalFlip_flag对应上下翻转
    label_offset = sample
    label_offset = np.flip(label_offset, axis=0)
    ignoreIndex = np.where(abs(label_offset[:, :, 1]) == 500)
    label_offset[:, :, 1] = -label_offset[:, :, 1]
    label_offset[:, :, 1][ignoreIndex] = 500
    # sample['label_offset'] = label_offset
    return label_offset

def RandRotate90n(sample,rand_angle):
    def flip180(arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        return new_arr

    def flip90_left(arr):
        new_arr = np.transpose(arr)
        new_arr = new_arr[::-1]
        return new_arr

    def flip90_right(arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        new_arr = np.transpose(new_arr)[::-1]
        return new_arr

    label_offset = sample
    if 0 == rand_angle:
        label_offset_ro=label_offset
    elif 3 == rand_angle:
        label_offset_c0=flip90_left(label_offset[:,:,0])
        label_offset_c1=flip90_left(label_offset[:,:,1])
        label_offset_c2=flip90_left(label_offset[:,:,2])
        label_offset_ro=np.concatenate((label_offset_c1[:,:,np.newaxis],-label_offset_c0[:,:,np.newaxis],label_offset_c2[:,:,np.newaxis]),axis=2)

    elif 2 == rand_angle:
        label_offset_c0=flip180(label_offset[:,:,0])
        label_offset_c1=flip180(label_offset[:,:,1])
        label_offset_c2=flip180(label_offset[:,:,2])
        label_offset_ro=np.concatenate((-label_offset_c0[:,:,np.newaxis],-label_offset_c1[:,:,np.newaxis],label_offset_c2[:,:,np.newaxis]),axis=2)
    else:
        label_offset_c0=flip90_right(label_offset[:,:,0])
        label_offset_c1=flip90_right(label_offset[:,:,1])
        label_offset_c2=flip90_right(label_offset[:,:,2])
        label_offset_ro=np.concatenate((-label_offset_c1[:,:,np.newaxis],label_offset_c0[:,:,np.newaxis],label_offset_c2[:,:,np.newaxis]),axis=2)

    # sample['label_offset']=label_offset_ro
    return label_offset_ro

# def ToTensor(sample):
#     label_offset=sample['label_offset']
#     label_offset=np.ascontiguousarray(label_offset)
#     label_offset=torch.from_numpy(label_offset)
#     label_offset=label_offset.permute(2,0,1)
#     if not isinstance(label_offset,torch.FloatTensor):
#         label_offset=label_offset.float()
#     sample['label_offset']=label_offset
#
#     if 'label_offset_b' in sample:
#         label_offset_b=sample['label_offset_b']
#         label_offset_b=np.ascontiguousarray(label_offset_b)
#         label_offset_b=torch.from_numpy(label_offset_b)
#         label_offset_b=label_offset_b.permute(2,0,1)
#         if not isinstance(label_offset_b,torch.FloatTensor):
#             label_offset_b=label_offset_b.float()
#         sample['label_offset_b']=label_offset_b
#
#     if 'label_len_angle' in sample:
#         label_len_angle=sample['label_len_angle']
#         label_len_angle=np.ascontiguousarray(label_len_angle)
#         label_len_angle=torch.from_numpy(label_len_angle)
#         label_len_angle=label_len_angle.permute(2,0,1)
#         if not isinstance(label_len_angle,torch.FloatTensor):
#             label_len_angle=label_len_angle.float()
#         sample['label_len_angle']=label_len_angle
#     return sample




def torch_tensor_to_label255(lab):
    lab = lab.numpy()
    print("np.unique(lab) = {}".format(np.unique(lab)))
    lab[lab == 1] = 50
    lab[lab == 2] = 100
    lab[lab == 3] = 150
    lab[lab==4]=200
    lab[lab==5]=200
    lab[lab==6]=200
    lab[lab==7]=200
    lab[lab==8]=200
    lab = lab.astype('uint8')
    return lab






def leng_loss_with_ignore(inpu,target,upsample_flag=False):
    #sample to make the input and target has the same H and W
    if upsample_flag:
        #bilinear upsample the input
        inpu=F.interpolate(inpu,size=(target.shape[2],target.shape[3]),mode='bilinear')
    else:
        target=F.interpolate(target.float(),size=(inpu.shape[2],inpu.shape[3]),mode='nearest')
    labelA=target.cpu().numpy()
    ignore_mask=np.ones((labelA.shape[0],labelA.shape[2],labelA.shape[3]))
    for index in range(labelA.shape[0]):
        ignore_index1=(abs(labelA[index][0,:,:])==500)
        ignore_index1=np.where(ignore_index1)
        ignore_index2=(abs(labelA[index][0,:,:])==400)
        ignore_index2=np.where(ignore_index2)
        ignore_index3=np.where(abs(labelA[index][2,:,:])==255)
        ignore_mask[index][ignore_index1]=0
        ignore_mask[index][ignore_index2]=0
        ignore_mask[index][ignore_index3]=0
    ignore_mask=torch.from_numpy(ignore_mask).float()
    ignore_mask=ignore_mask.type(target.type())
    target_c1=target[:,0,:,:]*ignore_mask
    input_c1=inpu[:,0,:,:]*ignore_mask
    d=input_c1-target_c1
    n_pixels=d.shape[1]*d.shape[2]
    term_1 = torch.pow(d.view(-1, n_pixels),2).mean(dim=1).sum()
    term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1),2)/(2*(n_pixels**2))).sum()
    loss=term_1-term_2
    return loss



def get_weights_map_from_line56(sample,special_weights):#对不同额区域的损失做加权，整幅图都为1，指定区域的损失权重为指定值weights
    #根据缩放倍数,获得5，6膨胀后的区域
    def get_four_special_area(image_edge5c_t,image_ABC_t,multiple,expand=15):#image是
        img_edge5c=image_edge5c_t.copy()
        img_edge5c=img_edge5c.astype(np.uint8)
        img_ABC=image_ABC_t.copy()
        img_ABC=img_ABC.astype(np.uint8)

        assert set(np.unique(img_edge5c))<={0,1,2}#我们只关心5,6边,他们的label是1,2
        dilate=int(expand*multiple)
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        dilate_img_edge5c = cv2.dilate(img_edge5c, kernel, 1)
        #
        _5_back=np.where(np.bitwise_and(dilate_img_edge5c==1,img_ABC==0))
        _5_roof=np.where(np.bitwise_and(dilate_img_edge5c==1,img_ABC==1))

        _6_roof=np.where(np.bitwise_and(dilate_img_edge5c==2,img_ABC==1))
        _6_side=np.where(np.bitwise_and(dilate_img_edge5c==2,img_ABC==2))
        _5_contour=np.where(img_edge5c==1)
        _6_contour=np.where(img_edge5c==2)
        return _5_roof,_5_back,_6_roof,_6_side,_5_contour,_6_contour

    label_edge5cls=sample['label_edge5cls'].cpu().numpy().copy()
    label_roofside=sample['label_roofside'].cpu().numpy().copy()
    weights_mask=np.ones(((label_edge5cls.shape[0],label_edge5cls.shape[1],label_edge5cls.shape[2])))
    for index in range(label_edge5cls.shape[0]):
        image_edge5c=label_edge5cls[index]
        image_edge5c[image_edge5c==3]=0
        image_edge5c[image_edge5c==4]=0
        image_edge5c[image_edge5c==255]=0

        image_roofside=label_roofside[index]
        multiple=max(sample['scale_factor'][2][index],sample['scale_factor'][3][index]).cpu().item()#sample['scale_factor']的组成为[tensor([b1,b2,b3]),tensor([b1,b2,b3]),tensor([b1,b2,b3]),tensor([b1,b2,b3])]
        _5_roof,_5_back,_6_roof,_6_side,_5_contour,_6_contour=get_four_special_area(image_edge5c,image_roofside,multiple,expand=13)
        for special_area in [_5_roof,_5_back,_5_contour]:
            weights_mask[index][special_area]=special_weights#0
    return weights_mask
    # loss=






def EPE_loss_with_special_weights(inpu,target,sample,special_weights,upsample_flag=False):
    #先获得不加权重的损失
    loss=EPE_loss_with_ignore(inpu,target,upsample_flag,mean_flag=False)
    #整张图乘以权重，
    weights_mask=get_weights_map_from_line56(sample,special_weights)
    weights_mask_t=torch.from_numpy(weights_mask).float()
    weights_mask_t=weights_mask_t.type(target.type())
    # if upsample_flag:
    #     pass
    # else:
    #     weights_mask_t=F.interpolate(weights_mask_t.float().unsqueeze(1),size=(inpu.shape[2],inpu.shape[3]),mode='nearest')[:,0,:,:]

    assert loss.shape==weights_mask_t.shape
    loss=(loss*weights_mask_t).mean()
    #损失求mean
    return loss,weights_mask


def EPE_loss_with_ignore(inpu,target,upsample_flag=False,mean_flag=True):
    #sample to make the input and target has the same H and W
    if upsample_flag:
        #bilinear upsample the input
        inpu=F.interpolate(inpu,size=(target.shape[2],target.shape[3]),mode='bilinear')
    else:
        target=F.interpolate(target.float(),size=(inpu.shape[2],inpu.shape[3]),mode='nearest')
    labelA=target.cpu().numpy()
    ignore_mask=np.ones((labelA.shape[0],labelA.shape[2],labelA.shape[3]))
    for index in range(labelA.shape[0]):
        ignore_index1=np.bitwise_and(abs(labelA[index][0,:,:])==500,abs(labelA[index][1,:,:])==500)
        ignore_index1=np.where(ignore_index1)
        ignore_index2=np.bitwise_and(abs(labelA[index][0,:,:])==400,abs(labelA[index][1,:,:])==400)
        ignore_index2=np.where(ignore_index2)
        ignore_index3=np.where(abs(labelA[index][2,:,:])==255)
        ignore_mask[index][ignore_index1]=0
        ignore_mask[index][ignore_index2]=0
        ignore_mask[index][ignore_index3]=0
    ignore_mask=torch.from_numpy(ignore_mask).float()
    ignore_mask=ignore_mask.type(target.type())
    target_c1=target[:,0,:,:]*ignore_mask
    input_c1=inpu[:,0,:,:]*ignore_mask
    target_c2=target[:,1,:,:]*ignore_mask
    input_c2=inpu[:,1,:,:]*ignore_mask
    #Todo:visualize the target,ignoremask,targt*ignoremask,inpu
    inpu=torch.cat((input_c1.unsqueeze(1),input_c2.unsqueeze(1)),dim=1)
    targ=torch.cat((target_c1.unsqueeze(1),target_c2.unsqueeze(1)),dim=1)
    if mean_flag:
        loss=torch.norm(inpu-targ,p=2,dim=1).mean()
    else:
        loss=torch.norm(inpu-targ,p=2,dim=1)
    return loss



def show_depth(depth_re,ignoreValue=(500,500),max_depth=560,min_depth=0):
    if isinstance(depth_re,torch.Tensor):
        depth=depth_re.clone()
        depth=depth.cpu().numpy().transpose(1,2,0)
    else:
        depth=depth_re.copy()

    depth_only=depth[:,:,0]

    if ignoreValue is not None and depth.shape[2]==3:
        #500是被忽略,400是该点计算时出现异常
        _=(abs(depth[:,:,0])==ignoreValue[0])
        ignoreIndex=np.where(_)
        depth_only[ignoreIndex]=0
        ignoreValue=(400,400)
        _=(abs(depth[:,:,0])==ignoreValue[0])
        ignoreIndex2=np.where(_)
        depth_only[ignoreIndex2]=0
    print('depth_only:',np.max(depth_only),np.min(depth_only))
    img_depth=np.zeros((depth_only.shape[0],depth_only.shape[1],3),np.uint8)
    depth_only=depth_only*(np.log(max_depth-min_depth+1))
    depth_only=np.exp(depth_only)-1
    img_depth[:,:,0],img_depth[:,:,1],img_depth[:,:,2]=depth_only.astype(np.uint8),depth_only.astype(np.uint8),depth_only.astype(np.uint8)




    if ignoreValue is not None and depth.shape[2]==3:
        img_depth[ignoreIndex]=(150,0,0)
        img_depth[ignoreIndex2]=(150,0,0)

    if depth.shape[2]==3:#如果多出忽略的分支
        img_depth[np.where(abs(depth[:,:,2])==255)]=(0,150,0)
    return img_depth










def show_flow_hsv(flow_re, show_style=1,ignoreValue=(500,500),hsv_flag=False):
    if isinstance(flow_re,torch.Tensor):
        flow=flow_re.clone()
        flow=flow.cpu().numpy().transpose(1,2,0)
    else:
        flow=flow_re.copy()
    if ignoreValue is not None and flow.shape[2]==3:
        #500是被忽略,400是该点计算时出现异常
        _=np.bitwise_and(abs(flow[:,:,0])==ignoreValue[0],abs(flow[:,:,0])==ignoreValue[1])
        ignoreIndex=np.where(_)
        flow[ignoreIndex]=0
        ignoreValue=(400,400)
        _=np.bitwise_and(abs(flow[:,:,0])==ignoreValue[0],abs(flow[:,:,0])==ignoreValue[1])
        ignoreIndex2=np.where(_)
        flow[ignoreIndex2]=0
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])#将直角坐标系光流场转成极坐标系
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    #光流可视化的颜色模式
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2 #angle弧度转角度
        hsv[..., 1] = 255
        hsv[..., 2] =np.clip((mag*3),0,255).astype(np.uint8) #magnitude归到0～255之间
        #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)#magnitude归到0～255之间
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
    #hsv转bgr
    if hsv_flag:
        bgr=hsv
    else:
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if ignoreValue is not None and flow.shape[2]==3:
        bgr[ignoreIndex]=255
        bgr[ignoreIndex2]=155
    if flow.shape[2]==3:#如果多出忽略的分支
        bgr[np.where(abs(flow[:,:,2])==255)]=150
    return bgr

def torch_tensor_to_ImageRGB(img,mean,std):
    for t,m,s in zip(img,mean,std):
        t.mul_(s).add_(m)
    img = img.numpy().transpose((1, 2, 0))
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img
# def image_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     image = []
#     for n in range(len(path)):
#         with open(path[n], 'rb') as f:
#             if debug_flag:
#                 cvImg=cv2.imread(path[n])
#                 cv2.imwrite('offsetDataVisForAug2/originImg/'+str(index_global)+'.jpg',cvImg)
#             img = Image.open(f)
#             image.append(img.convert('RGB'))
#     return image

def label_loader(path):
    label = []
    for n in range(len(path)):
        lab = np.load(path[n])
        label.append(lab)
    return label


def make_dataset(num_label_per_sample, data_root=None, data_list=None):
    data_root = '' if data_root is None else data_root
    assert num_label_per_sample >= 0
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()#将所有行的数据都读入list_read中作为一个列表包含了空格符
    print("Totally {} samples.".format(len(list_read)))
    print("Starting Checking image&label 'pair'...")

    line1 = list_read[0].strip().split(',')#strip移除字符串首尾指定的字符,默认为空格符号和换行符
    assert len(line1) > num_label_per_sample
    num_image_per_sample = len(line1) - num_label_per_sample#获得样本中图像的个数

    for line in list_read:
        line = line.strip()
        line_split = line.split(',')
        if len(line_split) != num_image_per_sample + num_label_per_sample:
            raise (RuntimeError("Image list file read line error : " + line + "\n"))
        image_name = []
        label_name = []
        for n in range(num_image_per_sample):
            image_name.append(os.path.join(data_root, line_split[n]))
        if num_label_per_sample > 0:
            for n in range(num_label_per_sample):
                label_name.append(os.path.join(data_root, line_split[n+num_image_per_sample]))
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair 'list'!")
    return image_label_list, num_image_per_sample


class OffsetFieldData(Dataset):
    def __init__(self, num_label_per_sample=1, data_root=None, data_list=None, transform=None):
        self.num_label_per_sample = num_label_per_sample
        self.data_list, self.num_image_per_sample = make_dataset(num_label_per_sample, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """self.data_list[index]:  (['/mnt/lustre/menglingxuan/buildingwolf/20200329/shanghai_18/google/RGBImg/L18_107040_219600.jpg'], ['/mnt/lustre/menglingxuan/buildingwolf/20200329/shanghai_18/google/annoV2/OffsetField/Npy/L18_107040_219600.npy'])
        """
        image_path, label_path = self.data_list[index]
        label = label_loader(label_path)

        if 0 == self.num_label_per_sample:
            label = None
        elif 1 == self.num_label_per_sample:
            label = label[0]
            # if debug_flag:
            #     label_copy=label.copy()
            #     label_bgr=show_flow_hsv(label_copy)
            #     global index_global
            #     cv2.imwrite('offsetDataVisForAug2/originOffset/'+str(index_global)+'.jpg',label_bgr)
            #     index_global=index_global+1
        else:
            raise(RuntimeError("num_label_per_sample > 1 not supported\n"))

        return label


# if __name__ == '__main__':
#     train_dict = {}
#     train_dict['max_color_shift'] = 20
#     train_dict['min_contrast'] = 0.8
#     train_dict['max_contrast'] = 1.2
#     train_dict['max_brightness_shift'] = 10
#     train_dict['max_pixel_noise'] = 20
#     train_dict['max_jitter'] = 0.3
#     value_scale = 255
#     mean = [127, 127, 127]
#     std = [0.25, 0.25, 0.25]
#     std = [item * value_scale for item in std]
#
#     train_transform = Compose([
#             Crop([1500,1300], crop_type='rand', padding=[0,0,0], ignore_label=255, rand_pair_trans_offset=0),
#             ColorJitter(train_dict),
#             RandomGaussianBlur(),
#             RandomHorizontalFlip(),
#             RandomVerticalFlip(),
#             RandPixelNoise(train_dict),
#             ToTensor(),
#             Normalize(mean=mean,std=std)
#             ])
#     #data_list='/mnt/lustre/menglingxuan/buildingwolf/20200329/20200409/util/offset2.csv'
#     data_list='/mnt/lustre/menglingxuan/buildingwolf/20200329/offsetField.csv'
#     offset_filed_data=OffsetFieldData(num_label_per_sample=1,data_root=None,data_list=data_list,transform=train_transform)
#
#     train_loader = torch.utils.data.DataLoader(offset_filed_data, batch_size=1, shuffle=False)
#
#
#     # for i, sample in enumerate(train_loader):
#     #     label=sample['label']
#     #     loss=L1_loss_with_ignore(label[:,0:2,:,:]+i,label)
#     #     print(loss)
#     #     if i==15:
#     #         break
#     for i, sample in enumerate(train_loader):
#         #可视化显示
#         image=sample['image'][0] #image的shape为[batch,channels,H,W]
#         image=image[0]
#         image=torch_tensor_to_ImageRGB(image,mean,std)
#         label=sample['label']
#         label=label[0]
#         label=show_flow_hsv(label)
#         cv2.imwrite('offsetDataVisForAug2/augImg/'+str(i)+'.jpg',np.asarray(image))
#         cv2.imwrite('offsetDataVisForAug2/augOffset/'+str(i)+'.jpg',label)
#
#         if 9 == i:
#             break


