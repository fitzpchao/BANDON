import pdb
import torch
import numpy as np
import cv2
import os
import rasterio
from affine import Affine
import torch.distributed as dist
from rasterio.plot import reshape_as_raster, reshape_as_image
import torch.nn.functional as F
# from ...models.losses.angle_loss_v2 import get_interpolated_angle
# from ...models.losses.angle_loss_v3 import get_interpolated_angle
# from ...models.losses.angle_loss_72channels import get_angle_cls_resi
# from ..pipelines.orientation_util import OrientationUtil

global imgs_vis,iters,g_save_freq
global use_show_flag
use_show_flag=True
iters=0
COLORS_DICT = {'Blue': (0, 130, 200), 'Red': (230, 25, 75), 'Yellow': (255, 225, 25), 'Green': (60, 180, 75), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Lavender': (230, 190, 255), 'Lime': (210, 245, 60), 'Teal': (0, 128, 128), 'Pink': (250, 190, 190), 'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)}

def cls_resi_to_angle(cls_result,reg_score):
    if not isinstance(cls_result,torch.Tensor):
        cls_result=torch.from_numpy(cls_result)
        cls_result=cls_result.unsqueeze(0)
    if not isinstance(reg_score,torch.Tensor):
        reg_score=torch.from_numpy(reg_score)
        reg_score=reg_score.unsqueeze(0)
    pred_angle=get_angle_cls_resi(cls_result,reg_score)
    return pred_angle


def classify_to_angle(cls_score_t):
    #根据回归的结果转变为角度
    if not isinstance(cls_score_t,torch.Tensor):
        cls_score_t=torch.from_numpy(cls_score_t)
    cls_score_t=cls_score_t.unsqueeze(0)
    cls_score=F.softmax(cls_score_t,dim=1)
    B,C,H,W=cls_score.shape

    orient=OrientationUtil()
    rep_norm_vec_map_h,rep_norm_vec_map_v=orient.get_rep_norm_vec_map(1,H,W)
    rep_norm_vec_map=np.concatenate([rep_norm_vec_map_h[0],rep_norm_vec_map_v[0]],axis=0)
    rep_norm_vec_map=torch.from_numpy(rep_norm_vec_map).type(cls_score_t.type()).unsqueeze(0)
    interpolated_orient_angle,loss_mask=get_interpolated_angle(cls_score,rep_norm_vec_map)
    return interpolated_orient_angle

def save_to_tif(img,save_path,dtype='float32'):
    if isinstance(img,torch.Tensor):
        img=img.clone().cpu().detach().numpy()
    pdb.set_trace()
    if len(img.shape)==2:
        img=img[np.newaxis,:]
        C,H,W=img.shape
    else:
        C,H,W=img.shape
    with rasterio.Env():
        profile={'driver': 'GTiff', 'dtype': dtype, 'nodata': None, 'width': W, 'height': H, 'count': C, 'crs': None, 'transform': Affine(1.0, 0.0, 0.0,0.0, 1.0, 0.0), 'tiled': False, 'interleave': 'pixel'}
        profile.update(dtype=rasterio.float32,compress='lzw')
        with rasterio.open(save_path,'w',**profile) as dst_dataset:
            dst_dataset.write(img)



def stitch_img(imgList,hor_flag=1,margin=0,outpath=None):
    """
    stich images in imgList horizontal
    hor_flag:1 horizontal stitch else vertical stitch
    """
    imgList=[img if len(img.shape)==3 else np.array([img,img,img]).transpose(1,2,0) for img in imgList ]
    channels=[_.shape[2] for _ in imgList]
    imgs_len=len(imgList)
    assert (np.array(channels)==3).all() or (np.array(channels)==1).all()
    hor_shape=[_.shape[1] for _ in imgList]
    ver_shape=[_.shape[0] for _ in imgList]
    hor=np.sum(hor_shape)+margin*(imgs_len-1) if hor_flag==1 else np.max(hor_shape)
    ver=np.max(ver_shape) if hor_flag==1 else np.sum(ver_shape)+margin*(imgs_len-1)
    stitch=np.ones((ver,hor,3)).astype(np.uint8)
    stitch=stitch*255
    for num,img in enumerate(imgList):
        ver_start=0 if hor_flag==1 else np.sum(ver_shape[0:num])+margin*num
        ver_end=img.shape[0] if hor_flag==1 else np.sum(ver_shape[0:num+1])+margin*num
        hor_start=np.sum(hor_shape[0:num])+margin*num if hor_flag==1 else 0
        hor_end=np.sum(hor_shape[0:num+1])+margin*num if hor_flag==1 else img.shape[1]
        stitch[int(ver_start):int(ver_end),int(hor_start):int(hor_end)]=img
    return stitch
def color_val(color = None):
    color_nums=len(COLORS_DICT)
    if isinstance(color,str):
        color = color[0].upper() + color[1:].lower()
        return list(COLORS_DICT[color])[::-1]
    elif color == None:
        color_name = random.choice(list(COLORS_DICT.keys()))
        return list(COLORS_DICT[color_name])[::-1]
    elif type(color) == int:
        return list(COLORS_DICT[list(COLORS_DICT.keys())[color%color_nums]])[::-1]
    else:
        return list(COLORS_DICT['Red'])[::-1]

def angle_to_img( angle_t,max_angle=2*np.pi,ignore_index=-2,no_angle=-1):
    """
    label map to rgb color image, for display
    no_angle shows in grey and ignore shows in white
    """
    if isinstance(angle_t,torch.Tensor):
        angle=angle_t.clone().cpu().detach().numpy()
    else:
        angle=angle_t.copy()
    H, W = angle.shape

    color = np.zeros((H, W, 3), np.float)
    color[...,1] = 255
    color[...,0] = (angle/max_angle)*180
    color[...,2] = (np.logical_not(angle == no_angle)).astype(np.float)/2*255
    color = np.clip(color,0,255).astype('uint8')

    color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)

    color[angle==no_angle, :] = 0
    color[angle<0.1, :] = 0
    color[angle==ignore_index, :] = 255

    return color
def rgb_to_img(img):
    def torch_tensor_to_ImageRGB(img,mean,std):
        for t,m,s in zip(img,mean,std):
            t.mul_(s).add_(m)
        img = img.numpy().transpose((1, 2, 0))
        return img


    if isinstance(img,torch.Tensor):
        img=img.clone().cpu().detach().numpy()
        # mean=[0,0,0]
        # std=[255,255,255]
        # rgb_img=torch_tensor_to_ImageRGB(img,mean,std)
    if img.shape[0]==3:
        img=img.transpose(1,2,0)
    rgb_img=cv2.normalize(img,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return rgb_img
def loss_to_img(img):
    if isinstance(img,torch.Tensor):
        img=img.clone().detach().cpu().numpy()
    img=cv2.normalize(img,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return img

def seg_to_img(img,ignore_label=255):
    seg_result=img
    if type(seg_result)==torch.Tensor:
        seg_result=seg_result.clone().cpu().detach().numpy()
    if len(seg_result.shape)==3:#把各个通道的预测值转为预测结果（类别）
        seg_result2=np.argmax(seg_result,axis=0)
        seg_result=seg_result2
    assert len(seg_result.shape)==2
    ignore_mask=(seg_result==ignore_label)
    seg_result[seg_result==ignore_label]=0
    num_classes=int(np.max(seg_result))
    seg_result2=np.zeros((num_classes+1,seg_result.shape[0],seg_result.shape[1]))
    label_color=np.zeros((seg_result.shape[0],seg_result.shape[1],3))
    for index in range(num_classes+1):
        seg_result2[index][seg_result==index]=1
    seg_result=seg_result2
    _=np.array([seg_result[0] for i in range(3)]).transpose(1,2,0)
    # rgb_img_back=rgb_img*_
    for index in range(seg_result.shape[0]-1):
        index=index+1
        _=np.array([seg_result[index],seg_result[index],seg_result[index]]).transpose(1,2,0)
        color=np.array(color_val(index)).reshape(1,1,3)
        label_color=label_color+_*color
    label_color=label_color.astype(np.uint8)
    label_color[ignore_mask]=255
    return label_color

def reg_to_img(img,ignore_value=np.nan,norm_flag=False):
    flow_re=img
    ignore_where=None

    if isinstance(flow_re,torch.Tensor):
        flow_tensor=flow_re.clone().cpu().detach()
        if ignore_value is not None:
            if ignore_value is np.nan:
                ignore_where = np.where(torch.isnan(flow_tensor))
                flow_tensor[ignore_where]=0
        flow=flow_tensor.numpy()
    else:
        flow=flow_re.copy()
    if flow.shape[0]==2 or flow.shape[0]==3:
        flow=flow.transpose(1,2,0) #[C,H,W]转[H,W,C]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    if norm_flag:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv[..., 2] =np.clip((mag*3),0,255).astype(np.uint8) #magnitude归到0～255之间
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)#[H,W,3]
    if ignore_where is not None:
#ignore_where is 3 elements,means the ignore index in C,H,W channels; we only need H,W channels' ignore index to show where value is ignore_value
        bgr[ignore_where[1:3]]=255
    return bgr



#主函数：
def result_to_img(img,img_flag):
    if img_flag=='rgb_img':
        rgb_img=rgb_to_img(img)
        return rgb_img
    elif img_flag=='loss':#loss是单通道，且需要归一化，并在图中标明最大最小值,如果已有原图，需要和原图叠加后显示
        loss_img=loss_to_img(img)
        imgs_vis['loss_img']=loss_img
        if 'rgb_img' in imgs_vis.keys():
            loss_img_fusion=fusion_img(img_vis['rgb_img'],loss_img)
            imgs_vis['loss_img_fusion']=loss_img_fusion
        return loss_img
    elif img_flag=='feature':#多通道的feature可以先降维到3维度，然后显示
        pass
    elif img_flag=='seg_result':
        seg_img=seg_to_img(img)
        return seg_img
    elif img_flag=='reg_result':
        reg_img=reg_to_img(img)
        return reg_img
    elif img_flag=='angle_result':
        angle_img=angle_to_img(img)
        return angle_img
    else:
        pass



def init_vis(save_path,save_freq):
    global use_show_flag
    if not use_show_flag:
        return

    if dist.get_rank()!=0:
        return
    global g_save_path,g_save_freq
    g_save_freq=save_freq
    g_save_path=save_path
    if not os.path.isdir(g_save_path):
        os.mkdir(g_save_path)


def init_img():
    global use_show_flag
    if not use_show_flag:
        return

    if dist.get_rank()!=0:
        return
    global imgs_vis,iters
    imgs_vis={}
    iters=iters+1
    return imgs_vis,iters

def add_img(imgs,img_flags,line_flag=0,outpath=None):
    global use_show_flag
    if not use_show_flag:
        return
    """
    imgs:lists or img,
    end:是否保存，end=True表示图像添加结束，保存下来
    img_flags: RGBimg,loss,seg_result,reg_result
    line_flag:which line to add current image,line_flag=0 means newline;line_flag=-1 means the previsou line
    """
    if dist.get_rank()!=0:
        return
    global imgs_vis,iters
    if iters%g_save_freq !=0:
        return
    if isinstance(imgs,list):
        img_list=[result_to_img(img,img_flag) for (img,img_flag) in zip(imgs,img_flags)]
        img_big=stitch_img(img_list)
    else:
        img_big=result_to_img(imgs,img_flags)
    if 'stitched_big_img' in imgs_vis.keys():
        imgs_vis['stitched_big_img']=stitch_img([imgs_vis['stitched_big_img'],img_big],hor_flag=0)
    else:
        imgs_vis['stitched_big_img']=img_big
    if outpath is not None:
        cv2.imwrite(outpath,img_big)

    return img_big

def save_img():
    global use_show_flag
    if not use_show_flag:
        return

    if dist.get_rank()!=0:
        return

    global imgs_vis,iters,g_save_path
    if iters%g_save_freq !=0:
        return

    if os.path.isdir(g_save_path):
        outpath=os.path.join(g_save_path,str(iters)+'.jpg')
        cv2.imwrite(outpath,imgs_vis['stitched_big_img'])

    # if os.path.isdir(outpath):
    #     outpath=os.path.join(outpath,str(iters)+'.jpg')
    # if outpath is not None:
    #     cv2.imwrite(outpath,imgs_vis['stitched_big_img'])
    return imgs_vis


