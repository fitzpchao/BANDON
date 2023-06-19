# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:12:24 2019

@author: wujiang
"""
import os, sys
import os.path as osp
import numpy as np
import cv2

from skimage.io import imread, imsave

num_bin = 36
bin_width = 2*np.pi/num_bin
label_space_set_valid = set(list(range(1,num_bin+1)))
label_space_set_gt    = label_space_set_valid | {0}
label_space_set_pred  = label_space_set_gt | {255}


def angle_to_tangVec(angle):
    if isinstance(angle, list):
        angle = np.asarray(angle)
    elif not isinstance(angle, np.ndarray):
        raise (RuntimeError("angle_to_tangVec only support list or numpy array.\n"))    
    
    angle = angle%(np.pi*2)

    assert(np.amax(angle) < 2*np.pi)
    assert(np.amin(angle) >= 0)
    
    return np.cos(angle + np.pi/2.0), np.sin(angle + np.pi/2.0)


def angle_to_normVec(angle):
    if isinstance(angle, list):
        angle = np.asarray(angle)
    elif not isinstance(angle, np.ndarray):
        raise (RuntimeError("angle_to_normVec only support list or numpy array.\n"))    
    
    angle = angle%(np.pi*2)

    assert(np.amax(angle) < 2*np.pi)
    assert(np.amin(angle) >= 0)
    
    return np.cos(angle), np.sin(angle)
    
def angle_to_label(angle):
    if isinstance(angle, list):
        angle = np.asarray(angle)
    elif not isinstance(angle, np.ndarray):
        raise (RuntimeError("angle_to_label only support list or numpy array.\n"))    
    
    angle = angle%(np.pi*2)

    assert(np.amax(angle) < 2*np.pi)
    assert(np.amin(angle) >= 0)
    #label = np.round((angle - bin_width/2.0) / bin_width) + 1
    label = np.round(angle/bin_width) + 1
    label[label == 37] = 1
    assert set(np.unique(label)).issubset(label_space_set_valid)

    return label

def label_to_angle(label):
    if isinstance(label, list):
        label = np.asarray(label)
    elif not isinstance(label, np.ndarray):
        raise (RuntimeError("label_to_angle only support list or numpy array.\n"))
        
    assert set(np.unique(label)).issubset(label_space_set_pred)
    angle = np.float32(label-1)*bin_width
    angle[label==0]   = np.nan
    angle[label==255] = np.nan
    return angle

def label_to_color(label):
     
    angle = label_to_angle(label)
   
    H, W = label.shape

    color = np.zeros((H, W, 3), np.float)
    color[...,1] = 255
    color[...,0] = angle/(np.pi*2)*180
    color[...,2] = (np.logical_not(label == 0)).astype(np.float)/2*255
    color = np.clip(color,0,255).astype('uint8')
    
    color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
    
    color[label==255, :] = 255
    
    return color


#######################################################################

def draw_rectangle():
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    
    dH  = 100
    gap = 10
    out_H = (dH + gap) * (num_bin + 2)
    out_W = 600
    
    out_label = np.zeros((out_H, out_W), dtype=np.uint8)
    label_list = []
    
    for h in range(0, num_bin+1):
        h_bgn = (dH+gap) * h
        h_end = h_bgn + dH
        out_label[h_bgn:h_end,:] = h
        
        label_list.append(h)
        
    angle_list = label_to_angle(label_list).tolist()
        
    out_label[-dH:,:] = 255  
    
    color = label_to_color(out_label)
    
    color = Image.fromarray(color, 'RGB')
    draw = ImageDraw.Draw(color)
    font = ImageFont.truetype("arial.ttf", 40)

    for h in range(0, num_bin+1):
        h_bgn = (dH+gap) * h
        draw.text((round(out_W*0.1), h_bgn+round(dH*0.1)),
                  "label = {}, angle = {:.4f}".format(label_list[h], angle_list[h]),
                  'white', font=font)
                 
    color.save("orient_color_bin{}_rectangle.png".format(num_bin))
    
def draw_circle():
    
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    
    from skimage.draw import circle

    R = 300
    
    H = R*4
    W = R*4
    
    h0 = 2*R
    w0 = 2*R
    
    out_label = np.zeros((H, W), dtype=np.uint8)
    
    label_list = []
    for l in range(1, num_bin+1):    
        label_list.append(l)
        
    print(label_list)

    angle = label_to_angle(label_list).tolist()
    print(angle)
    norm_vec = angle_to_normVec(angle)
    
    for i, l in enumerate(label_list):
        h = h0 + round(norm_vec[0][i] * 1.5*R)
        w = w0 + round(norm_vec[1][i] * 1.5*R)
        rr, cc = circle(h, w, 30)
        out_label[rr, cc] = l
        
    color = label_to_color(out_label)
   #imsave("orient_color_bin{}_circle.png".format(num_bin), color)


    color = Image.fromarray(color, 'RGB')
    draw = ImageDraw.Draw(color)
    font = ImageFont.truetype("arial.ttf", 40)
    
    for i, l in enumerate(label_list):
        h = h0 + round(norm_vec[0][i] * 1.5*R) - 20
        w = w0 + round(norm_vec[1][i] * 1.5*R) - 20
        draw.text((w, h), str(l),
              'white', font=font)
        
                         
    color.save("orient_color_bin{}_circle.png".format(num_bin))

if __name__ == "__main__":
 
    draw_rectangle()
    draw_circle()
