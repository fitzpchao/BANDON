# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:12:24 2019

@author: wujiang
"""

import numpy as np
import cv2

class OrientationUtil(object):
    """
    utils for orientation related
    """
    def __init__(self, num_bin=36, max_angle=2*np.pi):
        self.num_bin = num_bin
        self.max_angle = max_angle
        self.bin_width = self.max_angle/self.num_bin
        self.label_space_set_valid = set(list(range(1,self.num_bin+1)))
        self.label_space_set_gt    = self.label_space_set_valid | {0}
        self.label_space_set_pred  = self.label_space_set_gt | {255}

        self.repAngle_list = self.label_to_angle(list(self.label_space_set_valid))
        self.repNormVec_list = self.angle_to_normVec(list(self.repAngle_list))
    def angle_to_tangVec(self, angle):
        """
        angle to tangent vector (counter-clockwise for the outer boundary)
        Args:
            angle: in radius
        """
        if isinstance(angle, list):
            angle = np.asarray(angle)
        elif not isinstance(angle, np.ndarray):
            raise (RuntimeError("angle_to_tangVec only support list or numpy array.\n"))    
        
        angle = angle%(np.pi*2)
    
        assert(np.amax(angle) < 2*np.pi, "np.amax(angle) = {}".format(np.amax(angle)))
        assert(np.amin(angle) >= 0,      "np.amin(angle) = {}".format(np.amin(angle)))
        
        return np.cos(angle + np.pi/2.0), np.sin(angle + np.pi/2.0)


    def angle_to_normVec(self, angle):
        """
        angle to normal vector (pointing outer)
        Args:
            angle: in radius
        """
        if isinstance(angle, list):
            angle = np.asarray(angle)
        elif not isinstance(angle, np.ndarray):
            raise (RuntimeError("angle_to_normVec only support list or numpy array.\n"))    
        
        angle = angle%(np.pi*2)
    
        assert(np.amax(angle) < 2*np.pi, "np.amax(angle) = {}".format(np.amax(angle)))
        assert(np.amin(angle) >= 0,      "np.amin(angle) = {}".format(np.amin(angle)))
        
        return np.cos(angle), np.sin(angle)
        
    def angle_to_label(self, angle):
        """
        angle to label
        Args:
            angle: in radius
        """
        if isinstance(angle, list):
            angle = np.asarray(angle)
        elif not isinstance(angle, np.ndarray):
            raise (RuntimeError("angle_to_label only support list or numpy array.\n"))    
        
        angle = angle%(np.pi*2)
    
        assert(np.amax(angle) < 2*np.pi)
        assert(np.amin(angle) >= 0)
        label = np.round(angle/self.bin_width) + 1
        label[label == self.num_bin+1] = 1
        assert set(np.unique(label)).issubset(self.label_space_set_valid)
    
        return label
    
    def label_to_angle(self, label):
        """
        label to angle
        """
        if isinstance(label, list):
            label = np.asarray(label)
        elif not isinstance(label, np.ndarray):
            raise (RuntimeError("label_to_angle only support list or numpy array.\n"))
            
        assert(set(np.unique(label)).issubset(self.label_space_set_pred),
               "np.unique(label) is {}\n, label_space_set_pred is {}".format(np.unique(label), self.label_space_set_pred))
        angle = np.float32(label-1)*self.bin_width
        angle[label==0]   = np.nan
        angle[label==255] = np.nan
        return angle
    
    def label_to_color(self, label):
        """
        label map to rgb color image, for display
        """
         
        angle = self.label_to_angle(label)
       
        H, W = label.shape
    
        color = np.zeros((H, W, 3), np.float)
        color[...,1] = 255
        color[...,0] = angle/(self.max_angle)*180
        color[...,2] = (np.logical_not(label == 0)).astype(np.float)/2*255
        color = np.clip(color,0,255).astype('uint8')
        
        color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        
        color[label==255, :] = 255
        
        return color

    def get_range_from_angle(self, angle, hf_angle_thr):
        assert(hf_angle_thr >= 0)
        assert(hf_angle_thr < np.pi*2)
        hf_angle_thr = hf_angle_thr + 0.001
        #print("hf_angle_thr = {}".format(hf_angle_thr))
        
        if isinstance(angle, list):
            angle = np.asarray(angle)
        elif not isinstance(angle, np.ndarray):
            raise (RuntimeError("get_range_from_angle only support list or numpy array.\n"))
    
        
        angle = angle%(np.pi*2)
        norm_vec = self.angle_to_normVec(angle)
    
        rep_angle_in_range = []
        for ag in self.repAngle_list:
            ag_norm_vec = self.angle_to_normVec([ag])
            if np.arccos(np.clip(norm_vec[0]*ag_norm_vec[0] 
                       + norm_vec[1]*ag_norm_vec[1], -1, 1)) < hf_angle_thr:
                rep_angle_in_range.append(ag)
           
        label_in_range = self.angle_to_label(rep_angle_in_range).tolist()
        
        return label_in_range, rep_angle_in_range
    
    def get_reversed_from_label(self, label, hf_angle_thr):
        assert(hf_angle_thr >= 0)
        assert(hf_angle_thr < np.pi*2)
        hf_angle_thr = hf_angle_thr + 0.001
    
        angle = self.label_to_angle(label)
        rev_angle = (angle + np.pi)%(np.pi*2)
        
        return self.get_range_from_angle(rev_angle, hf_angle_thr)
           
    def get_rep_norm_vec_map(self, N, H, W):
        rep_norm_vec_map_h = np.empty((N, self.num_bin, H, W))
        rep_norm_vec_map_v = np.empty((N, self.num_bin, H, W))
        for b in range(self.num_bin):
            rep_norm_vec_map_h[:,b,:,:] = self.repNormVec_list[0][b]
            rep_norm_vec_map_v[:,b,:,:] = self.repNormVec_list[1][b]
        return rep_norm_vec_map_h, rep_norm_vec_map_v

    def get_interpolated_angle(self, edge_orient_feat, top_k=3):
        """
        interpolate the angle from top_k channel
        Args:
            edge_orient_feat (np.ndarray): probability (logits after softmax),
                size: N x C x H x W
            top_k (int): top_k channel for interpolation.
        """ 
        if top_k not in {3}:
            raise NotImplementedError
        assert isinstance(edge_orient_feat, np.ndarray)
        assert edge_orient_feat.ndim == 4
        N, C, H, W = edge_orient_feat.shape
        assert self.num_bin + 1 == C

        edge_orient_label = np.argmax(edge_orient_feat, axis=1).astype(np.uint8)
        assert set(np.unique(edge_orient_label)).issubset(set(range(0, self.num_bin + 1)))

        edge_orient_mat_topk = edge_orient_feat[:, 1:, :, :]
        for c in range(self.num_bin):
            if c == 0:
                c1 = int(self.num_bin - 1)
                c2 = int(c)
                c3 = int(c + 1)
            elif c == self.num_bin - 1:
                c1 = int(c - 1)
                c2 = int(c)
                c3 = int(0)
            else:
                c1 = int(c - 1)
                c2 = int(c)
                c3 = int(c + 1)

            assert set([c1, c2, c3]).issubset(set(range(0, self.num_bin)))
            topk_mask = np.logical_or.reduce((
                (edge_orient_label == c1 + 1),
                (edge_orient_label == c2 + 1),
                (edge_orient_label == c3 + 1),
            )).astype(np.float)
            edge_orient_mat_topk[:, c, :, :] = np.multiply(
                edge_orient_mat_topk[:, c, :, :],
                topk_mask)

        rep_norm_vec_map_h, rep_norm_vec_map_v = self.get_rep_norm_vec_map(N, H, W)
        norm_vec_map_h = np.multiply(rep_norm_vec_map_h, edge_orient_mat_topk)
        norm_vec_map_v = np.multiply(rep_norm_vec_map_v, edge_orient_mat_topk)
        norm_vec_map_h = np.sum(norm_vec_map_h, axis=1)
        norm_vec_map_v = np.sum(norm_vec_map_v, axis=1)

#       edge_feat_topk = np.sum(edge_orient_mat_topk, axis=2)
#       assert(np.max(edge_feat_topk) <= 1.0)
#       assert(np.min(edge_feat_topk) >= 0.0)
#       edge_feat_topk = (edge_feat_topk*255).astype(np.uint8)

        interpolated_orient_angle = np.arctan2(norm_vec_map_v, norm_vec_map_h)
        interpolated_orient_angle = interpolated_orient_angle%(np.pi*2)
        assert(np.amax(interpolated_orient_angle) < 2*np.pi)
        assert(np.amin(interpolated_orient_angle) >= 0)

        return interpolated_orient_angle

#######################################################################
def label_to_color__folder(src_dir, dst_dir, num_bin=36):
    from os import makedirs
    import os.path as osp
    from glob import glob
    from skimage.io import imread, imsave

    ortUtil = orientation_util(num_bin)

    makedirs(dst_dir, exist_ok=True)
    fn_lst = glob(osp.join(src_dir, "*.png"))
    for fn in fn_lst:
        fn_base = osp.basename(fn)
        fn_save = osp.join(dst_dir, fn_base)
        if osp.isfile(fn_save):
            continue
        print("converting {} from label to color ...".format(fn_base))

        label = imread(fn)
        color = ortUtil.label_to_color(label)
        imsave(fn_save, color)

#######################################################################

def draw_rectangle(ortUtil):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    
    dH  = 100
    gap = 10
    out_H = (dH + gap) * (ortUtil.num_bin + 2)
    out_W = 600
    
    out_label = np.zeros((out_H, out_W), dtype=np.uint8)
    label_list = []
    
    for h in range(0, ortUtil.num_bin+1):
        h_bgn = (dH+gap) * h
        h_end = h_bgn + dH
        out_label[h_bgn:h_end,:] = h
        
        label_list.append(h)
        
    angle_list = ortUtil.label_to_angle(label_list).tolist()
        
    out_label[-dH:,:] = 255  
    
    color = ortUtil.label_to_color(out_label)
    
    color = Image.fromarray(color, 'RGB')
    draw = ImageDraw.Draw(color)
    font = ImageFont.truetype("arial.ttf", 40)

    for h in range(0, ortUtil.num_bin+1):
        h_bgn = (dH+gap) * h
        draw.text((round(out_W*0.1), h_bgn+round(dH*0.1)),
                  "label = {}, angle = {:.4f}".format(label_list[h], angle_list[h]),
                  'white', font=font)
                 
    color.save("orient_color_bin{}_rectangle.png".format(ortUtil.num_bin))
    
def draw_circle(ortUtil):
    
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
    for l in range(1, ortUtil.num_bin+1):    
        label_list.append(l)
        
    print(label_list)

    angle = ortUtil.label_to_angle(label_list).tolist()
    print(angle)
    norm_vec = ortUtil.angle_to_normVec(angle)
    
    for i, l in enumerate(label_list):
        h = h0 + round(norm_vec[0][i] * 1.5*R)
        w = w0 + round(norm_vec[1][i] * 1.5*R)
        rr, cc = circle(h, w, 8*180.0/float(ortUtil.num_bin))
        out_label[rr, cc] = l
        
    color = ortUtil.label_to_color(out_label)
   #imsave("orient_color_bin{}_circle.png".format(num_bin), color)


    color = Image.fromarray(color, 'RGB')
    draw = ImageDraw.Draw(color)
    font = ImageFont.truetype("arial.ttf", 40)
    
    for i, l in enumerate(label_list):
        h = h0 + round(norm_vec[0][i] * 1.5*R) - 20
        w = w0 + round(norm_vec[1][i] * 1.5*R) - 20
        
        if ortUtil.num_bin > 40:
            if i%10 == 0:
                draw.text((w, h), str(l), 'white', font=font)
        else:
            draw.text((w, h), str(l), 'white', font=font)
        
                         
    color.save("orient_color_bin{}_circle.png".format(ortUtil.num_bin))

if __name__ == "__main__":
 
    #ortUtil = orientation_util(num_bin=36)
    #draw_rectangle(ortUtil)
    #draw_circle(ortUtil)
    #print(ortUtil.repNormVec_list)

#   src_dir = "caffe_output_orientation/luosida190719"
#   src_dir = "_convert_orientation/sdk_out__M_Remote_Segment_Building_1.1.1.model__luosida190719_2"
#   src_dir = "_convert_orientation/sdk_out__M_Remote_Segment_Building_1.1.1.model__luosida190719_1.0"
    src_dir = "_convert_orientation/sdk_out__M_Remote_Segment_Building_1.1.1.model__luosida190719_0.666666666667"
    dst_dir = src_dir + "_color"
    label_to_color__folder(src_dir, dst_dir)
