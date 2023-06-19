import random
import math
import numpy as np
import numbers
import collections
from PIL import Image, ImageOps, ImageFilter

import torch
import torchvision

import skimage.filters
from skimage.segmentation import find_boundaries
from . import utils_orient



class Compose(object):
    """
    Composes several geometric_transforms together.

    Args:
        geometric_transforms (List[Transform]): list of geometric_transforms to compose.

    Example:
        geometric_transforms.Compose([
            geometric_transforms.CenterCrop(10),
            geometric_transforms.XXX()])
    """
    def __init__(self, geometric_transforms):
        self.geometric_transforms = geometric_transforms

    def __call__(self, sample):
        for t in self.geometric_transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    # Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        for n in range(len(image)):
            if isinstance(image[n], Image.Image):
                image[n] = np.asarray(image[n])/255
            if not  isinstance(image[n], np.ndarray):
                raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                    "[eg: data readed by PIL.Image.open()].\n"))
        if label is not None:
            if isinstance(label, Image.Image):
                label = np.asarray(label)
            if not isinstance(label, np.ndarray):
                raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                    "[eg: data readed by PIL.Image.open()].\n"))

        for n in range(len(image)):
            if len(image[n].shape) > 3 or len(image[n].shape) < 2:
                raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
            if len(image[n].shape) == 2:
                image[n] = np.expand_dims(image[n], axis=2)

        if label is not None:
            if not len(label.shape) == 2:
                raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        for n in range(len(image)):
            image[n] = torch.from_numpy(image[n].transpose((2, 0, 1)))
            if not isinstance(image[n], torch.FloatTensor):
                image[n] = image[n].float()

        if label is not None:
            label = torch.from_numpy(label)
            if not isinstance(label, torch.LongTensor):
                label = label.long()

        if 'label_roofcontour' in sample:
            label_roofcontour = torch.from_numpy(sample['label_roofcontour'])
            if not isinstance(label_roofcontour, torch.LongTensor):
                label_roofcontour = label_roofcontour.long()
            sample['label_roofcontour'] = label_roofcontour

        if 'label_edge_orient' in sample:
            label_edge_orient = torch.from_numpy(sample['label_edge_orient'])
            if not isinstance(label_edge_orient, torch.LongTensor):
                label_edge_orient = label_edge_orient.long()
            sample['label_edge_orient'] = label_edge_orient

        if 'label_roofside_origin' in sample:
            label_roofside_origin = torch.from_numpy(sample['label_roofside_origin'])
            if not isinstance(label_roofside_origin, torch.LongTensor):
                label_roofside_origin = label_roofside_origin.long()
            sample['label_roofside_origin'] = label_roofside_origin

        if 'label_roofside' in sample:
            label_roofside = torch.from_numpy(sample['label_roofside'])
            if not isinstance(label_roofside, torch.LongTensor):
                label_roofside = label_roofside.long()
            sample['label_roofside'] = label_roofside

        if 'label_foot' in sample:
            label_foot = torch.from_numpy(sample['label_foot'])
            if not isinstance(label_foot, torch.LongTensor):
                label_foot = label_foot.long()
            sample['label_foot'] = label_foot

        if 'label_edge5cls' in sample:
            label_edge5cls = torch.from_numpy(sample['label_edge5cls'])
            if not isinstance(label_edge5cls, torch.LongTensor):
                label_edge5cls = label_edge5cls.long()
            sample['label_edge5cls'] = label_edge5cls

        if 'label_roof' in sample:
            label_roof = torch.from_numpy(sample['label_roof'])
            if not isinstance(label_roof, torch.LongTensor):
                label_roof = label_roof.long()
            sample['label_roof'] = label_roof

        if 'label_side' in sample:
            label_side = torch.from_numpy(sample['label_side'])
            if not isinstance(label_side, torch.LongTensor):
                label_side = label_side.long()
            sample['label_side'] = label_side

        if 'label_offset_angle' in sample:
            label_offset_angle = torch.unsqueeze(torch.from_numpy(np.asarray(sample['label_offset_angle'])), 0)
            if not isinstance(label_offset_angle, torch.FloatTensor):
                label_offset_angle       = label_offset_angle.float()
                label_offset_angle =label_offset_angle*np.pi/180
                if label_offset_angle.item()>np.pi:
                     label_offset_angle=label_offset_angle-2*np.pi
                if label_offset_angle.item()<-np.pi:
                     label_offset_angle=label_offset_angle+2*np.pi
                assert -np.pi-0.01<=label_offset_angle.item()<=np.pi+0.01,label_offset_angle.item()
            sample['label_offset_angle'] = label_offset_angle

        if 'label_offset_angle_cls' in sample:
            label_offset_angle_cls = torch.unsqueeze(torch.from_numpy(np.asarray(sample['label_offset_angle_cls'])), 0)
            if not isinstance(label_offset_angle_cls, torch.LongTensor):
                label_offset_angle_cls = label_offset_angle.long()
            sample['label_offset_angle_cls'] = label_offset_angle_cls

        if 'label_offset_angle_cls_new' in sample:
            label_offset_angle_cls_new = torch.unsqueeze(torch.from_numpy(np.asarray(sample['label_offset_angle_cls_new'])), 0)
            if not isinstance(label_offset_angle_cls_new, torch.LongTensor):
                label_offset_angle_cls_new = label_offset_angle_cls_new.long()
            sample['label_offset_angle_cls_new'] = label_offset_angle_cls_new

        if 'side_area' in sample:
            side_area = torch.unsqueeze(torch.from_numpy(np.asarray(sample['side_area'])), 0)
            if not isinstance(side_area, torch.LongTensor):
                side_area = side_area.long()
            sample['side_area'] = side_area



        sample['image'] = image

        if label is not None:
            sample['label_seg'] = label
        else:
            del sample['label_seg']

        return sample


### added: calculating label_height_angle_cls according to label_offset_angle and label_side


class Normalize(object):
    """
    Given mean and std of each channel
    Will normalize each channel of the torch.*Tensor (C*H*W), i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']

        for n in range(len(image)):
            assert image[n].size(0) == len(self.mean)
            if self.std is None:
                for t, m in zip(image[n], self.mean):
                    t.sub_(m)
            else:
                for t, m, s in zip(image[n], self.mean, self.std):
                    t.sub_(m).div_(s)


        sample['image'] = image
        return sample

#class Resize(object):
#    """
#    Resize the input PIL Image to the given size.
#    'size' is a 2-element tuple or list in the order of (h, w)
#    """
#    def __init__(self, size):
#        assert (isinstance(size, collections.Iterable) and len(size) == 2)
#        self.size = size
#
#    def __call__(self, sample):
#        image = sample['image']
#        label = sample['label']
#
#        for n in range(len(image)):
#            image[n] = image[n].resize(self.size[::-1], Image.BILINEAR)
#        label = label.resize(self.size[::-1], Image.NEAREST)
#
#        sample['image'] = image
#        sample['label'] = label
#        return sample



### for both segmentation and angle prediction
class RandScale(object):
    """
    Randomly resize image & label with scale factor in [scale_min, scale_max]
    """
    def __init__(self, scale, aspect_ratio=None):
        assert None == aspect_ratio  # currently, aspect ratio is not supported
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] <= scale[1]: ### changed
            self.scale = scale
        else:
            raise (RuntimeError("segtransforms.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] <= aspect_ratio[1]: ### changed
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransforms.RandScale() aspect_ratio param error.\n"))

    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']

        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale * temp_aspect_ratio
        w, h = image[0].size
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)

        for n in range(len(image)):
            image[n] = image[n].resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)
        sample['image'] = image
        sample['label_seg'] = label
        sample['scale_factor']=(w,h,scale_factor_w,scale_factor_h)
        return sample


### for both segmentation and angle prediction
class Crop(object):
    """Crops the given PIL Image.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size,crop_coords=None, crop_type='center', padding=None, ignore_label=255, rand_pair_trans_offset=0):
        self.crop_coords=crop_coords
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))
        if isinstance(rand_pair_trans_offset, int):
            self.rand_pair_trans_offset = rand_pair_trans_offset
        else:
            raise (RuntimeError("rand_pair_trans_offset should be an integer number\n"))


    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        if self.crop_coords is not None:#crop the image and label according to the specify cords
            for n in range(len(image)):
                image[n] = image[n].crop(self.crop_coords)
            label = label.crop(self.crop_coords)


            sample['image'] = image
            sample['label_seg'] = label
            sample['crop_coords']=self.crop_coords
            return sample

        else:
            w, h = image[0].size
            pad_h = max(self.crop_h - h - 2*self.rand_pair_trans_offset, 0)
            pad_w = max(self.crop_w - w - 2*self.rand_pair_trans_offset, 0)
            pad_h_half = int(pad_h / 2)
            pad_w_half = int(pad_w / 2)
            if pad_h > 0 or pad_w > 0:
                if self.padding is None:
                    raise (RuntimeError("segtransforms.Crop() need padding while padding argument is None\n"))
                border = (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half)
                for n in range(len(image)):
                    image[n] = ImageOps.expand(image[n], border=border, fill=tuple([int(item) for item in self.padding]))
                label = ImageOps.expand(label, border=border, fill=self.ignore_label)
            w, h = image[0].size
            if self.crop_type == 'rand':
                h_off = random.randint(self.rand_pair_trans_offset, h - self.crop_h - self.rand_pair_trans_offset)
                w_off = random.randint(self.rand_pair_trans_offset, w - self.crop_w - self.rand_pair_trans_offset)
            else:
                h_off = (h - self.crop_h) // 2
                w_off = (w - self.crop_w) // 2

            if self.crop_type == 'rand':
                for n in range(len(image)):
                    h_off_n = h_off + random.randint( -1 * self.rand_pair_trans_offset, self.rand_pair_trans_offset)
                    w_off_n = w_off + random.randint( -1 * self.rand_pair_trans_offset, self.rand_pair_trans_offset)
                    image[n] = image[n].crop((w_off_n, h_off_n, w_off_n+self.crop_w, h_off_n+self.crop_h))
#               print("h_off_n = {}, w_off_n = {}".format(h_off_n, w_off_n))
            else:
                for n in range(len(image)):
                    image[n] = image[n].crop((w_off, h_off, w_off+self.crop_w, h_off+self.crop_h))
            label = label.crop((w_off, h_off, w_off+self.crop_w, h_off+self.crop_h))


            sample['image'] = image
            sample['label_seg'] = label
            sample['crop_coords']=(w_off,h_off,w_off+self.crop_w,h_off+self.crop_h)
            return sample



class RandRotate(object):
    """
    Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    """
    def __init__(self, rotate, padding, ignore_label=255):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransforms.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label

    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
        mask = Image.new('L', image[0].size, 255)
        mask = mask.rotate(angle, Image.NEAREST)

        image_bg = []

        for n in range(len(image)):
            image_bg.append(Image.new(image[n].mode, image[n].size, tuple([int(item) for item in self.padding])))
        label_bg = Image.new(label.mode, label.size, self.ignore_label)
        for n in range(len(image)):
            image_bg[n].paste(image[n].rotate(angle, Image.BILINEAR), mask)
        label_bg.paste(label.rotate(angle, Image.NEAREST), mask)

        label_offset_angle = sample['label_offset_angle']
        label_offset_angle_rotated = (label_offset_angle[0] + angle + 360) % 360

        sample['image'] = image_bg
        sample['label_seg'] = label_bg
        sample['label_offset_angle'] = [label_offset_angle_rotated]
        #sample['label_offset_angle_org'] = label_offset_angle
        sample['angle'] = angle
        return sample

class RandRotate90n(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        label_offset_angle = sample['label_offset_angle']

        rand_angle = random.randint(0,3)  ### changed
        if 0 == rand_angle:
#           print('no rotation')
            pass
        elif 1 == rand_angle:
#           print('rotate 90 degrees clockwise')
            for n in range(len(image)):
                image[n] = image[n].transpose(Image.ROTATE_90)
            label = label.transpose(Image.ROTATE_90)

        elif 2 == rand_angle:
#           print('rotate 180 degrees')
            for n in range(len(image)):
                image[n] = image[n].transpose(Image.ROTATE_180)
            label = label.transpose(Image.ROTATE_180)
        else:
#           print('rotate 270 degrees clockwise')
            for n in range(len(image)):
                image[n] = image[n].transpose(Image.ROTATE_270)
            label = label.transpose(Image.ROTATE_270)

        label_offset_angle_rotated = (label_offset_angle[0] + rand_angle*90) % 360

        sample['image'] = image
        sample['label_seg'] = label
        sample['label_offset_angle'] = [label_offset_angle_rotated]
        #sample['label_offset_angle_org'] = label_offset_angle
        sample['rand_angle'] = rand_angle
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        label_offset_angle = sample['label_offset_angle']

        label_offset_angle_rotated = int(label_offset_angle[0])
        hflip = 0

        sample['HorizontalFlip_flag']=False
        if random.random() < 0.5:
            sample['HorizontalFlip_flag']=True
            for n in range(len(image)):
                image[n] = image[n].transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label_offset_angle_rotated = (360 - label_offset_angle_rotated) % 360
            hflip = 1

        sample['image'] = image
        sample['label_seg'] = label
        sample['label_offset_angle'] = [label_offset_angle_rotated]
        #sample['label_offset_angle_org'] = label_offset_angle
        #sample['hflip'] = hflip

        return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        label_offset_angle = sample['label_offset_angle']

        label_offset_angle_rotated = int(label_offset_angle[0])
        vflip = 0

        sample['VerticalFlip_flag']=False
        if random.random() < 0.5:
            sample['VerticalFlip_flag']=True
            for n in range(len(image)):
                image[n] = image[n].transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
            label_offset_angle_rotated = (180 - label_offset_angle_rotated + 360) % 360
            vflip = 1

        sample['image'] = image
        sample['label_seg'] = label
        sample['label_offset_angle'] = [label_offset_angle_rotated]
        #sample['label_offset_angle_org'] = label_offset_angle
        #sample['vflip'] = vflip
        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample, radius=2):
        image = sample['image']

        for n in range(len(image)):
            if random.random() < 0.5:
                image[n] = image[n].filter(ImageFilter.GaussianBlur(radius))

        sample['image'] = image
        return sample

class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, sample):
        image = sample['image']

        for n in range(len(image)):
           r, g, b = image[n].split()
           image[n] = Image.merge('RGB', (b, g, r))

        sample['image'] = image

        return sample

class ColorJitter(object):
    def __init__(self, attr_dict):
        self.max_jitter = attr_dict['max_jitter']
        assert (self.max_jitter >= 0) and (self.max_jitter <= 1)
    def __call__(self, sample):
        image = sample['image']

        cj = torchvision.transforms.ColorJitter(brightness=self.max_jitter, contrast=self.max_jitter, saturation=self.max_jitter, hue=0.5*self.max_jitter)
        for n in range(len(image)):
            image[n] = cj(image[n])

        sample['image'] = image
        return sample

class RandChannelShiftScale(object):
    def __init__(self, attr_dict):
        self.max_color_shift = attr_dict['max_color_shift']
        self.min_contrast =  attr_dict['min_contrast']
        self.max_contrast =  attr_dict['max_contrast']
        self.max_brightness_shift = attr_dict['max_brightness_shift']

    def _color_shift(self, image):
        if self.max_color_shift > 0 and random.randint(0,1) == 1:
            C = image.shape[2]
            for c in range(C):
                shift = random.randint(0,self.max_color_shift)
                sign = random.randint(0,1)
                if sign == 1:
                    image[:,:,c] = image[:,:,c] + np.array([shift]).astype(np.float)
                else:
                    image[:,:,c] = image[:,:,c] - np.array([shift]).astype(np.float)
        return image

    def _contrast_brightness(self, image):
        if (not self.min_contrast==self.max_contrast==1) and random.randint(0,1) == 1:
            alpha = random.uniform(self.min_contrast,self.max_contrast)
            beta = random.randint(-self.max_brightness_shift,self.max_brightness_shift)
            image = image * alpha + beta
        return image

    def __call__(self, sample):
        image = sample['image']

        for n in range(len(image)):
            if isinstance(image[n], Image.Image):
                image[n] = np.asarray(image[n]).astype('int')
            if not  isinstance(image[n], np.ndarray):
                raise (RuntimeError("geometric_transforms.RandPixelNoise() only handle PIL Image and np.ndarray"
                                    "[eg: data readed by PIL.Image.open()].\n"))
        for n in range(len(image)):
            image[n] = self._color_shift(image[n])
            image[n] = self._contrast_brightness(image[n])
            image[n] = np.clip(image[n],0,255)
            image[n] = Image.fromarray(image[n].astype('uint8'), 'RGB')

        sample['image'] = image
        return sample

class RandPixelNoise(object):
    def __init__(self, attr_dict):
        self.max_pixel_noise = abs(round(attr_dict['max_pixel_noise']))
        if 'max_pixel_noise_instance_adjust' in attr_dict:
            self.max_pixel_noise_instance_adjust = attr_dict['max_pixel_noise_instance_adjust']
        else:
            self.max_pixel_noise_instance_adjust = 1  # set True as default
    def __call__(self, sample):
        image = sample['image']

        for n in range(len(image)):
            if isinstance(image[n], Image.Image):
                image[n] = np.asarray(image[n]).astype('int')
            if not  isinstance(image[n], np.ndarray):
                raise (RuntimeError("geometric_transforms.RandPixelNoise() only handle PIL Image and np.ndarray"
                                    "[eg: data readed by PIL.Image.open()].\n"))

        if self.max_pixel_noise > 0:
            for n in range(len(image)):
                if self.max_pixel_noise_instance_adjust:
                    noise = np.random.randint( random.randint( -1 * self.max_pixel_noise, 0) , random.randint( 1, self.max_pixel_noise), size=image[n].shape, dtype='int')
                else:
                    noise = np.random.randint(-1 * self.max_pixel_noise, self.max_pixel_noise, size=image[n].shape, dtype='int')

                image[n] = image[n] + noise
                image[n] = np.clip(image[n],0,255)

        for n in range(len(image)):
            image[n] = Image.fromarray(image[n].astype('uint8'), 'RGB')

        sample['image'] = image
        return sample


##### added
class label_final_angle_float_to_angle_cls(object):
    #
    def __init__(self):
        pass
    def __call__(self, sample):
        image = sample['image']
        label = sample['label_seg']
        label_side = sample['label_side']
        label_roof = sample['label_roof']

        label_offset_angle = sample['label_offset_angle']

        if isinstance(label, Image.Image):
            label = np.asarray(label)

        elif not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                "[eg: data readed by PIL.Image.open()].\n"))

        label_side_tmp = np.zeros_like(label_side)
        label_side_tmp[label_side==1] = 1
        area_side = sum(sum(label_side_tmp.astype(np.int32)))
        sample['area_side'] = area_side

        label_roof_tmp = np.zeros_like(label_roof)
        label_roof_tmp[label_roof==1] = 1
        area_roof = sum(sum(label_roof_tmp.astype(np.int32)))
        sample['area_roof'] = area_roof

        label_ignore_tmp = np.zeros_like(label)
        label_ignore_tmp[label==255] = 1
        area_ignore = sum(sum(label_ignore_tmp.astype(np.int32)))
        sample['area_ignore'] = area_ignore

        bin_angle = 10.0
        label_offset_angle_cls = int(float(label_offset_angle[0]) / bin_angle)
        sample['label_offset_angle_cls'] = [label_offset_angle_cls]

        ### threshold definition
        w, h = image[0].size
        total_area = w * h
        thr_ignore = 0.5
        thr_btm_side = 0.01
        thr_top_side = 0.03

        label_offset_angle_cls_new = label_offset_angle_cls + 1

        if (area_ignore > 0 and area_roof == 0):
            label_offset_angle_cls_new = 255
        elif (area_side > total_area * thr_btm_side and area_side < total_area * thr_top_side):
            label_offset_angle_cls_new = 255
        elif (area_roof == 0):
            label_offset_angle_cls_new = 0
        elif (area_side <= total_area * thr_btm_side):
            label_offset_angle_cls_new = 0

        sample['label_offset_angle_cls_new'] = [label_offset_angle_cls_new]

        return sample


def label1234_add_edgelabel(label):
    """
    input:
        label:np.array,contains{0,1,2,3,4,255}
    return:
        label_roofside_origin:only contain back_roof_side
        label_roofside_5edge:contain back_roof_side and 5 edges
    """
    img_orglabel=label.copy()

    img_rooflabel = img_orglabel.copy()
    img_rooflabel[img_orglabel==2] = 0
    img_rooflabel[img_orglabel==3] = 1
    img_rooflabel[img_orglabel==4] = 0

    img_sidelabel = img_orglabel.copy()
    img_sidelabel[img_orglabel==1] = 0
    img_sidelabel[img_orglabel==2] = 1
    img_sidelabel[img_orglabel==3] = 0
    img_sidelabel[img_orglabel==4] = 1

    img_roofsidelabel = img_orglabel.copy()
    img_roofsidelabel[img_orglabel==3] = 1
    img_roofsidelabel[img_orglabel==4] = 2

    img_footlabel = img_orglabel.copy()
    img_footlabel[img_orglabel==2] = 1
    img_footlabel[img_orglabel==3] = 0
    img_footlabel[img_orglabel==4] = 0

    img_edge5c = np.zeros(img_orglabel.shape)
    img_edge5c[img_orglabel==255] = 255
    img_final_edge5c = img_orglabel.copy()

    img_rooflabel[img_rooflabel==255] = 0
    roof_contour = find_boundaries(img_rooflabel,mode='outer')
    img_sidelabel[img_sidelabel==255] = 0
    side_contour = find_boundaries(img_sidelabel,mode='outer')
    img_footlabel[img_footlabel==255] = 0
    foot_contour = find_boundaries(img_footlabel,mode='outer')

    edge_roofside = roof_contour.copy()
    edge_roofside[img_sidelabel==0] = 0
    edge_roofback = roof_contour.copy()
    edge_roofback[img_sidelabel==1] = 0

    edge_sidenotroof = side_contour.copy()
    edge_sidenotroof[img_rooflabel==1] = 0
    edge_sideback = edge_sidenotroof.copy()
    edge_sideback[foot_contour==1] = 0
    edge_sidefoot = edge_sidenotroof.copy()
    edge_sidefoot[edge_sideback==1] = 0

    img_edge5c[edge_roofback==1] = 1
    img_edge5c[edge_roofside==1] = 2
    img_edge5c[edge_sideback==1] = 3
    img_edge5c[edge_sidefoot==1] = 4
    img_final_edge5c[edge_roofback==1] = 5
    img_final_edge5c[edge_roofside==1] = 6
    img_final_edge5c[edge_sideback==1] = 7
    img_final_edge5c[edge_sidefoot==1] = 8
    # sample['label_seg']=img_final_edge5c
    # sample['label_roofside_origin']=img_orglabel
    label_roofside_origin=img_orglabel
    label_roofside_5edge=img_final_edge5c
    return label_roofside_origin,label_roofside_5edge




class label1234_2_label12345678(object):#将标签backroofside1234转为backroofside1234+4条边5678
    def __call__(self,sample):
        label = sample
        unique_set = set(np.unique(label))
        assert unique_set.issubset(set([0,1,2,3,4,255]))
        label_roofside_origin,label_roofside_5edge=label1234_add_edgelabel(label)
        label_roofside_origin[label_roofside_origin==3]=1
        label_roofside_origin[label_roofside_origin==4]=2
        # sample['label_roofside_origin']=label_roofside_origin
        # sample['label_seg']=label_roofside_5edge
        return label_roofside_origin, label_roofside_5edge

##### added
class label_final_edge5cls_to_roof_roofcontour_roofside_edge5cls_edge1cls(object):
    #
    def __init__(self):
        pass
    def __call__(self, sample):

        label = sample

        unique_set = set(np.unique(label))
        assert unique_set.issubset(set([0,1,2,3,4,5,6,7,8,255]))

        # ------------ label_roof (background/roof/ignore: 0/1/255)
        label_roof = np.zeros_like(label)
        label_roof[label==0] = 0
        label_roof[label==1] = 1
        label_roof[label==2] = 0
        label_roof[label==3] = 1
        label_roof[label==4] = 0
        label_roof[label==5] = 0
        label_roof[label==6] = 0
        label_roof[label==7] = 0
        label_roof[label==8] = 0
        label_roof[label==255] = 255

        # ------------ label_roofcontour (background/contour/ignore: 0/1/255)
        label_roofcontour = np.zeros_like(label)
        label_roofcontour[label==0] = 0
        label_roofcontour[label==1] = 0
        label_roofcontour[label==2] = 0
        label_roofcontour[label==3] = 0
        label_roofcontour[label==4] = 0
        label_roofcontour[label==5] = 1
        label_roofcontour[label==6] = 1
        label_roofcontour[label==7] = 0
        label_roofcontour[label==8] = 0
        label_roofcontour[label==255] = 255

        # ------------ label_roofside (background/roof/side/ignore: 0/1/2/255)
        label_roofside = np.zeros_like(label)
        label_roofside[label==0] = 0
        label_roofside[label==1] = 1
        label_roofside[label==2] = 2
        label_roofside[label==3] = 1
        label_roofside[label==4] = 2
        label_roofside[label==5] =255
        label_roofside[label==6] =255
        label_roofside[label==7] =255
        label_roofside[label==8] = 255
        label_roofside[label==255] = 255

        # ------------ label_edge5cls (background/edge5/edge6/edge7/edge8/ignore: 0/1/2/3/4/255)
        label_edge5cls = np.zeros_like(label)
        label_edge5cls[label==0] = 0
        label_edge5cls[label==1] = 0
        label_edge5cls[label==2] = 0
        label_edge5cls[label==3] = 0
        label_edge5cls[label==4] = 0
        label_edge5cls[label==5] = 1
        label_edge5cls[label==6] = 2
        label_edge5cls[label==7] = 3
        label_edge5cls[label==8] = 4
        label_edge5cls[label==255] = 255

        # ------------ label_edge1cls (background/edge5678/ignore: 0/1/255)
        label_edge1cls = np.zeros_like(label)
        label_edge1cls[label==0] = 0
        label_edge1cls[label==1] = 0
        label_edge1cls[label==2] = 0
        label_edge1cls[label==3] = 0
        label_edge1cls[label==4] = 0
        label_edge1cls[label==5] = 1
        label_edge1cls[label==6] = 1
        label_edge1cls[label==7] = 1
        label_edge1cls[label==8] = 1
        label_edge1cls[label==255] = 255

        ### added (label_side)
        # ------------ label_side (background/side/ignore: 0/1/255)
        label_side = np.zeros_like(label)
        label_side[label==0] = 0
        label_side[label==1] = 0
        label_side[label==2] = 1
        label_side[label==3] = 0
        label_side[label==4] = 1
        label_side[label==5] = 0
        label_side[label==6] = 0
        label_side[label==7] = 0
        label_side[label==8] = 0
        label_side[label==255] = 255

        # ------------ label_foot (background/foot/ignore: 0/1/255)
        label_foot = np.zeros_like(label)
        label_foot[label==0] = 0
        label_foot[label==3] = 0
        label_foot[label==4] = 0
        label_foot[label==5] =255
        label_foot[label==6] =255
        label_foot[label==7] = 0
        label_foot[label==8] =1
        label_foot[label==1] = 1
        label_foot[label==2] = 1
        label_foot[label==255] = 255

        # -------------------------------------------------------
        # sample['label_roof'] = label_roof
        # sample['label_foot'] = label_foot#use
        # sample['label_roofcontour'] = label_roofcontour
        # sample['label_roofside'] = label_roofside  ### use
        # sample['label_edge5cls'] = label_edge5cls  ### use
        # sample['label_edge1cls'] = label_edge1cls
        # sample['label_side'] = label_side

        return label_roof,label_roofside,label_edge5cls,label_side


### use this
class label_roof_side_to_edgeorient(object):
    #
    def __init__(self, wmap_vec=[1,1,1,1,1], vertex_wmap=1):
        self.wmap_vec    = wmap_vec
        self.vertex_wmap = vertex_wmap
    def __call__(self, label_roof,label_side,label_seg):
        # image = sample['image']
        # label_roof = sample['label_roof']
        # label_side = sample['label_side']
        # label_seg = sample['label_seg']

        if isinstance(label_roof, Image.Image):
            label_roof = np.asarray(label_roof)
        elif not isinstance(label_roof, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                "[eg: data readed by PIL.Image.open()].\n"))

        if isinstance(label_side, Image.Image):
            label_side = np.asarray(label_side)
        elif not isinstance(label_side, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                "[eg: data readed by PIL.Image.open()].\n"))

        if isinstance(label_seg, Image.Image):
            label_seg = np.asarray(label_seg)
        elif not isinstance(label_seg, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle PIL Image and np.ndarray"
                                "[eg: data readed by PIL.Image.open()].\n"))


        unique_set = set(np.unique(label_roof))


        # ------------ label_edge_orient
        if unique_set.issubset(set([0,255])):  ### prev: [0,1,2,3,255]
            orient_label = np.ones_like(label_roof) * 255
            sobel_h_roof = np.zeros_like(label_roof)
            sobel_v_roof = np.zeros_like(label_roof)
            sobel_h_side = np.zeros_like(label_roof)
            sobel_v_side = np.zeros_like(label_roof)
            is_grad_valid = np.array([0])

        else:
            ### convert side to neg0pos1
            labelside_NEG0POS1 = np.zeros_like(label_side, dtype=np.uint8)
            labelside_NEG0POS1[label_side==0] = 0
            labelside_NEG0POS1[label_side==1] = 255
            labelside_NEG0POS1[label_side==255] = 0

            bw_build_side = (labelside_NEG0POS1 == 255)
            bw_edge_side = find_boundaries(bw_build_side, mode='outer')
            ### calculate orient_angle for side
            labelside_NEG0POS1_filter = skimage.filters.gaussian(labelside_NEG0POS1, sigma=2)
            #labelside_NEG0POS1_filter = labelside_NEG0POS1

            sobel_h_side = skimage.filters.sobel_h(labelside_NEG0POS1_filter)
            sobel_v_side = skimage.filters.sobel_v(labelside_NEG0POS1_filter)

            orient_angle_side = np.arctan2(sobel_v_side, sobel_h_side) + np.pi
            orient_angle_side[orient_angle_side == 2*np.pi] -= 2*np.pi

            assert(np.amax(orient_angle_side) < 2*np.pi)
            assert(np.amin(orient_angle_side) >= 0)

            orient_label_side = utils_orient.angle_to_label(orient_angle_side)

            orient_label_side[label_side == 255] = 255
            #orient_label_side[bw_edge_side == 0] = 0  ### to check
            orient_label_side[label_seg <= 6] = 0


            ### convert roof to neg0pos1
            labelroof_NEG0POS1 = np.zeros_like(label_roof, dtype=np.uint8)
            labelroof_NEG0POS1[label_roof==0] = 0
            labelroof_NEG0POS1[label_roof==1] = 255
            labelroof_NEG0POS1[label_roof==255] = 0

            bw_build_roof = (labelroof_NEG0POS1 == 255)
            bw_edge_roof = find_boundaries(bw_build_roof, mode='outer')

            ### calculate orient_angle for roof
            labelroof_NEG0POS1_filter = skimage.filters.gaussian(labelroof_NEG0POS1, sigma=2)
            #labelroof_NEG0POS1_filter = labelroof_NEG0POS1

            sobel_h_roof = skimage.filters.sobel_h(labelroof_NEG0POS1_filter)
            sobel_v_roof = skimage.filters.sobel_v(labelroof_NEG0POS1_filter)
            is_grad_valid = np.array([1])

            orient_angle_roof = np.arctan2(sobel_v_roof, sobel_h_roof) + np.pi
            orient_angle_roof[orient_angle_roof == 2*np.pi] -= 2*np.pi

            assert(np.amax(orient_angle_roof) < 2*np.pi)
            assert(np.amin(orient_angle_roof) >= 0)

            orient_label_roof = utils_orient.angle_to_label(orient_angle_roof)

            #orient_label_roof[bw_edge_roof == 0] = 0  ### to check
            orient_label_roof[label_seg <= 4] = 0
            orient_label_roof[label_seg >= 7] = 0
            orient_label_roof[label_roof == 255] = 255


            ### merge orient_label_side and orient_label_roof into orient_label
            orient_label = orient_label_side.copy()
            orient_label[orient_label_roof > 0] = orient_label_roof[orient_label_roof > 0]


            ###  assign label to ignore and non-contour pixels.
            #orient_label[label_roof == 255] = 255
            #orient_label[label_side == 255] = 255
            #orient_label[bw_edge == 0] = 0  ### to check


        # -------------------------------------------------------
        # sample['label_edge_orient'] = orient_label   ### add to toTensor
        # sample['is_grad_valid'] = is_grad_valid
        # sample['grad_h_roof'] = sobel_h_roof
        # sample['grad_v_roof'] = sobel_v_roof
        # sample['grad_h_side'] = sobel_h_side
        # sample['grad_v_side'] = sobel_v_side

        return orient_label


if __name__ == '__main__':

    def torch_tensor_to_ImageRGB(img):
        img = img.numpy().transpose((1, 2, 0)).astype('uint8')
        img = Image.fromarray(img, 'RGB')
        return img

    def torch_tensor_to_gray(lab):
        lab = lab.numpy()
        lab = Image.fromarray(lab.astype('uint8'), 'L')
        return lab

    def torch_tensor_to_label_0_1_255(lab):
        lab = lab.numpy()
        print("np.unique(lab) = {}".format(np.unique(lab)))
        lab[lab == 0] = 0
        lab[lab == 1] = 127
        lab = Image.fromarray(lab.astype('uint8'), 'L')
        return lab

    def torch_tensor_to_label_0_1_2_255(lab):
        lab = lab.numpy()
        lab[lab == 0] = 0
        lab[lab == 1] = 85
        lab[lab == 2] = 170
        lab = Image.fromarray(lab.astype('uint8'), 'L')
        return lab

    train_dict = {}
    train_dict['max_color_shift'] = 20
    train_dict['min_contrast'] = 0.8
    train_dict['max_contrast'] = 1.2
    train_dict['max_brightness_shift'] = 10
    train_dict['max_pixel_noise'] = 20
    train_dict['max_jitter'] = 0.1

    train_transform = Compose([
        RandRotate([-180, 180], padding=[0, 0, 0], ignore_label=0),
        RandScale([1.0, 2.0],[0.8, 1.2]),
        Crop([576, 576], crop_type='rand', padding=[0, 0, 0], ignore_label=0),
#       ColorJitter(train_dict),
#       RandomGaussianBlur(),
#       RandomHorizontalFlip(),
#       RandomVerticalFlip(),
#       RandRotate90n(),
#       RandChannelShiftScale(train_dict),
#       RandPixelNoise(train_dict),
        label_final_edge5cls_to_roof_roofcontour_roofside_edge5cls_edge1cls(),
        label_final_angle_float_to_angle_cls(),
        label_roof_side_to_edgeorient(wmap_vec=[1,1,1,1,1]),
        ToTensor()
    ])

    import segdata as datasets
    train_data = datasets.SegData(num_label_per_sample=1, data_root='',
            data_list='/mnt/lustre/wujiang/greenhouse/data/list_train_45678_X10.txt', ###
            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)



    import os
    import os.path as osp
    save_path1 = 'tmp/rgb'
    save_path2 = 'tmp/label_building'
    save_path3 = 'tmp/vrtx_label'
    save_path4 = 'tmp/vrtx_orient_label'
    save_path5 = 'tmp/edge_label'
    save_path6 = 'tmp/edge_orient_label'
    os.makedirs(save_path1, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)
    os.makedirs(save_path3, exist_ok=True)
    os.makedirs(save_path4, exist_ok=True)
    os.makedirs(save_path5, exist_ok=True)
    os.makedirs(save_path6, exist_ok=True)

    save_path7  = 'tmp/lab_sobel_h'
    save_path8  = 'tmp/lab_sobel_v'
    save_path9  = 'tmp/lab_sobel_h_sk'
    save_path10 = 'tmp/lab_sobel_v_sk'
    os.makedirs(save_path7 , exist_ok=True)
    os.makedirs(save_path8 , exist_ok=True)
    os.makedirs(save_path9 , exist_ok=True)
    os.makedirs(save_path10, exist_ok=True)

    for i, sample in enumerate(train_loader):
        image = sample['image']

        # ------- from skimage
        #lab_sobel_h_sk = skimage.filters.sobel_h(sample['label'].float()[0, ...].numpy())
        #lab_sobel_v_sk = skimage.filters.sobel_v(sample['label'].float()[0, ...].numpy())
        lab_sobel_h_sk = sample['grad_h']
        lab_sobel_v_sk = sample['grad_v']

        is_grad_valid = sample['is_grad_valid']

        if not is_grad_valid.byte().any():
            continue

        lab_sobel_h_sk = lab_sobel_h_sk.float()[0, ...].numpy()
        lab_sobel_v_sk = lab_sobel_v_sk.float()[0, ...].numpy()
        print("np.max(lab_sobel_h_sk) = {}".format(np.max(lab_sobel_h_sk)))
        print("np.min(lab_sobel_h_sk) = {}".format(np.min(lab_sobel_h_sk)))

        lab_sobel_h_sk = (lab_sobel_h_sk + 1)/2.0 * 255
        lab_sobel_v_sk = (lab_sobel_v_sk + 1)/2.0 * 255

        lab_sobel_h_sk = Image.fromarray(lab_sobel_h_sk.astype('uint8'), 'L')
        lab_sobel_v_sk = Image.fromarray(lab_sobel_v_sk.astype('uint8'), 'L')

        lab_sobel_h_sk.save(osp.join(save_path9, "{}.png".format(i)))
        lab_sobel_v_sk.save(osp.join(save_path10, "{}.png".format(i)))


        # ------- from conv
        import operation_via_conv as OvC
        lab_sobel_h = OvC.sobel_h(sample['label'].float(), cuda=False)
        lab_sobel_v = OvC.sobel_v(sample['label'].float(), cuda=False)
        print("torch.max(lab_sobel_h) = {}".format(torch.max(lab_sobel_h)))
        print("torch.min(lab_sobel_h) = {}".format(torch.min(lab_sobel_h)))

        lab_sobel_h = (lab_sobel_h + 1)/2.0 * 255
        lab_sobel_v = (lab_sobel_v + 1)/2.0 * 255

        lab_sobel_h = torch_tensor_to_gray(lab_sobel_h[0, 0, ...])
        lab_sobel_v = torch_tensor_to_gray(lab_sobel_v[0, 0, ...])

        lab_sobel_h.save(osp.join(save_path7, "{}.png".format(i)))
        lab_sobel_v.save(osp.join(save_path8, "{}.png".format(i)))

        # ---------------------------------------------------------------------
        rgb = torch_tensor_to_ImageRGB(image[0][0,...])
        lab = torch_tensor_to_label_0_1_255(sample['label'][0,...])
        vrtx_label = torch_tensor_to_label_0_1_2_255(sample['label_vertex'][0,...])
        edge_label = torch_tensor_to_label_0_1_255(sample['label_oth0ct1ig255'][0,...])

        label_vertex_orient = sample['label_vertex_orient'][0,...].numpy()
        vrtx_orient_color = utils_orient.label_to_color(label_vertex_orient)
        vrtx_orient_color = Image.fromarray(vrtx_orient_color.astype('uint8'), 'RGB')

        label_edge_orient = sample['label_edge_orient'][0,...].numpy()
        edge_orient_color = utils_orient.label_to_color(label_edge_orient)
        edge_orient_color = Image.fromarray(edge_orient_color.astype('uint8'), 'RGB')


        rgb.save(osp.join(save_path1, "{}.jpg".format(i)))
        lab.save(osp.join(save_path2, "{}.png".format(i)))
        vrtx_label.save(osp.join(save_path3, "{}.png".format(i)))
        vrtx_orient_color.save(osp.join(save_path4, "{}.png".format(i)))
        edge_label.save(osp.join(save_path5, "{}.png".format(i)))
        edge_orient_color.save(osp.join(save_path6, "{}.png".format(i)))


#       if np.unique(label_vertex_orient).size > 5:
#           break
        if 50 == i:
            break



'''
if __name__ == '__main__':

    def torch_tensor_to_ImageRGB(img):
        img = img.numpy().transpose((1, 2, 0))
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        return img
    def torch_tensor_to_label255(lab):
        lab = lab.numpy()
        print("np.unique(lab) = {}".format(np.unique(lab)))
        lab[lab == 1] = 50
        lab[lab == 2] = 100
        lab[lab == 3] = 150
        lab = Image.fromarray(lab.astype('uint8'), 'L')
        return lab

    def torch_tensor_to_grad255(grad):
        grad = grad.numpy().transpose((1, 2, 0))
        print("np.amax(grad) = {}".format(np.amax(grad)))
        print("np.amin(grad) = {}".format(np.amin(grad)))
        grad = (grad - np.amin(grad)) / (np.amax(grad) - np.amin(grad)) * 255.0
        grad = Image.fromarray(grad.astype('uint8'), 'RGB')
        return grad

    train_dict = {}
    train_dict['max_color_shift'] = 10
    train_dict['min_contrast'] = 0.9
    train_dict['max_contrast'] = 1.1
    train_dict['max_brightness_shift'] = 10
    train_dict['max_pixel_noise'] = 10
    train_dict['max_jitter'] = 0.3

    train_transform = Compose([
            RandRotate90n(),
            RandScale([0.5, 2.0]),
            RandRotate([-180, 180], padding=[100,200,50], ignore_label=255),
#           RandomGaussianBlur(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandChannelShiftScale(train_dict),
            RandPixelNoise(train_dict),
            ColorJitter(train_dict),
            Crop([1024, 1024], crop_type='rand', padding=[100,200,50], ignore_label=255, rand_pair_trans_offset=10),
#           SobelGradient_mag_sq_crs_scaled(),
#           label_bg0att1ch2att3_to_BG0CH1(),
            ToTensor()
            ])

    import geometric_data as datasets
    train_data = datasets.Geometric_Data(num_label_per_sample=6, data_root='',
            data_list='/mnt/lustre/wujiang/airbus/FR01/geometric_info/list_tophalf22.txt',
            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    for i, sample in enumerate(train_loader):
        if 0 == i:
            break

    image = sample['image']

    print(sample['image_path'][0][0])
    print("INCIDENCE_ANGLE_ALONG_TRACK  = {}".format(sample['INCIDENCE_ANGLE_ALONG_TRACK'][0][0]))
    print("INCIDENCE_ANGLE_ACROSS_TRACK = {}".format(sample['INCIDENCE_ANGLE_ACROSS_TRACK'][0][0]))
    print("INCIDENCE_ANGLE              = {}".format(sample['INCIDENCE_ANGLE'][0][0]))
    print("SUN_AZIMUTH                  = {}".format(sample['SUN_AZIMUTH'][0][0]))
    print("SUN_ELEVATION                = {}".format(sample['SUN_ELEVATION'][0][0]))

    print("incidence_vector_h = {}".format(sample['incidence_vector_h'][0][0]))
    print("incidence_vector_w = {}".format(sample['incidence_vector_w'][0][0]))
    print("incidence_vector_n = {}".format(sample['incidence_vector_n'][0][0]))
    print("sun_vector_h       = {}".format(sample['sun_vector_h'][0][0]))
    print("sun_vector_w       = {}".format(sample['sun_vector_w'][0][0]))
    print("sun_vector_n       = {}".format(sample['sun_vector_n'][0][0]))

    rgb0_0 = torch_tensor_to_ImageRGB(image[0][0,...])
#   rgb0_1 = torch_tensor_to_ImageRGB(image[1][0,...])
'''


