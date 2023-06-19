import os
import numpy as np
import cv2
import cv2
from tqdm import tqdm
import argparse
# import torch
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def intersectionAndUnionWithPred(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target,area_output

def getlabelpath(a,root,flag='test_ood',index=0):
    flags = a.split("/")

    fn = flags[-1]


    if(flag=='test_ood'):
        path1 = os.path.join(root, 'BANDON_test_ood/gray')
    else:
        path1 = os.path.join(root,'BANDON_test/gray')


    path = os.path.join(path1,str(index) + '_' +fn)
    return path

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def cal_matrix(root_dataset,pred_path,target_txt,flag='test_ood'):


    list_read = open(target_txt).readlines()
    print("Totally {} samples.".format(len(list_read)))
    print("Starting Checking image&label 'pair'...")




    fn_list = []
    for line in list_read:
        line = line.strip()
        if ',' in line:
            line_split = line.split(',')
        else:
            line_split = line.split(' ')
        line_split = [s[1:] for s in line_split if s.startswith('/')]
        line_split = [os.path.join(root_dataset,flag, s) for s in line_split]
        fn_list.append(line_split[2])


    intersection_seg_meter = AverageMeter()
    union_seg_meter = AverageMeter()
    target_seg_meter = AverageMeter()
    pred_seg_meter = AverageMeter()
    count = 0
    for i, fn in tqdm(enumerate(fn_list)):
        pred_fn = getlabelpath(fn, pred_path,flag,i)

        count += 1

        target = cv2.imread(fn, 0)
        # target[target==255]=1

        if not os.path.exists(pred_fn):
            print(fn, pred_fn)
            print('ONE Missing!')
            continue

        pred = cv2.imread(pred_fn, 0)
        pred[pred == 255] = 1

        intersection_seg, union_seg, target_seg, pred_seg = intersectionAndUnionWithPred(pred, target, 2, 255)
        intersection_seg_meter.update(intersection_seg)
        union_seg_meter.update(union_seg)
        target_seg_meter.update(target_seg)
        pred_seg_meter.update(pred_seg)

    iou_class = intersection_seg_meter.sum / (union_seg_meter.sum + 1e-10)
    recall = intersection_seg_meter.sum / (target_seg_meter.sum + 1e-10)
    precision = intersection_seg_meter.sum / (pred_seg_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    allAcc = sum(intersection_seg_meter.sum) / (sum(target_seg_meter.sum) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall)
    return (iou_class,recall,precision,mIoU,allAcc,f1)
def print_results(resluts,flag):
    iou_class,recall,precision,mIoU,allAcc,f1=resluts
    print(flag)
    print("EVAL Iou class:{}".format(iou_class))
    print("EVAL Recall class:{}".format(recall))
    print("EVAL precision class:{}".format(precision))
    print("EVAL mIoU:{}".format(mIoU))
    print("EVAL F1:{}".format(f1))
def print_results_sample(resluts,flag):
    iou_class,recall,precision,mIoU,allAcc,f1=resluts
    print(flag)
    print("Iou/F1/Recall/Prec.: {:0.0f} {:0.0f} {:0.0f} {:0.0f}".format(iou_class[1] * 10000, f1[1] * 10000,
                                                                        recall[1] * 10000, precision[1] * 10000))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")


    parser.add_argument("--test_list_path", default='./lists/list_BANDON_test.txt',type=str, help="the path of testset list")
    parser.add_argument("--test_ood_list_path", default='./lists/list_BANDON_test_ood.txt', type=str, help="the path of test_ood set list")
    parser.add_argument("--pred_path", default='/root/BANDON/workdirs_bandon/MTGCDNet/config_GPU8/iter_40000', type=str, help="the path of predict label")
    parser.add_argument("--root_dataset", default='/remote-home/pangchao/data/BANDON', type=str, help="the root of BANDON dataset")


    args = parser.parse_args()

    root_dataset = args.root_dataset
    pred_path = args.pred_path
    target_txt_1 = args.test_list_path
    target_txt_2 = args.test_ood_list_path
    print('process testset...')
    results1 = cal_matrix(root_dataset,pred_path, target_txt_1,flag='test' )
    print('process test_ood set...')
    results2 = cal_matrix(root_dataset,pred_path, target_txt_2, flag='test_ood')
    print('+'*100)

    print_results_sample(results1, 'test')
    print('+' * 100)
    print_results_sample(results2, 'test_ood')
    print('+' * 100)




