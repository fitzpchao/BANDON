import numpy as np

from mmseg.core.evaluation import mean_iou


def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = np.bincount(inds, minlength=n**2).reshape(n, n)

    return mat


# This func is deprecated since it's not memory efficient
def legacy_mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(
            results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    iou = np.diag(total_mat) / (
        total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    return all_acc, acc, iou


def test_mean_iou():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    all_acc, acc, iou = mean_iou(results, label, num_classes, ignore_index)
    all_acc_l, acc_l, iou_l = legacy_mean_iou(results, label, num_classes,
                                              ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)
