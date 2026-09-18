"""评测指标计算"""
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def classification_metrics(y_true, y_score, threshold=0.5):
    """图像级分类指标"""
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
    return {"Acc": acc, "AUC": auc, "F1": f1, "R.Acc": r_acc, "F.Acc": f_acc}


def localization_metrics(heatmaps, masks, thresholds=None):
    """像素级定位指标"""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 256)
    best_f1 = 0
    best_iou = 0
    for th in thresholds:
        preds = (heatmaps >= th).astype(np.float32)
        tp = (preds * masks).sum()
        fp = (preds * (1 - masks)).sum()
        fn = ((1 - preds) * masks).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
        if iou > best_iou:
            best_iou = iou
    return {"mIoU": best_iou, "F1": best_f1}


def mask_area_bin(mask, ratio=True):
    """计算mask面积占比并分档"""
    if isinstance(mask, np.ndarray):
        area = mask.mean() if ratio else mask.sum()
    else:
        area = mask.float().mean().item() if ratio else mask.sum().item()
    if area < 0.02:
        return "极小(<2%)", area
    elif area < 0.05:
        return "小(2-5%)", area
    elif area < 0.15:
        return "中(5-15%)", area
    else:
        return "大(>15%)", area
