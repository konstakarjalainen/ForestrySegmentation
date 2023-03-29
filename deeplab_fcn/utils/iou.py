from torchmetrics.functional import jaccard_index


def iou_pytorch(n_cls,preds,trues):
    iou = jaccard_index(preds, trues, num_classes=n_cls)
    return iou.item()
