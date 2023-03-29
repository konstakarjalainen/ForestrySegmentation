import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.mean_iou import mIOU


def evaluate(net, dataloader, device, n_cls):
    net.eval()
    num_val_batches = len(dataloader)
    iou = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)['out']
            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred = mask_pred.argmax(dim=1)
            iou += mIOU(n_cls, mask_pred, mask_true)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return iou
    return iou / num_val_batches
