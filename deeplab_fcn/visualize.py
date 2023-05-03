import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.models.segmentation as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

colormap_RUGD = np.array([[0, 0, 0],[108, 64, 20],[255, 229, 204],[0, 102, 0],[0, 255, 0],[0, 153, 153],[0, 128, 255],
                 [0, 0, 255],[255, 255, 0],[255, 0, 127],[64, 64, 64],[255, 128, 0],[255, 0, 0],[153, 76, 0],
                 [102, 102, 0],[102, 0, 0],[0, 255, 128],[204, 153, 255],[255, 153, 204],[0, 102, 102],[153, 204, 255],
                 [102, 255, 255],[101, 101, 11],[114, 85, 47]])

colormap_FOREST = np.array([[170, 170, 170], [0, 255, 0], [102, 102, 51], [0, 60, 0], [0, 120, 255], [0, 0, 0]])

if __name__ == '__main__':
    RUGD = True
    RUGD_img = Path('/home/konsta/PycharmProjects/Segmentation/RUGD_img.png')
    RUGD_mask = Path('/home/konsta/PycharmProjects/Segmentation/RUGD_mask.png')
    FOREST_img = Path('/home/konsta/PycharmProjects/Segmentation/FOREST_img.jpg')
    FOREST_mask = Path('/home/konsta/PycharmProjects/Segmentation/FOREST_mask.png')
    path_model_RUGD = Path('checkpoints/FCN_RUGD30.pth')
    path_model_FOREST = Path('checkpoints/FCN_Forest30.pth')

    if RUGD:
        cmap = colormap_RUGD
        num_classes = 25
        path_img = RUGD_img
        path_mask = RUGD_mask
        path_model = path_model_RUGD
    else:
        cmap = colormap_FOREST
        num_classes = 6
        path_img = FOREST_img
        path_mask = FOREST_mask
        path_model = path_model_FOREST

    #model = models.deeplabv3_resnet50(num_classes=num_classes)
    model = models.fcn_resnet50(num_classes=num_classes)
    image = Image.open(path_img)
    mask_img = Image.open(path_mask)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    t_img = transform(image)
    mask = transform(mask_img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device=device)
    img = t_img.to(device=device, dtype=torch.float32)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        mask_pred = model(img.unsqueeze(0))['out']
        mask_pred = F.softmax(mask_pred, dim=1)
        mask_pred = mask_pred.argmax(dim=1).cpu().numpy().squeeze()
    end_time = time.time()
    color_pred = cmap[mask_pred]
    print(end_time - start_time)
    plt.figure()
    plt.imshow(mask_img)
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(color_pred)
    plt.show()
