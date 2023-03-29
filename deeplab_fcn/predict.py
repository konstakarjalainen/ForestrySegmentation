import matplotlib.image as mpimg

import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == '__main__':
    pth = Path('./wandb/latest-run/files/media/images/')
    img_files = list(pth.glob('**/images_*'))
    id = 0
    for file in img_files:
        id_n = int(file.name.split('_')[1])
        if id_n > id:
            id = id_n
    id = str(id)

    img_pth = list(pth.glob('**/images_' + id + '*'))[0]
    pred_pth = list(pth.glob('**/pred_' + id + '*'))[0]
    true_pth = list(pth.glob('**/true_' + id + '*'))[0]

    img = mpimg.imread(img_pth)
    pred = mpimg.imread(pred_pth)
    true = mpimg.imread(true_pth)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Original image')
    ax = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(pred[:,:,0])
    ax.set_title('Predicted mask')
    ax = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(true[:,:,0])
    ax.set_title('Ground truth')
    plt.show()
