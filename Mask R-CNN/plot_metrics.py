import matplotlib.pyplot as plt
import numpy as np
import json


#metrics_file = 'outputs/cana100/metrics.json'
#metrics_file = 'outputs/cana100_synth/metrics.json'
metrics_file = 'outputs/log/metrics.json'
losses = []
ap = 0
ap50 = 0
ap75 = 0
ap_box = 0
ap50_box = 0
ap75_box = 0

with open(metrics_file, 'r') as file:

    for line in file:
        metrics = json.loads(line)
        if 'total_loss' in metrics:
            losses.append(metrics['total_loss'])
        if 'segm/AP' in metrics:
            if metrics['segm/AP'] > ap:
                ap = metrics['segm/AP']
            if metrics['segm/AP50'] > ap50:
                ap50 = metrics['segm/AP50']
            if metrics['segm/AP75'] > ap75:
                ap75 = metrics['segm/AP75']
            if metrics['bbox/AP'] > ap_box:
                ap_box = metrics['segm/AP']
            if metrics['bbox/AP50'] > ap50_box:
                ap50_box = metrics['segm/AP50']
            if metrics['bbox/AP75'] > ap75_box:
                ap75_box = metrics['segm/AP75']

print('Mask   AP:', ap, 'AP50:', ap50, 'AP75:', ap75)
print('Bbox   AP:', ap_box, 'AP50:', ap50_box, 'AP75:', ap75_box)
plt.figure()
plt.plot(losses)
plt.title('Training loss')
plt.xlabel('Iterations x 20')
plt.ylabel('Loss')
plt.show()
