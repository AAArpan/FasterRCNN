import torch, wavemix
from wavemix.classification import WaveMix
import torch

model = WaveMix(
    num_classes= 1000, 
    depth= 16,
    mult= 2,
    ff_channel= 192,
    final_dim= 192,
    dropout= 0.5,
    level=3,
    patch_size=4,
)
img = torch.randn(1, 3, 640, 480).cuda()

preds = model.cuda()(img)