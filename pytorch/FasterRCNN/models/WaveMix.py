
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision
from enum import Enum
from ..datasets import image
from .backbone import Backbone
from torch.hub import load_state_dict_from_url
from wavemix import Level4Waveblock, Level3Waveblock, Level2Waveblock, Level1Waveblock
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchsummary 
from torchsummary import summary

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        num_classes=1000,
        depth = 16,
        mult = 2,
        ff_channel = 192,
        final_dim = 192,
        dropout = 0.5,
        level = 3,
        initial_conv = 'pachify', # or 'strided'
        patch_size = 4,
        stride = 2,

    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth): 
                if level == 4:
                    self.layers.append(Level4Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 3:
                    self.layers.append(Level3Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 2:
                    self.layers.append(Level2Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                else:
                    self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.extra = nn.Sequential(
            # nn.Identity(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        
        if initial_conv == 'strided':
            self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, stride, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, stride, 1)
        )
        else:
            self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/4),3, 1, 1),
            nn.Conv2d(int(final_dim/4), int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, patch_size, patch_size),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
            )
        

    def forward(self, image_data):
        print(image_data.shape)
        x = self.conv(image_data)   
            
        for attn in self.layers:
            x = attn(x) + x

        out = self.extra(x)

        return out


class PoolToFeatureVector(nn.Module):
  def __init__(self, dropout_probability):
    super().__init__()

    # Define network layers
    self._fc1 = nn.Linear(in_features = 512 * 7 * 7, out_features = 4096)
    self._fc2 = nn.Linear(in_features = 4096, out_features = 4096)

    # Dropout layers
    self._dropout1 = nn.Dropout(p = dropout_probability)
    self._dropout2 = nn.Dropout(p = dropout_probability)

  def forward(self, rois):
    """
    Converts RoI-pooled features into a linear feature vector suitable for use
    with the detector heads (classifier and regressor).

    Parameters
    ----------
    rois : torch.Tensor
      Output of RoI pool layer, of shape (N, 512, 7, 7).

    Returns
    -------
    torch.Tensor
      Feature vector of shape (N, 4096).
    """

    rois = rois.reshape((rois.shape[0], 512*7*7))  # flatten each RoI: (N, 512*7*7)
    y1o = F.relu(self._fc1(rois))
    y1 = self._dropout1(y1o)
    y2o = F.relu(self._fc2(y1))
    y2 = self._dropout2(y2o)

    return y2


class WaveMixBackbone(Backbone):
  def __init__(self, dropout_probability):
    super().__init__()

    # Backbone properties
    self.feature_map_channels = 512
    self.feature_pixels = 16
    self.feature_vector_size = 4096
    self.image_preprocessing_params = image.PreprocessingParams(channel_order = image.ChannelOrder.BGR, scaling = 1.0, means = [ 103.939, 116.779, 123.680 ], stds = [ 1, 1, 1 ])
    
    state_dict = load_state_dict_from_url('https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/imagenet_71.49.pth')
    WaveMix = FeatureExtractor()
    WaveMix.load_state_dict(state_dict, strict=False)
    print("Loaded IMAGENET pre-trained weights for WaveMix backbone")

    # Assign the loaded model to the feature_extractor
    self.feature_extractor = WaveMix

    print(summary(WaveMix.cuda(), (3, 224, 224)))

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(dropout_probability=dropout_probability)

  def compute_feature_map_shape(self, image_shape):
    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, image_height // self.feature_pixels, image_width // self.feature_pixels)
