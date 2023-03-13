import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class b6_seg_model(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()  
        self.model = smp.UnetPlusPlus (# UnetPlusPlus / DeepLabV3Plus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        x = torch.mean(x, axis=[-1,-2])

        return x