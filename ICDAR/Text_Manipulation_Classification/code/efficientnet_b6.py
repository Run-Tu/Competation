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
        model_out_put = self.model(x)
        cls_out_put = torch.mean(model_out_put, axis=[-1,-2])

        return cls_out_put


class b6_seg_cls_model(nn.Module):
    def __init__(self, CFG):
        super().__init__()  
        self.model = smp.UnetPlusPlus (# UnetPlusPlus / DeepLabV3Plus
                encoder_name=CFG.backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
            )
        self.fine_tune = nn.Sequential(
            nn.Linear(3*CFG.img_size*CFG.img_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(512, CFG.num_classes)
        )
    def forward(self, x):
        #with autocast():
        seg_out_put = self.model(x) # seg task
        cls_out_put = self.fine_tune(torch.flatten(seg_out_put, start_dim=1)) # cls task

        return seg_out_put, cls_out_put