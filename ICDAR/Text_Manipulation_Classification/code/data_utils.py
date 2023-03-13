import cv2
import timm
import random
import torch
import sys
sys.path.append("/root/autodl-tmp/ICDAR/Text_Manipulation_Classification")
import numpy as np
import albumentations as A # Augmentations
from torch.utils.data import Dataset, DataLoader
from vit_model import vit_base_patch16_224_in21k
from efficientnet_b6 import b6_seg_model
from losses.dice_loss import DiceLoss
from losses.soft_ce import SoftCrossEntropyLoss 


def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(CFG):
    data_transforms = {
        "train" : A.Compose([
                    A.Resize(height=CFG.img_size, width=CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
                    # Rotate
                    A.HorizontalFlip(p=0.5),
                    A.OneOf([
                             A.VerticalFlip(p=0.5),
                             A.RandomRotate90(p=0.5),
                             A.RandomBrightnessContrast(p=0.5),
                             A.HueSaturationValue(p=0.5),
                             A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                             #A.CoarseDropout(p=0.2),
                             A.Transpose(p=0.5)
                            ]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ], p=1.0),
        "valid" : A.Compose([
                    A.Resize(height=CFG.img_size, width=CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ], p=1.0)
    }

    return data_transforms


class build_dataset(Dataset):
    def __init__(self, df, transforms, train_val_flag=True):
        self.df = df
        self.train_val_flag = train_val_flag
        self.img_paths = df["img_path"].tolist()
        self.img_names = df["img_name"].tolist()
        self.transforms = transforms

        if self.train_val_flag:
            self.labels = df["img_label"].tolist()


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [h, w, c]

        if self.train_val_flag:
            data = self.transforms(image=img)
            img = np.transpose(data["image"], (2,0,1)) # [c, h, w]
            label = self.labels[index]

            return torch.tensor(img), torch.tensor(int(label)), img_name
        
        else:
            data = self.transforms(image=img)
            img = np.transpose(data["image"], (2,0,1)) # [c, h, w]

            return torch.tensor(img), img_name


def build_dataloader(df, fold, data_transforms, CFG, train=True):
    if train:
        train_df = df[df["fold"]!=fold].reset_index(drop=True)
        valid_df = df[df["fold"]==fold].reset_index(drop=True)

        train_dataset = build_dataset(train_df, transforms=data_transforms["train"], train_val_flag=True)
        valid_dataset = build_dataset(valid_df, transforms=data_transforms["valid"], train_val_flag=True)

        train_dataloader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True, drop_last=False)
        
        return train_dataloader, valid_dataloader
    
    else:
        test_dataset = build_dataset(df, transforms=data_transforms["valid"], train_val_flag=False)
        test_dataloader = DataLoader(test_dataset, batch_size=CFG.test_bs, num_workers=0, shuffle=False, pin_memory=True, drop_last=False)

        return test_dataloader


def build_model(CFG, pretrain_flag=False):
    """
        Use timm for loading pre_trained model
    """
    if CFG.backbone == "efficientnet_b0":
        model = timm.create_model(CFG.backbone, pretrained=pretrain_flag, num_classes=CFG.num_classes)
    if CFG.backbone == "vit_model":
        model = vit_base_patch16_224_in21k(CFG, pretrain_flag=pretrain_flag)
    if CFG.backbone == "efficientnet_b6":
        model = b6_seg_model(model_name="efficientnet-b6", n_class=CFG.num_classes)
        
    model.to(CFG.device)

    return model


def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    DICELoss = DiceLoss(mode="multiclass")
    SoftCrossEntropy = SoftCrossEntropyLoss(smooth_factor=0.1)

    return {"CELoss":CELoss, "DICELoss":DICELoss, "SoftCrossEntropy":SoftCrossEntropy}


def build_metric(preds, valids):
    tampers = valids[valids[:, 1].astype(int) == 1]
    untampers = valids[valids[:, 1].astype(int) == 0]
    pred_tampers = preds[np.in1d(preds[:, 0], tampers[:, 0])]
    pred_untampers = preds[np.in1d(preds[:, 0], untampers[:, 0])]
    thres = np.percentile(pred_untampers[:, 1].astype(float), np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers[:, 1][:, np.newaxis].astype(float), thres).mean(axis=0))
    
    return recall * 100
