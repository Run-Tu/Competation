import cv2
import timm
import random
import torch
import numpy as np
from config import CFG
import albumentations as A # Augmentations
from torch.utils.data import Dataset, DataLoader


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
                    A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
                    A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
                                    min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),
        "valid" : A.Compose([
                    A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ], p=1.0)
    }

    return data_transforms


class build_dataset(Dataset):
    def __init__(self, df, transforms, train_val_flag=True):
        self.df = df
        self.train_val_flag = train_val_flag,
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

            return torch.tensor(img), torch.tensor(int(label))
        
        else:
            data = self.transforms(image=img)
            img = np.transpose(data["image"], (2,0,1)) # [c, h, w]

            return torch.tensor(img), img_name


def build_dataloader(df, fold, data_transforms, CFG):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    train_dataset = build_dataset(train_df, transforms=data_transforms["train"], train_val_flag=True)
    valid_dataset = build_dataset(valid_df, transforms=data_transforms["vaild"], train_val_flag=True)

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True, drop_last=False)
    
    return train_dataloader, valid_dataloader


def build_model(CFG, pretrain_flag=False):
    """
        Use timm for loading pre_trained model
    """
    if pretrain_flag:
        pretrain_weights = "imagenet"
    else:
        pretrain_weights = False
    model = timm.create_model(CFG.backbone, pretrained=pretrain_flag, num_classes=CFG.num_classes)
    model.to(CFG.device)

    return model


def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()

    return {"CELoss":CELoss}


def build_metric():

    pass