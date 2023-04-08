"""
    A40
    train efficientnet_b6 two task command
    python run.py --do_train --ckpt_fold ckpt_0316_efficient_b6 -tb 32 --epoch 30 --backbone efficientnet_b6_two_task --two_tasks --img_size 512
"""
import os
import time
import logging
import datetime
import pandas as pd
import argparse
from config import CFG
from sklearn.model_selection import StratifiedGroupKFold, KFold
from data_utils import (set_seed, build_transforms, build_dataloader_seg_cls, 
                        build_model, build_loss)
from train_eval import *
import requests

# logging
TODAY = datetime.date.today()
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y/%m/%d %H:%M:%S %P"
if not os.path.exists("../logs"):
    os.makedirs("../logs")
logging.basicConfig(filename=f"../logs/{TODAY}_two_task.log", level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action="store_true", default=False, help="Whether to run training.")
parser.add_argument("--ckpt_fold", type=str, default="ckpt_ddt1", help="where to save model checkpoint")
parser.add_argument("--tampered_img_paths", type=str, default="../data/train/tampered/imgs")
parser.add_argument("--untampered_img_paths", type=str, default="../data/train/untampered/imgs")
parser.add_argument("--tampered_mask_img_path", type=str, default="../data/train/tampered/masks")
parser.add_argument("--untampered_mask_img_path", type=str, default="../data/train/untampered/masks")
parser.add_argument("--test_img_paths", type=str, default="../data/test/")
# hyper-parameter
parser.add_argument("--n_fold", type=int, default=4)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("-tb", "--train_bs", help="Batch size for training", type=int, default=32)
parser.add_argument("--test_bs", help="Batch size for test", type=int, default=64*2)
# model parameter
parser.add_argument("--backbone", help="[efficientnet_b0, vit_model, efficientnet-b6]", type=str, default="efficientnet_b0")
parser.add_argument("--two_tasks", action="store_true", default=False, help="whether to use two_tasks mode training")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--epoch", type=int, default=12)
parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr_drop", type=float, default=8)
parser.add_argument("--threshold", type=float, default=0.5)

args = parser.parse_args()


def train_entry(CFG):
    col_name = ["img_name", "img_path", "mask_path", "img_label"]
    imgs_info = []  # img_name, img_path, img_label
    for img_name, mask_name in zip(sorted(os.listdir(CFG.tampered_img_paths)), sorted(os.listdir(CFG.tampered_mask_img_path))):
        if img_name.endswith(".jpg") and mask_name.endswith(".png"): # pass other files
            imgs_info.append(["p_"+img_name, os.path.join(CFG.tampered_img_paths, img_name), os.path.join(CFG.tampered_mask_img_path, mask_name),1])
            
    for img_name, mask_name in zip(sorted(os.listdir(CFG.untampered_img_paths)), sorted(os.listdir(CFG.untampered_mask_img_path))):
        if img_name.endswith('.jpg') and mask_name.endswith(".png"): # pass other files
            imgs_info.append(["n_"+img_name, os.path.join(CFG.untampered_img_paths, img_name), os.path.join(CFG.untampered_mask_img_path, mask_name), 0])
    
    df = pd.DataFrame(imgs_info, columns=col_name)
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = fold

    # mkdir for ckpt file
    set_seed(CFG.seed)
    ckpt_path = f"../{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    data_transforms = build_transforms(CFG) 
    for fold in range(CFG.n_fold):
        print(f'#'*40, flush=True)
        print(f'###### Fold: {fold}', flush=True)
        print(f'#'*40, flush=True)

        train_dataloader, valid_dataloader = build_dataloader_seg_cls(df, fold, data_transforms, CFG)
        model = build_model(CFG, pretrain_flag=True) # model
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
        loss_dict = build_loss()
        best_val_recall = 0

        for epoch in range(CFG.epoch):
            start_time = time.time()
            current_lr, losses_all = train(CFG, model, train_dataloader, loss_dict, optimizer)
            lr_scheduler.step()
            val_recall = valid(model, valid_dataloader, CFG)

            is_best = (val_recall > best_val_recall)
            best_val_recall = max(best_val_recall, val_recall)

            if is_best:
                save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                if os.path.isfile(save_path):
                    os.remove(save_path) 
                torch.save(model.state_dict(), save_path)
            epoch_time = time.time() - start_time
            
            # WeChat remind
            requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                            json={"token": "1ff3eac8fd3b",
                                "title": f"ICDAR Resize{CFG.img_size}",
                                "name": f"ICDAR {CFG.backbone}",
                                "content":f"epoch:{epoch}, best_recall:{best_val_recall:.6f}, current_lr:{current_lr}, loss:{losses_all}"})
            logging.info(f"epoch:{epoch}, time:{epoch_time}, best_recall:{best_val_recall:.6f}, current_lr:{current_lr}, loss:{losses_all}")


def test_entry(CFG):
    col_name = ['img_name', 'img_path', 'pred_prob']
    imgs_info = []  # img_name, img_path, pred_prob
    test_imgs =  os.listdir(CFG.test_img_path)
    test_imgs.sort(key=lambda x: x[:-4]) 
    for img_name in test_imgs:
        if img_name.endswith(".jpg"):
            imgs_info.append([img_name, os.path.join(CFG.test_img_path, img_name), 0])
    
    test_df = pd.DataFrame(imgs_info, columns=col_name)
    # prepare test_dataloader
    data_transforms = build_transforms(CFG)
    test_dataloader = build_dataloader_seg_cls(test_df, 1, data_transforms, CFG, train=False)
    # prepare trained model for infer
    model = build_model(CFG, pretrain_flag=False)
    ckpt_paths = ["/root/autodl-tmp/ICDAR/Text_Manipulation_Classification/ckpt_0313_efficient_b6/efficientnet_b6_img768_bs10/best_fold0_epoch11.pth"] # ckpt path
    # submit result
    test_df = test(test_df, test_dataloader, model, ckpt_paths, CFG)
    submit_df = test_df.loc[:, ['img_name', 'pred_prob']]
    submit_df.to_csv(f"../{TODAY}_b6net_fold0_epoch11.csv", header=False, index=False, sep=' ')


if __name__ == "__main__":
    config = CFG(args)

    if args.do_train:
        train_entry(config)
    else:
        test_entry(config)