import os
import time
import pandas as pd
import argparse
from config import CFG
from sklearn.model_selection import StratifiedGroupKFold, KFold
from data_utils import *
from train_eval import *


parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="Whether to run training.")
parser.add_argument("--ckpt_fold", type=str, default="ckpt_ddt1", help="where to save model checkpoint")
parser.add_argument("--tampered_img_paths", type=str, default="../data/train/tampered/imgs")
parser.add_argument("--untampered_img_paths", type=str, default="../data/train/untampered/")
parser.add_argument("--test_img_paths", type=str, default="../data/imgs/")
# hyper-parameter
parser.add_argument("--n_fold", type=int, default=4)
parser.add_argument("--img_size", nargs='+', default=[224,224])
parser.add_argument("-tb", "--train_bs", help="Batch size for training", type=int, default=256)
parser.add_argument("--test_bs", help="Batch size for test", type=int, default=256*2)
# model parameter
parser.add_argument("--backbone", help="BackBone for pre-training", type=str, default="efficientnet_b0")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--epoch", type=int, default=12)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5)
parser.add_argument("--lr_drop", help="", type=float, default=8)
parser.add_argument("--threshold", type=float, default=0.5)

args = parser.parse_args()


def train_entry(CFG):
    col_name = ['img_name', 'img_path', 'img_label']
    imgs_info = []  # img_name, img_path, img_label
    for img_name in os.listdir(CFG.tampered_img_paths):
        if img_name.endswith('.jpg'): # pass other files
            imgs_info.append(["p_"+img_name, os.path.join(CFG.tampered_img_paths, img_name), 1])
            
    for img_name in os.listdir(CFG.untampered_img_paths):
        if img_name.endswith('.jpg'): # pass other files
            imgs_info.append(["n_"+img_name, os.path.join(CFG.untampered_img_paths, img_name), 0])
         
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

        train_dataloader, valid_dataloader = build_dataloader(df, fold, data_transforms, CFG)
        model = build_model(CFG, pretrain_flag=True) # model
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop)
        loss_dict = build_loss()

        best_val_acc = 0
        best_epoch = 0

        for epoch in range(CFG.epoch):
            start_time = time.time()
            train(model, train_dataloader, optimizer, loss_dict, CFG)
            lr_scheduler.step()
            val_acc = valid(model, valid_dataloader, CFG)

            is_best = (val_acc > best_val_acc)
            best_val_acc = max(best_val_acc, val_acc)
            if is_best:
                save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                if os.path.isfile(save_path):
                    os.remove(save_path) 
                torch.save(model.state_dict(), save_path)
            epoch_time = time.time() - start_time
            print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_acc), flush=True)


def test_entry(CFG):
    col_name = ['img_name', 'img_path', 'pred_prob']
    imgs_info = []  # img_name, img_path, pred_prob
    test_imgs =  os.listdir(CFG.test_img_path)
    test_imgs.sort(key=lambda x: x[:-4]) 
    for img_name in test_imgs:
        if img_name.endswith(".jpg"):
            imgs_info.append(img_name, os.path.join(CFG.test_img_path, img_name), 0)
    
    test_df = pd.DataFrame(imgs_info, columns=col_name)
    # prepare test_dataloader
    data_transforms = build_transforms(CFG)
    test_dataloader = build_dataloader(test_df, False, None, data_transforms)
    # prepare trained model for infer
    model = build_model(CFG, pretrain_flag=True)
    ckpt_paths = ["dummy_test"] # ckpt path
    # submit result
    test_df = test(test_df, test_dataloader, model, ckpt_paths, CFG)
    submit_df = test_df.loc[:, ['img_name', 'pred_prob']]
    submit_df.to_csv("submit_dummy.csv", header=False, index=False, sep=' ')


if __name__ == "__main__":
    config = CFG(args)

    if args.do_train():
        train_entry(config)
    else:
        test_entry(config)