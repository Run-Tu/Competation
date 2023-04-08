# coding: UTF-8
import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
from data_utils import build_metric
import torch.nn.functional as F


def train(CFG, model, train_dataloader, losses_dict, optimizer):
    """
        Use amp.GradScaler() for half-precision learning
    """
    # half-precision learning
    model.train()
    scaler = amp.GradScaler()
    losses_all = 0

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train ')
    if CFG.two_tasks:
        for _, (img_batch, img_batch_labels, img_batch_masks, img_name) in pbar:
            optimizer.zero_grad()

            img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
            img_batch_labels = img_batch_labels.to(CFG.device)
            img_batch_masks = img_batch_masks.to(CFG.device) # [b ,3, h, w]

            with amp.autocast(enabled=True):
                seg_out_put, cls_out_put = model(img_batch)
                cls_y_pred = F.softmax(cls_out_put, dim=-1)
                losses = 0.3*losses_dict["DICELoss"](cls_y_pred, img_batch_labels.long())+ \
                        losses_dict["SoftCrossEntropy"](cls_y_pred, img_batch_labels.long())+ \
                        0.1*(losses_dict["SEG_DICELoss"](seg_out_put, img_batch_masks.long())/3)


            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_all += losses.item() / img_batch.shape[0]
            current_lr = optimizer.param_groups[0]['lr']

        return current_lr, losses_all
    
    else:
        for _, (img_batch, img_batch_labels, img_name) in pbar:
            optimizer.zero_grad()

            img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
            img_batch_labels = img_batch_labels.to(CFG.device)

            with amp.autocast(enabled=True):
                y_preds = model(img_batch) 
                losses = 5.*losses_dict["DICELoss"](F.softmax(y_preds, dim=-1), img_batch_labels.long())+ \
                        losses_dict["SoftCrossEntropy"](F.softmax(y_preds, dim=-1), img_batch_labels.long())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            losses_all += losses.item() / img_batch.shape[0]
            current_lr = optimizer.param_groups[0]['lr']

        return current_lr, losses_all


@torch.no_grad()
def valid(model, valid_dataloader, CFG):
    """
        Verified once per epoch
    """
    model.eval()

    preds = [] # [[img_name, y_preds]]
    valids = []# [[img_name, y_labels]]
    pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc='Valid ')
    if CFG.two_tasks:
        for _, (img_batch, img_batch_labels, img_batch_masks, img_names) in pbar:
            img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
            img_batch_labels = img_batch_labels.to(CFG.device, dtype=torch.int)

            seg_out_put, cls_out_put = model(img_batch)
            y_preds = F.softmax(cls_out_put, dim=-1) # [b, num_class]
            """
                torch.max will return two tensors, 
                the first tensor is the maximum value of each row; 
                the second tensor is the index of the maximum value of each row.
            """
            _, y_preds = torch.max(y_preds.data, dim=-1) # [b, num_class]

            for img_name, y_pred, y_label in zip(img_names, y_preds.cpu().numpy().tolist(), img_batch_labels.cpu().numpy().tolist()):
                preds.append([img_name, y_pred])
                valids.append([img_name, y_label])
            
        recall = build_metric(np.array(preds), np.array(valids))

        return recall

    else:
        for _, (img_batch, img_batch_labels, img_names) in pbar:
            img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
            img_batch_labels = img_batch_labels.to(CFG.device, dtype=torch.int)
        
            y_preds = F.softmax(model(img_batch), dim=-1) # [b, num_class]
            """
                torch.max will return two tensors, 
                the first tensor is the maximum value of each row; 
                the second tensor is the index of the maximum value of each row.
            """
            _, y_preds = torch.max(y_preds.data, dim=-1) # [b, num_class]

            for img_name, y_pred, y_label in zip(img_names, y_preds.cpu().numpy().tolist(), img_batch_labels.cpu().numpy().tolist()):
                preds.append([img_name, y_pred])
                valids.append([img_name, y_label])
            
        recall = build_metric(np.array(preds), np.array(valids))

        return recall


@torch.no_grad()
def test(test_df, test_dataloader, model, ckpt_paths, CFG):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test: ')

    for _, (img_batch, img_batch_name) in pbar:
        img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
        
        # Cross Validation Infer
        for sub_ckpt_path in ckpt_paths:
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            if CFG.two_tasks:
                seg_preds, cls_preds = model(img_batch) # [b, c, w, h]
            else:
                cls_preds = model(img_batch)
                
            prob = F.softmax(cls_preds, dim=-1)[:,1].detach().cpu().numpy()
            test_df.loc[test_df['img_name'].isin(img_batch_name), 'pred_prob'] = prob

    return test_df
