# coding: UTF-8
import torch
from torch.cuda import amp
from tqdm import tqdm


def train(CFG, model, train_dataloader, losses_dict, optimizer):
    """
        Use amp.GradScaler() for half-precision learning
    """
    # half-precision learning
    model.train()
    scaler = amp.GradScaler()
    losses_all, ce_all = 0, 0

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train ')
    for _, (img_batch, img_batch_labels) in pbar:
        optimizer.zero_grad()

        img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
        img_batch_labels = img_batch_labels.to(CFG.device)

        with amp.autocast(enabled=True):
            y_preds = model(img_batch) 
            ce_loss = losses_dict["CELoss"](y_preds, img_batch_labels.long())
            losses = ce_loss

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_all += losses.item() / img_batch.shape[0]
        ce_all += ce_loss.item() / img_batch.shape[0]

        current_lr = optimizer.param_groups[0]['lr']

    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all, ce_all), flush=True)


@torch.no_grad()
def valid(model, valid_dataloader, CFG):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc='Valid ')
    for _, (img_batch, img_batch_labels) in pbar:
        img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
        img_batch_labels = img_batch_labels.to(CFG.device)
        
        y_preds = model(img_batch)
        """
            torch.max will return two tensors, 
            the first tensor is the maximum value of each row; 
            the second tensor is the index of the maximum value of each row.
        """
        _, y_preds = torch.max(y_preds.data, dim=1)

        correct += (y_preds==img_batch_labels).sum()
        total += img_batch_labels.shape[0]

    val_acc = correct/total
    print("val_acc: {:.2f}".format(val_acc), flush=True)
    
    return val_acc


@torch.no_grad()
def test(test_df, test_dataloader, model, ckpt_paths, CFG):
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test: ')

    for _, (img_batch, img_batch_name) in pbar:
        img_batch = img_batch.to(CFG.device, dtype=torch.float) # [b, c, h, w]
        
        # Cross Validation Infer
        for sub_ckpt_path in ckpt_paths:
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            y_preds = model(img_batch) # [b, c, w, h]
            prob = torch.nn.functionl.softmax(y_preds, dim=-1)[:,1].detach().cpu().numpy()

            test_df.loc[test_df['img_name'].isin(img_batch_name), 'pred_prob'] = prob

    return test_df
