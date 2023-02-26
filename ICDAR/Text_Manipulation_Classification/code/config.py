import torch


class CFG:
    def __init__(self, args):
        # step1: hyper-parameter
        self.seed = args.seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ckpt_fold = args.ckpt_fold
        self.ckpt_name = f"{args.backbone}_img224224_bs{args.train_bs}"  # for submit.
        self.tampered_img_paths = args.tampered_img_paths
        self.untampered_img_paths = args.untampered_img_paths
        self.test_img_paths = args.test_img_paths
        # step2: data
        self.n_fold = args.n_fold
        self.img_size = args.img_size
        self.train_bs = args.train_bs
        self.valid_bs = args.train_bs * 2
        self.test_bs = args.test_bs
        # step3: model
        self.backbone = args.backbone
        self.num_classes = args.num_classes
        # step4: optimizer
        self.epoch = args.epoch
        self.lr = args.lr
        self.wd = args.wd
        self.lr_drop = args.lr_drop
        # step5: infer
        self.threshold = args.threshold