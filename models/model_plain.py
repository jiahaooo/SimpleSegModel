from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torch.nn.parallel import DataParallel  # , DistributedDataParallel
from models.select_network import define_Net
from models.model_base import ModelBase
from models.loss import DiceLoss, FocalLoss



class ModelPlain(ModelBase):
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.net = define_Net(opt).to(self.device)
        self.net = DataParallel(self.net)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.opt_train = self.opt['train']    # training option
        self.load()                           # load model
        self.net.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path = self.opt['path']['pretrained_net']
        if load_path is not None:
            print('Loading model for G [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net)

    # ----------------------------------------
    # save model
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.net, iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        lossfn_type = self.opt_train['lossfn_type']
        if lossfn_type in ['CrossEntropyLoss', 'crossentropyloss']:
            self.lossfn = nn.CrossEntropyLoss()
        elif lossfn_type in ['FocalLoss', 'focalloss']:
            self.lossfn = FocalLoss()
        elif lossfn_type in ['DiceLoss', 'diceloss']:
            self.lossfn = DiceLoss(n_classes=2)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.optimizer = Adam(optim_params, lr=self.opt_train['optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer,
                                                        self.opt_train['scheduler_milestones'],
                                                        self.opt_train['scheduler_gamma']
                                                        ))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_label=True):
        self.Input = data['Input'].to(self.device)
        if need_label:
            self.Label = data['Label'].to(self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.optimizer.zero_grad()
        self.Output, self.Mask = self.net(self.Input)
        loss = self.lossfn(self.Output, self.Label)
        loss.backward()

        self.optimizer.step()

        self.log_dict['loss'] = loss.item()

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.net.eval()
        with torch.no_grad():
            self.Output, self.Mask = self.net(self.Input)
        self.net.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get Input, Output, Label image
    # ----------------------------------------
    def current_visuals(self, need_Label=True):
        out_dict = OrderedDict()
        out_dict['Input'] = self.Input.detach()[0].float().cpu()
        out_dict['Output'] = self.Output.detach()[0].float().cpu()
        out_dict['Mask'] = self.Mask.detach()[0].cpu()
        if need_Label:
            out_dict['Label'] = self.Label.detach()[0].cpu()
        return out_dict

    # ----------------------------------------
    # get Input, Output, Mask, Label batch images
    # ----------------------------------------
    def current_results(self, need_Label=True):
        out_dict = OrderedDict()
        out_dict['Input'] = self.Input.detach().float().cpu()
        out_dict['Output'] = self.Output.detach().float().cpu()
        out_dict['Mask'] = self.Mask.detach().cpu()
        if need_Label:
            out_dict['Label'] = self.Label.detach().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of network
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.net)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.net)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.net)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.net)
        return msg
