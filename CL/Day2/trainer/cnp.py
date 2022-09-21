import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch import optim
import  numpy as np

from    copy import deepcopy
from trainer.trainer_factory import GenericTrainer


class CNP_Trainer(GenericTrainer):
    """
    Meta Learner
    """
    def __init__(self, model, args):
        """
        :param args:
        """
        super(CNP_Trainer, self).__init__(model, args)

        self.meta_optim = optim.Adam(self.net.parameters(), lr=args['meta_lr'])
        self.loss = nn.MSELoss()
            

    def Meta_Train(self, x_spt, y_spt, x_qry, y_qry, train=True):
        """
        :param x_spt:   [tasknum, setsz, 1]
         - Training input data
        :param y_spt:   [tasknum, setsz, 1]
         - Training target data
        :param x_qry:   [tasknum, querysz, 1]
         - Test input data
        :param y_qry:   [tasknum, querysz, 1]
         - Test target data
        :return: 'loss' (a scalar)
        """
        
        logits = self.net(x_spt, y_spt, x_qry)
        loss = self.loss(logits, y_qry)
        if train:
            self.meta_optim.zero_grad()
            loss.backward()
            self.meta_optim.step()
        
        return [loss.item()]


    def Meta_test(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [tasknum, setsz, 1]
         - Training input data
        :param y_spt:   [tasknum, setsz, 1]
         - Training target data
        :param x_qry:   [tasknum, querysz, 1]
         - Test input data
        :param y_qry:   [tasknum, querysz, 1]
         - Test target data
        :return: 'loss' (a scalar)
        """
        
        logits = self.net(x_spt.unsqueeze(0), y_spt.unsqueeze(0), x_qry.unsqueeze(0))
        loss = self.loss(logits, y_qry.unsqueeze(0))
        
        return [loss.item()]