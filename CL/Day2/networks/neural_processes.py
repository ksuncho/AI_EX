import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



class EncoderDecoder(nn.Module):
    def __init__(self, ninput=1, noutput=1):
        super(EncoderDecoder, self).__init__()
        
        ntotal = ninput+noutput
        
        self.fc_enc_1 = nn.Linear(ntotal,128)
        self.fc_enc_2 = nn.Linear(128,128)
        self.fc_enc_3 = nn.Linear(128,128)
        self.fc_enc_4 = nn.Linear(128,128)
        
        self.fc_dec_1 = nn.Linear(128+ninput,128)
        self.fc_dec_2 = nn.Linear(128,128)
        self.fc_dec_3 = nn.Linear(128,noutput)
        
        self.relu = torch.nn.ReLU()
        
    def forward(self, x_spt, y_spt, x_qry, encoder=False):
        """
        :param x_spt: [tasknum, num_spt, 1]
        :param y_spt: [tasknum, num_spt, 1]
        :param x_qry: [tasknum, num_qry, 1]
        :return: output(encoder==False), representation(encoder==True)
        """
        
        input = torch.cat((x_spt,y_spt), dim=-1)
        
        h = self.fc_enc_1(input)
        h = self.relu(h)
        h = self.fc_enc_2(h)
        h = self.relu(h)
        h = self.fc_enc_3(h)
        h = self.relu(h)
        h = self.fc_enc_4(h)
        
        representation = h.mean(dim=1, keepdim=True)
        
        if encoder:
            return representation
        
        num_qry = x_qry.size()[1]
        tasknum, _, dim = representation.size()
        representation = representation.expand(tasknum, num_qry, dim)
        
        input = torch.cat((x_qry, representation), dim=-1)
        
        h = self.fc_dec_1(input)
        h = self.relu(h)
        h = self.fc_dec_2(h)
        h = self.relu(h)
        h = self.fc_dec_3(h)

        return h
