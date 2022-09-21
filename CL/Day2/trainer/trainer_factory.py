import torch
import numpy as np
from torch.nn import functional as F
from copy import deepcopy
    
class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, model, args):
        
        self.net = model
        
        self.device = args['device']
        # N way K shots
        self.n_way = args['n_way']
        self.k_spt = args['k_spt']
        self.k_qry = args['k_qry']
        self.tasknum = args['tasknum']
        
        self.epochs = args['epochs']
        self.inner_lr = args['inner_lr']        
        self.meta_lr = args['meta_lr']
        self.inner_step = args['inner_steps']
        self.inner_step_test = 1
    
    def train(self, dataloader, epoch):
        eval_flag = True
        for step in range(epoch):
            x_spt, y_spt, x_qry, y_qry = dataloader.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(self.device), torch.from_numpy(y_spt).to(self.device), \
            torch.from_numpy(x_qry).to(self.device), torch.from_numpy(y_qry).to(self.device)
            accs = self.Meta_Train(x_spt, y_spt, x_qry, y_qry)
            eval_flag = False
            
            if step % 50 == 0:
                print('step:', step, '\ttraining loss:', accs)

            if step % 500 == 0:
                accs = []
                for _ in range(1000//self.tasknum):
                    # test
                    x_spt, y_spt, x_qry, y_qry = dataloader.next('test')
                    x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(self.device), torch.from_numpy(y_spt).to(self.device), \
                    torch.from_numpy(x_qry).to(self.device), torch.from_numpy(y_qry).to(self.device)

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        test_acc = self.Meta_Train(x_spt_one.unsqueeze(0),
                                                  y_spt_one.unsqueeze(0),
                                                  x_qry_one.unsqueeze(0),
                                                  y_qry_one.unsqueeze(0), train=False)
                        accs.append( test_acc )

                # [b, inner_step+1]
                accs = np.array(accs).mean(axis=0).astype(np.float16)
                print('Test loss:', accs)
            
    def _train_epoch(self, x_spt, y_spt, x_qry, y_qry):
        pass

    def _finetunning(self, x_spt, y_spt, x_qry, y_qry):
        pass
