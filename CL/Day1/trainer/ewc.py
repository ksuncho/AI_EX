from __future__ import print_function
from tqdm import tqdm
import torch
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, model, args, optimizer, evaluator, task_info):
        super().__init__(model, args, optimizer, evaluator, task_info)
        
        self.lamb=args['lamb']
        

    def train(self, train_loader, test_loader, t, device = None):
        
        self.device = device
        self.setup_training(self.lr)
        # Do not update self.t
        if t>0: # update fisher before starting training new task
            self.update_frozen_model()
            self.update_fisher()
        
        # Now, you can update self.t
        self.t = t
        
        self.train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=self.batch_size, shuffle=True)
        self.test_iterator = torch.utils.data.DataLoader(test_loader, 100, shuffle=False)
        self.fisher_iterator = torch.utils.data.DataLoader(train_loader, batch_size=20, shuffle=True)
        
        
        for epoch in range(self.epochs):
            self.model.train()
            for samples in tqdm(self.train_iterator):
                data, target = samples
                if device is not None:
                    data, target = data.to(device), target.to(device)
                batch_size = data.shape[0]

                output = self.model(data)[t]
                loss_CE = self.criterion(output,target)

                self.optimizer.zero_grad()
                (loss_CE).backward()
                self.optimizer.step()
            self.scheduler.step()

            train_loss,train_acc = self.evaluator.evaluate(self.model, self.train_iterator, t, self.device)
            num_batch = len(self.train_iterator)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch+1,train_loss,100*train_acc),end='')
            test_loss,test_acc=self.evaluator.evaluate(self.model, self.test_iterator, t, self.device)
            print(' Test: loss={:.3f}, acc={:5.1f}% |'.format(test_loss,100*test_acc),end='')
            print()
        
    def criterion(self,output,targets):
        """
        Arguments: output (The output logit of self.model), targets (Ground truth label)
        Return: loss function for the regularization-based continual learning
        
        For the hyperparameter on regularization, please use self.lamb
        """
        
        #######################################################################################
        
        loss_reg=0
        if self.t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.ce(output,targets)+self.lamb*loss_reg
        
        
        #######################################################################################
    
    def compute_diag_fisher(self):
        """
        Arguments: None. Just use global variables (self.model, self.criterion, ...)
        Return: Diagonal Fisher matrix. 
        
        This function will be used in the function 'update_fisher'
        """
        #######################################################################################
        # Write youre code here
        fisher={}
        for n,p in self.model.named_parameters():
            fisher[n]=0*p.data
        # Compute
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        total = 0
        for samples in tqdm(self.fisher_iterator):
            data, target = samples
            total += data.shape[0]
            # Forward and backward
            self.model.zero_grad()
            outputs = self.model.forward(data)[self.t]
            loss=self.criterion(outputs, target)
            loss.backward()

            # Get gradients
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n]+=self.args.batch_size*p.grad.data.pow(2)
        # Mean
        with torch.no_grad():
            for n,_ in self.model.named_parameters():
                fisher[n]=fisher[n]/total
        return fisher        

        
        #######################################################################################        
    
    def update_fisher(self):
        
        """
        Arguments: None. Just use global variables (self.model, self.fisher, ...)
        Return: None. Just update the global variable self.fisher
        Use 'compute_diag_fisher' to compute the fisher matrix
        """
        
        #######################################################################################
        
        if self.t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=self.fisher_matrix_diag()
        if self.t>0:
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*self.t)/(self.t+1)
        
        #######################################################################################