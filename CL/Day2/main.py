import  torch, os
import  numpy as np
import data_handler
from argument import get_args
import trainer
import networks
def main():
    args = get_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    device = torch.device('cpu')  

    args.device = device
    myModel = networks.ModelFactory.get_model(args.dataset, args.n_way).to(device)

    myTrainer = trainer.TrainerFactory.get_trainer(myModel, args)
    dataloader = data_handler.get_dataset(args)
    myTrainer.train(dataloader, args.epoch)
    print('done!')

if __name__ == '__main__':
    main()
