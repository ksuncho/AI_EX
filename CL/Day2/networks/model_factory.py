import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset,n_way=5):
        
        if dataset == 'maml':
            import networks.optimization_based as net
            return net.mlp(1,1)
        elif dataset == 'cnp':
            import networks.neural_processes as net
            return net.EncoderDecoder(1,1)
        
