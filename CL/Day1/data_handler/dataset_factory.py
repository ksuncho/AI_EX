import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, tasknum):
        if name == "CIFAR100":
            return data.CIFAR100(tasknum)
        elif name == 'MNIST':
            return data.MNIST(tasknum)
