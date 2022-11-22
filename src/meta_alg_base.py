import abc
import os
import torch


class BayesianMetaAlgBase(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, args):
        self.args = args
        self.model = self.construct_model()

    @abc.abstractmethod
    def construct_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, train_dataloader, val_dataloader):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, test_dataloader):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, dataloader, num_tasks, return_std=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_support_posterior(self, x_supp, y_supp):
        raise NotImplementedError

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, file_name))

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(os.path.join(self.args.model_dir, file_name)))
