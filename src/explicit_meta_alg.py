import abc
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
from torchmeta.utils import gradient_update_parameters
from src.utils import Checkpointer, split_task_batch, kld_normal_standard
from src.meta_alg_base import BayesianMetaAlgBase


class BMLAlgExplicit(BayesianMetaAlgBase):
    def __init__(self, args):
        super(BMLAlgExplicit, self).__init__(args)

    @abc.abstractmethod
    def meta_loss(self, qry_nll, *args):
        raise NotImplementedError

    def train(self, train_dataloader, val_dataloader):
        print('Training starts ...')

        meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.meta_lr)
        check_pointer = Checkpointer(self.save_model, self.args.algorithm.lower())

        running_loss = 0.
        running_acc = 0.

        # training loop
        for meta_idx, task_batch in enumerate(train_dataloader):

            if meta_idx >= self.args.meta_iter:
                break

            self.model.train()
            meta_optimizer.zero_grad()

            for x_supp, y_supp, x_qry, y_qry in zip(*split_task_batch(task_batch, self.args.device)):
                post_params, supp_nll = self.update_support_posterior(x_supp, y_supp)
                y_preds = self.model(x_qry, params=post_params)
                qry_nll = torch.stack([self.args.loss_fn(y_pred, y_qry) for y_pred in y_preds]).mean()
                meta_loss = self.meta_loss(qry_nll, supp_nll, post_params) / self.args.batch_size
                # torch.cuda.reset_peak_memory_stats()
                # import time
                # start_time = time.time()
                meta_loss.backward()
                # torch.cuda.synchronize()
                # print('Time:', time.time() - start_time)
                # print('Space:', torch.cuda.max_memory_allocated() / 1024 ** 3)

                with torch.no_grad():
                    running_loss += meta_loss.detach().item()
                    running_acc += (sum(y_preds).argmax(dim=1) == y_qry).detach().float().mean().item()

            meta_optimizer.step()

            # validation
            if (meta_idx + 1) % self.args.log_iter == 0:
                val_acc, val_nll = self.evaluate(val_dataloader, self.args.num_log_tasks)
                print('Meta-iter {0:d}: train loss = {1:.4f}, train acc = {2:.2f}%, val nll = {3:.4f}, val acc = {4:.2f}%'
                      .format(meta_idx + 1,
                              running_loss / (self.args.log_iter * self.args.batch_size),
                              running_acc / (self.args.log_iter * self.args.batch_size) * 100,
                              val_nll, val_acc * 100)
                      )

                running_loss = 0.
                running_acc = 0.

            # save
            if (meta_idx + 1) % self.args.save_iter == 0:
                val_acc, val_nll = self.evaluate(val_dataloader, self.args.num_val_tasks)
                check_pointer.update(val_acc)
                print('Checkpoint {0:d}: val nll = {1:.4f}, val acc = {2:.2f}%'
                      .format(check_pointer.counter, val_nll, val_acc * 100)
                      )

    def test(self, test_dataloader):
        acc_mean, nll_mean, acc_std, nll_std = self.evaluate(test_dataloader, self.args.num_ts_tasks, return_std=True)
        print('Test: nll = {0:.4f} +/- {1:.4f}, acc = {2:.2f}% +/- {3:.2f}%'
            .format(nll_mean, 1.96 * nll_std / np.sqrt(self.args.num_ts_tasks),
                    acc_mean * 100, 196 * acc_std / np.sqrt(self.args.num_ts_tasks))
              )

    def evaluate(self, dataloader, num_tasks, return_std=False):
        self.model.eval()
        acc_list = []
        nll_list = []

        for eval_idx, task_batch in enumerate(dataloader):
            if eval_idx >= num_tasks:
                break

            for x_supp, y_supp, x_qry, y_qry in zip(*split_task_batch(task_batch, self.args.device)):
                post_params, _ = self.update_support_posterior(x_supp, y_supp)
                with torch.no_grad():
                    y_preds = torch.stack(self.model(x_qry, params=post_params))
                    y_prob = F.softmax(y_preds, dim=-1).mean(dim=0)
                    acc_list.append((y_prob.argmax(dim=1) == y_qry).float().mean().item())
                    nll_list.append(self.args.loss_fn(y_prob.log(), y_qry).item())

        if return_std:
            return np.mean(acc_list), np.mean(nll_list), np.std(acc_list), np.std(nll_list)
        else:
            return np.mean(acc_list), np.mean(nll_list)

    def update_support_posterior(self, x_supp, y_supp):
        prior_mean, prior_log_var = self.get_distribution()
        y_preds = self.model(x_supp)
        nll = torch.stack([self.args.loss_fn(y_pred, y_supp) for y_pred in y_preds]).mean()
        relative_weight = self.args.kld_weight / self.args.num_supp / self.args.num_way
        params = gradient_update_parameters(self.model,
                                            nll,
                                            step_size=self.args.task_lr,
                                            first_order=not self.model.training)

        for _ in range(self.args.task_iter - 1):
            y_preds = self.model(x_supp, params=params)
            nll = torch.stack([self.args.loss_fn(y_pred, y_supp) for y_pred in y_preds]).mean()
            post_mean, post_log_var = self.get_distribution(params)
            kld = kld_normal_standard(post_mean - prior_mean, post_log_var - prior_log_var)
            task_loss = nll + kld * relative_weight
            params = gradient_update_parameters(self.model,
                                                task_loss,
                                                params=params,
                                                step_size=self.args.task_lr,
                                                first_order=not self.model.training)

        if self.model.training:
            y_preds = self.model(x_supp, params=params)
            nll = torch.stack([self.args.loss_fn(y_pred, y_supp) for y_pred in y_preds]).mean()
            return params, nll
        else:
            return params, None

    def get_distribution(self, params=None):
        if params is not None:
            if not isinstance(params, OrderedDict):
                params = OrderedDict(params)
            mean = self.model.get_subdict(params, 'model').values()
            log_var = self.model.get_subdict(params, 'log_var').values()
        else:
            mean = self.model.mean.values()
            log_var = self.model.log_var.values()

        mean = parameters_to_vector(mean)
        log_var = parameters_to_vector(log_var)

        return mean, log_var
