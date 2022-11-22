import abc
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
import torch.nn.functional as F
from torchmeta.utils import gradient_update_parameters
from src.utils import (Checkpointer, split_task_batch, kld_normal_standard, L_op,
                       detach_params_allow_grads_, cg, gradient_update_parameters_)
from src.meta_alg_base import BayesianMetaAlgBase


class BMLAlgImplicit(BayesianMetaAlgBase):
    def __init__(self, args):
        super(BMLAlgImplicit, self).__init__(args)

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
                meta_loss_post, meta_loss_prior = self.meta_loss(qry_nll, supp_nll, post_params)
                meta_loss_post.div_(self.args.batch_size), meta_loss_prior.div_(self.args.batch_size)
                # torch.cuda.reset_peak_memory_stats()
                # import time
                # start_time = time.time()
                self.implicit_backward(supp_nll, meta_loss_post, meta_loss_prior, post_params)
                # torch.cuda.synchronize()
                # print('Time:', time.time() - start_time)
                # print('Space:', torch.cuda.max_memory_allocated() / 1024 ** 3)

                with torch.no_grad():
                    running_loss += meta_loss_post.detach().item()
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
        print('Test: loss = {0:.4f} +/- {1:.4f}, acc = {2:.2f}% +/- {3:.2f}%'
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
                                            first_order=True)       # circumvent high-order derivative

        for _ in range(self.args.task_iter - 1):
            y_preds = self.model(x_supp, params=params)
            nll = torch.stack([self.args.loss_fn(y_pred, y_supp) for y_pred in y_preds]).mean()
            post_mean, post_log_var = self.get_distribution(params)
            kld = kld_normal_standard(post_mean - prior_mean, post_log_var - prior_log_var)
            task_loss = nll + kld * relative_weight
            gradient_update_parameters_(self.model,
                                        task_loss,
                                        params=params,
                                        step_size=self.args.task_lr,
                                        first_order=True)

        detach_params_allow_grads_(params)

        if self.model.training:
            y_preds = self.model(x_supp, params=params)
            nll = torch.stack([self.args.loss_fn(y_pred, y_supp) for y_pred in y_preds]).mean()
            return params, nll
        else:
            return params, None

    def get_distribution(self, params=None, vec=True):
        if params is not None:
            if not isinstance(params, OrderedDict):
                params = OrderedDict(params)
            mean = self.model.get_subdict(params, 'model').values()
            log_var = self.model.get_subdict(params, 'log_var').values()
        else:   # prior
                mean = self.model.mean.values()
                log_var = self.model.log_var.values()

        if vec:
            mean = parameters_to_vector(mean)
            log_var = parameters_to_vector(log_var)

        return mean, log_var

    def implicit_backward(self, supp_nll, meta_loss_post, meta_loss_prior, post_params):
        post_var_vec = parameters_to_vector(self.model.get_subdict(post_params, 'log_var').values()).exp()
        prior_mean, prior_log_var = self.get_distribution(vec=False)

        task_post_grad_vec = parameters_to_vector(
            torch.autograd.grad(supp_nll,
                                post_params.values(),
                                create_graph=True,
                                retain_graph=True,
                                allow_unused=True))
        half_offset = task_post_grad_vec.numel() // 2
        task_post_grad_vec[half_offset:].div(post_var_vec)          # deriv. of log d_t -> d_t

        if meta_loss_post is not None:
            with torch.no_grad():
                # implicit differentiation
                meta_post_grad_vec = parameters_to_vector(
                    torch.autograd.grad(meta_loss_post,
                                        post_params.values(),
                                        create_graph=False,
                                        retain_graph=True,
                                        allow_unused=True))
                meta_post_grad_vec[half_offset:].div_(post_var_vec) # deriv. of log d_t -> d_t

                prior_var_vec = parameters_to_vector(prior_log_var).exp()

                # conjugate gradient method
                prior_var_inv_vec = 1 / prior_var_vec
                diag_term = torch.cat([prior_var_inv_vec, 0.5 * (prior_var_inv_vec + 2 * post_var_vec) ** 2])
                def Avec_fn(vec):
                    Avec = L_op(task_post_grad_vec,
                                post_params.values(),
                                vec)
                    Avec[half_offset:].div_(post_var_vec)           # deriv. of log d_t -> d_t
                    Avec.add_(diag_term * vec)
                    return Avec

                prior_grad_vec = cg(A=Avec_fn,
                                    b=meta_post_grad_vec,
                                    x0=meta_post_grad_vec / diag_term,
                                    maxiter=self.args.cg_iter)

                prior_grad_vec.mul_(diag_term)
                prior_grad_vec[half_offset:].sub_(prior_grad_vec[:half_offset] * task_post_grad_vec[:half_offset])
                prior_grad_vec[half_offset:].mul_(prior_var_vec)    # deriv. of d -> log d

                # apply grads
                offset = 0
                for mean, log_var in zip(prior_mean, prior_log_var):
                    offset_new = offset + mean.numel()
                    if mean.grad is not None:
                        mean.grad += prior_grad_vec[offset:offset_new].view_as(mean)
                        log_var.grad += prior_grad_vec[half_offset+offset:half_offset+offset_new].view_as(log_var)
                    else:
                        mean.grad = prior_grad_vec[offset:offset_new].view_as(mean)
                        log_var.grad = prior_grad_vec[half_offset+offset:half_offset+offset_new].view_as(log_var)
                    offset = offset_new

        if meta_loss_prior is not None:
            meta_loss_prior.backward()
