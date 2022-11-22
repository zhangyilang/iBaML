import torch
from collections import OrderedDict
from torchmeta.modules import MetaModule


def split_task_batch(task_batch, device):
    x_supp_batch, y_supp_batch = task_batch['train']
    x_qry_batch, y_qry_batch = task_batch['test']

    x_supp_batch = x_supp_batch.to(device=device)
    y_supp_batch = y_supp_batch.to(device=device)
    x_qry_batch = x_qry_batch.to(device=device)
    y_qry_batch = y_qry_batch.to(device=device)

    return x_supp_batch, y_supp_batch, x_qry_batch, y_qry_batch


def kld_normal_standard(mean, log_var):
    kl_div = 0.5 * (log_var.exp().sum()
                    + (mean ** 2).sum()
                    - mean.numel()
                    - log_var.sum())

    return kl_div


class Checkpointer:
    def __init__(self, save_fn, alg_name):
        self.save_fn = save_fn
        self.alg_name = alg_name
        self.counter = 0
        self.best_acc = 0

    def update(self, acc):
        self.counter += 1
        self.save_fn(self.alg_name + '_{0:02d}.ct'.format(self.counter))

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_fn(self.alg_name + '_final.ct'.format(self.counter))


def detach_params_allow_grads_(params):
    for param in params.values():
        param.detach_().requires_grad_(True)


def R_op(ys, xs, vs):
    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y, requires_grad=True) for y in ys]
    else:
        ws = torch.zeros_like(ys, requires_grad=True)

    gs = torch.autograd.grad(ys,
                             xs,
                             grad_outputs=ws,
                             create_graph=True,
                             retain_graph=True,
                             allow_unused=True)

    re = torch.autograd.grad(gs,
                             ws,
                             grad_outputs=vs,
                             create_graph=False,
                             retain_graph=True,
                             allow_unused=True)

    return re


@torch.no_grad()
def L_op(ys, xs, ws, create_graph=False, retain_graph=True, vec=True):
    vJ = torch.autograd.grad(ys,
                             xs,
                             grad_outputs=ws,
                             create_graph=create_graph,
                             retain_graph=retain_graph,
                             allow_unused=True)

    if vec:
        vJ = torch.cat([j.detach().reshape(-1) for j in vJ])

    return vJ


# Modified from https://docs.backpack.pt/en/master/use_cases/example_cg_newton.html
@torch.no_grad()
def cg(A, b, x0=None, maxiter=None, tol=1e-5, atol=1e-8):
        r"""Solve :math:`Ax = b` for :math:`x` using conjugate gradient.

        The interface is similar to CG provided by :code:`scipy.sparse.linalg.cg`.

        The main iteration loop follows the pseudo code from Wikipedia:
            https://en.wikipedia.org/w/index.php?title=Conjugate_gradient_method&oldid=855450922

        Parameters
        ----------
        A : function
            Function implementing matrix-vector multiplication by `A`.
        b : torch.Tensor
            Right-hand side of the linear system.
        x0 : torch.Tensor
            Initialization estimate.
        maxiter: int
            Maximum number of iterations.
        tol: float
            Relative tolerance to accept convergence. Stop if
            :math:`|| A x - b || / || b || <` `tol`.
        atol: float
            Absolute tolerance to accept convergence. Stop if
            :math:`|| A x - b || <` `atol`

        Returns
        -------
        x (torch.Tensor): Approximate solution
            :math:`x` of the linear system
        """
        maxiter = b.numel() if maxiter is None else min(maxiter, b.numel())
        x = torch.zeros_like(b, requires_grad=False) if x0 is None else x0.detach()

        # initialize parameters
        r = b - A(x)
        p = r.clone()
        rs_old = (r**2).sum()

        # stopping criterion
        # rs_bound = max([tol * (b**2).sum(), atol])

        # iterate
        iterations = 0
        while True:
            Ap = A(p)
            alpha = rs_old / (p @ Ap)

            x += alpha * p
            r -= alpha * Ap
            rs_new = (r**2).sum()
            iterations += 1

            # if iterations > maxiter or rs_bound > rs_new:
            if iterations >= maxiter:
                return x

            p *= rs_new / rs_old
            p += r
            rs_old = rs_new

# Modified from https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/utils/gradient_based.py
def gradient_update_parameters_(model,
                                loss,
                                params=None,
                                step_size=0.5,
                                first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.iterms(), grads):
            param.sub_(step_size[name] * grad)

    else:
        for param, grad in zip(params.values(), grads):
            param.sub_(step_size * grad)
