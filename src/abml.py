import torch
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from src.implicit_meta_alg import BMLAlgImplicit
from src.explicit_meta_alg import BMLAlgExplicit
from src.utils import kld_normal_standard
from src.models import FourBlkCNN, ReparamMetaModule


class ABMLImplicit(BMLAlgImplicit):
    def __init__(self, args):
        super(ABMLImplicit, self).__init__(args)

        mean, log_var = self.get_distribution(self.model.meta_named_parameters())
        self.mean_meta_prior = Normal(torch.zeros_like(mean), torch.ones_like(mean))
        self.prec_meta_prior = Gamma(torch.ones_like(log_var), 0.01 * torch.ones_like(log_var))

    def construct_model(self):
        if self.args.dataset.lower() == 'miniimagenet':
            model = FourBlkCNN(self.args.num_way)
            model = ReparamMetaModule(model).to(self.args.device)
        elif self.args.dataset.lower() == 'omniglot':
            model = FourBlkCNN(self.args.num_way, in_channels=1, hidden_size=64, num_feat=64)
            model = ReparamMetaModule(model, num_samples_train=2, num_samples_eval=5).to(self.args.device)
        else:
            raise NotImplementedError

        return model

    def meta_loss(self, qry_nll, *args):
        supp_nll, post_params = args[0:2]
        supp_weight = self.args.num_supp / (self.args.num_supp + self.args.num_qry)
        qry_weight = 1 - supp_weight

        prior_mean, prior_log_var = self.get_distribution()
        post_mean, post_log_var = self.get_distribution(params=post_params)

        supp_kld = kld_normal_standard(post_mean - prior_mean, post_log_var - prior_log_var)
        meta_kld = -1 * (self.mean_meta_prior.log_prob(prior_mean).sum() +
                         self.prec_meta_prior.log_prob(torch.exp(-prior_log_var)).sum())

        task_relative_weight = self.args.kld_weight / (self.args.num_supp + self.args.num_qry) / self.args.num_way
        meta_relative_weight = task_relative_weight / self.args.meta_iter

        meta_loss_post = qry_nll * qry_weight + supp_nll * supp_weight + supp_kld * task_relative_weight
        meta_loss_prior = supp_kld * task_relative_weight + meta_kld * meta_relative_weight

        return meta_loss_post, meta_loss_prior


class ABMLExplicit(BMLAlgExplicit, ABMLImplicit):
    def __init__(self, args):
        super(ABMLExplicit, self).__init__(args)

    def construct_model(self):
        model = ABMLImplicit.construct_model(self)
        return model
        
    def meta_loss(self, qry_nll, *args):
        supp_nll, post_params = args[0:2]
        supp_weight = self.args.num_supp / (self.args.num_supp + self.args.num_qry)
        qry_weight = 1 - supp_weight

        prior_mean, prior_log_var = self.get_distribution()
        post_mean, post_log_var = self.get_distribution(params=post_params)

        supp_kld = kld_normal_standard(post_mean - prior_mean, post_log_var - prior_log_var)
        meta_kld = -1 * (self.mean_meta_prior.log_prob(prior_mean).sum() +
                         self.prec_meta_prior.log_prob(torch.exp(-prior_log_var)).sum())

        task_relative_weight = self.args.kld_weight / (self.args.num_supp + self.args.num_qry) / self.args.num_way
        meta_relative_weight = task_relative_weight / self.args.meta_iter

        meta_loss = qry_nll * qry_weight + supp_nll * supp_weight + supp_kld * task_relative_weight + meta_kld * meta_relative_weight

        return meta_loss
