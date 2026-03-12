import os
import sys
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F, init
from torch.distributions.studentT import StudentT

from src.utils import (
    bernoulli_log_pdf,
    masked_bernoulli_log_pdf,
    masked_gaussian_log_pdf,
    standard_normal_log_pdf,
    normal_log_pdf,
    kl_divergence_standard_normal_prior,
    kl_divergence_standard_student_prior,
    log_mean_exp,
    product_of_experts,
    aggregate_pseudo_obs_to_nig,
)


# from src.torch_core.flows import NormalizingFlows


class MLE_1PL(nn.Module):

    def __init__(
            self,
            latent_dim,
            num_person,
            num_item,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.ability_dim = latent_dim
        self.response_dim = 1
        self.num_person = num_person
        self.num_item = num_item

        self._set_item_feat_dim()

        self.ability = nn.Embedding(self.num_person, self.ability_dim)
        self.item_feat = nn.Embedding(self.num_item, self.item_feat_dim)

        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim

    def encode(self, index, response, mask):
        ability = self.ability(index)
        item_domain = torch.arange(self.num_item).unsqueeze(1).to(response.device)
        item_feat = self.item_feat(item_domain).squeeze(1)
        return ability, item_feat

    def decode(self, ability, item_feat):
        return irt_model_1pl(ability, item_feat)

    def forward(self, index, response, mask):
        ability, item_feat = self.encode(index, response, mask)
        response_mu = self.decode(ability, item_feat)
        return response_mu

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class MLE_2PL(MLE_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 1

    def decode(self, ability, item_feat):
        return irt_model_2pl(ability, item_feat)


class MLE_3PL(MLE_2PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 2

    def decode(self, ability, item_feat):
        return irt_model_3pl(ability, item_feat)


class VI_1PL(nn.Module):

    def __init__(
            self,
            latent_dim,
            num_person,
            num_item,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.ability_dim = latent_dim
        self.response_dim = 1
        self.num_person = num_person
        self.num_item = num_item

        self._set_item_feat_dim()

        self.ability_mu_lookup = nn.Embedding(self.num_person, self.ability_dim)
        self.ability_logvar_lookup = nn.Embedding(self.num_person, self.ability_dim)

        self.item_mu_lookup = nn.Embedding(self.num_item, self.item_feat_dim)
        self.item_logvar_lookup = nn.Embedding(self.num_item, self.item_feat_dim)

        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim

    def forward(self, index, response, mask):
        ability, ability_mu, ability_logvar, item_feat, item_feat_mu, item_feat_logvar \
            = self.encode(index, response, mask)
        response_mu = self.decode(ability, item_feat)

        return response, mask, response_mu, ability, ability_mu, ability_logvar, \
            item_feat, item_feat_mu, item_feat_logvar

    def encode(self, index, response, mask):
        device = response.device

        item_domain = torch.arange(self.num_item).unsqueeze(1).to(device)
        item_feat_mu = self.item_mu_lookup(item_domain).squeeze(1)
        item_feat_logvar = self.item_logvar_lookup(item_domain).squeeze(1)
        item_feat = self.reparameterize_gaussian(item_feat_mu, item_feat_logvar)

        ability_mu = self.ability_mu_lookup(index)
        ability_logvar = self.ability_logvar_lookup(index)
        ability = self.reparameterize_gaussian(ability_mu, ability_logvar)

        return ability, ability_mu, ability_logvar, \
            item_feat, item_feat_mu, item_feat_logvar

    def decode(self, ability, item_feat):
        return irt_model_1pl(ability, item_feat)

    def elbo(
            self,
            response,
            mask,
            response_mu,
            ability,
            ability_mu,
            ability_logvar,
            item_feat,
            item_feat_mu,
            item_feat_logvar,
            annealing_factor=1,
            use_kl_divergence=True,
    ):
        log_p_r_j_given_d_u = masked_bernoulli_log_pdf(response, mask, response_mu).sum()

        if use_kl_divergence:
            kl_q_u_p_u = kl_divergence_standard_normal_prior(ability_mu, ability_logvar).sum()
            kl_q_d_p_d = kl_divergence_standard_normal_prior(item_feat_mu, item_feat_logvar).sum()
            elbo = log_p_r_j_given_d_u - annealing_factor * kl_q_u_p_u - annealing_factor * kl_q_d_p_d
        else:
            log_p_u = standard_normal_log_pdf(ability).sum()
            log_p_d = standard_normal_log_pdf(item_feat).sum()
            log_q_u = normal_log_pdf(ability, ability_mu, ability_logvar).sum()
            log_q_d = normal_log_pdf(item_feat, item_feat_mu, item_feat_logvar).sum()

            model_log_prob_sum = log_p_r_j_given_d_u + log_p_u + log_p_d
            guide_log_prob_sum = log_q_u + log_q_d

            elbo = model_log_prob_sum - guide_log_prob_sum

        return -elbo

    def log_marginal(self, response, mask, num_samples=100):
        with torch.no_grad():
            log_weight = []
            for _ in range(num_samples):
                (
                    response,
                    mask,
                    response_mu,
                    ability,
                    ability_mu,
                    ability_logvar,
                    item_feat,
                    item_feat_mu,
                    item_feat_logvar,
                ) = self.forward(response, mask)

                log_w = -self.elbo(
                    response,
                    mask,
                    response_mu,
                    ability,
                    ability_mu,
                    ability_logvar,
                    item_feat,
                    item_feat_mu,
                    item_feat_logvar,
                    annealing_factor=1,
                    use_kl_divergence=False,
                )
                log_weight.append(log_w)

            log_weight = torch.stack(log_weight)
            logp = torch.logsumexp(log_weight, 0) - math.log(num_samples)

        return logp

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class VI_2PL(VI_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 1

    def decode(self, ability, item_feat):
        return irt_model_2pl(ability, item_feat)


class VI_3PL(VI_2PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 2

    def decode(self, ability, item_feat):
        return irt_model_3pl(ability, item_feat)


class VIBO_1PL(nn.Module):

    def __init__(
            self,
            ability_dim,
            num_item,
            Q_mat=None,
            hidden_dim=64,
            ability_merge='mean',
            conditional_posterior=False,
            generative_model='irt',
            response_dist='bernoulli',
            replace_missing_with_prior=True,
            n_norm_flows=0,
    ):
        super().__init__()

        assert ability_merge in ['mean', 'product']
        assert generative_model in ['irt', 'link', 'deep', 'residual']
        assert response_dist in ['bernoulli', 'gaussian']

        self.latent_dim = ability_dim
        self.ability_dim = ability_dim
        self.response_dim = 1
        self.hidden_dim = hidden_dim
        self.num_item = num_item
        self.ability_merge = ability_merge
        self.conditional_posterior = conditional_posterior
        self.generative_model = generative_model
        self.response_dist = response_dist
        self.replace_missing_with_prior = replace_missing_with_prior
        self.n_norm_flows = n_norm_flows
        if Q_mat is not None:
            q_array = np.asarray(Q_mat, dtype=np.float32)
            if q_array.shape[0] != self.num_item or q_array.shape[1] != self.ability_dim:
                raise ValueError("Q_mat shape must be (num_item, ability_dim)")
            self.register_buffer('Q_mat', torch.tensor(q_array))

        self._set_item_feat_dim()
        self._set_irt_num()

        if self.conditional_posterior:
            self.ability_encoder = ConditionalAbilityInferenceNetwork(
                self.ability_dim,
                self.response_dim,
                self.item_feat_dim,
                self.hidden_dim,
                ability_merge=self.ability_merge,
                replace_missing_with_prior=self.replace_missing_with_prior,
            )
        else:
            self.ability_encoder = AbilityInferenceNetwork(
                self.ability_dim,
                self.response_dim,
                self.hidden_dim,
                ability_merge=self.ability_merge,
                replace_missing_with_prior=self.replace_missing_with_prior,
            )

        self.item_encoder = ItemInferenceNetwork(self.num_item, self.item_feat_dim)

        # if self.n_norm_flows > 0:
        #     self.ability_norm_flows = NormalizingFlows(
        #         self.ability_dim,
        #         n_flows=self.n_norm_flows,
        #     )
        #     self.item_norm_flows = NormalizingFlows(
        #         self.item_feat_dim,
        #         n_flows=self.n_norm_flows,
        #     )

        if self.generative_model == 'link':
            self.decoder = LinkedIRT(
                irt_model=f'{self.irt_num}pl',
                hidden_dim=self.hidden_dim,
            )
        elif self.generative_model == 'deep':
            self.decoder = DeepIRT(
                self.ability_dim,
                irt_model=f'{self.irt_num}pl',
                hidden_dim=self.hidden_dim,
            )
        elif self.generative_model == 'residual':
            self.decoder = ResidualIRT(
                self.ability_dim,
                irt_model=f'{self.irt_num}pl',
                hidden_dim=self.hidden_dim,
            )

        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim
        # self.item_feat_dim = self.ability_dim

    def _set_irt_num(self):
        self.irt_num = 1

    def forward(self, response, mask):
        ability, ability_mu, ability_logvar, \
            item_feat, item_feat_mu, item_feat_logvar \
            = self.encode(response, mask)

        if self.n_norm_flows > 0:
            ability_k, ability_logabsdetjac = self.ability_norm_flows(ability)
            item_feat_k, item_feat_logabsdetjac = self.item_norm_flows(item_feat)
            response_mu = self.decode(ability_k, item_feat_k)
            return response, mask, response_mu, \
                ability_k, ability, ability_mu, ability_logvar, ability_logabsdetjac, \
                item_feat_k, item_feat, item_feat_mu, item_feat_logvar, item_feat_logabsdetjac

        else:
            response_mu = self.decode(ability, item_feat)
            return response, mask, response_mu, \
                ability, ability_mu, ability_logvar, \
                item_feat, item_feat_mu, item_feat_logvar

    def encode(self, response, mask):
        device = response.device

        item_domain = torch.arange(self.num_item).unsqueeze(1).to(device)
        item_feat_mu, item_feat_logvar = self.item_encoder(item_domain)
        item_feat = self.reparameterize_gaussian(item_feat_mu, item_feat_logvar)

        if self.conditional_posterior:
            ability_mu, ability_logvar = self.ability_encoder(response, mask, item_feat)
        else:
            ability_mu, ability_logvar = self.ability_encoder(response, mask)

        ability = self.reparameterize_gaussian(ability_mu, ability_logvar)

        return ability, ability_mu, ability_logvar, \
            item_feat, item_feat_mu, item_feat_logvar

    def decode(self, ability, item_feat):
        if self.generative_model == 'irt':
            response_mu = irt_model_1pl(ability, item_feat)
            # target_dtype = self.ability_agg_mat.weight.dtype
            # ability = ability.to(dtype=target_dtype)
            # item_feat = item_feat.to(dtype=target_dtype)
            # q_mat = self.Q_mat.to(dtype=target_dtype)
            # ability_agg = self.ability_agg_mat(ability.unsqueeze(1) * q_mat.unsqueeze(0))
            # item_feat_agg = self.item_agg_mat(item_feat.unsqueeze(0) * q_mat.unsqueeze(0))
            # response_mu = irt_model_1pl_q(ability_agg, item_feat_agg)
            return response_mu
        else:
            return self.decoder(ability, item_feat)

    def elbo(
            self,
            response,
            mask,
            response_mu,
            ability,
            ability_mu,
            ability_logvar,
            item_feat,
            item_feat_mu,
            item_feat_logvar,
            annealing_factor=1,
            use_kl_divergence=True,
            ability_k=None,
            item_feat_k=None,
            ability_logabsdetjac=None,
            item_logabsdetjac=None,
    ):
        if self.response_dist == 'bernoulli':
            log_p_r_j_given_d_u = masked_bernoulli_log_pdf(response, mask, response_mu).sum()
        elif self.response_dist == 'gaussian':
            response_logvar = 2. * torch.log(torch.ones_like(response_mu) * 0.1)
            log_p_r_j_given_d_u = masked_gaussian_log_pdf(response, mask, response_mu, response_logvar).sum()
        else:
            raise Exception(f'response_dist {self.response_dist} not supported.')

        if self.n_norm_flows > 0:
            assert ability_logabsdetjac is not None
            assert item_logabsdetjac is not None
            assert ability_k is not None
            assert item_feat_k is not None

            log_q_u_0 = normal_log_pdf(ability, ability_mu, ability_logvar).sum()
            log_q_d_0 = normal_log_pdf(item_feat, item_feat_mu, item_feat_logvar).sum()

            log_p_u_k = standard_normal_log_pdf(ability_k).sum()
            log_p_d_k = standard_normal_log_pdf(item_feat_k).sum()

            log_q_u_k = log_q_u_0 - ability_logabsdetjac.sum()
            log_q_d_k = log_q_d_0 - item_logabsdetjac.sum()

            model_log_prob_sum = log_p_r_j_given_d_u + log_p_u_k + log_p_d_k
            guide_log_prob_sum = log_q_u_k + log_q_d_k

            elbo = model_log_prob_sum - guide_log_prob_sum

        else:
            if use_kl_divergence:
                kl_q_u_p_u = kl_divergence_standard_normal_prior(ability_mu, ability_logvar).sum()
                kl_q_d_p_d = kl_divergence_standard_normal_prior(item_feat_mu, item_feat_logvar).sum()
                elbo = log_p_r_j_given_d_u - annealing_factor * kl_q_u_p_u - annealing_factor * kl_q_d_p_d

            else:
                log_p_u = standard_normal_log_pdf(ability).sum()
                log_p_d = standard_normal_log_pdf(item_feat).sum()
                log_q_u = normal_log_pdf(ability, ability_mu, ability_logvar).sum()
                log_q_d = normal_log_pdf(item_feat, item_feat_mu, item_feat_logvar).sum()

                model_log_prob_sum = log_p_r_j_given_d_u + log_p_u + log_p_d
                guide_log_prob_sum = log_q_u + log_q_d

                elbo = model_log_prob_sum - guide_log_prob_sum

        return -elbo

    def log_marginal(self, response, mask, num_samples=100):
        with torch.no_grad():
            log_weight = []
            for _ in range(num_samples):
                if self.n_norm_flows > 0:
                    (
                        response,
                        mask,
                        response_mu,
                        ability_k,
                        ability,
                        ability_mu,
                        ability_logvar,
                        ability_logabsdetjac,
                        item_feat_k,
                        item_feat,
                        item_feat_mu,
                        item_feat_logvar,
                        item_feat_logabsdetjac,
                    ) = self.forward(response, mask)
                else:
                    (
                        response,
                        mask,
                        response_mu,
                        ability,
                        ability_mu,
                        ability_logvar,
                        item_feat,
                        item_feat_mu,
                        item_feat_logvar,
                    ) = self.forward(response, mask)
                    ability_k = None
                    item_feat_k = None
                    ability_logabsdetjac = None
                    item_feat_logabsdetjac = None

                log_w = -self.elbo(
                    response,
                    mask,
                    response_mu,
                    ability,
                    ability_mu,
                    ability_logvar,
                    item_feat,
                    item_feat_mu,
                    item_feat_logvar,
                    annealing_factor=1,
                    use_kl_divergence=False,
                    ability_k=ability_k,
                    item_feat_k=item_feat_k,
                    ability_logabsdetjac=ability_logabsdetjac,
                    item_logabsdetjac=item_feat_logabsdetjac,
                )
                log_weight.append(log_w)

            log_weight = torch.stack(log_weight)
            logp = torch.logsumexp(log_weight, 0) - math.log(num_samples)

        return logp

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class VIBO_2PL(VIBO_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 1

    def _set_irt_num(self):
        self.irt_num = 2

    def decode(self, ability, item_feat):
        if hasattr(self, "Q_mat"):
            q_mat = self.Q_mat
            discrimination = item_feat[:, :self.ability_dim] * q_mat
            difficulty = item_feat[:, self.ability_dim:]
            item_feat = torch.cat([discrimination, difficulty], dim=1)
        if self.generative_model == 'irt':
            return irt_model_2pl(ability, item_feat)
        else:
            return self.decoder(ability, item_feat)


class VIBO_3PL(VIBO_2PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 2

    def _set_irt_num(self):
        self.irt_num = 3

    def decode(self, ability, item_feat):
        if self.generative_model == 'irt':
            return irt_model_3pl(ability, item_feat)
        else:
            return self.decoder(ability, item_feat)


class EvidIRT_1PL(nn.Module):

    def __init__(
            self,
            latent_dim,
            num_item,
            Q_mat=None,
            hidden_dim=64,
            ability_merge='sum',
            conditional_posterior=False,
            generative_model='irt',
            response_dist='bernoulli',
            replace_missing_with_prior=True,
            n_norm_flows=0,
    ):
        super().__init__()

        assert ability_merge in ['sum']
        assert generative_model in ['irt', 'link', 'deep', 'residual', 'neural', 'dina']
        assert response_dist in ['bernoulli', 'gaussian']

        self.latent_dim = latent_dim
        self.ability_dim = latent_dim
        self.response_dim = 1
        self.hidden_dim = hidden_dim
        self.num_item = num_item
        self.ability_merge = ability_merge
        self.conditional_posterior = conditional_posterior
        self.generative_model = generative_model
        self.response_dist = response_dist
        self.replace_missing_with_prior = replace_missing_with_prior
        self.n_norm_flows = n_norm_flows
        if Q_mat is not None:
            q_array = np.asarray(Q_mat, dtype=np.float32)
            if q_array.shape[0] != self.num_item or q_array.shape[1] != self.ability_dim:
                raise ValueError("Q_mat shape must be (num_item, ability_dim)")
            self.register_buffer('Q_mat', torch.tensor(q_array))

        self._set_item_feat_dim()
        self._set_irt_num()

        if self.conditional_posterior:
            pass
            # self.ability_encoder = ConditionalEvidentialAbilityInferenceNetwork(
            #     self.ability_dim, 
            #     self.response_dim, 
            #     self.item_feat_dim, 
            #     self.hidden_dim, 
            #     ability_merge = self.ability_merge,
            #     replace_missing_with_prior = self.replace_missing_with_prior,
            # )
        else:
            self.ability_encoder = EvidentialAbilityInferenceNetwork(
                self.ability_dim,
                self.response_dim,
                self.hidden_dim,
                ability_merge=self.ability_merge,
                replace_missing_with_prior=self.replace_missing_with_prior,
            )

        self.item_encoder = ItemInferenceNetwork(self.num_item, self.item_feat_dim)

        if self.generative_model == 'link':
            self.decoder = LinkedIRT(
                irt_model=f'{self.irt_num}pl',
                hidden_dim=self.hidden_dim,
            )
        elif self.generative_model == 'deep':
            self.decoder = DeepIRT(
                self.ability_dim,
                irt_model=f'{self.irt_num}pl',
                hidden_dim=self.hidden_dim,
            )
        elif self.generative_model == 'residual':
            self.decoder = ResidualIRT(
                self.ability_dim,
                irt_model=f'{self.irt_num}pl',
                hidden_dim=self.hidden_dim,
            )

        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim

    def _set_irt_num(self):
        self.irt_num = 1

    def forward(self, response, mask):
        ability, ability_gamma, ability_v, ability_alpha, ability_beta, \
            item_feat, item_feat_mu, item_feat_logvar \
            = self.encode(response, mask)

        response_gamma = self.decode(ability, item_feat)
        return response, mask, response_gamma, \
            ability, ability_gamma, ability_v, ability_alpha, ability_beta, \
            item_feat, item_feat_mu, item_feat_logvar

    def encode(self, response, mask):
        device = response.device

        item_domain = torch.arange(self.num_item).unsqueeze(1).to(device)
        item_feat_mu, item_feat_logvar = self.item_encoder(item_domain)
        item_feat = self.reparameterize_gaussian(item_feat_mu, item_feat_logvar)

        if self.conditional_posterior:
            pass
            # ability_mu, ability_logvar = self.ability_encoder(response, mask, item_feat) 
        else:
            ability_gamma, ability_v, ability_alpha, ability_beta = self.ability_encoder(response, mask)

        ability = self.reparameterize_student(ability_gamma, ability_v, ability_alpha, ability_beta)

        return ability, ability_gamma, ability_v, ability_alpha, ability_beta, \
            item_feat, item_feat_mu, item_feat_logvar

    def decode(self, ability, item_feat):
        if self.generative_model == 'irt':
            response_gamma = irt_model_1pl(ability, item_feat)
            return response_gamma
        else:
            return self.decoder(ability, item_feat)

    def elbo(
            self,
            response,
            mask,
            response_gamma,
            ability,
            ability_gamma,
            ability_v,
            ability_alpha,
            ability_beta,
            item_feat,
            item_feat_mu,
            item_feat_logvar,
            annealing_factor=0.1,
            use_kl_divergence=True,
            ability_k=None,
            item_feat_k=None,
            ability_logabsdetjac=None,
            item_logabsdetjac=None,
    ):
        if self.response_dist == 'bernoulli':
            log_p_r_j_given_d_u_ = masked_bernoulli_log_pdf(response, mask, response_gamma)
        elif self.response_dist == 'gaussian':
            response_logvar = 2. * torch.log(torch.ones_like(response_gamma) * 0.1)
            log_p_r_j_given_d_u_ = masked_gaussian_log_pdf(response, mask, response_gamma, response_logvar)
        else:
            raise Exception(f'response_dist {self.response_dist} not supported.')
        log_p_r_j_given_d_u = log_p_r_j_given_d_u_.sum()
        kl_q_d_p_d = kl_divergence_standard_normal_prior(item_feat_mu, item_feat_logvar).sum()
        kl_q_u_p_u = kl_divergence_standard_student_prior(ability_gamma, ability_v, ability_alpha, ability_beta).sum()
        # annealing_factor /= self.ability_dim
        # elbo = log_p_r_j_given_d_u - annealing_factor * kl_q_d_p_d - annealing_factor * kl_q_u_p_u
        unc = ability_v * (ability_alpha - 1) / (ability_beta * (ability_v + 1))
        unc_mean = unc.mean(dim=1).unsqueeze(-1)
        elbo = log_p_r_j_given_d_u + annealing_factor * (log_p_r_j_given_d_u_.sum(
            dim=1) * unc_mean).sum() - annealing_factor * kl_q_d_p_d - annealing_factor * kl_q_u_p_u
        # - annealing_factor * log_p_r_j_given_d_u * (ability_v * (ability_alpha - 1) / (ability_beta * (ability_v + 1)))
        # torch.log(2 * ability_v + ability_alpha)
        return -elbo

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    @staticmethod
    def reparameterize_student(gamma, v, alpha, beta):
        return StudentT(2 * alpha, loc=gamma, scale=torch.sqrt(beta * (1 + v) / (v * alpha))).rsample()

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class EvidIRT_2PL(EvidIRT_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 1

    def _set_irt_num(self):
        self.irt_num = 2

    def decode(self, ability, item_feat):
        q_mat = None
        if hasattr(self, "Q_mat"):
            q_mat = self.Q_mat
            discrimination = item_feat[:, :self.ability_dim] * q_mat
            difficulty = item_feat[:, self.ability_dim:]
            item_feat = torch.cat([discrimination, difficulty], dim=1)
        if self.generative_model == 'irt':
            return irt_model_2pl(ability, item_feat, q_mat)
        else:
            return self.decoder(ability, item_feat)


class EvidIRT_3PL(EvidIRT_2PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim + 2

    def _set_irt_num(self):
        self.irt_num = 3

    def decode(self, ability, item_feat):
        if self.generative_model == 'irt':
            return irt_model_3pl(ability, item_feat)
        else:
            return self.decoder(ability, item_feat)


class VIBO_DINA(VIBO_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = 2

    def decode(self, ability, item_feat):
        q_mat = self.Q_mat
        slip = torch.sigmoid(item_feat[:, 0])
        guess = torch.sigmoid(item_feat[:, 1])
        skill_prob = torch.sigmoid(ability).clamp(1e-6, 1 - 1e-6)
        log_eta = (q_mat.unsqueeze(0) * torch.log(skill_prob).unsqueeze(1)).sum(dim=-1)
        eta = torch.exp(log_eta).clamp(1e-6, 1 - 1e-6)
        prob = eta * (1 - slip) + (1 - eta) * guess
        return prob.unsqueeze(-1).clamp(0.0, 1.0)


class Evid_DINA(EvidIRT_1PL):

    def _set_item_feat_dim(self):
        self.item_feat_dim = 2

    def decode(self, ability, item_feat):
        q_mat = self.Q_mat
        slip = torch.sigmoid(item_feat[:, 0])
        guess = torch.sigmoid(item_feat[:, 1])
        skill_prob = torch.sigmoid(ability).clamp(1e-6, 1 - 1e-6)
        log_eta = (q_mat.unsqueeze(0) * torch.log(skill_prob).unsqueeze(1)).sum(dim=-1)
        eta = torch.exp(log_eta).clamp(1e-6, 1 - 1e-6)
        prob = eta * (1 - slip) + (1 - eta) * guess
        return prob.unsqueeze(-1).clamp(0.0, 1.0)


class VIBO_NCD(VIBO_1PL):
    def __init__(
            self,
            ability_dim,
            num_item,
            Q_mat=None,
            hidden_dim=64,
            ability_merge='mean',
            conditional_posterior=False,
            generative_model='irt',
            response_dist='bernoulli',
            replace_missing_with_prior=True,
            n_norm_flows=0,
    ):
        super().__init__(ability_dim, num_item, Q_mat, hidden_dim, ability_merge, conditional_posterior,
                         generative_model, response_dist, replace_missing_with_prior, n_norm_flows)
        self.net = nn.Sequential(
            PosLinear(self.latent_dim, self.latent_dim),
            nn.ELU(inplace=True),
            PosLinear(self.latent_dim, self.latent_dim),
            nn.ELU(inplace=True),
            PosLinear(self.latent_dim, 1),
            nn.Sigmoid(),
        )
        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim * 2

    def decode(self, ability, item_feat):
        q_mat = self.Q_mat
        ability = ability
        discrimination = torch.sigmoid(item_feat[:, :self.ability_dim])
        difficulty = torch.sigmoid(item_feat[:, self.ability_dim:])
        x = (ability[:, None, :] - difficulty[None, :, :]) * (discrimination * q_mat)[None, :, :]
        return self.net(x)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class Evid_NCD(EvidIRT_1PL):
    def __init__(
            self,
            latent_dim,
            num_item,
            Q_mat=None,
            hidden_dim=64,
            ability_merge='sum',
            conditional_posterior=False,
            generative_model='irt',
            response_dist='bernoulli',
            replace_missing_with_prior=True,
            n_norm_flows=0,
    ):
        super().__init__(latent_dim, num_item, Q_mat, hidden_dim, ability_merge, conditional_posterior,
                         generative_model, response_dist, replace_missing_with_prior, n_norm_flows)
        self.net = nn.Sequential(
            PosLinear(self.latent_dim, self.latent_dim),
            nn.ELU(inplace=True),
            PosLinear(self.latent_dim, self.latent_dim),
            nn.ELU(inplace=True),
            PosLinear(self.latent_dim, 1),
            nn.Sigmoid(),
        )
        self.apply(self.weights_init)

    def _set_item_feat_dim(self):
        self.item_feat_dim = self.latent_dim * 2

    def decode(self, ability, item_feat):
        q_mat = self.Q_mat
        ability = ability
        discrimination = torch.sigmoid(item_feat[:, :self.ability_dim])
        difficulty = torch.sigmoid(item_feat[:, self.ability_dim:])
        x = (ability[:, None, :] - difficulty[None, :, :]) * (discrimination * q_mat)[None, :, :]
        return self.net(x)

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class AbilityInferenceNetwork(nn.Module):

    def __init__(
            self,
            ability_dim,
            response_dim,
            hidden_dim=64,
            ability_merge='mean',
            replace_missing_with_prior=True,
    ):
        super().__init__()

        self.ability_dim = ability_dim
        self.response_dim = response_dim
        self.hidden_dim = hidden_dim
        self.ability_merge = ability_merge
        self.replace_missing_with_prior = replace_missing_with_prior

        getattr(self, f'_create_models_{self.ability_merge}')(
            self.response_dim,
            self.hidden_dim,
            self.ability_dim * 2,
        )

    def _create_models_product(self, input_dim, hidden_dim, output_dim):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def _create_models_mean(self, input_dim, hidden_dim, output_dim):
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def _forward_product(self, mlp_input, mask, num_person, num_item):
        has_missing = bool(torch.sum(1 - mask).item()) if mask is not None else False

        mu_flat, logvar_flat = torch.chunk(self.mlp(mlp_input), 2, dim=1)
        mu_set = mu_flat.view(num_person, num_item, self.ability_dim)
        logvar_set = logvar_flat.view(num_person, num_item, self.ability_dim)

        p_mu_set = torch.zeros_like(mu_set)
        p_logvar_set = torch.zeros_like(logvar_set)

        if has_missing:
            mu, logvar = [], []
            for i in range(num_person):
                if mask[i].sum().item() != num_item:
                    mask_i = mask[i].bool().repeat(1, self.ability_dim)
                    mu_set_i = mu_set[i][mask_i].view(-1, self.ability_dim)
                    logvar_set_i = logvar_set[i][mask_i].view(-1, self.ability_dim)
                    # replace all missing items with a prior score
                    if self.replace_missing_with_prior:
                        p_mu_set_i = p_mu_set[i][~mask_i].view(-1, self.ability_dim)
                        p_logvar_set_i = p_logvar_set[i][~mask_i].view(-1, self.ability_dim)
                        mu_set_i = torch.cat([mu_set_i, p_mu_set_i], dim=0)
                        logvar_set_i = torch.cat([logvar_set_i, p_logvar_set_i], dim=0)
                        assert mu_set_i.size(0) == num_item
                        assert logvar_set_i.size(0) == num_item
                else:
                    mu_set_i, logvar_set_i = mu_set[i], logvar_set[i]
                mu_i, logvar_i = product_of_experts(mu_set_i, logvar_set_i)
                mu.append(mu_i);
                logvar.append(logvar_i)
            mu, logvar = torch.stack(mu), torch.stack(logvar)
        else:
            mu, logvar = product_of_experts(mu_set.permute(1, 0, 2), logvar_set.permute(1, 0, 2))

        return mu, logvar

    def _forward_mean(self, mlp_input, mask, num_person, num_item):
        has_missing = bool(torch.sum(1 - mask).item()) if mask is not None else False

        hid = F.elu(self.mlp1(mlp_input))
        hid = hid.view(num_person, num_item, self.hidden_dim)
        if has_missing:
            hid_mean = []
            for i in range(num_person):
                hid_i = hid[i][mask[i].repeat(1, self.hidden_dim).bool()]
                num_i = mask[i].squeeze().sum().item()
                hid_i = hid_i.view(num_i, self.hidden_dim)
                hid_i_mean = hid_i.mean(0)
                hid_mean.append(hid_i_mean)
            hid_mean = torch.stack(hid_mean)
        else:
            hid_mean = hid.mean(1)

        mu, logvar = torch.chunk(self.mlp2(hid_mean), 2, dim=1)

        return mu, logvar

    def forward(self, response, mask):
        num_person, num_item, response_dim = response.size()
        mlp_input = response.view(num_person * num_item, response_dim)

        return getattr(self, f'_forward_{self.ability_merge}')(
            mlp_input,
            mask,
            num_person,
            num_item,
        )


class ConditionalAbilityInferenceNetwork(AbilityInferenceNetwork):

    def __init__(
            self,
            ability_dim,
            response_dim,
            item_feat_dim,
            hidden_dim=64,
            ability_merge='mean',
            replace_missing_with_prior=True,
    ):
        super().__init__(
            ability_dim,
            response_dim,
            hidden_dim=hidden_dim,
            ability_merge=ability_merge,
            replace_missing_with_prior=replace_missing_with_prior,
        )
        self.ability_dim = ability_dim
        self.response_dim = response_dim
        self.item_feat_dim = item_feat_dim
        self.hidden_dim = hidden_dim
        self.ability_merge = ability_merge
        self.replace_missing_with_prior = replace_missing_with_prior

        getattr(self, f'_create_models_{self.ability_merge}')(
            self.response_dim + self.item_feat_dim,
            self.hidden_dim,
            self.ability_dim * 2,
        )

    def forward(self, response, mask, item_feat):
        num_person, num_item, response_dim = response.size()
        item_feat_dim = item_feat.size(1)

        response_flat = response.view(num_person * num_item, response_dim)
        item_feat_flat = item_feat.unsqueeze(0).repeat(num_person, 1, 1)
        item_feat_flat = item_feat_flat.view(num_person * num_item, item_feat_dim)

        mlp_input = torch.cat([response_flat, item_feat_flat], dim=1)

        return getattr(self, f'_forward_{self.ability_merge}')(
            mlp_input,
            mask,
            num_person,
            num_item,
        )


class EvidentialAbilityInferenceNetwork(nn.Module):

    def __init__(
            self,
            ability_dim,
            response_dim,
            hidden_dim=64,
            ability_merge='sum',
            replace_missing_with_prior=True,
            m0=0.0,
            kappa0=0.1,
            alpha0=2.0,
            beta0=1.0,
    ):
        super().__init__()

        self.ability_dim = ability_dim
        self.response_dim = response_dim
        self.hidden_dim = hidden_dim
        self.ability_merge = ability_merge
        self.replace_missing_with_prior = replace_missing_with_prior

        def _as_prior_param(x):
            t = torch.as_tensor(x, dtype=torch.float32).view(-1)
            if t.numel() == 1:
                t = t.repeat(self.ability_dim)
            elif t.numel() != self.ability_dim:
                raise ValueError(f'prior param must be scalar or length={self.ability_dim}, got {t.numel()}')
            return t

        self.register_buffer('m0', _as_prior_param(m0))
        self.register_buffer('kappa0', _as_prior_param(kappa0))
        self.register_buffer('alpha0', _as_prior_param(alpha0))
        self.register_buffer('beta0', _as_prior_param(beta0))

        getattr(self, f'_create_models_{self.ability_merge}')(
            self.response_dim,
            self.hidden_dim,
            self.ability_dim * 3,
        )

    def _create_models_sum(self, input_dim, hidden_dim, output_dim):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def _forward_sum(self, mlp_input, mask, num_person, num_item):
        eps = 1e-6

        ztilde_flat, raw_v_flat, raw_c_flat = torch.chunk(self.mlp(mlp_input), 3, dim=1)
        v_flat = F.softplus(raw_v_flat) + eps
        c_flat = F.softplus(raw_c_flat) + eps

        ztilde_set = ztilde_flat.view(num_person, num_item, self.ability_dim)
        v_set = v_flat.view(num_person, num_item, self.ability_dim)
        c_set = c_flat.view(num_person, num_item, self.ability_dim)

        m, kappa, alpha, beta = aggregate_pseudo_obs_to_nig(
            ztilde_set=ztilde_set,
            v_set=v_set,
            c_set=c_set,
            mask=mask,
            prior_params=(self.m0, self.kappa0, self.alpha0, self.beta0),
            replace_missing_with_prior=self.replace_missing_with_prior,
            eps=eps,
        )

        return m, kappa, alpha, beta

    def forward(self, response, mask):
        num_person, num_item, response_dim = response.size()
        mlp_input = response.view(num_person * num_item, response_dim)

        return getattr(self, f'_forward_{self.ability_merge}')(
            mlp_input,
            mask,
            num_person,
            num_item,
        )


class ItemInferenceNetwork(nn.Module):

    def __init__(self, num_item, item_feat_dim):
        super().__init__()

        self.mu_lookup = nn.Embedding(num_item, item_feat_dim)
        self.logvar_lookup = nn.Embedding(num_item, item_feat_dim)

    def forward(self, item_index):
        item_index = item_index.squeeze(1)
        mu = self.mu_lookup(item_index.long())
        logvar = self.logvar_lookup(item_index.long())

        return mu, logvar


def irt_model_1pl_q(ability, item_feat, return_logit=False):
    difficulty = item_feat
    logit = ability + difficulty

    if return_logit:
        return logit
    else:
        response_mu = torch.sigmoid(logit)
        return response_mu


def irt_model_1pl(ability, item_feat, return_logit=False):
    difficulty = item_feat
    logit = (torch.sum(ability, dim=-1, keepdim=True) + difficulty.T).unsqueeze(2)

    if return_logit:
        return logit
    else:
        response_mu = torch.sigmoid(logit)
        return response_mu


def irt_model_2pl(ability, item_feat, Q_mat=None, return_logit=False):
    ability_dim = ability.size(1)
    discrimination = item_feat[:, :ability_dim]
    difficulty = item_feat[:, ability_dim:]
    if Q_mat is not None:
        logit = (ability @ discrimination.T / Q_mat.sum(dim=1) + difficulty.T).unsqueeze(2)
    else:
        logit = (ability @ discrimination.T + difficulty.T).unsqueeze(2)

    if return_logit:
        return logit
    else:
        response_mu = torch.sigmoid(logit)
        return response_mu


def irt_model_3pl(ability, item_feat, return_logit=False):
    ability_dim = ability.size(1)
    discrimination = item_feat[:, :ability_dim]
    difficulty = item_feat[:, ability_dim:ability_dim + 1]
    guess_logit = item_feat[:, ability_dim + 1:ability_dim + 2]
    guess = torch.sigmoid(guess_logit)
    logit = (torch.mm(ability, -discrimination.T) + difficulty.T).unsqueeze(2)

    if return_logit:
        return logit, guess
    else:
        guess = guess.unsqueeze(0)
        response_mu = guess + (1. - guess) * torch.sigmoid(logit)
        return response_mu


class LinkedIRT(nn.Module):

    def __init__(self, irt_model='1pl', hidden_dim=64):
        super().__init__()
        assert irt_model in ['1pl', '2pl', '3pl']
        self.irt_model = irt_model
        self.hidden_dim = hidden_dim
        self.link = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.apply(self.weights_init)

    def forward(self, ability, item_feat):
        if self.irt_model == '1pl':
            logit = irt_model_1pl(ability, item_feat, return_logit=True)
            response_mu = self.link(logit)

        elif self.irt_model == '2pl':
            logit = irt_model_2pl(ability, item_feat, return_logit=True)
            response_mu = self.link(logit)

        elif self.irt_model == '3pl':
            logit, guess = irt_model_3pl(ability, item_feat, return_logit=True)
            response_mu = guess + (1. - guess) * self.link(logit)

        else:
            raise Exception(f'Unsupported irt_model {self.irt_model}.')

        return response_mu

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class DeepIRT(nn.Module):

    def __init__(self, latent_dim, irt_model='1pl', hidden_dim=64):
        super().__init__()
        assert irt_model in ['1pl', '2pl', '3pl']
        self.latent_dim = latent_dim
        self.ability_dim = latent_dim
        self.irt_model = irt_model
        self.hidden_dim = hidden_dim

        if self.irt_model == '1pl':
            self.item_feat_dim = 1
        elif self.irt_model == '2pl':
            self.item_feat_dim = self.latent_dim + 1
        elif self.irt_model == '3pl':
            self.item_feat_dim = self.latent_dim + 2

        self.mlp_item_feat = nn.Sequential(
            nn.Linear(self.item_feat_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.mlp_ability = nn.Sequential(
            nn.Linear(self.ability_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.mlp_concat = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.apply(self.weights_init)

    def forward(self, ability, item_feat):
        num_person, num_item = ability.size(0), item_feat.size(0)

        hid_ability = self.mlp_ability(ability)
        hid_item_feat = self.mlp_item_feat(item_feat)

        hid_ability = hid_ability.unsqueeze(1).repeat(1, num_item, 1)
        hid_item_feat = hid_item_feat.unsqueeze(0).repeat(num_person, 1, 1)

        hid = torch.cat([hid_item_feat, hid_ability], dim=2)
        hid = self.mlp_concat(hid)
        response_mu = torch.sigmoid(hid)
        return response_mu

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class ResidualIRT(DeepIRT):

    def __init__(self, latent_dim, irt_model='1pl', hidden_dim=64):
        super().__init__(
            latent_dim,
            irt_model=irt_model,
            hidden_dim=hidden_dim,
        )
        self.apply(self.zero_init)

    def residual_forward(self, ability, item_feat):
        num_person, num_item = ability.size(0), item_feat.size(0)
        hid_ability = self.mlp_ability(ability)
        hid_item_feat = self.mlp_item_feat(item_feat)
        hid_ability = hid_ability.unsqueeze(1).repeat(1, num_item, 1)
        hid_item_feat = hid_item_feat.unsqueeze(0).repeat(num_person, 1, 1)
        hid = torch.cat([hid_item_feat, hid_ability], dim=2)
        return self.mlp_concat(hid)

    def forward(self, ability, item_feat):
        res_logit = self.residual_forward(ability, item_feat)

        if self.irt_model == '1pl':
            irt_logit = irt_model_1pl(ability, item_feat, return_logit=True)
            return torch.sigmoid(res_logit + irt_logit)

        elif self.irt_model == '2pl':
            irt_logit = irt_model_2pl(ability, item_feat, return_logit=True)
            return torch.sigmoid(res_logit + irt_logit)

        elif self.irt_model == '3pl':
            irt_logit, guess = irt_model_3pl(ability, item_feat, return_logit=True)
            return guess + (1. - guess) * torch.sigmoid(res_logit + irt_logit)

        else:
            raise Exception(f'Unsupported irt_model {self.irt_model}.')

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
