# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Restricted Boltzmann Machine"""
import numpy as np
import torch
import torch.nn.functional as F
from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class RestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Create a Restricted Boltzmann Machine.

    Args:
        num_visible (int): Number of visible nodes in the model.

        num_hidden (int): Number of hidden nodes in the model.

        quadratic_coef (torch.FloatTensor, optional): quadratic coefficent,
            shape is [num_visible, num_hidden]

        linear_bias (torch.FloatTensor, optional): linear bias, shape is [num_hidden]

        device (torch.device, optional): Device to construct tensors.
    """

    VISIBLE_GIBBS_REQUIRED_PURPOSES = ("general", "cd")
    VISIBLE_CUSTOMIZED_REQUIRED_PURPOSES = ("general",)

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        quadratic_coef: torch.FloatTensor = None,
        linear_bias: torch.FloatTensor = None,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__(
            num_visible=num_visible,
            num_hidden=num_hidden,
            quadratic_coef=quadratic_coef,
            linear_bias=linear_bias,
            dtype=dtype,
            device=device,
        )

    def energy(self, s_all: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """Compute the Hamiltonian.

        The energy function for RBM is defined as:
        E(v,h) = -sum_i,j J_{ij} v_i h_j - sum_k H_k s_k

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is the batch
                size, and N is the number of variables in the model.
            enable_grad (bool): Whether to enable gradient computation.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        context = torch.enable_grad if enable_grad else torch.no_grad
        with context():
            s_visible = s_all[:, : self.num_visible]
            s_hidden = s_all[:, self.num_visible :]
            return (
                -torch.sum(s_visible @ self.quadratic_coef * s_hidden, dim=-1)
                - s_all @ self.linear_bias
            )

    def marginal_energy_from_visible(self, s_visible: torch.Tensor) -> torch.Tensor:
        """Compute the Free Hamiltonian of given visible nodes.

        Args:
            s_visible (torch.tensor): State of visible nodes.

        Returns:
            torch.tensor: Free Hamiltonian of given visible nodes.

        Raises:
            ValueError: If s_visible is not a 2-D tensor or has inconsistent node numbers.
        """
        self._validate_visible_batch_state(s_visible)
        with torch.no_grad():
            return (
                -torch.sum(
                    F.softplus(s_visible @ self.quadratic_coef + self.hidden_bias),
                    dim=-1,
                )
                - s_visible @ self.visible_bias
            )

    def infer_from_visible(
        self,
        s_visible: torch.Tensor,
        requires_grad: bool = False,
        binarize: bool = True,
        no_random: bool = False,
    ) -> torch.Tensor:
        """Propagate visible spins to the hidden layer.

        Args:
            s_visible: Visible layer tensor.
            requires_grad: Whether to allow gradient backpropagation.
            binarize: Whether to binarize the hidden layer.
            no_random: Whether to disable random sampling.
        """
        self._validate_visible_batch_state(s_visible)
        context = torch.enable_grad if requires_grad else torch.no_grad
        with context():
            n_sample = s_visible.shape[0]
            s_all = torch.zeros(
                n_sample, self.num_nodes, device=self.device, dtype=self.dtype
            )
            s_all[:, : self.num_visible] = s_visible
            prob = torch.sigmoid(s_visible @ self.quadratic_coef + self.hidden_bias)
            if binarize:
                if no_random:
                    s_all[:, self.num_visible :] = (prob >= 0.5).to(self.dtype)
                else:
                    s_all[:, self.num_visible :] = (prob > torch.rand_like(prob)).to(
                        self.dtype
                    )
            else:
                s_all[:, self.num_visible :] = prob
            return s_all

    def get_hidden(
        self,
        s_visible: torch.Tensor,
        requires_grad: bool = False,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Backward-compatible alias for visible-to-hidden inference."""
        return self.infer_from_visible(
            s_visible,
            requires_grad=requires_grad,
            binarize=bernoulli,
        )

    def infer_from_hidden(
        self,
        s_hidden: torch.Tensor,
        binarize: bool = True,
        no_random: bool = False
    ) -> torch.Tensor:
        """Propagate hidden spins to the visible layer.

        Args:
            s_hidden: Hidden layer tensor.
            binarize: Whether to binarize the visible layer.
            no_random: Whether to disable random sampling.
        Returns:
            s_all: Tensor containing both visible and hidden layers.
        """
        with torch.no_grad():
            s_all = torch.zeros(
                s_hidden.shape[0], self.num_nodes, device=self.device, dtype=self.dtype
            )
            s_all[:, self.num_visible :] = s_hidden
            prob = torch.sigmoid(s_hidden @ self.quadratic_coef.t() + self.visible_bias)
            if binarize:
                if no_random:
                    s_all[:, : self.num_visible] = (prob >= 0.5).to(self.dtype)
                else:
                    s_all[:, : self.num_visible] = (prob > torch.rand_like(prob)).to(
                        self.dtype
                    )
            else:
                s_all[:, : self.num_visible] = prob
            return s_all

    def get_visible(
        self,
        s_hidden: torch.Tensor,
        bernoulli: bool = False,
    ) -> torch.Tensor:
        """Backward-compatible alias for hidden-to-visible inference."""
        return self.infer_from_hidden(s_hidden, binarize=bernoulli)

    def _to_ising_matrix(self):
        """Convert the Restricted Boltzmann Machine to Ising format."""
        num_nodes = self.linear_bias.shape[-1]
        with torch.no_grad():
            ising_mat = torch.zeros((num_nodes + 1, num_nodes + 1), device=self.device)
            # Restricted Boltzmann Machine: only connections between visible and hidden layers
            ising_mat[: self.num_visible, self.num_visible : -1] = (
                self.quadratic_coef / 8
            )
            ising_mat[self.num_visible : -1, : self.num_visible] = (
                self.quadratic_coef.t() / 8
            )
            ising_bias = self.linear_bias / 4 + ising_mat.sum(dim=0)[:-1]
            ising_mat[:num_nodes, -1] = ising_bias
            ising_mat[-1, :num_nodes] = ising_bias
            return ising_mat.detach().cpu().numpy()


    def gibbs_sample(
        self,
        n_step: int,
        n_burnin: int,
        sampler=None,
        n_sample: int | None = None,
    ) -> torch.Tensor:
        """Perform Gibbs sampling for the RBM.

        Args:
            n_step (int): Number of Gibbs sampling steps to perform.
            n_burnin (int): Number of initial steps to discard.
            sampler (optional): Sampler for initialization if provided.
            n_sample (int, optional): Number of samples to generate.

        Returns:
            torch.tensor: Samples generated through Gibbs sampling.

        Raises:
            ValueError: If neither sampler nor n_sample is provided.
        """
        if np.sum([sampler is not None, n_sample is not None]) != 1:
            raise ValueError("One and only one of sampler and n_sample is required.")
        with torch.no_grad():
            s_all = self._initialize_binary_sampling_state(sampler, n_sample)
            samples = []
            for no_step in range(n_step):
                s_visible = s_all[:, : self.num_visible]
                s_all = self.infer_from_visible(s_visible)
                s_hidden = s_all[:, self.num_visible :]
                s_all = self.infer_from_hidden(s_hidden)
                if no_step >= n_burnin - 1:
                    samples.append(s_all)
            return torch.concat(samples, dim=0)

    def sample(
        self,
        sampler=None,
        *,
        sampling_mode: str | None = None,
        sampling_purpose: str = "general",
        n_gibbs_step: int | None = None,
        n_gibbs_burnin: int | None = None,
        s_visible: torch.Tensor | None = None,
        n_sample: int | None = None,
    ) -> torch.Tensor:
        """Sample from the RBM using external, Gibbs, or conditional modes."""
        return self._run_visible_sampling(
            sampler=sampler,
            sampling_mode=sampling_mode,
            sampling_purpose=sampling_purpose,
            n_gibbs_step=n_gibbs_step,
            n_gibbs_burnin=n_gibbs_burnin,
            s_visible=s_visible,
            n_sample=n_sample,
            customized_alias="sampler",
        )

    def conditional_gibbs_sample(
        self,
        s_visible: torch.Tensor,
        no_random: bool = False,
    ) -> torch.Tensor:
        """Sample according to given visible nodes.

        Args:
            s_visible (torch.tensor): State of visible nodes.
            no_random (bool, optional): Whether to use deterministic sampling.
                Defaults to False.

        Returns:
            torch.tensor: Samples with visible nodes fixed and hidden nodes sampled.

        Raises:
            ValueError: If s_visible is not a 2-D tensor or has inconsistent node numbers.
        """
        self._validate_visible_batch_state(s_visible)
        with torch.no_grad():
            return self.infer_from_visible(s_visible, no_random=no_random)

    def cd_gibbs_sample(
        self,
        s_visible: torch.Tensor,
        n_step: int,
        n_burnin: int,
    ) -> torch.Tensor:
        """Sample from the Restricted Boltzmann Machine for contrastive divergence.

        Args:
            s_visible (torch.tensor): State of visible nodes.
            n_step (int): Number of Gibbs sampling steps.
            n_burnin (int): Number of initial steps to discard.

        Returns:
            torch.tensor: Samples generated through CD Gibbs sampling.

        Raises:
            ValueError: If s_visible is not a 2-D tensor or has inconsistent node numbers.
        """
        self._validate_visible_batch_state(s_visible)
        with torch.no_grad():
            s_all = torch.zeros(
                s_visible.shape[0], self.num_nodes, device=self.device, dtype=self.dtype
            )
            s_all[:, : self.num_visible] = s_visible
            samples = []
            for no_step in range(n_step):
                current_visible = s_all[:, : self.num_visible]
                s_all = self.infer_from_visible(current_visible)
                s_hidden = s_all[:, self.num_visible :]
                s_all = self.infer_from_hidden(s_hidden)
                if no_step >= n_burnin - 1:
                    samples.append(s_all)
            return torch.concat(samples, dim=0)


    def _sample_conditional_gibbs(
        self,
        *,
        s_visible: torch.Tensor,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        sampler,
    ) -> torch.Tensor:
        """Run conditional Gibbs sampling with visible nodes fixed."""
        del n_gibbs_step, n_gibbs_burnin, sampler
        return self.conditional_gibbs_sample(s_visible=s_visible)

    def _sample_cd_gibbs(
        self,
        *,
        s_visible: torch.Tensor,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
    ) -> torch.Tensor:
        """Run contrastive-divergence Gibbs sampling."""
        return self.cd_gibbs_sample(
            s_visible=s_visible,
            n_step=n_gibbs_step,
            n_burnin=n_gibbs_burnin,
        )

    def _sample_conditional_customized(
        self, *, sampler, s_visible: torch.Tensor
    ) -> torch.Tensor:
        """Run deterministic conditional inference for customized mode."""
        del sampler
        return self.conditional_gibbs_sample(s_visible=s_visible, no_random=True)

    def positive_phase_energy_expectation(
        self, s_visible: torch.Tensor, enable_grad: bool = True
    ) -> torch.Tensor:
        """Compute the positive-phase energy expectation from visible nodes.

        Args:
            s_visible (torch.Tensor): State of visible nodes.
            enable_grad (bool, optional): Whether to enable gradient computation.
                Defaults to True.

        Returns:
            torch.Tensor: Expected energy for the given visible nodes.

        Raises:
            ValueError: If s_visible is not a 2-D tensor or has inconsistent node numbers.
        """
        if len(s_visible.shape) != 2:
            raise ValueError("s_visible should be a 2-D tensor.")
        if s_visible.shape[1] != self.num_visible:
            raise ValueError("Inconsistent number of visible nodes.")
        context = torch.enable_grad if enable_grad else torch.no_grad
        with context():
            prob_hidden = self.infer_from_visible(
                s_visible,
                requires_grad=enable_grad,
                binarize=False,
            )[:, self.num_visible :]
            return (
                -torch.sum(s_visible @ self.quadratic_coef * prob_hidden, dim=-1)
                - s_visible @ self.visible_bias
                - prob_hidden @ self.hidden_bias
            )

    def sampling_energy(
        self,
        *,
        sampling_mode: str = "gibbs",
        sampling_purpose: str = "general",
        n_gibbs_step: int | None = None,
        n_gibbs_burnin: int | None = None,
        sampler=None,
        s_visible: torch.Tensor | None = None,
        n_sample: int | None = None,
        enable_grad: bool = False,
    ):
        """Compute the energy of samples generated by a sampling strategy.

        Args:
            sampling_mode (str, optional): Sampling mode (``"gibbs"`` or
                ``"customized"``). Defaults to ``"gibbs"``. Legacy
                ``"sampler"`` is still accepted as an alias.
            sampling_purpose (str, optional): Purpose of sampling
                (``"general"``, ``"conditional"`` or ``"cd"``).
                Defaults to ``"general"``.
            n_gibbs_step (int, optional): Number of Gibbs sampling steps.
            n_gibbs_burnin (int, optional): Number of burn-in steps.
            sampler (optional): External sampler used when
                ``sampling_mode="customized"``.
            s_visible (torch.Tensor, optional): Visible nodes used for
                conditional or CD sampling.
            n_sample (int, optional): Number of samples to generate for
                unconditional Gibbs sampling.
            enable_grad (bool, optional): Whether to enable gradient computation.
                Defaults to False.

        Returns:
            torch.Tensor: Energy of the sampled states.
        """
        sampling_kwargs = {
            "sampling_mode": sampling_mode,
            "sampling_purpose": sampling_purpose,
        }
        sampling_kwargs["sampler"] = sampler
        sampling_kwargs["n_gibbs_step"] = n_gibbs_step
        sampling_kwargs["n_gibbs_burnin"] = n_gibbs_burnin
        sampling_kwargs["s_visible"] = s_visible
        sampling_kwargs["n_sample"] = n_sample
        return self._sampling_energy_from_sample(
            enable_grad=enable_grad, **sampling_kwargs
        )
