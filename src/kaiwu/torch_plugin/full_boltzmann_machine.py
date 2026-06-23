# -*- coding: utf-8 -*-
"""Boltzmann Machine"""

import numpy as np
import torch

from .abstract_boltzmann_machine import AbstractBoltzmannMachine


class BoltzmannMachine(AbstractBoltzmannMachine):
    """Boltzmann Machine.

    Args:
        num_visible (int): Number of visible nodes in the model.

        num_hidden (int): Number of hidden nodes in the model.

        quadratic_coef (torch.FloatTensor, optional): quadratic coefficent,
            shape is [num_visible + num_hidden, num_visible + num_hidden]

        linear_bias (torch.FloatTensor, optional): linear bias, shape is
            [num_visible + num_hidden]

        device (torch.device, optional): Device for tensor construction.
        If ``None``, uses CPU.
    """

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
            quadratic_shape=(num_visible + num_hidden, num_visible + num_hidden),
        )


    def symmetrized_quadratic_coef(self):
        """Quadratic coefficient"""
        quadratic_coef = self.quadratic_coef.triu(1)
        return quadratic_coef + quadratic_coef.transpose(0, 1)

    def energy(self, s_all: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.tensor): Tensor of shape (B, N), where B is batch size,
                N is the number of variables in the model.
            enable_grad (bool): Whether to enable gradient computation.

        Returns:
            torch.tensor: Hamiltonian of shape (B,).
        """
        context = torch.enable_grad if enable_grad else torch.no_grad
        with context():
            return -s_all @ self.linear_bias - 0.5 * torch.sum(
                s_all.matmul(self.symmetrized_quadratic_coef()) * s_all, dim=-1
            )

    def _to_ising_matrix(self):
        """Convert Boltzmann Machine to Ising matrix."""
        with torch.no_grad():
            linear_bias = self.linear_bias
            quadratic_coef = self.symmetrized_quadratic_coef()  # quadratic_coef
            column_sums = torch.sum(quadratic_coef, dim=0)
            num_nodes = self.num_nodes

            ising_mat = torch.zeros(
                (num_nodes + 1, num_nodes + 1),
                device=self.device,
                dtype=linear_bias.dtype,
            )
            # Fill quadratic part
            ising_mat[:-1, :-1] = quadratic_coef / 8
            # Calculate ising_bias
            ising_bias = linear_bias / 4 + column_sums / 8
            # Fill bias part
            ising_mat[:num_nodes, -1] = ising_bias
            ising_mat[-1, :num_nodes] = ising_bias
            return ising_mat.cpu().numpy()

    def _hidden_to_ising_matrix(self, s_visible: torch.Tensor) -> np.ndarray:
        """Given visible nodes, convert the model to a submatrix in Ising format.

        Args:
            s_visible (torch.Tensor): State of the visible layer, shape (B, num_visible).

        Returns:
            np.ndarray: Submatrix in Ising format.
        """
        if len(s_visible.shape) != 1:
            raise ValueError("s_visible should be a 1-D tensor.")
        if s_visible.shape[0] != self.num_visible:
            raise ValueError("Inconsistent number of visible nodes.")

        with torch.no_grad():
            linear_bias = self.linear_bias
            quadratic_coef = self.symmetrized_quadratic_coef()
            n_vis = self.num_visible
            n_hid = self.num_hidden
            sub_quadratic = quadratic_coef[n_vis:, n_vis:]
            sub_column_sums = torch.sum(sub_quadratic, dim=0)
            sub_quadratic_vh = quadratic_coef[n_vis:, :n_vis]
            sub_linear = sub_quadratic_vh @ s_visible + linear_bias[n_vis:]

            ising_mat = torch.zeros(
                (n_hid + 1, n_hid + 1),
                device=self.device,
                dtype=sub_linear.dtype,
            )
            ising_mat[:-1, :-1] = sub_quadratic / 8
            ising_bias = sub_linear / 4 + sub_column_sums / 8
            ising_mat[:-1, -1] = ising_bias
            ising_mat[-1, :-1] = ising_bias
            return ising_mat.cpu().numpy()

    def gibbs_sample(
        self, num_steps: int = 100, s_visible: torch.Tensor = None, num_sample=None
    ) -> torch.Tensor:
        """Sample from the Boltzmann Machine.

        Args:
            num_steps (int): Number of Gibbs sampling steps.

            s_visible (torch.Tensor, optional): State of the visible layer,
                shape (B, num_visible). If ``None``, randomly initialize visible layer.

            num_sample (int, optional): Number of samples.
                If ``None``, uses batch size of s_visible.
        """
        with torch.no_grad():
            # Initialization: If neither visible unit state nor sample number is provided,
            # raise error
            if s_visible is None and num_sample is None:
                raise ValueError("Either s_visible or num_sample must be provided.")
            if s_visible is not None:
                self._validate_visible_batch_state(s_visible)
                # Initialize all units (visible + hidden) with Bernoulli(0.5)
                s_all = torch.bernoulli(
                    torch.full(
                        (s_visible.size(0), self.num_nodes),
                        0.5,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                # Replace visible part with given visible unit state
                s_all[:, : self.num_visible] = s_visible.clone()
                n_vis = self.num_visible
            else:
                # If no visible units, initialize all randomly
                s_all = torch.bernoulli(
                    torch.full(
                        (num_sample, self.num_nodes),
                        0.5,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                n_vis = 0

            q_coef = self.symmetrized_quadratic_coef()
            for _ in range(num_steps):
                # Random update order (Gibbs sampling)
                update_order = torch.randperm(self.num_nodes, device=self.device)
                for unit in update_order:
                    if unit < n_vis:
                        # Skip visible units (only sample hidden units)
                        continue
                    # Compute activation value (logit of conditional probability)
                    activation = (
                        torch.matmul(s_all, q_coef[:, unit]) + self.linear_bias[unit]
                    )
                    # Get activation probability via sigmoid
                    prob = torch.sigmoid(activation)
                    # Sample current unit state according to probability
                    s_all[:, unit] = (prob > torch.rand_like(prob)).float()
            # Return sampled states of all units
            return s_all

    def conditional_sample(self, sampler, s_visible, dtype=torch.float32) -> torch.Tensor:
        """Sample from the Boltzmann Machine given some nodes.

        Args:
            sampler (kaiwu.core.Optimizer): Optimizer used for sampling from the model.
            s_visible: State of the visible layer.

        Returns:
            torch.Tensor: Spins sampled from the model
                (shape determined by ``sampler`` and ``sample_params``).
        """
        self._validate_visible_batch_state(s_visible)
        solutions = []
        for i in range(s_visible.size(0)):
            ising_mat = self._hidden_to_ising_matrix(s_visible[i])
            solution = sampler.solve(ising_mat)
            solution = (solution[:, :-1] * solution[:, [-1]] + 1) / 2
            solution = torch.tensor(solution, dtype=dtype, device=self.device)
            solution = torch.cat(
                [s_visible[i].unsqueeze(0).expand(solution.shape[0], -1), solution],
                dim=-1,
            )
            solutions.append(solution)
        solutions = torch.cat(solutions, dim=0)
        return solutions

    def condition_sample(self, sampler, s_visible, dtype=torch.float32) -> torch.Tensor:
        """Backward-compatible alias for conditional sampling."""
        return self.conditional_sample(sampler, s_visible, dtype=dtype)

    def conditional_gibbs_sample(
        self,
        s_visible: torch.Tensor,
        n_step: int,
        n_burnin: int,
        sampler=None,
    ) -> torch.Tensor:
        """Sample from the Boltzmann Machine with visible nodes fixed.

        Args:
            s_visible (torch.Tensor): State of the visible layer,
                shape (B, num_visible).
            n_step (int): Number of Gibbs sampling steps.
            n_burnin (int): Number of burn-in steps.
            sampler (optional): Sampler for initialization.

        Returns:
            torch.tensor: Samples with visible nodes fixed and hidden nodes sampled.

        Raises:
            ValueError: If s_visible is not a 2-D tensor or has inconsistent node numbers.
        """
        self._validate_visible_batch_state(s_visible)
        with torch.no_grad():
            if sampler is not None:
                s_all = self.conditional_sample(sampler, s_visible)
            else:
                s_all = torch.bernoulli(
                    torch.full(
                        (s_visible.shape[0], self.num_nodes),
                        0.5,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                s_all[:, : self.num_visible] = s_visible
            samples = []
            for no_step in range(n_step):
                update_order = (
                    torch.randperm(self.num_hidden, device=self.device)
                    + self.num_visible
                )
                quadratic_coef = self.symmetrized_quadratic_coef()
                for unit in update_order:
                    activation = (
                        s_all @ quadratic_coef[:, unit] + self.linear_bias[unit]
                    )
                    prob = torch.sigmoid(activation)
                    s_all[:, unit] = (prob > torch.rand_like(prob)).to(self.dtype)
                if no_step >= n_burnin - 1:
                    samples.append(s_all.clone())
            return torch.concat(samples, dim=0)

    def cd_gibbs_sample(
        self,
        s_visible: torch.Tensor,
        n_step: int,
        n_burnin: int,
    ) -> torch.Tensor:
        """Sample from the Boltzmann Machine for contrastive divergence.

        Args:
            s_visible (torch.Tensor): State of visible nodes.
            n_step (int): Number of Gibbs sampling steps.
            n_burnin (int): Number of initial steps to discard.

        Returns:
            torch.Tensor: Samples generated through CD Gibbs sampling.

        Raises:
            ValueError: If s_visible is not a 2-D tensor or has inconsistent node numbers.
        """
        self._validate_visible_batch_state(s_visible)
        with torch.no_grad():
            s_all = torch.bernoulli(
                torch.full(
                    (s_visible.shape[0], self.num_nodes),
                    0.5,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            s_all[:, : self.num_visible] = s_visible
            samples = []
            for no_step in range(n_step):
                hidden_order = (
                    torch.randperm(self.num_hidden, device=self.device)
                    + self.num_visible
                )
                visible_order = torch.randperm(self.num_visible, device=self.device)
                update_order = torch.cat((hidden_order, visible_order))
                quadratic_coef = self.symmetrized_quadratic_coef()
                for unit in update_order:
                    activation = (
                        s_all @ quadratic_coef[:, unit] + self.linear_bias[unit]
                    )
                    prob = torch.sigmoid(activation)
                    s_all[:, unit] = (prob > torch.rand_like(prob)).to(self.dtype)
                if no_step >= n_burnin - 1:
                    samples.append(s_all.clone())
            return torch.concat(samples, dim=0)
