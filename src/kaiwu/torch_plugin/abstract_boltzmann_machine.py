# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0


"""Abstract base class for Boltzmann Machines."""
import torch


class AbstractBoltzmannMachine(torch.nn.Module):
    """Abstract base class for Boltzmann Machines.

    This class provides common functionality and interface definitions for
    concrete Boltzmann machine implementations. It handles device management
    and dtype specification.

    Args:
        dtype (torch.dtype, optional): Data type for tensor construction.
            Defaults to torch.float32.
        device (torch.device, optional): Device for tensor construction.
            If ``None``, uses CUDA when available, otherwise CPU.

    """

    VISIBLE_GIBBS_REQUIRED_PURPOSES: tuple[str, ...] = ()
    VISIBLE_CUSTOMIZED_REQUIRED_PURPOSES: tuple[str, ...] = ()


    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        quadratic_coef: torch.Tensor = None,
        linear_bias: torch.Tensor = None,
        dtype: torch.dtype = torch.float32,
        device=None,
        quadratic_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.dtype = dtype
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._initialize_visible_hidden_layout(num_visible, num_hidden)
        if quadratic_shape is None:
            quadratic_shape = (num_visible, num_hidden)

        self.quadratic_coef = torch.nn.Parameter(
            self._prepare_quadratic_coef_parameter(
                quadratic_coef, expected_shape=quadratic_shape
            )
        )
        self.linear_bias = torch.nn.Parameter(
            self._prepare_linear_bias_parameter(linear_bias)
        )

    def clip_parameters(self, linear_range, quadratic_range) -> None:
        """Clip linear and quadratic bias weights in-place.

        Args:
            linear_range (tuple[float, float]): Range for linear weights. for example, [-1, 1]
            quadratic_range (tuple[float, float]): Range for quadratic weights. for example, [-1, 1]
        """
        self.get_parameter("linear_bias").data.clamp_(*linear_range)
        self.get_parameter("quadratic_coef").data.clamp_(*quadratic_range)

    @property
    def hidden_bias(self) -> torch.Tensor:
        """Return the hidden bias."""
        return self.linear_bias[self.num_visible :]

    @property
    def visible_bias(self) -> torch.Tensor:
        """Return the visible bias."""
        return self.linear_bias[: self.num_visible]


    def to(self, device=..., dtype=..., non_blocking=...):
        """Moves the model to the specified device.

        Args:
            device: Target device.
            dtype: Target data type.
            non_blocking: Whether the operation should be non-blocking.

        Returns:
            AbstractBoltzmannMachine: The model on the target device.
        """
        self.device = device
        return super().to(device)

    def forward(self, s_all: torch.Tensor) -> torch.Tensor:
        """Computes the Hamiltonian.

        This method should compute the Hamiltonian of the system given the
        state tensor.

        Args:
            s_all (torch.Tensor): Input tensor representing the state of all nodes.

        Returns:
            torch.Tensor: Computed Hamiltonian values.
        """
        return self.energy(s_all, enable_grad=True)

    def energy(self, s_all: torch.Tensor, enable_grad: bool = False) -> torch.Tensor:
        """Compute the Hamiltonian.

        Args:
            s_all (torch.Tensor): Tensor of shape (B, N), where B is batch size,
                N is the number of variables in the model.
            enable_grad (bool): Whether to enable gradient computation.

        Returns:
            torch.Tensor: Computed Hamiltonian values.
        """
        raise NotImplementedError("Subclasses must implement energy method")


    def get_ising_matrix(self):
        """Converts the model to Ising format.

        Returns:
            torch.Tensor: Ising matrix.
        """
        return self._to_ising_matrix()

    def _to_ising_matrix(self):
        """Converts the model to Ising format.

        Returns:
            torch.Tensor: Ising matrix.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError("Subclasses must implement _ising method")

    def objective(
        self,
        s_positive: torch.Tensor,
        s_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Objective function whose gradient is equivalent to the gradient of
        negative log-likelihood.

        Args:
            s_positive (torch.Tensor): Tensor of observed spins (data), shape (b1, N),
                            where b1 is batch size and N is the number of variables.
            s_negative (torch.Tensor): Tensor of spins sampled from the model, shape (b2, N),
                            where b2 is batch size and N is the number of variables.

        Returns:
            torch.Tensor: Scalar difference between data and model average energy.
        """
        return self(s_positive).mean() - self(s_negative).mean()

    def sample(self, sampler) -> torch.Tensor:
        """Samples from the Boltzmann Machine.

        Args:
            sampler (kaiwu.core.OptimizerBase): Optimizer used for sampling from the model.
                The sampler can be kaiwuSDK's CIM or other solvers.

        Returns:
            torch.Tensor: Spins sampled from the model.
        """
        ising_mat = self.get_ising_matrix()
        solution = sampler.solve(ising_mat)
        solution = (solution[:, :-1] * solution[:, [-1]] + 1) / 2
        solution = torch.FloatTensor(solution)
        solution = solution.to(self.device)
        return solution
    # ==========================================helper methods=================================================
    # region helper methods

    @staticmethod
    def _validate_2d_batch_state(
        state: torch.Tensor,
        state_name: str,
        expected_width: int,
        inconsistent_message: str,
    ) -> None:
        """Validate a batched 2-D state tensor."""
        if len(state.shape) != 2:
            raise ValueError(f"{state_name} should be a 2-D tensor.")
        if state.shape[1] != expected_width:
            raise ValueError(inconsistent_message)

    def _validate_visible_batch_state(self, s_visible: torch.Tensor) -> None:
        """Validate a batched visible-layer state tensor."""
        self._validate_2d_batch_state(
            s_visible,
            state_name="s_visible",
            expected_width=self.num_visible,
            inconsistent_message="Inconsistent number of visible nodes.",
        )

    def _initialize_visible_hidden_layout(
        self, num_visible: int, num_hidden: int
    ) -> None:
        """Initialize the common visible/hidden layout metadata."""
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_nodes = self.num_visible + self.num_hidden

    def _prepare_optional_parameter(
        self,
        parameter: torch.Tensor | None,
        *,
        expected_shape: tuple[int, ...],
        default_factory,
        parameter_name: str,
    ) -> torch.Tensor:
        """Prepare an optional tensor parameter for registration."""
        if parameter is None:
            return default_factory()
        if parameter.shape != expected_shape:
            raise ValueError(
                f"{parameter_name} should have shape {expected_shape}."
            )
        return parameter.detach().clone().to(device=self.device, dtype=self.dtype)

    def _prepare_linear_bias_parameter(
        self, linear_bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Prepare the standard linear-bias parameter."""
        return self._prepare_optional_parameter(
            linear_bias,
            expected_shape=(self.num_nodes,),
            default_factory=lambda: torch.zeros(
                self.num_nodes, dtype=self.dtype, device=self.device
            ),
            parameter_name="linear_bias",
        )

    def _prepare_quadratic_coef_parameter(
        self,
        quadratic_coef: torch.Tensor | None,
        *,
        expected_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Prepare the standard quadratic-coefficient parameter."""
        return self._prepare_optional_parameter(
            quadratic_coef,
            expected_shape=expected_shape,
            default_factory=lambda: (
                torch.randn(*expected_shape, dtype=self.dtype, device=self.device)
                * 0.01
            ),
            parameter_name="quadratic_coef",
        )

    def _initialize_binary_sampling_state(
        self,
        sampler=None,
        n_sample: int | None = None,
    ) -> torch.Tensor:
        """Initialize a binary state either from a sampler or Bernoulli noise."""
        if sampler is not None:
            return AbstractBoltzmannMachine.sample(self, sampler)
        return torch.bernoulli(
            torch.full(
                (n_sample, self.num_nodes),
                0.5,
                device=self.device,
                dtype=self.dtype,
            )
        )

    @staticmethod
    def _normalize_sampling_mode(
        sampling_mode: str | None,
        *,
        customized_alias: str | None = None,
    ) -> str:
        """Normalize a public sampling mode value."""
        if sampling_mode is None:
            return "gibbs"
        if customized_alias is not None and sampling_mode == customized_alias:
            return "customized"
        return sampling_mode

    @staticmethod
    def _validate_sampling_mode_and_purpose(
        sampling_mode: str,
        sampling_purpose: str,
        *,
        valid_modes: tuple[str, ...],
        valid_purposes: tuple[str, ...] = ("general", "conditional", "cd"),
    ) -> None:
        """Validate shared sampling mode/purpose enums."""
        if sampling_mode not in valid_modes:
            supported_modes = "', '".join(valid_modes)
            raise ValueError(
                f"Supported sampling modes include '{supported_modes}'."
            )
        if sampling_purpose not in valid_purposes:
            supported_purposes = "', '".join(valid_purposes)
            raise ValueError(
                f"Supported sampling purposes include '{supported_purposes}'."
            )

    @staticmethod
    def _validate_customized_cd_conflict(
        sampling_mode: str, sampling_purpose: str
    ) -> None:
        """Reject customized sampling for contrastive-divergence requests."""
        if sampling_mode == "customized" and sampling_purpose == "cd":
            raise ValueError(
                "Conflict between sampling mode 'customized' and "
                "sampling purpose 'cd'."
            )

    @staticmethod
    def _validate_gibbs_schedule(
        *,
        sampling_mode: str,
        sampling_purpose: str,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        required_purposes: tuple[str, ...],
    ) -> None:
        """Validate Gibbs schedule arguments for selected sampling purposes."""
        if sampling_mode != "gibbs" or sampling_purpose not in required_purposes:
            return
        if n_gibbs_step is None:
            raise ValueError(
                f"n_gibbs_step is required for sampling mode 'gibbs', "
                f"sampling purpose '{sampling_purpose}'."
            )
        if n_gibbs_burnin is None:
            raise ValueError(
                f"n_gibbs_burnin is required for sampling mode 'gibbs', "
                f"sampling purpose '{sampling_purpose}'."
            )

    @staticmethod
    def _validate_sampler_requirement(
        *,
        sampling_mode: str,
        sampling_purpose: str,
        sampler,
        required_purposes: tuple[str, ...],
    ) -> None:
        """Validate sampler availability for customized sampling."""
        if sampling_mode == "customized" and sampling_purpose in required_purposes:
            if sampler is None:
                joined_purposes = "', '".join(required_purposes)
                raise ValueError(
                    "sampler is required for sampling mode 'customized', "
                    f"sampling purpose in '{joined_purposes}'."
                )

    @staticmethod
    def _validate_visible_sampling_state(
        sampling_purpose: str, s_visible: torch.Tensor | None
    ) -> None:
        """Validate visible-state requirements for conditional sampling."""
        if sampling_purpose in {"conditional", "cd"} and s_visible is None:
            raise ValueError(
                f"s_visible is required for sampling purpose '{sampling_purpose}'."
            )

    def _validate_standard_sampling_request(
        self,
        *,
        sampling_mode: str,
        sampling_purpose: str,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        sampler,
        gibbs_required_purposes: tuple[str, ...],
        customized_required_purposes: tuple[str, ...],
    ) -> None:
        """Validate the standard Gibbs/customized sampling controls."""
        self._validate_sampling_mode_and_purpose(
            sampling_mode,
            sampling_purpose,
            valid_modes=("gibbs", "customized"),
        )
        self._validate_customized_cd_conflict(sampling_mode, sampling_purpose)
        self._validate_gibbs_schedule(
            sampling_mode=sampling_mode,
            sampling_purpose=sampling_purpose,
            n_gibbs_step=n_gibbs_step,
            n_gibbs_burnin=n_gibbs_burnin,
            required_purposes=gibbs_required_purposes,
        )
        self._validate_sampler_requirement(
            sampling_mode=sampling_mode,
            sampling_purpose=sampling_purpose,
            sampler=sampler,
            required_purposes=customized_required_purposes,
        )

    def _validate_visible_sampling_request(
        self,
        *,
        sampling_mode: str,
        sampling_purpose: str,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        sampler,
        s_visible: torch.Tensor | None,
        gibbs_required_purposes: tuple[str, ...],
        customized_required_purposes: tuple[str, ...],
    ) -> None:
        """Validate standard visible-state sampling controls."""
        self._validate_standard_sampling_request(
            sampling_mode=sampling_mode,
            sampling_purpose=sampling_purpose,
            n_gibbs_step=n_gibbs_step,
            n_gibbs_burnin=n_gibbs_burnin,
            sampler=sampler,
            gibbs_required_purposes=gibbs_required_purposes,
            customized_required_purposes=customized_required_purposes,
        )
        self._validate_visible_sampling_state(sampling_purpose, s_visible)

    def _validate_sampling_request(
        self,
        *,
        sampling_mode: str,
        sampling_purpose: str,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        sampler,
        s_visible: torch.Tensor | None,
    ) -> None:
        """Validate a standard visible-state sampling request."""
        self._validate_visible_sampling_request(
            sampling_mode=sampling_mode,
            sampling_purpose=sampling_purpose,
            n_gibbs_step=n_gibbs_step,
            n_gibbs_burnin=n_gibbs_burnin,
            sampler=sampler,
            s_visible=s_visible,
            gibbs_required_purposes=self.VISIBLE_GIBBS_REQUIRED_PURPOSES,
            customized_required_purposes=self.VISIBLE_CUSTOMIZED_REQUIRED_PURPOSES,
        )

    def _run_visible_sampling(
        self,
        *,
        sampler,
        sampling_mode: str | None,
        sampling_purpose: str,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        s_visible: torch.Tensor | None,
        n_sample: int | None,
        customized_alias: str | None,
    ) -> torch.Tensor:
        """Run the shared visible-layer sampling pipeline."""
        if sampler is not None and sampling_mode is None:
            return AbstractBoltzmannMachine.sample(self, sampler)
        sampling_mode = self._normalize_sampling_mode(
            sampling_mode, customized_alias=customized_alias
        )
        self._validate_sampling_mode_and_purpose(
            sampling_mode,
            sampling_purpose,
            valid_modes=("gibbs", "customized"),
        )
        self._validate_sampling_request(
            sampling_mode=sampling_mode,
            sampling_purpose=sampling_purpose,
            n_gibbs_step=n_gibbs_step,
            n_gibbs_burnin=n_gibbs_burnin,
            sampler=sampler,
            s_visible=s_visible,
        )
        if sampling_mode == "gibbs":
            if sampling_purpose == "general":
                return self.gibbs_sample(
                    n_step=n_gibbs_step,
                    n_burnin=n_gibbs_burnin,
                    sampler=sampler,
                    n_sample=n_sample,
                )
            if sampling_purpose == "conditional":
                return self._sample_conditional_gibbs(
                    s_visible=s_visible,
                    n_gibbs_step=n_gibbs_step,
                    n_gibbs_burnin=n_gibbs_burnin,
                    sampler=sampler,
                )
            return self._sample_cd_gibbs(
                s_visible=s_visible,
                n_gibbs_step=n_gibbs_step,
                n_gibbs_burnin=n_gibbs_burnin,
            )
        if sampling_purpose == "general":
            return AbstractBoltzmannMachine.sample(self, sampler)
        if sampling_purpose == "conditional":
            return self._sample_conditional_customized(
                sampler=sampler, s_visible=s_visible
            )
        raise ValueError(
            "Sampling purpose 'cd' is not supported for customized sampling mode."
        )

    def _sample_conditional_gibbs(
        self,
        *,
        s_visible: torch.Tensor,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
        sampler,
    ) -> torch.Tensor:
        """Subclass hook for conditional Gibbs sampling."""
        raise NotImplementedError

    def _sample_cd_gibbs(
        self,
        *,
        s_visible: torch.Tensor,
        n_gibbs_step: int | None,
        n_gibbs_burnin: int | None,
    ) -> torch.Tensor:
        """Subclass hook for contrastive-divergence Gibbs sampling."""
        raise NotImplementedError

    def _sample_conditional_customized(
        self, *, sampler, s_visible: torch.Tensor
    ) -> torch.Tensor:
        """Subclass hook for customized conditional sampling."""
        raise NotImplementedError

    def _sampling_energy_from_sample(
        self, *, enable_grad: bool = False, **sample_kwargs
    ) -> torch.Tensor:
        """Compute sampling energy through the public sampling entrypoint."""
        s_all = self.sample(**sample_kwargs)
        return self.energy(s_all, enable_grad)

    # endregion helper methods
