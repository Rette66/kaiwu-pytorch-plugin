"""Offline tests for the MDLM adapter and NLP evaluation helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from example.qdiffusion.nlp.evaluation import (
    causal_lm_nll,
    compute_generative_perplexity,
    compute_token_id_perplexity,
)
from example.qdiffusion.nlp.eval_text_quality import (
    compute_mean_token_entropy,
    compute_text_quality,
)
from example.qdiffusion.nlp import eval_energy_ranking
from example.qdiffusion.nlp.edlm_sampling import EDLMDDPMCacheSampler
from example.qdiffusion.nlp.smoke_generate import load_prompts, parse_args
from example.qdiffusion.nlp.train_energy import (
    candidate_recovery_scores,
    candidate_teacher_nll,
    recovery_ranking_objective,
)
from example.qdiffusion.nlp.builder import build_mdlm_qdiffusion
from example.qdiffusion.nlp.checkpoint import (
    load_energy_weights,
    read_energy_checkpoint,
    save_energy_checkpoint,
)
from example.qdiffusion.nlp.models.mdlm import (
    MDLMBackbone,
    build_mdlm_token_spec,
)
from example.qdiffusion.nlp.models.bm import MDLMConditionedEnergyModel
from example.qdiffusion.nlp.models.edlm import MDLMScalarEnergyModel


class FakeMDLM(nn.Module):
    def __init__(self, *, time_conditioning: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            time_conditioning=time_conditioning,
            vocab_size=6,
            hidden_size=1,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        input_ids,
        timesteps,
        output_hidden_states,
        return_dict,
    ):
        assert timesteps.shape == (input_ids.size(0),)
        assert return_dict
        logits = (
            torch.arange(
                self.config.vocab_size,
                device=input_ids.device,
                dtype=torch.float32,
            )
            .view(1, 1, -1)
            .expand(*input_ids.shape, -1)
        )
        hidden_states = None
        if output_hidden_states:
            hidden_states = (input_ids.float().unsqueeze(-1),)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class CountingFakeMDLM(FakeMDLM):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batch_sizes = []

    def forward(self, **kwargs):
        self.forward_batch_sizes.append(kwargs["input_ids"].size(0))
        return super().forward(**kwargs)


class BFloat16CountingFakeMDLM(CountingFakeMDLM):
    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        outputs.logits = outputs.logits.to(torch.bfloat16)
        return outputs


class LayeredFakeMDLM(FakeMDLM):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList(
            [nn.Linear(1, 1) for _ in range(3)]
        )


class FakeVocabEmbed(nn.Module):
    def __init__(self, vocab_size: int = 6, hidden_size: int = 1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class FakeSigmaMap(nn.Module):
    def __init__(self, cond_dim: int = 1) -> None:
        super().__init__()
        self.projection = nn.Linear(1, cond_dim)

    def forward(self, timesteps):
        return self.projection(timesteps.unsqueeze(-1))


class FakeRotary(nn.Module):
    def forward(self, hidden_states):
        del hidden_states
        return None


class FakeConditionedBlock(nn.Module):
    def __init__(self, hidden_size: int = 1) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, rotary_cos_sin, conditioning, seqlens):
        del rotary_cos_sin, conditioning, seqlens
        return self.projection(hidden_states)


class FakeConditionedOutput(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

    def forward(self, hidden_states, conditioning):
        del conditioning
        return self.linear(hidden_states)


class EDLMFakeMDLM(FakeMDLM):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.vocab_embed = FakeVocabEmbed()
        self.backbone.sigma_map = FakeSigmaMap()
        self.backbone.rotary_emb = FakeRotary()
        self.backbone.blocks = nn.ModuleList([FakeConditionedBlock()])
        self.backbone.output_layer = FakeConditionedOutput(1, 6, 1)


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2


class FakeCausalTokenizer:
    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, max_length, return_tensors
        rows = [[1, 2, 3] if text == "long" else [1, 2, 0] for text in texts]
        masks = [[1, 1, 1] if text == "long" else [1, 1, 0] for text in texts]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }


class FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict
        logits = torch.zeros(*input_ids.shape, 4, device=input_ids.device)
        return SimpleNamespace(logits=logits)


class FakeBMSampler:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, ising_matrix):
        self.calls += 1
        return np.ones((2, ising_matrix.shape[0]), dtype=np.float32)


class RecordingMDLMEnergy(MDLMConditionedEnergyModel):
    def score_visible_logits(self, visible_logits):
        self.last_visible_logits = visible_logits
        return visible_logits.sum(dim=-1, keepdim=True)


class FixedDDPMProposal(nn.Module):
    def __init__(self, mask_id: int = 3, vocab_size: int = 4) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, tokens):
        self.calls += 1
        logits = torch.full(
            (*tokens.shape, self.vocab_size),
            -torch.inf,
            device=tokens.device,
        )
        masked = tokens.eq(self.mask_id)
        logits[..., 0] = torch.where(
            masked,
            torch.log(torch.tensor(0.75)),
            logits[..., 0],
        )
        logits[..., 1] = torch.where(
            masked,
            torch.log(torch.tensor(0.25)),
            logits[..., 1],
        )
        for token_id in range(self.mask_id):
            logits[..., token_id] = torch.where(
                tokens.eq(token_id),
                torch.zeros_like(logits[..., token_id]),
                logits[..., token_id],
            )
        return logits


class CountingCandidateEnergy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def score_candidates_conditioned(
        self,
        noisy_tokens,
        candidate_tokens,
        attention_mask,
    ):
        del noisy_tokens, attention_mask
        self.calls += 1
        return candidate_tokens.float().sum(dim=-1)


def test_mdlm_subs_parameterization_suppresses_mask_and_copies_clean_tokens():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    input_ids = torch.tensor([[1, 5, 3]])

    log_probs = backbone(input_ids)

    assert torch.isneginf(log_probs[..., 5]).all()
    assert log_probs[0, 0, 1].item() == 0.0
    assert log_probs[0, 2, 3].item() == 0.0
    assert torch.isneginf(log_probs[0, 0, 0])
    assert torch.logsumexp(log_probs[0, 1, :5], dim=-1).item() == pytest.approx(0.0)


def test_edlm_ddpm_cache_preserves_unmasked_tokens():
    proposal = FixedDDPMProposal()
    sampler = EDLMDDPMCacheSampler(
        proposal,
        mask_id=proposal.mask_id,
    )
    tokens = torch.tensor([[2, proposal.mask_id, proposal.mask_id]])
    torch.manual_seed(4)

    sampled = sampler.sample(tokens, num_steps=8)

    assert sampled[0, 0].item() == 2
    assert not sampled.eq(proposal.mask_id).any()
    assert sampler.last_stats["proposal_forwards"] <= 8


def test_edlm_importance_window_uses_early_reverse_steps():
    proposal = FixedDDPMProposal()
    energy = CountingCandidateEnergy()
    sampler = EDLMDDPMCacheSampler(
        proposal,
        mask_id=proposal.mask_id,
        energy_model=energy,
        num_candidates=2,
        importance_start_t=1.0,
        importance_end_t=0.8,
    )
    tokens = torch.full((1, 8), proposal.mask_id)
    torch.manual_seed(7)

    sampler.sample(tokens, num_steps=4)

    assert sampler.last_stats["guided_steps"] == 1
    assert energy.calls == 1


def test_edlm_token_entropy_matches_reference_definition():
    entropy = compute_mean_token_entropy(
        [[1, 1, 2, 2], [3, 3, 3, 3]]
    )

    assert entropy == pytest.approx(0.5)


def test_token_id_perplexity_uses_first_eos_and_non_eos_targets():
    model = FakeCausalLM()
    result = compute_token_id_perplexity(
        [[1, 2, 2, 3]],
        model,
        eos_token_id=2,
        batch_size=1,
    )

    assert result.num_tokens == 2
    assert result.perplexity == pytest.approx(4.0)


def test_mdlm_adapter_exposes_hidden_states_and_token_spec():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    input_ids = torch.tensor([[1, 5, 3]])

    hidden_states = backbone.encode_tokens(input_ids)
    token_spec = build_mdlm_token_spec(backbone)

    assert torch.equal(hidden_states.squeeze(-1), input_ids.float())
    assert token_spec.mask_id == 5
    assert token_spec.pad_id == 0
    assert token_spec.bos_id == 1
    assert token_spec.eos_id == 2


def test_mdlm_adapter_rejects_time_conditioned_checkpoint():
    with pytest.raises(ValueError, match="time_conditioning=False"):
        MDLMBackbone(FakeMDLM(time_conditioning=True), FakeTokenizer(), mask_id=5)


def test_mdlm_energy_backbone_only_unfreezes_requested_final_blocks():
    backbone = MDLMBackbone(
        LayeredFakeMDLM(),
        FakeTokenizer(),
        mask_id=5,
    )

    trainable_names = backbone.train_last_blocks(2)

    assert trainable_names
    assert all(
        name.startswith("model.backbone.blocks.1.")
        or name.startswith("model.backbone.blocks.2.")
        for name in trainable_names
    )
    assert not any(
        parameter.requires_grad
        for parameter in backbone.model.backbone.blocks[0].parameters()
    )


def test_mdlm_conditioned_bm_is_the_qdiffusion_energy_path():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = RecordingMDLMEnergy(
        backbone,
        bm_num_visible=2,
        bm_num_hidden=1,
        sampler=object(),
    )
    noisy_tokens = torch.tensor([[1, 3, 0]])
    candidate_tokens = torch.tensor([[2, 4, 0]])
    attention_mask = torch.tensor([[True, True, False]])

    scores = energy_model.score_conditioned(
        noisy_tokens,
        candidate_tokens,
        attention_mask,
    )
    scores.sum().backward()

    assert scores.shape == (1, 1)
    assert energy_model.last_visible_logits.shape == (1, 2)
    assert energy_model.feature_projector.weight.grad is not None


def test_edlm_scalar_energy_matches_joint_pair_pooling_and_scalar_head():
    backbone = MDLMBackbone(EDLMFakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = MDLMScalarEnergyModel(backbone)
    noisy_tokens = torch.tensor([[1, 5, 5]])
    candidate_tokens = torch.tensor([[1, 3, 4]])

    energy = energy_model.score_conditioned(
        noisy_tokens,
        candidate_tokens,
        torch.ones_like(candidate_tokens, dtype=torch.bool),
    )
    energy.sum().backward()

    assert energy.shape == (1, 1)
    assert energy_model.conditioned_encoder.input_projection.weight.grad is not None
    assert energy_model.energy_head[2].weight.grad is not None


def test_bm_can_replace_only_the_edlm_scalar_head():
    backbone = MDLMBackbone(EDLMFakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = RecordingMDLMEnergy(
        backbone,
        bm_num_visible=2,
        bm_num_hidden=1,
        sampler=object(),
        feature_mode="edlm_pair",
    )
    noisy_tokens = torch.tensor([[1, 5, 5]])
    candidate_tokens = torch.tensor([[1, 3, 4]])

    energy = energy_model.score_conditioned(
        noisy_tokens,
        candidate_tokens,
        torch.ones_like(candidate_tokens, dtype=torch.bool),
    )
    energy.sum().backward()

    assert energy.shape == (1, 1)
    assert energy_model.feature_projector.in_features == backbone.hidden_size
    assert energy_model.conditioned_encoder.input_projection.weight.grad is not None


def test_mdlm_candidate_scoring_encodes_noisy_context_once():
    model = CountingFakeMDLM()
    backbone = MDLMBackbone(model, FakeTokenizer(), mask_id=5)
    energy_model = RecordingMDLMEnergy(
        backbone,
        bm_num_visible=2,
        bm_num_hidden=1,
        sampler=object(),
    )
    noisy_tokens = torch.tensor([[1, 5, 5]])
    candidate_tokens = torch.tensor(
        [[[2, 3, 0], [4, 2, 0], [3, 4, 2]]]
    )
    attention_mask = candidate_tokens.ne(0)

    scores = energy_model.score_candidates_conditioned(
        noisy_tokens,
        candidate_tokens,
        attention_mask,
    )

    assert scores.shape == (1, 3)
    assert model.forward_batch_sizes == [1, 3]
    assert energy_model.last_visible_logits.shape == (3, 2)


def test_exact_bm_scoring_is_differentiable_and_skips_sampler():
    sampler = FakeBMSampler()
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = MDLMConditionedEnergyModel(
        backbone,
        bm_num_visible=2,
        bm_num_hidden=2,
        sampler=sampler,
        scoring_mode="exact",
    ).to("cpu")
    visible_logits = torch.tensor(
        [[-1.0, 0.5], [0.25, 1.0]],
        requires_grad=True,
    )

    scores = energy_model.score_visible_logits(visible_logits)
    scores.sum().backward()

    assert scores.shape == (2, 1)
    assert visible_logits.grad is not None
    assert energy_model.energy_bm.quadratic_coef.grad is not None
    assert sampler.calls == 0


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("sigmoid", torch.tensor([[0.5, 0.880797]])),
        ("identity", torch.tensor([[0.0, 2.0]])),
    ],
)
def test_mdlm_visible_transform_modes(mode, expected):
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = MDLMConditionedEnergyModel(
        backbone,
        bm_num_visible=2,
        bm_num_hidden=1,
        sampler=object(),
        visible_transform=mode,
    )

    visible = energy_model.discretize_visible_state(
        torch.tensor([[0.0, 2.0]])
    )

    assert torch.allclose(visible, expected, atol=1e-6)


def test_mdlm_layernorm_visible_transform_preserves_gradient_and_scale():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = MDLMConditionedEnergyModel(
        backbone,
        bm_num_visible=3,
        bm_num_hidden=1,
        sampler=object(),
        visible_transform="layernorm",
    )
    logits = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)

    visible = energy_model.discretize_visible_state(logits)
    visible.square().sum().backward()

    assert torch.allclose(visible.mean(dim=-1), torch.zeros(1), atol=1e-6)
    assert logits.grad is not None
    assert energy_model.visible_transform.scale.grad is not None


def test_builder_and_checkpoint_keep_baseline_and_guided_paths_distinct(tmp_path):
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    baseline = build_mdlm_qdiffusion(
        backbone,
        use_energy=False,
        dtype=torch.float32,
    )
    guided = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        bm_num_visible=2,
        bm_num_hidden=1,
        bm_sampler=object(),
        num_candidates=4,
        dtype=torch.float32,
    )

    checkpoint_path = tmp_path / "energy.pt"
    save_energy_checkpoint(guided, checkpoint_path, epoch=1, metric=0.5)
    checkpoint = read_energy_checkpoint(checkpoint_path)
    with torch.no_grad():
        guided.energy_model.feature_projector.weight.zero_()
    load_energy_weights(guided, checkpoint)

    assert baseline.energy_model is None
    assert not baseline.config.use_energy
    assert guided.energy_model is not None
    assert guided.config.use_energy
    assert guided.config.num_candidates == 4
    assert torch.count_nonzero(guided.energy_model.feature_projector.weight).item() > 0


def test_scalar_energy_checkpoint_round_trip(tmp_path):
    backbone = MDLMBackbone(EDLMFakeMDLM(), FakeTokenizer(), mask_id=5)
    scalar = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        energy_type="scalar",
        energy_feature_mode="edlm_pair",
        num_candidates=4,
        dtype=torch.float32,
    )
    checkpoint_path = tmp_path / "scalar_energy.pt"
    original_weight = scalar.energy_model.energy_head[2].weight.detach().clone()

    save_energy_checkpoint(scalar, checkpoint_path, epoch=1, metric=0.5)
    checkpoint = read_energy_checkpoint(checkpoint_path)
    with torch.no_grad():
        scalar.energy_model.energy_head[2].weight.zero_()
    load_energy_weights(scalar, checkpoint)

    assert checkpoint["metadata"]["energy_type"] == "scalar"
    assert checkpoint["metadata"]["feature_mode"] == "edlm_pair"
    assert torch.equal(
        scalar.energy_model.energy_head[2].weight,
        original_weight,
    )


def test_builder_exposes_proposal_resampling_controls():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)

    baseline = build_mdlm_qdiffusion(
        backbone,
        use_energy=False,
        disable_resample=False,
        resample_ratio=0.15,
        resample_top_p=0.9,
        dtype=torch.float32,
    )

    assert not baseline.config.disable_resample
    assert baseline.config.resample_ratio == pytest.approx(0.15)
    assert baseline.config.resample_top_p == pytest.approx(0.9)


def test_enabled_resampler_reenters_proposal_model():
    proposal = CountingFakeMDLM()
    backbone = MDLMBackbone(proposal, FakeTokenizer(), mask_id=5)
    baseline = build_mdlm_qdiffusion(
        backbone,
        use_energy=False,
        proposal_noise_scale=0.0,
        disable_resample=False,
        resample_ratio=0.5,
        dtype=torch.float32,
    )
    input_tokens = torch.tensor([[1, 5, 5, 5, 5]])
    fixed_prompt = torch.tensor([[True, False, False, False, False]])

    baseline.generate(
        input_tokens,
        max_steps=1,
        partial_masks=fixed_prompt,
    )

    assert proposal.forward_batch_sizes == [1, 1]


def test_resampler_casts_bfloat16_scores_to_generation_score_dtype():
    proposal = BFloat16CountingFakeMDLM()
    backbone = MDLMBackbone(proposal, FakeTokenizer(), mask_id=5)
    baseline = build_mdlm_qdiffusion(
        backbone,
        use_energy=False,
        proposal_noise_scale=0.0,
        disable_resample=False,
        resample_ratio=0.5,
        dtype=torch.float32,
    )
    input_tokens = torch.tensor([[1, 5, 5, 5, 5]])
    fixed_prompt = torch.tensor([[True, False, False, False, False]])

    output = baseline.generate(
        input_tokens,
        max_steps=1,
        partial_masks=fixed_prompt,
    )

    assert output.shape == input_tokens.shape
    assert proposal.forward_batch_sizes == [1, 1]


def test_smoke_cli_exposes_proposal_resampling_controls(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "smoke_generate",
            "--enable-resample",
            "--resample-ratio",
            "0.15",
            "--resample-top-p",
            "0.9",
        ],
    )

    args = parse_args()

    assert args.enable_resample
    assert args.resample_ratio == pytest.approx(0.15)
    assert args.resample_top_p == pytest.approx(0.9)


def test_guided_generation_reaches_conditioned_bm_sampler():
    sampler = FakeBMSampler()
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    guided = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        bm_num_visible=2,
        bm_num_hidden=1,
        bm_sampler=sampler,
        num_candidates=2,
        dtype=torch.float32,
    )
    input_tokens = torch.tensor([[1, 5, 5]])
    fixed_prompt = torch.tensor([[True, False, False]])

    output = guided.generate(
        input_tokens,
        max_steps=1,
        partial_masks=fixed_prompt,
    )

    assert output.shape == input_tokens.shape
    assert sampler.calls == 2


def test_causal_nll_ignores_padding_and_first_token():
    logits = torch.zeros(2, 3, 4)
    input_ids = torch.tensor([[1, 2, 3], [1, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    nll, num_tokens = causal_lm_nll(logits, input_ids, attention_mask)

    assert int(num_tokens) == 3
    assert float(nll) == pytest.approx(3 * torch.log(torch.tensor(4.0)).item())


def test_generative_perplexity_is_token_weighted():
    result = compute_generative_perplexity(
        ["long", "short"],
        FakeCausalLM(),
        FakeCausalTokenizer(),
        batch_size=2,
    )

    assert result.perplexity == pytest.approx(4.0)
    assert result.num_tokens == 3
    assert result.num_sequences == 2


def test_prompt_suite_and_text_quality_metrics(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt":"First prompt"}\n{"prompt":"Second prompt"}\n',
        encoding="utf-8",
    )

    prompts = load_prompts(None, prompts_path)
    metrics = compute_text_quality(
        ["alpha beta alpha beta", "gamma delta"]
    )

    assert prompts == ["First prompt", "Second prompt"]
    assert metrics["num_sequences"] == 2
    assert metrics["distinct_1"] == pytest.approx(4 / 6)
    assert metrics["repetition_2"] > 0


def test_candidate_teacher_nll_returns_one_score_per_candidate():
    candidate_tokens = torch.tensor(
        [[[1, 2, 3], [1, 2, 0]]],
        dtype=torch.long,
    )

    nll = candidate_teacher_nll(
        candidate_tokens,
        FakeCausalLM(),
        eos_token_id=0,
    )

    assert nll.shape == (1, 2)
    assert torch.allclose(nll, torch.full_like(nll, torch.log(torch.tensor(4.0))))


def test_recovery_ranking_prefers_candidates_matching_masked_targets():
    outputs = {
        "targets": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.tensor([[False, True, True, False]]),
        "candidate_tokens": torch.tensor(
            [[[1, 2, 3, 4], [1, 2, 0, 4], [1, 0, 0, 4]]]
        ),
    }

    scores = candidate_recovery_scores(outputs)
    aligned_energies = torch.tensor([[-2.0, 0.0, 2.0]])
    reversed_energies = -aligned_energies

    aligned_loss = recovery_ranking_objective(
        aligned_energies,
        scores,
        target_temperature=0.1,
        energy_temperature=1.0,
    )
    reversed_loss = recovery_ranking_objective(
        reversed_energies,
        scores,
        target_temperature=0.1,
        energy_temperature=1.0,
    )

    assert torch.allclose(scores, torch.tensor([[1.0, 0.5, 0.0]]))
    assert aligned_loss < reversed_loss


def test_energy_ranking_diagnostic_compares_candidate_orderings(monkeypatch):
    class FakeProposal:
        def eval(self):
            return self

    class FakeGenerator:
        eos_id = 0
        proposal_model = FakeProposal()
        config = SimpleNamespace(num_candidates=3)

        def eval(self):
            return self

        def objective(self, batch, **kwargs):
            del batch, kwargs
            return {
                "candidate_tokens": torch.zeros(2, 3, 2, dtype=torch.long),
                "targets": torch.zeros(2, 2, dtype=torch.long),
                "loss_mask": torch.ones(2, 2, dtype=torch.bool),
                "candidate_energies": torch.tensor(
                    [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
                ),
                "positive_energy_mean": torch.tensor(-1.0),
                "negative_energy_mean": torch.tensor(1.0),
            }

    monkeypatch.setattr(
        eval_energy_ranking,
        "candidate_teacher_nll",
        lambda *args, **kwargs: torch.tensor(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
        ),
    )

    metrics = eval_energy_ranking.evaluate_ranking(
        FakeGenerator(),
        [{"targets": torch.zeros(2, 2, dtype=torch.long)}],
        teacher_model=object(),
    )

    assert metrics["ranking_top1_accuracy"] == 0.5
    assert metrics["ranking_pairwise_accuracy"] == 0.5
    assert metrics["ranking_spearman"] == 0.0
    assert metrics["selected_teacher_nll"] == 1.0
    assert metrics["oracle_teacher_nll"] == 0.0
    assert metrics["candidate_mean_teacher_nll"] == 1.0
    assert metrics["energy_margin"] == 2.0
