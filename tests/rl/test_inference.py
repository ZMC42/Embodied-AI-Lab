"""Check the RL flow head's trajectory and actor/critic gradient contracts."""

import random
import unittest

import torch
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from rlinf.models.embodiment.gr00t.gr00t_n1d6.gr00t_action_model import (
    FlowMatchingActionHeadForRLActionPrediction,
)
from transformers.feature_extraction_utils import BatchFeature


def small_head_config():
    return Gr00tN1d6Config(
        hidden_size=8, input_embedding_dim=8, backbone_embedding_dim=8,
        max_action_dim=4, max_state_dim=4, max_num_embodiments=1,
        action_horizon=3, max_seq_len=8, num_inference_timesteps=4,
        use_alternate_vl_dit=True,
        diffusion_model_cfg={
            "attention_head_dim": 4, "num_attention_heads": 2,
            "output_dim": 8, "num_layers": 2, "dropout": 0.0,
            "final_dropout": False, "interleave_self_attention": True,
            "norm_type": "ada_norm", "positional_embeddings": None,
        },
    )


class InferenceTest(unittest.TestCase):
    def test_bfloat16_eval_trajectory_matches_official_euler(self):
        torch.manual_seed(7)
        cfg = small_head_config()
        official = Gr00tN1d6ActionHead(cfg).bfloat16().eval()
        rl = FlowMatchingActionHeadForRLActionPrediction(
            cfg, {"noise_method": "flow_sde", "add_value_head": False},
            output_action_chunks=3,
        )
        for name, module in official.named_children():
            setattr(rl, name, module)
        rl.eval()
        features = torch.randn(1, 5, 8, dtype=torch.bfloat16)
        inputs = BatchFeature({"state": torch.randn(1, 1, 4, dtype=torch.bfloat16),
                               "embodiment_id": torch.tensor([0])})

        def backbone():
            return BatchFeature({"backbone_features": features.clone(),
                                 "backbone_attention_mask": torch.ones(1, 5, dtype=torch.bool),
                                 "image_mask": torch.tensor([[True, True, False, False, False]])})

        with torch.inference_mode():
            torch.manual_seed(42)
            expected = official.get_action(backbone(), inputs)["action_pred"]
            expected_rng = torch.get_rng_state()
            torch.manual_seed(42)
            actual, _ = rl.get_rl_action(backbone(), inputs, mode="eval")
        torch.testing.assert_close(actual["action_pred"], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng),
                        "Eval must consume only the initial noise, like the exported policy")


class ValueGradientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        random.seed(7)
        cfg = small_head_config()
        official = Gr00tN1d6ActionHead(cfg)
        self.head = FlowMatchingActionHeadForRLActionPrediction(
            cfg, {"noise_method": "flow_sde", "add_value_head": True,
                  "value_vlm_mode": "mean_token"}, output_action_chunks=3,
        )
        for name, module in official.named_children():
            setattr(self.head, name, module)
        self.head.requires_grad_(True).eval()
        self.features = torch.randn(2, 5, 8, requires_grad=True)
        self.inputs = BatchFeature({"state": torch.randn(2, 1, 4),
                                    "embodiment_id": torch.zeros(2, dtype=torch.long)})
        with torch.no_grad():
            _, self.rollout = self.head.get_rl_action(self.backbone(), self.inputs)

    def backbone(self):
        return BatchFeature({
            "backbone_features": self.features.clone(),
            "backbone_attention_mask": torch.ones(2, 5, dtype=torch.bool),
            "image_mask": torch.tensor([[True, True, False, False, False]] * 2),
        })

    def forward(self):
        return self.head(self.backbone(), self.inputs, self.rollout["chains"],
                         self.rollout["denoise_inds"])

    def assert_module_gradient(self, module, expected):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if expected:
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.square().sum().item() for g in grads), 0)
        else:
            self.assertEqual(grads, [])

    def test_critic_gradient_respects_detach_for_each_value_mode(self):
        for use_vlm in (False, True):
            # Rebuild only the value head for its different input width.
            self.head.rl_config["use_vlm_value"] = use_vlm
            value_cfg = dict(self.head.rl_config)
            self.head.value_head = FlowMatchingActionHeadForRLActionPrediction(
                small_head_config(), value_cfg,
            ).value_head
            for mode in ("mean_token", "first_token", "last_token"):
                self.head.rl_config["value_vlm_mode"] = mode
                for detach in (None, False, True):
                    with self.subTest(use_vlm=use_vlm, mode=mode, detach=detach):
                        self.head.rl_config.pop("detach_critic_input", None)
                        if detach is not None:
                            self.head.rl_config["detach_critic_input"] = detach
                        self.head.zero_grad(set_to_none=True)
                        self.features.grad = None
                        _, values = self.forward()
                        (values - 1).square().mean().backward()
                        self.assert_module_gradient(self.head.value_head, True)
                        self.assert_module_gradient(self.head.vlln, not detach)
                        self.assert_module_gradient(self.head.state_encoder,
                                                    not detach and not use_vlm)
                        self.assert_module_gradient(self.head.action_decoder, False)
                        if detach:
                            self.assertIsNone(self.features.grad)
                        else:
                            self.assertGreater(self.features.grad.abs().sum().item(), 0)

    def test_detached_critic_preserves_actor_gradients_in_combined_loss(self):
        self.head.rl_config["detach_critic_input"] = False
        logprobs, _ = self.forward()
        (-logprobs.mean()).backward()
        actor_grads = {name: p.grad.clone() for name, p in self.head.named_parameters()
                       if p.grad is not None}
        for module in (self.head.state_encoder, self.head.vlln,
                       self.head.action_decoder, self.head.model):
            self.assert_module_gradient(module, True)
        self.assert_module_gradient(self.head.value_head, False)
        self.head.zero_grad(set_to_none=True)
        self.head.rl_config["detach_critic_input"] = True
        logprobs, values = self.forward()
        (-logprobs.mean() + (values - 1).square().mean()).backward()
        for name, p in self.head.named_parameters():
            if name in actor_grads:
                torch.testing.assert_close(p.grad, actor_grads[name], rtol=0, atol=0)
        self.assert_module_gradient(self.head.value_head, True)

    def test_detach_preserves_replay_and_rollout_outputs_and_rng(self):
        results = []
        for detach in (False, True):
            self.head.rl_config["detach_critic_input"] = detach
            with torch.no_grad():
                replay = self.forward()
                torch.manual_seed(42)
                action, rollout = self.head.get_rl_action(
                    self.backbone(), self.inputs, mode="eval", compute_values=True,
                )
            results.append((*replay, action["action_pred"], rollout["prev_values"],
                            torch.get_rng_state()))
        for before, after in zip(*results):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
