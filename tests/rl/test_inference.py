"""The RL flow head's deterministic trajectory must match the exported GR00T head."""

import unittest

import torch
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from rlinf.models.embodiment.gr00t.gr00t_n1d6.gr00t_action_model import (
    FlowMatchingActionHeadForRLActionPrediction,
)
from transformers.feature_extraction_utils import BatchFeature


class InferenceTest(unittest.TestCase):
    def test_bfloat16_eval_trajectory_matches_official_euler(self):
        torch.manual_seed(7)
        cfg = Gr00tN1d6Config(
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


if __name__ == "__main__":
    unittest.main()
