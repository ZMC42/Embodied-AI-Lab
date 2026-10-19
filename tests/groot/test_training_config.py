"""Check local training overrides at the upstream CLI and checkpoint-loading seam."""

import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.setup import Gr00tN1d6Pipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from gr00t.model.modules.eagle_backbone import EagleBackbone


GROOT_ROOT = Path(__file__).resolve().parents[2] / "third_party/IsaacLab-Arena/submodules/Isaac-GR00T"


class TrainingConfigTest(unittest.TestCase):
    def test_flow_time_sampling_after_bfloat16_loading(self):
        config = Gr00tN1d6Config(
            hidden_size=8, input_embedding_dim=8, max_state_dim=8,
            max_action_dim=8, max_num_embodiments=2, max_seq_len=16,
        )
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with patch(
                "gr00t.model.gr00t_n1d6.gr00t_n1d6.AlternateVLDiT",
                return_value=nn.Linear(1, 1),
            ):
                head = Gr00tN1d6ActionHead(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        sample = head.sample_time(16, "cpu", torch.bfloat16)
        self.assertEqual(sample.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(sample).all())
        self.assertTrue(((sample >= 0) & (sample <= 1)).all())

    def test_cli_overrides_reach_checkpoint_loader(self):
        args = [
            "launch_finetune.py", "--base-model-path", "unused-model",
            "--dataset-path", "unused-dataset", "--embodiment-tag", "NEW_EMBODIMENT",
            "--no-tune-llm", "--no-tune-visual", "--tune-top-llm-layers", "0",
            "--model-dtype", "bfloat16",
        ]
        with patch.object(sys, "argv", args), patch("gr00t.experiment.experiment.run") as run:
            runpy.run_path(str(GROOT_ROOT / "gr00t/experiment/launch_finetune.py"), run_name="__main__")
        config = run.call_args.args[0]
        model = nn.Linear(1, 1)
        model.config = config.model
        with tempfile.TemporaryDirectory() as directory, patch(
            "gr00t.model.gr00t_n1d6.setup.AutoModel.from_pretrained",
            return_value=(model, {"missing_keys": []}),
        ) as load:
            Gr00tN1d6Pipeline(config, Path(directory))._create_model()
        self.assertEqual(load.call_args.args[0], "unused-model")
        self.assertEqual(load.call_args.kwargs["tune_top_llm_layers"], 0)
        self.assertIs(load.call_args.kwargs["torch_dtype"], torch.bfloat16)
        self.assertFalse(load.call_args.kwargs["tune_llm"])
        self.assertFalse(load.call_args.kwargs["tune_visual"])

    def test_zero_top_layers_freezes_the_whole_backbone(self):
        backbone = EagleBackbone.__new__(EagleBackbone)
        nn.Module.__init__(backbone)
        backbone.model = nn.Module()
        backbone.model.language_model = nn.Module()
        backbone.model.language_model.model = nn.Module()
        backbone.model.language_model.model.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(6)])
        backbone.model.vision_model = nn.Linear(2, 2)
        backbone.model.mlp1 = nn.Linear(2, 2)
        backbone.set_trainable_parameters(False, False, 0)
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
        backbone.set_trainable_parameters(False, False, 4)
        layers = backbone.model.language_model.model.layers
        self.assertFalse(any(p.requires_grad for p in layers[:2].parameters()))
        self.assertTrue(all(p.requires_grad for p in layers[2:].parameters()))
        self.assertFalse(any(p.requires_grad for p in backbone.model.vision_model.parameters()))

    def test_upstream_training_defaults_are_preserved(self):
        config = FinetuneConfig("unused-model", "unused-dataset", EmbodimentTag.NEW_EMBODIMENT)
        self.assertEqual(config.tune_top_llm_layers, 4)
        self.assertEqual(config.model_dtype, "float32")


if __name__ == "__main__":
    unittest.main()
