"""CPU coverage of the real embodied update loop and PPO metric mask reduction."""

import inspect
import math
import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.registry import policy_loss
from rlinf.utils.utils import masked_mean
from rlinf.workers.actor import embodied_fsdp_actor_worker as worker


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.epoch = 0

    def forward(self, forward_inputs, **kwargs):
        ratios = forward_inputs['ratios'][:, self.epoch]
        logprobs = ratios.log().unsqueeze(-1).repeat(1, 2) / 2
        return {'logprobs': logprobs + self.parameter * 0,
                'prev_logprobs': torch.zeros_like(logprobs),
                'values': torch.zeros_like(logprobs[:, :1]) + self.parameter * 0}


def make_actor(*, epochs=3, micro=1, enabled=True, all_masked=False):
    cfg = OmegaConf.create({
        'actor': {'seed': 7, 'micro_batch_size': micro, 'global_batch_size': 4,
                  'model': {'model_type': 'gr00t_n1d6', 'action_dim': 2}},
        'algorithm': {'loss_type': 'actor_critic', 'adv_type': 'gae', 'reward_type': 'chunk_level',
                      'logprob_type': 'chunk_level', 'update_epoch': epochs,
                      'log_epoch_metrics': enabled, 'clip_ratio_high': .2,
                      'clip_ratio_low': .2, 'value_clip': .2, 'huber_delta': 1., 'entropy_bonus': 0},
        'env': {'train': {'max_episode_steps': 1500}},
        'runner': {'task_type': 'embodied'},
    })
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    mask = torch.tensor([False] * 4 if all_masked else [True, True, False, True])
    actor = SimpleNamespace(
        cfg=cfg, model=model, optimizer=optimizer,
        lr_scheduler=SimpleNamespace(step=lambda: None),
        grad_scaler=SimpleNamespace(scale=lambda value: value),
        is_weight_offloaded=False, is_optimizer_offloaded=False,
        device='cpu', _rank=0, _world_size=1, gradient_accumulation=4 // micro,
        optimizer_steps=0, critic_warmup_steps=0, enable_sft_co_train=False,
        amp_context=nullcontext(), loss_agg_func=masked_mean,
        torch_platform=SimpleNamespace(empty_cache=lambda: None),
        before_micro_batch=lambda *args, **kwargs: nullcontext(),
        rollout_batch={
            'prev_logprobs': torch.zeros(1, 4, 2),
            'advantages': torch.tensor([[[1.], [-1.], [1.], [1.]]]),
            'loss_mask': mask.reshape(1, 4, 1),
            'returns': torch.arange(4.).reshape(1, 4, 1),
            'prev_values': torch.zeros(2, 4, 1),
            'forward_inputs': {'ratios': torch.tensor([
                [[1., 1.5, 1.1], [1., .5, .9], [1., 2., 2.], [1., 1., 1.]]])},
        },
    )

    def optimizer_step():
        optimizer.step()
        model.epoch += 1
        actor.optimizer_steps += 1
        return 0., [.01]

    actor.optimizer_step = optimizer_step
    actor.train_micro_batch = MethodType(worker.EmbodiedFSDPActor.train_micro_batch, actor)
    return actor


def run_actor(actor, reduce=None):
    with patch.object(worker, 'clear_memory'), patch.object(
        worker.torch.distributed, 'get_world_size', return_value=1
    ), patch.object(worker.Worker.torch_platform, 'current_device', return_value='cpu'), patch.object(
        worker, 'all_reduce_dict', side_effect=reduce or (lambda data, **kwargs: data)
    ):
        return inspect.unwrap(worker.EmbodiedFSDPActor.run_training)(actor)


class EpochMetricsTest(unittest.TestCase):
    def test_real_loop_separates_epoch_one_and_excludes_zero_mask(self):
        result = run_actor(make_actor())
        self.assertEqual(result['actor/epoch_1/clip_fraction'], 0)
        self.assertEqual(result['actor/epoch_1/ratio'], 1)
        self.assertEqual(result['actor/epoch_1/approx_kl'], 0)
        self.assertEqual(result['actor/epoch_2/valid_count'], 3)
        self.assertAlmostEqual(result['actor/epoch_2/clip_fraction'], 2 / 3)
        self.assertAlmostEqual(result['actor/epoch_2/ratio'], 1)
        self.assertAlmostEqual(result['actor/epoch_2/approx_kl'], -math.log(.75) / 3)
        self.assertEqual(result['actor/epoch_3/clip_fraction'], 0)
        self.assertEqual(result['actor/epoch_ge_2/valid_count'], 6)
        self.assertAlmostEqual(result['actor/epoch_ge_2/clip_fraction'], 1 / 3)
        self.assertFalse(any('_ppo_metric_count' in name for name in result))

    def test_microbatch_partition_does_not_change_new_weighted_metrics(self):
        singles = run_actor(make_actor(micro=1))
        pairs = run_actor(make_actor(micro=2))
        for name in singles:
            if '/epoch_' in name:
                self.assertAlmostEqual(singles[name], pairs[name], places=7, msg=name)

    def test_switch_preserves_all_legacy_metrics_and_optimizer_steps(self):
        enabled, disabled = make_actor(), make_actor(enabled=False)
        with_metrics = run_actor(enabled)
        legacy = run_actor(disabled)
        self.assertEqual({k: v for k, v in with_metrics.items() if '/epoch_' not in k}, legacy)
        self.assertFalse(any('/epoch_' in name for name in legacy))
        self.assertEqual(enabled.optimizer_steps, disabled.optimizer_steps)
        self.assertTrue(torch.equal(enabled.model.parameter, disabled.model.parameter))
        single = run_actor(make_actor(epochs=1))
        self.assertFalse(any('/epoch_ge_2/' in name for name in single))

    def test_rank_aggregation_sums_counts_before_division(self):
        def reduce(data, *, op, **kwargs):
            if op == torch.distributed.ReduceOp.AVG or not any('/epoch_' in k for k in data):
                return data
            self.assertEqual(op, torch.distributed.ReduceOp.SUM)
            self.assertEqual(kwargs['dtype'], torch.float64)
            # A second rank has one valid sample in each epoch; it clips in epoch2.
            remote = {'valid_count': 1., 'ratio': 1., 'clip_fraction': 0., 'approx_kl': 0.}
            result = {}
            for key, value in data.items():
                name = key.rsplit('/', 1)[-1]
                extra = remote[name]
                if key == 'actor/epoch_2/clip_fraction':
                    extra = 1.
                result[key] = value + extra
            return result

        result = run_actor(make_actor(epochs=2), reduce=reduce)
        self.assertEqual(result['actor/epoch_2/valid_count'], 4)
        self.assertAlmostEqual(result['actor/epoch_2/clip_fraction'], .75)
        self.assertAlmostEqual(result['actor/epoch_ge_2/clip_fraction'], .75)

    def test_all_masked_reports_zero_count_and_undefined_mean(self):
        result = run_actor(make_actor(all_masked=True))
        self.assertEqual(result['actor/epoch_ge_2/valid_count'], 0)
        self.assertTrue(math.isnan(result['actor/epoch_ge_2/clip_fraction']))
        self.assertTrue(math.isnan(result['actor/epoch_2/approx_kl']))

    def test_actual_loss_count_uses_broadcast_token_mask(self):
        logprobs = torch.full((2, 6), .1, requires_grad=True)
        _, metrics = policy_loss(
            task_type='embodied', loss_type='actor', logprob_type='token_level',
            reward_type='action_level', single_action_dim=3,
            logprobs=logprobs, old_logprobs=torch.zeros_like(logprobs),
            advantages=torch.ones(2, 2), loss_mask=torch.tensor([[True, False], [True, True]]),
            loss_mask_sum=torch.tensor([[1500., 10.], [600., 1.]]), max_episode_steps=1500,
            clip_ratio_high=.2, clip_ratio_low=.2, log_epoch_metrics=True,
        )
        self.assertEqual(metrics['_ppo_metric_count'], 9)
        self.assertAlmostEqual(metrics['actor/ratio'], math.exp(.1), places=6)
        self.assertAlmostEqual(metrics['actor/approx_kl'], -.1, places=6)


if __name__ == '__main__':
    unittest.main()
