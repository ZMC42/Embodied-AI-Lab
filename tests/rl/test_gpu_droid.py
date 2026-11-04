"""Maple table Stage 0: real observation capture and fixed-noise action parity."""

import argparse
import importlib
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
ATOL = RTOL = 1e-5
SEED = 42


def sources():
    return {name: str(Path(importlib.import_module(name).__file__).resolve())
            for name in ('embodied_ai_lab', 'rlinf', 'gr00t', 'isaaclab',
                         'isaaclab_arena', 'isaaclab_arena_gr00t')}


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if hasattr(value, 'items'):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(item) for item in value]
    return value


def capture(args):
    from embodied_ai_lab.rlinf.env import make_sim
    cfg = OmegaConf.load(args.config)
    env, app = make_sim(cfg.env.train, 4, 1000)
    try:
        obs, _ = env.reset(seed=1000)
        base = env.env
        assert base.max_episode_length == 1056
        assert abs(base.step_dt - 1 / 15) < 1e-12
        assert base.cfg.compute_final_obs
        assert not base.episode_signals()['fall_once'].any()
        assert obs['task_descriptions'] == [cfg.env.train.language_instruction] * 4
        for name in ('main_images', 'wrist_images'):
            assert obs[name].shape == (4, 720, 1280, 3)
            assert obs[name].float().std() > 1
        torch.save(obs, args.output / 'observations.pt')
        report = dict(sources=sources(), language=obs['task_descriptions'],
                      step_dt=base.step_dt, episode_steps=base.max_episode_length,
                      fall_once=base.episode_signals()['fall_once'].tolist(),
                      image_shapes={k: list(obs[k].shape) for k in ('main_images', 'wrist_images')})
        (args.output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        env.close()
    app.close()


def reference_observation(obs, policy):
    from embodied_ai_lab.environments.droid_io import joint_configs
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_policy_observations
    from types import SimpleNamespace
    joints, _, state = joint_configs()
    return build_gr00t_policy_observations(
        [obs[k].numpy() for k in ('main_images', 'wrist_images')], obs['states'].numpy(),
        obs['task_descriptions'][0], SimpleNamespace(target_image_size=(180, 320, 3)),
        state, joints, policy.modality_configs,
    )


def inference(args):
    from embodied_ai_lab.environments import droid_io
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_action_np
    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import TaskMode
    obs = torch.load(args.output / 'observations.pt', weights_only=False)
    records = []
    if args.phase == 'reference':
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag
        model_policy = Gr00tPolicy(EmbodimentTag.OXE_DROID,
                                  str(ROOT / 'models/GR00T-N1.6-DROID'), device='cuda:0')
        model = model_policy.model
    else:
        from embodied_ai_lab.rlinf.extension import build_model, register
        register()
        cfg = OmegaConf.load(args.config)
        model = build_model(cfg.actor.model, torch.bfloat16).to('cuda:0')
        model.eval()
        # Eval processor must remain deterministic, as in the official policy.
        assert not model._modality_transform.training
    model.eval()
    original_prepare = model.prepare_input
    prepared = []

    def prepare(inputs):
        result = original_prepare(inputs)
        # The reference action input retains unused processor fields; RLinf
        # retains only action fields. Compare the tensors consumed by each head.
        prepared.append(cpu({
            'backbone': {key: result[0][key] for key in
                         ('input_ids', 'attention_mask', 'pixel_values')},
            'action': {key: result[1][key] for key in ('state', 'embodiment_id')},
        }))
        return result

    model.prepare_input = prepare
    original_randn = torch.randn
    noises = []

    def randn(*a, **kw):
        result = original_randn(*a, **kw)
        noises.append(cpu(result))
        return result

    for i in range(4):
        sample = {key: value[i:i+1] for key, value in obs.items()}
        prepared.clear()
        noises.clear()
        random.seed(SEED + i)
        np.random.seed(SEED + i)
        torch.manual_seed(SEED + i)
        with torch.inference_mode(), patch('torch.randn', side_effect=randn):
            if args.phase == 'reference':
                inputs = reference_observation(sample, model_policy)
                actions, _ = model_policy.get_action(inputs)
                joints, action_joints, _ = droid_io.joint_configs()
                sim_actions = build_gr00t_action_np(
                    actions, TaskMode.DROID_MANIPULATION, joints, action_joints,
                    embodiment_tag='OXE_DROID',
                ).astype(np.float32)
            else:
                normalized = []
                original_decode = model._get_unnormalized_action
                def decode(action, state=None):
                    result = original_decode(action, state=state)
                    normalized.append(cpu(result))
                    return result
                model._get_unnormalized_action = decode
                sim_actions, _ = model.predict_action_batch(sample, mode='eval')
                actions = normalized[0]
                model._get_unnormalized_action = original_decode
        assert len(prepared) == 1
        assert len(noises) == 1, 'Eval must draw only the shared initial action noise'
        assert prepared[0]['action']['embodiment_id'].tolist() == [16]
        records.append(dict(inputs=prepared[0], initial_noise=noises[0],
                            decoded=cpu(actions), sim=sim_actions,
                            cuda_rng=torch.cuda.get_rng_state(), cpu_rng=torch.get_rng_state()))
    torch.save(records, args.output / f'{args.phase}.pt')
    (args.output / f'{args.phase}.json').write_text(json.dumps(dict(
        sources=sources(), samples=4, embodiment_id=16, seed=SEED,
        denoising_steps=model.action_head.num_inference_timesteps,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    ), indent=2) + '\n')


def compare(args):
    reference = torch.load(args.output / 'reference.pt', weights_only=False)
    actual = torch.load(args.output / 'rlinf.pt', weights_only=False)
    errors = []
    checks = {}
    def check(a, b, name):
        if isinstance(a, dict):
            assert set(a) == set(b), (name, set(a), set(b))
            for key in a:
                check(a[key], b[key], f'{name}/{key}')
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for i, (x, y) in enumerate(zip(a, b)):
                check(x, y, f'{name}/{i}')
        elif isinstance(a, (torch.Tensor, np.ndarray)):
            a, b = torch.as_tensor(a), torch.as_tensor(b)
            assert a.shape == b.shape, (name, a.shape, b.shape)
            floating = a.is_floating_point()
            delta = (a.double() - b.double()).abs().max().item() if a.numel() else 0
            exact = not floating or name.endswith('initial_noise')
            equal = torch.equal(a, b) if exact else torch.allclose(
                a.double(), b.double(), atol=ATOL, rtol=RTOL)
            checks[name] = dict(shape=list(a.shape), max_abs_delta=delta, passed=equal)
            if not equal:
                errors.append(name)
        else:
            assert a == b, (name, a, b)
    check(reference, actual, 'samples')
    report = dict(passed=not errors, atol=ATOL, rtol=RTOL, errors=errors, checks=checks)
    (args.output / 'parity.json').write_text(json.dumps(report, indent=2) + '\n')
    assert not errors, f'Action parity failed: {errors}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['capture', 'reference', 'rlinf', 'compare'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'capture': capture, 'reference': inference, 'rlinf': inference, 'compare': compare}[args.phase](args)


if __name__ == '__main__':
    main()
