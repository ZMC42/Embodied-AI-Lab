"""Summarize complete CPU/GPU baseline runs with Wilson 95% intervals."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def failure_category(record, navigation_threshold=0.5):
    if record["success"]:
        return "success"
    if record["termination_terms"]["object_dropped"]:
        return "dropped_box"
    lifted = any(
        event["predicate_index"] == 1 and event["criteria_name"] == "pick_and_place"
        for event in record["progress"]["events"]
    )
    if lifted:
        return "not_placed"
    if record["min_base_box_xy_distance_m"] <= navigation_threshold:
        return "not_lifted"
    return "not_reached"


def summarize_backend(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    records = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    assert len(records) == protocol["episodes"] == complete["episodes"]
    assert len({(record["env_id"], record["episode_in_env"]) for record in records}) == len(records)
    assert all(record["seed"] == protocol["seed"] for record in records)
    assert all(0 < record["episode_length"] <= protocol["max_episode_steps"] for record in records)
    successes = sum(record["success"] for record in records)
    falls = sum(record["fall_once"] for record in records)
    assert successes == complete["successes"]
    categories = Counter(failure_category(record, protocol["navigation_threshold_m"]) for record in records)
    steps = sum(record["episode_length"] for record in records)
    with (directory.parent / f"gpu-{directory.name}.csv").open() as stream:
        gpu_samples = list(csv.reader(stream))[1:]
    peak_gpu_mib = max(float(row[1].strip().split()[0]) for row in gpu_samples)
    return {
        "protocol": protocol,
        **complete,
        "success_rate": successes / len(records),
        "falls": falls,
        "fall_rate": falls / len(records),
        "wilson_95_ci": wilson_interval(successes, len(records)),
        "categories": dict(categories),
        "termination_counts": dict(Counter(
            name for record in records for name, active in record["termination_terms"].items() if active
        )),
        "episode_control_steps": steps,
        "episode_steps_per_second": steps / complete["elapsed_s"],
        "peak_host_single_gpu_memory_mib": peak_gpu_mib,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    results = {backend: summarize_backend(args.run_dir / backend) for backend in ("cpu", "gpu")}
    cpu, gpu = results["cpu"], results["gpu"]
    for key in ("seed", "episodes", "num_envs", "max_episode_steps", "action_chunk_length", "checkpoint"):
        assert cpu["protocol"][key] == gpu["protocol"][key], key
    overlap = max(cpu["wilson_95_ci"][0], gpu["wilson_95_ci"][0]) <= min(cpu["wilson_95_ci"][1], gpu["wilson_95_ci"][1])
    summary = {
        **results,
        "cpu_gpu_ci_overlap": overlap,
        "requires_backend_investigation": not overlap,
        "failure_classification": {
            "dropped_box": "Official object_dropped termination; takes priority among failures.",
            "not_placed": "Official lift predicate completed, but final task success did not.",
            "not_lifted": "Base came within 0.5 m XY of initial box, but official lift predicate did not complete.",
            "not_reached": "Base never came within 0.5 m XY of initial box; navigation proxy.",
        },
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = [
        "# CPU/GPU checkpoint 基线",
        "",
        "| 物理 | 成功数 | 成功率 | Wilson 95% CI | 摔倒数 | 摔倒比例 | rollout 耗时 | 主机单卡显存采样峰值 (MiB) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for backend, result in results.items():
        lo, hi = result["wilson_95_ci"]
        rows.append(
            f"| {backend.upper()} | {result['successes']}/{result['episodes']} | "
            f"{result['success_rate']:.1%} | [{lo:.1%}, {hi:.1%}] | "
            f"{result['falls']} | {result['fall_rate']:.1%} | {result['elapsed_s'] / 60:.1f} min | "
            f"{result['peak_host_single_gpu_memory_mib']:.0f} |"
        )
    rows += ["", f"CPU/GPU 区间重叠：{overlap}。", "", "失败分类："]
    for backend, result in results.items():
        rows.append(f"- {backend.upper()}: {json.dumps(result['categories'], ensure_ascii=False)}")
    rows += [
        "",
        "显存为 nvidia-smi 返回的所有 GPU、所有采样时刻中的单卡最大占用，包含推理服务器和其他进程；不是多卡总和或评测进程独占用量。",
        "每 5 秒采样可能遗漏瞬时峰值。",
        "成功沿用 Arena 原生判定；抬起沿用 pick_and_place 的第 2 个 predicate。",
        "抬起 predicate 只表示官方进度条件；reset 后的箱体回弹也可能触发，不能据此证明已抓稳。",
        "没走到／没抓起按底盘距箱子初始 XY 是否达到 0.5 m 区分，是导航代理指标。",
        "掉箱以官方 object_dropped 终止为准；其余已抬起的失败归为没放进料箱。",
        "摔倒为一局内骨盆相对默认站立位置下降超过 0.4 m；独立于成功和失败分类，仅作诊断。",
        "随机种子固定并不保证 PhysX 位级可复现；并行环境中的策略采样随 episode 结束顺序变化。",
    ]
    (args.run_dir / "report.md").write_text("\n".join(rows) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
