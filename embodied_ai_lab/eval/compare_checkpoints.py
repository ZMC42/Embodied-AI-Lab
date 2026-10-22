"""Summarize closed-loop evaluations of an SFT checkpoint and its RL policy."""

import argparse
import json
from pathlib import Path

from embodied_ai_lab.eval.summary import summarize_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("backend", choices=("gpu", "cpu", "both"), nargs="?", default="gpu")
    args = parser.parse_args()
    root = args.run_dir
    backends = ("cpu", "gpu") if args.backend == "both" else (args.backend,)
    results = {}
    rows = [
        "# SFT / RL checkpoint 评测",
        "",
        "| 物理后端 | 模型 | 成功数 | 成功率 | Wilson 95% CI | 摔倒率 | rollout 耗时 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for backend in backends:
        pair = {name: summarize_backend(root / name / backend) for name in ("sft", "rl")}
        delta = pair["rl"]["success_rate"] - pair["sft"]["success_rate"]
        results[backend] = {**pair, "success_rate_delta": delta}
        for name, result in pair.items():
            lo, hi = result["wilson_95_ci"]
            rows.append(
                f"| {backend.upper()} | {name.upper()} | "
                f"{result['successes']}/{result['episodes']} | "
                f"{result['success_rate']:.1%} | [{lo:.1%}, {hi:.1%}] | "
                f"{result['fall_rate']:.1%} | {result['elapsed_s'] / 60:.1f} min |"
            )
    rows += ["", "RL 相对 SFT 的成功率变化："]
    for backend, result in results.items():
        rows.append(f"- {backend.upper()}: {result['success_rate_delta'] * 100:+.1f} 个百分点")
    rows += ["", "失败分类（success 为成功局，其余类别互斥）："]
    for backend, pair in results.items():
        for name in ("sft", "rl"):
            rows.append(f"- {backend.upper()} {name.upper()}: "
                        f"{json.dumps(pair[name]['categories'], ensure_ascii=False)}")
    rows += [
        "",
        "dropped_box=掉箱；not_placed=触发抬起进度但未完成放置；"
        "not_lifted=接近但未触发抬起；not_reached=未接近。",
        "接近和抬起仅作进度诊断；摔倒率独立于成功与失败分类。",
        "成功沿用 Arena 原生判定。固定 seed 不保证 PhysX 位级复现。",
        "成功率差值是描述性结果；单个模型的 Wilson 区间不是差值的置信区间，不能据此宣称显著提升。",
    ]
    (root / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    report = "\n".join(rows) + "\n"
    (root / "report.md").write_text(report)
    print(report)
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
