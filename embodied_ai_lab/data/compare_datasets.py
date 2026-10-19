"""Validate a converted dataset and compare it with the official LeRobot metadata."""

import argparse
import json
from pathlib import Path
import subprocess

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ("observation.state", "action")


def read_json(path):
    return json.loads(path.read_text())


def read_episodes(directory):
    return [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]


def validate_dataset(dataset, expected_episodes):
    info = read_json(dataset / "meta/info.json")
    episodes = read_episodes(dataset / "meta")
    assert info["fps"] == 50, "Expected 50 fps"
    assert len(episodes) == info["total_episodes"] == expected_episodes, "Episode count mismatch"
    assert info["total_videos"] == expected_episodes, "Video count mismatch"
    assert info["total_chunks"] == (expected_episodes + info["chunks_size"] - 1) // info["chunks_size"]
    assert info["splits"] == {"train": f"0:{expected_episodes}"}, "Invalid train split"
    normalization = read_json(dataset / "meta/stats.json")
    values = {key: [] for key in FEATURES}
    for episode in episodes:
        index = episode["episode_index"]
        chunk = index // info["chunks_size"]
        parquet = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        frame = pd.read_parquet(parquet)
        length = episode["length"]
        assert len(frame) == length, f"Row count mismatch: {parquet}"
        np.testing.assert_array_equal(frame["frame_index"], np.arange(length))
        np.testing.assert_array_equal(frame["episode_index"], np.full(length, index))
        offset = sum(ep["length"] for ep in episodes[:index])
        np.testing.assert_array_equal(frame["index"], np.arange(offset, offset + length))
        np.testing.assert_allclose(frame["timestamp"], np.arange(length) / info["fps"])
        assert frame["next.done"].sum() == 1 and frame["next.done"].iloc[-1], "Invalid episode end"
        for key, feature in info["features"].items():
            if "float" in feature["dtype"]:
                array = np.stack(frame[key])
                assert np.isfinite(array).all(), f"Nonfinite values in {key}"
                for statistic in ("mean", "std", "min", "max", "q01", "q99"):
                    assert np.isfinite(normalization[key][statistic]).all(), f"Invalid stats for {key}"
        for key in FEATURES:
            array = np.stack(frame[key]).astype(np.float64)
            assert array.shape == (length, 43) and np.isfinite(array).all(), f"Invalid {key}"
            values[key].append(array)
        video = dataset / info["video_path"].format(
            episode_chunk=chunk, episode_index=index, video_key="observation.images.ego_view"
        )
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=nb_frames,width,height,r_frame_rate", "-of", "json", str(video),
        ]))["streams"][0]
        assert int(probe["nb_frames"]) == length, f"Video frame count mismatch: {video}"
        assert (probe["width"], probe["height"]) == (640, 480), f"Invalid resolution: {video}"
        numerator, denominator = map(int, probe["r_frame_rate"].split("/"))
        assert numerator / denominator == info["fps"], f"Invalid video fps: {video}"
    assert sum(ep["length"] for ep in episodes) == info["total_frames"], "Total frame count mismatch"
    assert len(list((dataset / "data").rglob("*.parquet"))) == expected_episodes
    assert len(list((dataset / "videos").rglob("*.mp4"))) == expected_episodes
    return info, episodes, {key: np.concatenate(arrays) for key, arrays in values.items()}


def length_statistics(episodes):
    lengths = np.array([episode["length"] for episode in episodes])
    return {
        "episodes": len(lengths), "frames": int(lengths.sum()),
        "min": int(lengths.min()), "mean": float(lengths.mean()),
        "median": float(np.median(lengths)), "p95": float(np.quantile(lengths, 0.95)),
        "max": int(lengths.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=ROOT / "data/locomanipulation/generated/lerobot")
    parser.add_argument("--official", type=Path, default=ROOT / "data/locomanipulation/official/lerobot/meta")
    parser.add_argument("--hdf5", type=Path, default=ROOT / "data/locomanipulation/generated.hdf5")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/dataset-comparison")
    parser.add_argument("--expected-episodes", type=int, default=100)
    args = parser.parse_args()
    with h5py.File(args.hdf5, "r") as source:
        assert len(source["data"]) == args.expected_episodes, "HDF5 episode count mismatch"
        assert all(bool(ep.attrs["success"]) for ep in source["data"].values()), "Unsuccessful HDF5 demo"
        source_lengths = [len(ep["actions"]) - 1 for ep in source["data"].values()]
    info, episodes, values = validate_dataset(args.generated, args.expected_episodes)
    assert [ep["length"] for ep in episodes] == source_lengths, "HDF5 and converted lengths differ"
    official_info = read_json(args.official / "info.json")
    official_episodes = read_episodes(args.official)
    assert len(official_episodes) == official_info["total_episodes"]
    assert sum(ep["length"] for ep in official_episodes) == official_info["total_frames"]
    assert info["fps"] == official_info["fps"]
    official_stats = read_json(args.official / "stats.json")
    summary = {"generated": length_statistics(episodes), "official": length_statistics(official_episodes)}
    computed_stats = {}
    comparisons = []
    for key, array in values.items():
        names = info["features"][key]["names"]
        assert names == official_info["features"][key]["names"], f"Joint ordering mismatch: {key}"
        stats = {"min": array.min(0), "max": array.max(0), "mean": array.mean(0), "std": array.std(0)}
        stats.update(q01=np.quantile(array, 0.01, axis=0), q99=np.quantile(array, 0.99, axis=0))
        computed_stats[key] = {name: value.tolist() for name, value in stats.items()}
        reference = official_stats[key]
        for joint_index, joint_name in enumerate(names):
            row = {"feature": key, "joint": joint_name}
            for statistic in stats:
                row[f"generated_{statistic}"] = float(stats[statistic][joint_index])
                row[f"official_{statistic}"] = float(reference[statistic][joint_index])
            # A constant official joint has no meaningful standard-deviation scale.
            std = row["official_std"]
            row["mean_shift_in_official_std"] = (
                (row["generated_mean"] - row["official_mean"]) / std if std > 1e-6 else None
            )
            comparisons.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "state-action-stats.json").write_text(json.dumps(computed_stats, indent=2) + "\n")
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output / "joint-distributions.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    axes[0].hist([[ep["length"] for ep in source] for source in (official_episodes, episodes)],
                 bins=20, label=["Official", "Generated"])
    axes[0].set(xlabel="Frames per episode", ylabel="Episodes", title="Trajectory lengths (50 fps)")
    axes[0].legend()
    for axis, key in zip(axes[1:], FEATURES):
        rows = comparison[comparison.feature == key].dropna(subset=["mean_shift_in_official_std"])
        rows = rows.loc[rows.mean_shift_in_official_std.abs().nlargest(10).index].iloc[::-1]
        axis.barh(rows.joint, rows.mean_shift_in_official_std)
        axis.set(xlabel="Mean shift / official std", title=key)
    figure.savefig(args.output / "comparison.png", dpi=150)
    plt.close(figure)
    report = ["# 数据集对比", "", "| 指标 | 官方数据 | 自己生成 |", "| --- | ---: | ---: |"]
    labels = {"episodes": "轨迹数", "frames": "总帧数", "min": "最短轨迹（帧）", "mean": "平均长度（帧）",
              "median": "长度中位数（帧）", "p95": "长度 P95（帧）", "max": "最长轨迹（帧）"}
    for metric in summary["official"]:
        report.append(f"| {labels[metric]} | {summary['official'][metric]:.2f} | {summary['generated'][metric]:.2f} |")
    report += ["", "![数据分布](comparison.png)", "",
               "验证通过：HDF5 成功标记、转换前后轨迹长度与条数、43 维状态和动作、关节顺序、索引、",
               "时间戳、归一化统计，以及全部视频的帧数、640×480 分辨率和 50 fps。",
               "", "逐关节均值、标准差、范围和分位数见 [CSV](joint-distributions.csv)。",
               "官方对照使用固定 revision 公开的统计元数据，未重新计算其状态和动作统计。",
               "标准化均值偏移跳过官方标准差 ≤ 1e-6 的关节。图中展示偏移最大的 10 个关节。",
               "该比较描述逐维边缘分布，不能用于判断时序相似度或策略质量。"]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Validation passed. Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
