"""Single-checkpoint evaluation entry point.

This module adapts the generic evaluator to result directories shaped like:

```
step_000355617/
  final_dataset_eval_10_sampled/
    videos/
      gt/
      pred/
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .io import count_video_frames, discover_video_pairs, pairing_report_to_jsonable
from .serialization import write_json


DEFAULT_METRICS = ("psnr", "ssim", "lpips", "fid", "fvd")
GT_SUFFIXES = ("_gt", "_ground_truth", "-gt")
GENERATED_SUFFIXES = (
    "_generated",
    "_gen",
    "_pred",
    "_prediction",
    "-generated",
    "-gen",
    "-pred",
)
GENERATED_DIR_NAMES = ("pred", "generated", "gen")


@dataclass(frozen=True)
class StepVideoDirs:
    videos_root: Path
    gt_dir: Path
    generated_dir: Path


def _parse_metrics(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = raw
    metrics = [str(value).strip().lower() for value in values if str(value).strip()]
    if not metrics:
        raise ValueError("At least one metric must be selected")
    return metrics


def _resolve_step_dir(step_dir: str | Path) -> Path:
    resolved = Path(step_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Step directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected a step directory: {resolved}")
    return resolved


def _generated_dir_for(videos_root: Path, preferred_name: str | None) -> Path | None:
    if preferred_name:
        candidate = videos_root / preferred_name
        return candidate if candidate.is_dir() else None

    for name in GENERATED_DIR_NAMES:
        candidate = videos_root / name
        if candidate.is_dir():
            return candidate
    return None


def _is_video_root(videos_root: Path, generated_subdir: str | None) -> bool:
    return (videos_root / "gt").is_dir() and _generated_dir_for(
        videos_root, generated_subdir
    ) is not None


def infer_step_video_dirs(
    step_dir: str | Path,
    *,
    dataset_subdir: str | None = None,
    generated_subdir: str | None = None,
) -> StepVideoDirs:
    """Infer GT and generated video directories from a checkpoint step dir."""

    step_path = _resolve_step_dir(step_dir)
    candidates: list[Path] = []

    if dataset_subdir:
        candidates.append(step_path / dataset_subdir / "videos")
    else:
        candidates.append(step_path / "videos")
        candidates.extend(sorted(step_path.glob("final_dataset*/videos")))
        candidates.extend(sorted(step_path.glob("*/videos")))

    seen: set[Path] = set()
    valid: list[StepVideoDirs] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_dir() or not _is_video_root(candidate, generated_subdir):
            continue
        generated_dir = _generated_dir_for(candidate, generated_subdir)
        if generated_dir is None:
            continue
        valid.append(
            StepVideoDirs(
                videos_root=candidate,
                gt_dir=candidate / "gt",
                generated_dir=generated_dir,
            )
        )

    if not valid:
        raise FileNotFoundError(
            "Could not find a videos root with 'gt' and generated videos under "
            f"{step_path}. Expected something like "
            "<step>/final_dataset_eval_*/videos/{gt,pred}."
        )
    if len(valid) > 1:
        roots = "\n".join(f"  - {item.videos_root}" for item in valid)
        raise ValueError(
            "Found multiple candidate video roots. Pass --dataset-subdir or "
            f"--generated-subdir to disambiguate:\n{roots}"
        )
    return valid[0]


def _frame_count_report(
    *,
    gt_dir: Path,
    generated_dir: Path,
    video_glob: str,
    max_videos: int | None,
    fail_on_missing_pairs: bool,
) -> dict[str, Any]:
    pairing_report = discover_video_pairs(
        gt_dir,
        generated_dir,
        pattern=video_glob,
        gt_suffixes=GT_SUFFIXES,
        generated_suffixes=GENERATED_SUFFIXES,
        case_sensitive=True,
        max_videos=max_videos,
        fail_on_missing_pairs=fail_on_missing_pairs,
    )

    records: list[dict[str, Any]] = []
    totals = {
        "num_pairs_discovered": len(pairing_report.pairs),
        "num_missing_generated": len(pairing_report.missing_generated),
        "num_missing_gt": len(pairing_report.missing_gt),
        "num_duplicate_gt": len(pairing_report.duplicate_gt),
        "num_duplicate_generated": len(pairing_report.duplicate_generated),
        "num_equal_frame_pairs": 0,
        "num_frame_mismatches": 0,
        "num_frame_count_errors": 0,
        "total_gt_frames": 0,
        "total_generated_frames": 0,
    }

    for pair in pairing_report.pairs:
        record: dict[str, Any] = {
            "name": pair.name,
            "gt_path": pair.gt_path,
            "generated_path": pair.generated_path,
        }
        try:
            gt_frames = count_video_frames(pair.gt_path)
            generated_frames = count_video_frames(pair.generated_path)
            record["gt_num_frames"] = gt_frames
            record["generated_num_frames"] = generated_frames
            totals["total_gt_frames"] += gt_frames
            totals["total_generated_frames"] += generated_frames
            if gt_frames == generated_frames:
                record["status"] = "ok"
                totals["num_equal_frame_pairs"] += 1
            else:
                record["status"] = "frame_mismatch"
                record["error"] = (
                    f"Frame-count mismatch: gt={gt_frames}, "
                    f"generated={generated_frames}"
                )
                totals["num_frame_mismatches"] += 1
        except Exception as exc:
            record["status"] = "frame_count_error"
            record["error"] = str(exc)
            totals["num_frame_count_errors"] += 1
        records.append(record)

    return {
        "gt_dir": gt_dir,
        "generated_dir": generated_dir,
        "video_glob": video_glob,
        "totals": totals,
        "pairing": pairing_report_to_jsonable(pairing_report),
        "records": records,
    }


def _load_eval_config() -> Any:
    from omegaconf import OmegaConf

    config_path = Path(__file__).resolve().parent / "configs" / "eval.yaml"
    return OmegaConf.load(config_path)


def _build_eval_config(
    *,
    step_dir: Path,
    video_dirs: StepVideoDirs,
    output_dir: Path,
    metrics: Sequence[str],
    max_videos: int | None,
    video_glob: str,
    device: str,
    skip_first_n_frames: int,
    continue_on_metric_error: bool,
    batch_size_lpips: int,
    batch_size_fid: int,
    batch_size_fvd: int,
    lpips_net: str,
    fvd_target_len: int | None,
    fail_on_missing_pairs: bool,
) -> Any:
    cfg = _load_eval_config()
    cfg.input_dir = str(video_dirs.videos_root)
    cfg.gt_dir = str(video_dirs.gt_dir)
    cfg.generated_dir = str(video_dirs.generated_dir)
    cfg.output_dir = str(output_dir)
    cfg.video_glob = video_glob
    cfg.max_videos = max_videos
    cfg.metrics = list(metrics)
    cfg.device = device
    cfg.skip_first_n_frames = skip_first_n_frames
    cfg.continue_on_metric_error = continue_on_metric_error
    cfg.pairing.gt_suffixes = list(GT_SUFFIXES)
    cfg.pairing.generated_suffixes = list(GENERATED_SUFFIXES)
    cfg.strict.require_equal_frames = True
    cfg.strict.fail_on_missing_pairs = fail_on_missing_pairs
    cfg.strict.fail_on_frame_mismatch = False
    cfg.batch_size.lpips = batch_size_lpips
    cfg.batch_size.fid = batch_size_fid
    cfg.batch_size.fvd = batch_size_fvd
    cfg.lpips.net = lpips_net
    cfg.fvd.target_len = fvd_target_len
    cfg.step_dir = str(step_dir)
    return cfg


def _print_totals(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key, value in values.items():
        print(f"  {key}: {value}")


def _metric_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    scalar = summary.get("scalar_metrics", {})
    distribution = summary.get("distribution_metrics", {})
    snapshot: dict[str, Any] = {}
    for key in ("psnr_mean", "ssim_mean", "lpips_mean", "mse_mean"):
        if key in scalar and "mean" in scalar[key]:
            snapshot[key] = scalar[key]["mean"]
    for key in ("fid", "fvd"):
        if key in distribution:
            snapshot[key] = distribution[key]
    return snapshot


def evaluate_step_dir(
    step_dir: str | Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    output_dir: str | Path | None = None,
    summary_name: str = "evaluation_summary.json",
    summary_path: str | Path | None = None,
    dataset_subdir: str | None = None,
    generated_subdir: str | None = None,
    max_videos: int | None = None,
    video_glob: str = "**/*.mp4",
    device: str = "auto",
    skip_first_n_frames: int = 0,
    continue_on_metric_error: bool = False,
    batch_size_lpips: int = 64,
    batch_size_fid: int = 64,
    batch_size_fvd: int = 8,
    lpips_net: str = "alex",
    fvd_target_len: int | None = None,
    allow_frame_mismatch: bool = False,
    allow_missing_pairs: bool = False,
) -> dict[str, Any]:
    """Evaluate a single checkpoint step directory and write summaries."""

    step_path = _resolve_step_dir(step_dir)
    selected_metrics = _parse_metrics(metrics)
    video_dirs = infer_step_video_dirs(
        step_path,
        dataset_subdir=dataset_subdir,
        generated_subdir=generated_subdir,
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else step_path / "video_model_eval"
    )
    resolved_summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path
        else step_path / summary_name
    )

    frame_report = _frame_count_report(
        gt_dir=video_dirs.gt_dir,
        generated_dir=video_dirs.generated_dir,
        video_glob=video_glob,
        max_videos=max_videos,
        fail_on_missing_pairs=not allow_missing_pairs,
    )
    frame_report_path = resolved_output_dir / "frame_count_report.json"
    write_json(frame_report_path, frame_report)
    _print_totals("Frame-count totals:", frame_report["totals"])

    frame_totals = frame_report["totals"]
    if (
        frame_totals["num_frame_mismatches"] > 0
        or frame_totals["num_frame_count_errors"] > 0
    ) and not allow_frame_mismatch:
        summary = {
            "step_dir": step_path,
            "status": "failed_frame_count_check",
            "videos_root": video_dirs.videos_root,
            "gt_dir": video_dirs.gt_dir,
            "generated_dir": video_dirs.generated_dir,
            "metrics_requested": selected_metrics,
            "frame_count_report": frame_report_path,
            "frame_count_totals": frame_totals,
            "output_dir": resolved_output_dir,
        }
        write_json(resolved_summary_path, summary)
        raise RuntimeError(
            "Frame-count check failed. Fix mismatched videos or rerun with "
            f"--allow-frame-mismatch. Summary: {resolved_summary_path}"
        )

    cfg = _build_eval_config(
        step_dir=step_path,
        video_dirs=video_dirs,
        output_dir=resolved_output_dir,
        metrics=selected_metrics,
        max_videos=max_videos,
        video_glob=video_glob,
        device=device,
        skip_first_n_frames=skip_first_n_frames,
        continue_on_metric_error=continue_on_metric_error,
        batch_size_lpips=batch_size_lpips,
        batch_size_fid=batch_size_fid,
        batch_size_fvd=batch_size_fvd,
        lpips_net=lpips_net,
        fvd_target_len=fvd_target_len,
        fail_on_missing_pairs=not allow_missing_pairs,
    )
    from .evaluator import evaluate_from_config

    eval_summary = evaluate_from_config(cfg)
    snapshot = _metric_snapshot(eval_summary)

    top_level_summary = {
        "step_dir": step_path,
        "status": "ok",
        "videos_root": video_dirs.videos_root,
        "gt_dir": video_dirs.gt_dir,
        "generated_dir": video_dirs.generated_dir,
        "metrics_requested": selected_metrics,
        "metric_snapshot": snapshot,
        "frame_count_report": frame_report_path,
        "frame_count_totals": frame_totals,
        "output_dir": resolved_output_dir,
        "detailed_summary": resolved_output_dir / "summary.json",
        "evaluation": eval_summary,
    }
    write_json(resolved_summary_path, top_level_summary)

    eval_totals = {
        "num_pairs_discovered": eval_summary.get("num_pairs_discovered"),
        "num_videos_evaluated": eval_summary.get("num_videos_evaluated"),
        "num_videos_skipped": eval_summary.get("num_videos_skipped"),
        "num_missing_generated": eval_summary.get("num_missing_generated"),
        "num_missing_gt": eval_summary.get("num_missing_gt"),
    }
    _print_totals("Evaluation totals:", eval_totals)
    _print_totals("Metric summary:", snapshot)
    print(f"Summary written to: {resolved_summary_path}")
    return top_level_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one checkpoint step directory containing "
            "final_dataset_eval_*/videos/{gt,pred}."
        )
    )
    parser.add_argument("step_dir", help="Path to a single step_... directory")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics. Valid: psnr,ssim,lpips,fid,fvd,mse.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Detailed output dir. Defaults to <step_dir>/video_model_eval.",
    )
    parser.add_argument(
        "--summary-name",
        default="evaluation_summary.json",
        help="Top-level summary filename written inside step_dir.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help=(
            "Explicit top-level summary path. Overrides --summary-name and "
            "allows writing summaries outside step_dir."
        ),
    )
    parser.add_argument(
        "--dataset-subdir",
        default=None,
        help="Dataset subdir under step_dir, e.g. final_dataset_eval_10_sampled.",
    )
    parser.add_argument(
        "--generated-subdir",
        default=None,
        help="Generated-video subdir under videos/. Defaults to pred/generated/gen.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-glob", default="**/*.mp4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-first-n-frames", type=int, default=0)
    parser.add_argument("--continue-on-metric-error", action="store_true")
    parser.add_argument("--batch-size-lpips", type=int, default=64)
    parser.add_argument("--batch-size-fid", type=int, default=64)
    parser.add_argument("--batch-size-fvd", type=int, default=8)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument(
        "--fvd-target-len",
        type=int,
        default=None,
        help="Pad/truncate FVD clips to this frame count. Defaults to original length.",
    )
    parser.add_argument(
        "--allow-frame-mismatch",
        action="store_true",
        help="Continue after frame-count mismatches; mismatched pairs are skipped.",
    )
    parser.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="Continue when GT or generated videos are missing their counterpart.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    evaluate_step_dir(
        args.step_dir,
        metrics=_parse_metrics(args.metrics),
        output_dir=args.output_dir,
        summary_name=args.summary_name,
        summary_path=args.summary_path,
        dataset_subdir=args.dataset_subdir,
        generated_subdir=args.generated_subdir,
        max_videos=args.max_videos,
        video_glob=args.video_glob,
        device=args.device,
        skip_first_n_frames=args.skip_first_n_frames,
        continue_on_metric_error=args.continue_on_metric_error,
        batch_size_lpips=args.batch_size_lpips,
        batch_size_fid=args.batch_size_fid,
        batch_size_fvd=args.batch_size_fvd,
        lpips_net=args.lpips_net,
        fvd_target_len=args.fvd_target_len,
        allow_frame_mismatch=args.allow_frame_mismatch,
        allow_missing_pairs=args.allow_missing_pairs,
    )


if __name__ == "__main__":
    main()
