"""Video discovery, pairing, and decoding utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoPair:
    """A matched ground-truth/generated trajectory pair."""

    name: str
    gt_path: Path
    generated_path: Path


@dataclass(frozen=True)
class PairingReport:
    """Diagnostics from video-pair discovery."""

    pairs: list[VideoPair]
    missing_generated: list[str]
    missing_gt: list[str]
    duplicate_gt: list[str]
    duplicate_generated: list[str]


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_video_dirs(
    input_dir: str | Path,
    gt_dir: str | Path | None,
    generated_dir: str | Path | None,
    gt_subdir: str,
    generated_subdir: str,
) -> tuple[Path, Path]:
    """Resolve GT and generated video directories from config values."""

    root = _as_path(input_dir)
    resolved_gt = _as_path(gt_dir) if gt_dir else root / gt_subdir
    resolved_generated = (
        _as_path(generated_dir) if generated_dir else root / generated_subdir
    )
    return resolved_gt, resolved_generated


def _remove_known_suffix(stem: str, suffixes: Sequence[str]) -> str:
    for suffix in suffixes:
        if suffix and stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _normalization_key(
    path: Path,
    root: Path,
    suffixes: Sequence[str],
    *,
    case_sensitive: bool,
) -> str:
    rel = path.relative_to(root)
    normalized_stem = _remove_known_suffix(rel.stem, suffixes)
    key = (rel.parent / normalized_stem).as_posix()
    return key if case_sensitive else key.lower()


def _discover_mp4s(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _index_videos(
    root: Path,
    pattern: str,
    suffixes: Sequence[str],
    *,
    case_sensitive: bool,
) -> tuple[dict[str, Path], list[str]]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in _discover_mp4s(root, pattern):
        key = _normalization_key(
            path, root, suffixes, case_sensitive=case_sensitive
        )
        if key in index:
            duplicates.append(key)
            logger.warning(
                "Duplicate normalized video key '%s': %s and %s",
                key,
                index[key],
                path,
            )
            continue
        index[key] = path

    return index, duplicates


def discover_video_pairs(
    gt_dir: str | Path,
    generated_dir: str | Path,
    *,
    pattern: str = "**/*.mp4",
    gt_suffixes: Sequence[str] = (),
    generated_suffixes: Sequence[str] = (),
    case_sensitive: bool = True,
    max_videos: int | None = None,
    fail_on_missing_pairs: bool = False,
) -> PairingReport:
    """Discover matching videos by normalized relative path.

    Examples:
    ``gt/task/000_gt.mp4`` and ``generated/task/000_pred.mp4`` both normalize to
    ``task/000`` when the suffix lists contain ``_gt`` and ``_pred``.
    """

    gt_root = _as_path(gt_dir)
    generated_root = _as_path(generated_dir)

    gt_index, duplicate_gt = _index_videos(
        gt_root, pattern, gt_suffixes, case_sensitive=case_sensitive
    )
    generated_index, duplicate_generated = _index_videos(
        generated_root,
        pattern,
        generated_suffixes,
        case_sensitive=case_sensitive,
    )

    gt_keys = set(gt_index)
    generated_keys = set(generated_index)
    matched_keys = sorted(gt_keys & generated_keys)

    missing_generated = sorted(gt_keys - generated_keys)
    missing_gt = sorted(generated_keys - gt_keys)

    if max_videos is not None:
        matched_keys = matched_keys[:max_videos]

    if (missing_generated or missing_gt) and fail_on_missing_pairs:
        raise ValueError(
            "Missing video pairs: "
            f"{len(missing_generated)} missing generated, "
            f"{len(missing_gt)} missing ground truth"
        )

    pairs = [
        VideoPair(name=key, gt_path=gt_index[key], generated_path=generated_index[key])
        for key in matched_keys
    ]

    logger.info("Ground-truth video dir: %s", gt_root)
    logger.info("Generated video dir: %s", generated_root)
    logger.info(
        "Discovered %d matched pairs (%d GT-only, %d generated-only)",
        len(pairs),
        len(missing_generated),
        len(missing_gt),
    )

    return PairingReport(
        pairs=pairs,
        missing_generated=missing_generated,
        missing_gt=missing_gt,
        duplicate_gt=duplicate_gt,
        duplicate_generated=duplicate_generated,
    )


def _ensure_rgb_uint8(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"Expected video with shape (T,H,W,C), got {frames.shape}")
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif frames.shape[-1] == 4:
        frames = frames[..., :3]
    elif frames.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {frames.shape[-1]}")

    if frames.dtype == np.uint8:
        return frames

    if np.issubdtype(frames.dtype, np.floating):
        if frames.max(initial=0.0) <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255)
    return frames.astype(np.uint8)


def read_video(path: str | Path) -> np.ndarray:
    """Read an MP4 as ``uint8`` RGB frames with shape ``(T, H, W, 3)``."""

    video_path = _as_path(path)
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu(0))
        frames = reader.get_batch(range(len(reader))).asnumpy()
        return _ensure_rgb_uint8(frames)
    except ImportError:
        logger.debug("decord is not installed; falling back to imageio")
    except Exception as exc:
        logger.warning(
            "decord failed to read %s (%s); falling back to imageio",
            video_path,
            exc,
        )

    import imageio.v3 as iio

    frames = [np.asarray(frame) for frame in iio.imiter(video_path)]
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    return _ensure_rgb_uint8(np.stack(frames, axis=0))


def count_video_frames(path: str | Path) -> int:
    """Count frames using decord when available, otherwise decode with imageio."""

    video_path = _as_path(path)
    try:
        from decord import VideoReader, cpu

        return int(len(VideoReader(str(video_path), ctx=cpu(0))))
    except Exception:
        return int(read_video(video_path).shape[0])


def to_float_video(video: np.ndarray) -> np.ndarray:
    """Convert RGB video frames to ``float32`` in ``[0, 1]``."""

    video = _ensure_rgb_uint8(video)
    return video.astype(np.float32) / 255.0


def resize_video_to(video: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize a video to ``(height, width)`` using bilinear interpolation."""

    import torch
    import torch.nn.functional as F

    target_h, target_w = size_hw
    x = torch.from_numpy(to_float_video(video)).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).clamp(0, 1).numpy()
    return (x * 255.0).round().astype(np.uint8)


def safe_name(name: str) -> str:
    """Make a normalized pair name safe for filenames."""

    return name.replace("/", "__").replace("\\", "__").replace(" ", "_")


def pairing_report_to_jsonable(report: PairingReport) -> dict:
    return {
        "num_pairs": len(report.pairs),
        "pairs": [
            {
                "name": pair.name,
                "gt_path": str(pair.gt_path),
                "generated_path": str(pair.generated_path),
            }
            for pair in report.pairs
        ],
        "missing_generated": report.missing_generated,
        "missing_gt": report.missing_gt,
        "duplicate_gt": report.duplicate_gt,
        "duplicate_generated": report.duplicate_generated,
    }
