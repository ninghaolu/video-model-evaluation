"""Image and frame-distribution metrics for video model evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

logger = logging.getLogger(__name__)


def _ensure_batch(video: np.ndarray) -> np.ndarray:
    if video.ndim == 4:
        return video[np.newaxis]
    if video.ndim == 5:
        return video
    raise ValueError(f"Expected video shape (T,H,W,C) or (N,T,H,W,C), got {video.shape}")


def _to_float(video: np.ndarray) -> np.ndarray:
    if video.dtype == np.uint8:
        return video.astype(np.float32) / 255.0
    video = video.astype(np.float32)
    if video.max(initial=0.0) > 1.0:
        video = video / 255.0
    return np.clip(video, 0.0, 1.0)


def compute_psnr(
    gt_video: np.ndarray,
    generated_video: np.ndarray,
    *,
    per_frame: bool = True,
) -> dict[str, float | list[float]]:
    """Compute MSE and PSNR over aligned videos.

    ``psnr_mean`` is the mean of per-frame PSNR values. ``psnr_global`` is
    computed from the globally aggregated MSE across all pixels and frames.
    """

    gt = _to_float(gt_video)
    generated = _to_float(generated_video)

    mse_per_frame = np.mean((gt - generated) ** 2, axis=(1, 2, 3))
    with np.errstate(divide="ignore"):
        psnr_per_frame = 10.0 * np.log10(1.0 / mse_per_frame)
    psnr_per_frame = np.where(mse_per_frame == 0, np.inf, psnr_per_frame)

    global_mse = float(np.mean(mse_per_frame))
    psnr_global = float("inf") if global_mse == 0 else float(10.0 * np.log10(1.0 / global_mse))

    result: dict[str, float | list[float]] = {
        "mse_mean": float(np.mean(mse_per_frame)),
        "mse_std": float(np.std(mse_per_frame)),
        "psnr_mean": float(np.mean(psnr_per_frame)),
        "psnr_std": float(np.std(psnr_per_frame)),
        "psnr_global": psnr_global,
    }
    if per_frame:
        result["mse_per_frame"] = mse_per_frame.astype(float).tolist()
        result["psnr_per_frame"] = psnr_per_frame.astype(float).tolist()
    return result


def _ssim_win_size(height: int, width: int) -> int | None:
    min_dim = min(height, width)
    if min_dim < 7:
        if min_dim < 3:
            raise ValueError(
                f"SSIM requires frames at least 3x3 pixels, got {height}x{width}"
            )
        return min_dim if min_dim % 2 == 1 else min_dim - 1
    return None


def compute_ssim(
    gt_video: np.ndarray,
    generated_video: np.ndarray,
    *,
    per_frame: bool = True,
) -> dict[str, float | list[float]]:
    """Compute Structural Similarity Index over aligned RGB videos."""

    from skimage.metrics import structural_similarity

    gt = _to_float(gt_video)
    generated = _to_float(generated_video)
    win_size = _ssim_win_size(gt.shape[1], gt.shape[2])

    values: list[float] = []
    for gt_frame, generated_frame in zip(gt, generated):
        kwargs = {"channel_axis": 2, "data_range": 1.0}
        if win_size is not None:
            kwargs["win_size"] = win_size
        values.append(float(structural_similarity(gt_frame, generated_frame, **kwargs)))

    result: dict[str, float | list[float]] = {
        "ssim_mean": float(np.mean(values)),
        "ssim_std": float(np.std(values)),
    }
    if per_frame:
        result["ssim_per_frame"] = values
    return result


class LPIPSMetric:
    """LPIPS metric wrapper with batched frame evaluation."""

    def __init__(self, *, device: torch.device, net: str = "alex") -> None:
        import lpips

        self.device = device
        self.net = net
        self.model = lpips.LPIPS(net=net).to(device).eval()
        logger.info("Initialized LPIPS with %s backbone", net)

    def compute(
        self,
        gt_video: np.ndarray,
        generated_video: np.ndarray,
        *,
        batch_size: int = 64,
        per_frame: bool = True,
    ) -> dict[str, float | list[float]]:
        gt = torch.from_numpy(_to_float(gt_video)).permute(0, 3, 1, 2).float()
        generated = torch.from_numpy(_to_float(generated_video)).permute(0, 3, 1, 2).float()

        gt = gt * 2.0 - 1.0
        generated = generated * 2.0 - 1.0

        values: list[float] = []
        with torch.no_grad():
            for start in range(0, gt.shape[0], batch_size):
                end = start + batch_size
                gt_batch = gt[start:end].to(self.device)
                generated_batch = generated[start:end].to(self.device)
                distances = self.model(gt_batch, generated_batch)
                values.extend(distances.reshape(-1).cpu().numpy().astype(float).tolist())

        result: dict[str, float | list[float]] = {
            "lpips_mean": float(np.mean(values)),
            "lpips_std": float(np.std(values)),
        }
        if per_frame:
            result["lpips_per_frame"] = values
        return result


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Compute Frechet distance between two multivariate Gaussians."""

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        logger.warning("FID covariance product was singular; adding %.1e to diagonal", eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            logger.warning("FID covariance square root has non-trivial imaginary part")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


class FIDMetric:
    """Frechet Inception Distance over flattened video frames."""

    def __init__(self, *, device: torch.device) -> None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = device
        self.model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()
        logger.info("Initialized InceptionV3 for FID")

    def _extract_feature_batch(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        tensors = []
        for frame in frames:
            frame_tensor = torch.from_numpy(_to_float(frame)).permute(2, 0, 1).unsqueeze(0)
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(299, 299),
                mode="bilinear",
                align_corners=False,
            )
            tensors.append(frame_tensor)

        batch = torch.cat(tensors, dim=0).float().to(self.device)
        batch = (batch - mean) / std
        return self.model(batch).cpu().numpy()

    def extract_features(
        self,
        frames: np.ndarray | Sequence[np.ndarray],
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        if isinstance(frames, np.ndarray):
            if frames.ndim != 4:
                raise ValueError(f"Expected frame batch shape (N,H,W,C), got {frames.shape}")
            frame_list = [frames[index] for index in range(len(frames))]
        else:
            frame_list = list(frames)

        features: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(frame_list), batch_size):
                features.append(
                    self._extract_feature_batch(frame_list[start : start + batch_size])
                )

        return np.concatenate(features, axis=0)

    def extract_video_features(
        self,
        videos: Sequence[np.ndarray],
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        frame_batches: list[np.ndarray] = []
        pending: list[np.ndarray] = []

        with torch.no_grad():
            for video in videos:
                for frame in _to_float(video):
                    pending.append(frame)
                    if len(pending) >= batch_size:
                        frame_batches.append(self._extract_feature_batch(pending))
                        pending = []
            if pending:
                frame_batches.append(self._extract_feature_batch(pending))

        return np.concatenate(frame_batches, axis=0)

    def compute(
        self,
        gt_videos: Sequence[np.ndarray],
        generated_videos: Sequence[np.ndarray],
        *,
        batch_size: int = 64,
    ) -> float:
        gt_num_frames = sum(int(video.shape[0]) for video in gt_videos)
        generated_num_frames = sum(int(video.shape[0]) for video in generated_videos)

        if gt_num_frames < 2 or generated_num_frames < 2:
            raise ValueError("FID requires at least two frames in each distribution")

        logger.info(
            "Extracting InceptionV3 features for FID (gt=%d frames, generated=%d frames)",
            gt_num_frames,
            generated_num_frames,
        )
        gt_features = self.extract_video_features(gt_videos, batch_size=batch_size)
        generated_features = self.extract_video_features(
            generated_videos, batch_size=batch_size
        )

        mu_gt, sigma_gt = gt_features.mean(axis=0), np.cov(gt_features, rowvar=False)
        mu_generated, sigma_generated = (
            generated_features.mean(axis=0),
            np.cov(generated_features, rowvar=False),
        )
        return _frechet_distance(mu_gt, sigma_gt, mu_generated, sigma_generated)


def aggregate_scalar_metrics(per_video_metrics: Sequence[dict]) -> dict[str, dict[str, float | int]]:
    """Aggregate scalar metric values from per-video metric dictionaries."""

    scalar_keys: set[str] = set()
    for metrics in per_video_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scalar_keys.add(key)

    summary: dict[str, dict[str, float | int]] = {}
    for key in sorted(scalar_keys):
        raw_values = [
            float(metrics[key])
            for metrics in per_video_metrics
            if isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
        ]
        finite_values = np.asarray([value for value in raw_values if np.isfinite(value)])
        if finite_values.size == 0:
            summary[key] = {"num_values": len(raw_values), "num_finite": 0}
            continue
        summary[key] = {
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "num_values": len(raw_values),
            "num_finite": int(finite_values.size),
        }
    return summary


@dataclass(frozen=True)
class MetricSelection:
    names: set[str]

    @classmethod
    def from_config(cls, metrics: str | Iterable[str]) -> "MetricSelection":
        if isinstance(metrics, str):
            names = {name.strip().lower() for name in metrics.split(",") if name.strip()}
        else:
            names = {str(name).strip().lower() for name in metrics if str(name).strip()}
        valid = {"psnr", "ssim", "lpips", "fid", "fvd", "mse"}
        unknown = sorted(names - valid)
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown}. Valid metrics: {sorted(valid)}")
        return cls(names=names)

    def wants(self, name: str) -> bool:
        return name in self.names
