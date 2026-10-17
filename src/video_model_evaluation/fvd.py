"""FVD utilities backed by the TorchScript I3D checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch

try:
    from scipy import linalg as scipy_linalg
except ImportError:  # pragma: no cover - scipy is a declared dependency.
    scipy_linalg = None


logger = logging.getLogger(__name__)

FVD_REPO_ID = "flateon/FVD-I3D-torchscript"
FVD_FILENAME = "i3d_torchscript.pt"
DEFAULT_FVD_MODEL_PATH = os.environ.get(
    "VIDEO_MODEL_EVAL_FVD_MODEL_PATH",
    str(Path.home() / ".cache" / "video_model_evaluation" / "fvd_i3d" / FVD_FILENAME),
)


def download_fvd_i3d_checkpoint(
    output_path: Union[str, os.PathLike] = DEFAULT_FVD_MODEL_PATH,
    *,
    force: bool = False,
) -> str:
    """Download the I3D TorchScript checkpoint used for FVD."""

    output = Path(output_path)
    if output.exists() and not force:
        return str(output)

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=FVD_REPO_ID, filename=FVD_FILENAME)
        tmp_path = output.with_suffix(output.suffix + ".tmp")
        with open(downloaded, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp_path, output)
    except Exception:
        url = f"https://huggingface.co/{FVD_REPO_ID}/resolve/main/{FVD_FILENAME}"
        fd, tmp_name = tempfile.mkstemp(prefix=output.name, suffix=".tmp", dir=output.parent)
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, tmp_name)
            os.replace(tmp_name, output)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    return str(output)


class FVDMetric:
    """Compute Frechet Video Distance from video-level I3D features."""

    def __init__(
        self,
        model_path: Union[str, os.PathLike] = DEFAULT_FVD_MODEL_PATH,
        *,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 8,
        target_len: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            download_fvd_i3d_checkpoint(self.model_path)

        device_name = str(device)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.batch_size = int(batch_size)
        self.target_len = target_len
        self.model = torch.jit.load(str(self.model_path), map_location=self.device).eval()
        logger.info("Initialized TorchScript I3D feature extractor for FVD")

    def compute(
        self,
        gt_videos: Sequence[np.ndarray] | np.ndarray,
        pred_videos: Sequence[np.ndarray] | np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> float:
        gt_video_list = self._as_video_list(gt_videos)
        pred_video_list = self._as_video_list(pred_videos)

        if len(gt_video_list) != len(pred_video_list):
            raise ValueError(
                f"GT/pred video count mismatch: {len(gt_video_list)} vs {len(pred_video_list)}"
            )
        if len(gt_video_list) < 2:
            raise ValueError("FVD requires at least two video pairs.")

        logger.info(
            "Extracting I3D features for FVD (gt=%d videos, generated=%d videos)",
            len(gt_video_list),
            len(pred_video_list),
        )
        gt_features = self.extract_features(gt_video_list, batch_size=batch_size)
        pred_features = self.extract_features(pred_video_list, batch_size=batch_size)
        return self.calculate_frechet_distance(gt_features, pred_features)

    @torch.no_grad()
    def extract_features(
        self,
        videos: Sequence[np.ndarray] | np.ndarray,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        video_list = self._as_video_list(videos)
        if not video_list:
            raise ValueError("FVD feature extraction requires at least one video.")

        effective_batch_size = int(batch_size or self.batch_size)
        features: list[np.ndarray] = []
        for start in range(0, len(video_list), effective_batch_size):
            batch_videos = video_list[start : start + effective_batch_size]
            batch = torch.cat([self._video_to_tensor(video) for video in batch_videos], dim=0)
            output = self._forward_i3d(batch)
            features.append(output.detach().cpu().float().numpy())
        return np.concatenate(features, axis=0)

    @staticmethod
    def _as_video_list(videos: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        if isinstance(videos, np.ndarray):
            if videos.ndim == 4:
                return [videos]
            if videos.ndim == 5:
                return [videos[index] for index in range(len(videos))]
            raise ValueError(
                f"Expected videos with shape (T,H,W,C) or (N,T,H,W,C), got {videos.shape}"
            )
        return list(videos)

    def _video_to_tensor(self, video: np.ndarray) -> torch.Tensor:
        """Convert one video from [T, H, W, C] uint8/[0,1] to [1, C, T, H, W]."""

        if video.ndim != 4 or video.shape[-1] < 3:
            raise ValueError(f"Expected video shape [T, H, W, C>=3], got {video.shape}")
        if video.shape[0] == 0:
            raise ValueError("Expected video with at least one frame.")

        tensor = torch.as_tensor(video[..., :3]).float()
        min_value = float(tensor.min())
        max_value = float(tensor.max())
        if min_value < 0:
            raise ValueError("Expected non-negative RGB values for FVD video input.")
        if max_value <= 1.0:
            tensor = tensor * 255.0
        elif max_value > 255.0:
            raise ValueError(f"Expected RGB values in [0, 255], got max value {max_value}")
        if self.target_len is not None:
            target_len = int(self.target_len)
            if target_len <= 0:
                raise ValueError(f"target_len must be positive or None, got {target_len}")
            if tensor.shape[0] < target_len:
                pad = tensor[-1:].expand(target_len - tensor.shape[0], -1, -1, -1)
                tensor = torch.cat([tensor, pad], dim=0)
            elif tensor.shape[0] > target_len:
                tensor = tensor[:target_len]
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        return tensor.to(self.device)

    def _forward_i3d(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            output = self.model(batch, rescale=True, resize=True, return_features=True)
        except TypeError:
            output = self.model(batch)

        if isinstance(output, dict):
            tensors = [value for value in output.values() if torch.is_tensor(value)]
            if not tensors:
                raise TypeError("FVD I3D model returned a dict without tensor values.")
            output = tensors[0]
        elif isinstance(output, (tuple, list)):
            output = output[0]

        if not torch.is_tensor(output):
            raise TypeError(f"FVD I3D model returned unsupported output type: {type(output)}")

        if output.ndim > 2:
            output = output.mean(dim=tuple(range(2, output.ndim)))
        return output.flatten(start_dim=1)

    @staticmethod
    def calculate_frechet_distance(features_real: np.ndarray, features_fake: np.ndarray) -> float:
        if features_real.shape[0] < 2 or features_fake.shape[0] < 2:
            raise ValueError("Frechet distance needs at least two feature vectors per distribution.")

        features_real = np.asarray(features_real, dtype=np.float64)
        features_fake = np.asarray(features_fake, dtype=np.float64)

        mu_real = np.mean(features_real, axis=0)
        mu_fake = np.mean(features_fake, axis=0)
        sigma_real = np.cov(features_real, rowvar=False)
        sigma_fake = np.cov(features_fake, rowvar=False)

        diff = mu_real - mu_fake
        trace_covmean = FVDMetric._trace_sqrt_product(sigma_real, sigma_fake)
        fvd = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * trace_covmean
        return float(np.real(fvd))

    @staticmethod
    def _trace_sqrt_product(sigma_real: np.ndarray, sigma_fake: np.ndarray) -> float:
        cov_prod = sigma_real.dot(sigma_fake)
        if scipy_linalg is not None:
            covmean, _ = scipy_linalg.sqrtm(cov_prod, disp=False)
            if not np.isfinite(covmean).all():
                eps = 1e-6
                offset = np.eye(sigma_real.shape[0]) * eps
                covmean = scipy_linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            return float(np.trace(covmean))

        eigvals = np.linalg.eigvals(cov_prod)
        if not np.isfinite(eigvals).all():
            eps = 1e-6
            offset = np.eye(sigma_real.shape[0]) * eps
            eigvals = np.linalg.eigvals((sigma_real + offset).dot(sigma_fake + offset))
        eigvals = np.real(eigvals)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        return float(np.sum(np.sqrt(eigvals)))


def compute_fvd(
    generated_videos: Sequence[np.ndarray] | np.ndarray,
    gt_videos: Sequence[np.ndarray] | np.ndarray,
    *,
    device: Union[str, torch.device],
    batch_size: int = 8,
    target_len: int | None = None,
    model_path: Union[str, os.PathLike] = DEFAULT_FVD_MODEL_PATH,
) -> float:
    """Compute Frechet Video Distance for generated and ground-truth videos."""

    return FVDMetric(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        target_len=target_len,
    ).compute(
        gt_videos,
        generated_videos,
        batch_size=batch_size,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="FVD I3D utility")
    parser.add_argument("command", choices=["download"], help="Utility command to run.")
    parser.add_argument("--output_path", default=DEFAULT_FVD_MODEL_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        print(download_fvd_i3d_checkpoint(args.output_path, force=args.force))


if __name__ == "__main__":
    _main()
