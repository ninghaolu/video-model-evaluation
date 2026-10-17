# video-model-evaluation

Evaluation code for comparing generated video model MP4 videos
against ground-truth MP4 videos. The evaluator pairs trajectories, checks frame
counts, computes selected metrics, and writes both per-video and aggregate
results.

## Expected Input

By default, the input directory should contain:

```text
<input_dir>/
  gt/
    trajectory_000.mp4
    trajectory_001.mp4
  generated/
    trajectory_000.mp4
    trajectory_001.mp4
```

Videos are paired by relative path and filename stem. Common suffixes such as
`_gt`, `_generated`, `_gen`, and `_pred` are stripped, so
`gt/task/000_gt.mp4` will match `generated/task/000_pred.mp4`.

## Install

```bash
cd /path/to/video-model-evaluation
pip install -e .
```

FID uses torchvision pretrained weights. FVD uses a TorchScript I3D checkpoint;
set `VIDEO_MODEL_EVAL_FVD_MODEL_PATH` to a local checkpoint or let the first run
download it into the user cache.

## Run

Evaluate one checkpoint/result directory:

```bash
video-model-eval-step /path/to/step_or_eval_dir
```

Or run the Hydra/config entry point directly:

```bash
video-model-eval \
  input_dir=/path/to/videos \
  output_dir=/path/to/eval_outputs \
  metrics='[psnr,ssim,lpips,fid,fvd]'
```

For Slurm sweeps, use the placeholder sample as a starting point and keep local
copies ignored:

```bash
cp scripts/submit_metrics_sample.sh scripts/submit_metrics_local.sh
bash scripts/submit_metrics_local.sh --dry-run
```

## Configuration

The default config is in `configs/eval.yaml` and is also packaged under
`src/video_model_evaluation/configs/eval.yaml`.

Important options:

- `metrics`: any subset of `ssim`, `psnr`, `lpips`, `fid`, `fvd`, `mse`.
- `strict.require_equal_frames`: skip mismatched trajectories when `true`.
- `strict.fail_on_frame_mismatch`: raise immediately on a frame mismatch.
- `strict.require_equal_resolution`: skip resolution mismatches when `true`.
- `skip_first_n_frames`: ignore leading context frames during metric computation.
- `continue_on_metric_error`: record metric failures instead of aborting.

## Outputs

The output directory contains:

- `summary.json`: aggregate scalar metrics, FID/FVD, counts, and errors.
- `per_video_metrics.jsonl`: one record per discovered pair.
- `per_video/*.json`: detailed metrics for each trajectory.
- `pairing_report.json`: matched, missing, and duplicate video diagnostics.
- `skipped_videos.json`: frame mismatches, resolution mismatches, and read errors.
- `evaluation.log`: console-equivalent log output.

## Metric Semantics

Videos are first paired by normalized relative path. Suffixes such as `_gt`,
`_generated`, `_gen`, and `_pred` are stripped before matching. By default,
missing pairs, frame-count mismatches, and resolution mismatches are reported
and skipped or failed according to the `strict.*` config values before metrics
are aggregated.

Scalar metrics are computed per aligned video pair:

- `mse_mean`: mean squared error over all pixels and frames in one video pair.
- `psnr_mean`: mean of per-frame PSNR values for one video pair.
- `psnr_global`: PSNR from that video pair's global mean MSE.
- `ssim_mean`: mean of per-frame RGB SSIM values with `data_range=1.0`.
- `lpips_mean`: mean of per-frame LPIPS distances after mapping RGB frames from
  `[0, 1]` to `[-1, 1]`.

`summary.json` aggregates scalar values by taking the mean/std/min/max across
video pairs. Therefore `scalar_metrics.psnr_global.mean` is the mean of
per-video global PSNR values; it is not one dataset-wide PSNR from all pixels in
all videos.

Distribution metrics are computed once per eval run:

- `fid`: Frechet distance between InceptionV3 feature distributions for all
  valid GT frames and all valid generated frames.
- `fvd`: Frechet distance between TorchScript I3D feature distributions for all
  valid GT videos and all valid generated videos. This is a distribution-level
  score over the whole eval set, not an average of per-video FVD values.

FVD uses each clip's original frame count unless `fvd.target_len` or
`--fvd-target-len` is set, in which case clips are padded or truncated to the
target length before I3D feature extraction.
