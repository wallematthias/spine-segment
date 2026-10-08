# Changelog

## 0.1.6 (2026-10-07)

- Rebalance native PyTorch sequence selection toward confident detections so
  curved-spine spacing does not replace strong candidates with neighboring-label
  alternatives. Keep the existing geometry penalties and boundary exclusion.
- Expose `--sequence-confidence-weight` (default `0.8`, historical `0.2`) and
  record it in localization metadata.
- Add regression coverage for retained curved-spine detections, implausible
  candidate connections, invalid weights, and CLI propagation.

## 0.1.5

- Add centroid-driven vertebral level segmentation with
  `--level-only --centroids CENTROIDS_JSON`.
- Validate supplied centroid labels and coordinates against the input CT
  geometry while preserving stable localization metadata.
- Preserve supplied landmarks that yield no output voxels and mark each
  centroid-driven result as `segmented` or `missing`.
- Keep the existing localization-backed `--level-only` behavior when no
  centroid artifact is supplied.
