# Changelog

## 0.1.5

- Add centroid-driven vertebral level segmentation with
  `--level-only --centroids CENTROIDS_JSON`.
- Validate supplied centroid labels and coordinates against the input CT
  geometry while preserving stable localization metadata.
- Keep the existing localization-backed `--level-only` behavior when no
  centroid artifact is supplied.
