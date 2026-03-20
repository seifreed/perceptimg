# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-20

### Added

- Initial release
- Perceptual, policy-driven image optimization
- Multi-format support: JPEG, PNG, WebP, AVIF, JXL, HEIF, GIF, TIFF, APNG
- Policy-based constraints: max_size_kb, min_ssim, preserve_text, preserve_faces
- Batch processing with parallel execution
- Enterprise features: checkpoint/resume, retry with backoff, rate limiting
- Prometheus metrics collection
- Clean Architecture implementation with dependency inversion
- CLI tool for command-line usage
- Python API for programmatic usage
- SSIM and PSNR perceptual metrics
- Content-aware analysis (edge density, color variance, text/face detection)

### Changed

- N/A (initial release)

### Deprecated

- N/A (initial release)

### Removed

- N/A (initial release)

### Fixed

- N/A (initial release)

### Security

- Use `secrets.randbelow()` for cryptographically secure random number generation in retry logic

## [Unreleased]

### Added

- N/A

### Changed

- N/A

### Deprecated

- N/A

### Removed

- N/A

### Fixed

- N/A

### Security

- N/A

---

[0.1.0]: https://github.com/seifreed/perceptimg/releases/tag/v0.1.0