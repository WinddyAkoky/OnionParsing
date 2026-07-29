# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-01

### Added

- Initial release of OnionParsing
- Coarse-grained and fine-grained layout detection
- XY-Cut reading order sorting
- Smart region cropping with column expansion and filtering
- OCR recognition via PaddleOCR-VL with vLLM acceleration
- NSP-based reading-order reordering
- Text direction detection (LTR/RTL)
- 4-level configuration system (runtime > env vars > YAML > defaults)
- Registry-based processor/model architecture
- CLI entry point (`onion-parsing`)
- Python API (`Pipeline` class)
