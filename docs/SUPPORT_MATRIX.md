# ATOM OSS SDK Support Matrix

## Python
- Supported: 3.9, 3.10, 3.11, 3.12
- Best-effort: 3.13 (not release-blocking)

## Platforms
- Linux (primary)
- macOS (CPU + Apple Silicon)
- Windows (best effort)

## Hardware
- CPU: supported
- CUDA GPU: supported via optional GPU extras
- MPS (Apple): supported where backend components permit

## Dependency Profiles
- Core SDK: `pip install -e .`
- Web/API/UI extras: `pip install -e ".[web]"`
- GPU extras: `pip install -e ".[gpu]"`
- Dev tooling: `pip install -e ".[dev]"`

## Notes
- Web/API routes requiring FastAPI stack are unavailable unless `[web]` extra is installed.
- Director-pack endpoint is disabled by default; enable explicitly with `ATOM_ENABLE_DIRECTOR_PACK=1`.
- Release evidence and packet generation are strict in `--profile release`.
