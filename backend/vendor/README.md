# Vendored inference sources

The `Fun-CosyVoice3-0.5B` engine depends on two upstream repositories that
ship no installable package (no `setup.py` / `pyproject.toml`), so the
inference-only subset is vendored here:

| Directory | Upstream | License | Notes |
|-----------|----------|---------|-------|
| `cosyvoice/` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0 | Inference-only modules (`cli`, `flow`, `hifigan`, `llm`, `tokenizer`, `transformer`, `utils`). Training/dataset/vllm code is omitted. |
| `matcha/` | [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | MIT | Only the components CosyVoice uses (`models/components/*`, `utils/audio.py`). |

## Local modifications (vs upstream)

- `cosyvoice/cli/cosyvoice.py`
  - Removed the `modelscope` import and `snapshot_download()` fallbacks —
    Voicebox always downloads the model via `huggingface_hub.snapshot_download`
    and passes a local `model_dir`.
  - Added `_load_inference_configs()` which strips training-only yaml keys
    (`hifigan`/GAN discriminators, `parquet_opener`/dataset pipelines,
    `train_conf`) before `load_hyperpyyaml()`. This avoids pulling in
    `matcha.hifigan`, `pyarrow`, and `pyworld` at load time.
- `cosyvoice/llm/llm.py`
  - `Qwen2Encoder.forward_one_step()` now builds a full-length all-ones
    attention mask when a `past_key_values` cache is present. Transformers
    >= 4.5x pads a short 2D mask with zeros in `prepare_padding_mask()`, so the
    upstream `(1, 1)` mask silently masked out every cached position and the
    LLM decoded while attending to the SOS token only (intelligible but
    hallucinated content). Verified against a no-cache full recompute; matches
    the pinned upstream `transformers==4.51.3` behavior.
- `cosyvoice/utils/file_utils.py`
  - `load_wav()` now reads via `soundfile` instead of
    `torchaudio.load(backend='soundfile')`, which torchaudio 2.11 removed in
    favor of the optional `torchcodec` dependency.
- `matcha/utils/pylogger.py`
  - Vendored without the `lightning` `rank_zero_only` wrapper (Voicebox runs
    single-process).

Do not run `ruff` on this directory — it is excluded via `backend/pyproject.toml`.
