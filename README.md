# ComfyUI DramaBox

A ComfyUI custom node wrapping [resemble-ai/DramaBox](https://github.com/resemble-ai/DramaBox) — prompt-driven expressive TTS with optional voice cloning.

## Nodes

- **DramaBox TTS** — generate speech from a descriptive prompt, with optional reference audio for voice cloning
- **DramaBox Unload** — free all DramaBox models from VRAM on demand

## Requirements

1. Clone DramaBox into your `custom_nodes` directory:
   ```
   git clone https://github.com/resemble-ai/DramaBox <your_comfyui>/custom_nodes/DramaBox
   ```

2. Install DramaBox dependencies:
   ```
   pip install -r <your_comfyui>/custom_nodes/DramaBox/requirements.txt
   ```

3. Clone this node:
   ```
   git clone https://github.com/nimblecloud13/Dramabox_Nimble_Wrapper <your_comfyui>/custom_nodes/comfyui_dramabox
   ```

## Usage

### DramaBox TTS

| Input | Description |
|---|---|
| `prompt` | Describe the speaker, emotion, and style. Put dialogue in quotes: `A woman speaks warmly, "Hello!"` |
| `pace_wpm` | Speaking pace in words-per-minute (1–500). Higher = faster/shorter audio. |
| `cfg_scale` | Classifier-free guidance strength (default 2.5) |
| `stg_scale` | Style-transfer guidance scale (default 1.5) |
| `gen_duration` | Override output duration in seconds. Set to 0 to auto-calculate from word count + pace. |
| `ref_duration` | Seconds of reference audio to use for voice cloning (default 10s) |
| `num_steps` | Diffusion steps — more = better quality but slower (default 30) |
| `seed` | Reproducibility seed |
| `free_memory_after_generate` | Unload models from VRAM after each generation |
| `voice_ref` *(optional)* | Reference AUDIO for voice cloning — ~10s of clean speech works best |

Connect the `audio` output to a **Save Audio** node.

### DramaBox Unload

Frees all DramaBox models from VRAM. Can sit inline in a chain via the `passthrough_audio` input.

## Why pace_wpm?

DramaBox requires you to specify the output duration upfront before it generates — unlike most TTS systems that just produce however much audio the text needs. That means without some help, you'd have to manually guess how many seconds your prompt will take to speak, try it, check the result, and adjust.

The `pace_wpm` slider attempts to solve this by estimating duration automatically: it extracts the quoted dialogue from your prompt, counts the words, and divides by your expected speaking rate to get a target duration before generation starts. That way the model has a realistic length to aim for without you doing the math yourself. The `gen_duration` override is still there if you want exact control, but for most uses the auto-estimate gets you close enough on the first try.

## Notes

- Models load on first generation and are cached in VRAM unless `free_memory_after_generate` is enabled.
- Requires a CUDA GPU. Runs in `bf16` with 4-bit quantization by default.
- On Blackwell (RTX 50xx) GPUs, xformers attention is automatically disabled and falls back to PyTorch SDPA.
