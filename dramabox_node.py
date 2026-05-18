"""
DramaBox ComfyUI Node — wraps resemble-ai/DramaBox TTSServer.

Expects DramaBox cloned at custom_nodes/DramaBox/:
  git clone https://github.com/resemble-ai/DramaBox custom_nodes/DramaBox
"""
import gc
import os
import re
import sys
import logging
import tempfile

import torch
import torchaudio
import folder_paths

logger = logging.getLogger("DramaBox")

# ---------------------------------------------------------------------------
# Locate DramaBox source (cloned sibling directory)
# ---------------------------------------------------------------------------
_DB_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "DramaBox"))
_DB_SRC  = os.path.join(_DB_ROOT, "src")

for _p in (_DB_ROOT, _DB_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from inference_server import TTSServer
    from model_downloader import get_all_paths
    _DB_AVAILABLE = True
except ImportError as e:
    logger.error(f"[DramaBox] Could not import TTSServer: {e}")
    logger.error(
        "[DramaBox] Clone the repo into custom_nodes/DramaBox/:\n"
        "  git clone https://github.com/resemble-ai/DramaBox "
        "<your_comfyui>/custom_nodes/DramaBox"
    )
    _DB_AVAILABLE = False
    TTSServer = None
    get_all_paths = None


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------
_server_cache: dict = {}


def _free_all_servers():
    """Delete all cached TTSServer instances and release VRAM."""
    for key in list(_server_cache.keys()):
        server = _server_cache.pop(key)
        for attr in (
            "_prompt_encoder", "_audio_conditioner", "_transformer",
            "_audio_decoder", "_warm_text_encoder", "_warm_embeddings_processor",
            "_perth",
        ):
            try:
                sub = getattr(server, attr, None)
                if sub is not None:
                    setattr(server, attr, None)
                    del sub
            except Exception:
                pass
        del server

    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    logger.info("[DramaBox] VRAM freed.")


def _get_server() -> "TTSServer":
    key = "bf16"
    if key not in _server_cache:
        logger.info("[DramaBox] Loading TTSServer (dtype=bf16) …")

        # xformers doesn't support dense tensor attn_bias on Blackwell (sm_120+).
        # Null it out so DramaBox falls back to PytorchAttention (SDPA).
        try:
            import ltx_core.model.transformer.attention as _attn_mod
            _attn_mod.memory_efficient_attention = None
            logger.info("[DramaBox] Using PyTorch SDPA attention backend.")
        except ImportError:
            pass

        logger.info("[DramaBox] Resolving model paths via get_all_paths() …")
        paths = get_all_paths()

        # Suppress verbose third-party logging during model load
        _noisy = ["transformers", "accelerate", "torch"]
        _saved = {n: logging.getLogger(n).level for n in _noisy}
        for n in _noisy:
            logging.getLogger(n).setLevel(logging.ERROR)
        _root = logging.getLogger()
        _root_saved = _root.level
        _root.setLevel(logging.ERROR)

        try:
            _server_cache[key] = TTSServer(
                checkpoint=paths["transformer"],
                full_checkpoint=paths["audio_components"],
                gemma_root=paths["gemma_root"],
                device="cuda",
                dtype="bf16",
                compile_model=False,
                bnb_4bit=True,
            )
        finally:
            _root.setLevel(_root_saved)
            for n, lvl in _saved.items():
                logging.getLogger(n).setLevel(lvl)

        logger.info("[DramaBox] TTSServer ready.")
    return _server_cache[key]


# ---------------------------------------------------------------------------
# Duration estimation from quoted speech text
# ---------------------------------------------------------------------------

def _estimate_gen_duration(prompt: str, pace_wpm: float) -> float:
    """Extract quoted speech from prompt, estimate duration from word count."""
    quoted = re.findall(r'"([^"]*)"', prompt)
    text = " ".join(quoted) if quoted else prompt
    words = len(text.split())
    if words == 0:
        return 5.0
    actual_wpm = float(pace_wpm)
    seconds = (words / actual_wpm) * 60.0 * 1.10  # 10% pad
    return max(3.0, round(seconds, 1))


# ---------------------------------------------------------------------------
# Main generation node
# ---------------------------------------------------------------------------

class DramaBoxNode:
    """
    DramaBox expressive TTS with optional voice cloning.
    Returns a ComfyUI AUDIO tensor — connect to SaveAudio for file output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": 'A woman speaks warmly, "Hello, how are you today?"',
                    "tooltip": (
                        "Describe the speaker, emotion, pace, and style. "
                        "Put the dialogue in quotes."
                    ),
                }),
                "pace_wpm": ("FLOAT", {
                    "default": 190.0, "min": 1.0, "max": 500.0, "step": 1.0,
                    "tooltip": (
                        "Speaking pace in words-per-minute. Higher = faster/shorter. Lower = slower/longer. "
                        "Adjust until generated duration matches your voice."
                    ),
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance strength.",
                }),
                "stg_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Style-transfer guidance scale.",
                }),
                "gen_duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Override target duration in seconds. 0 = auto-calculate from prompt word count + pace.",
                }),
                "ref_duration": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Seconds of the reference audio used for voice cloning.",
                }),
                "num_steps": ("INT", {
                    "default": 30, "min": 1, "max": 0x7FFFFFFF, "step": 1,
                    "tooltip": "Diffusion denoising steps. More steps = better quality but slower. 30 is the default.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**32 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Generation seed for reproducibility.",
                }),
                "free_memory_after_generate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload all DramaBox models from VRAM after generation completes.",
                }),
            },
            "optional": {
                "voice_ref": ("AUDIO", {
                    "tooltip": (
                        "Optional reference audio for voice timbre cloning. "
                        "~10 s of clean speech works best."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "generate"
    CATEGORY      = "audio/tts"
    DESCRIPTION   = (
        "DramaBox (resemble-ai) — prompt-driven expressive TTS with optional "
        "voice cloning. Connect audio output to SaveAudio for file saving."
    )

    def generate(
        self,
        prompt: str,
        cfg_scale: float,
        stg_scale: float,
        pace_wpm: float,
        gen_duration: float,
        ref_duration: float,
        num_steps: int,
        seed: int,
        free_memory_after_generate: bool,
        voice_ref=None,
    ):
        if not _DB_AVAILABLE:
            raise RuntimeError(
                "[DramaBox] DramaBox is not installed. "
                "Clone it into custom_nodes/DramaBox/:\n"
                "  git clone https://github.com/resemble-ai/DramaBox "
                "<your_comfyui>/custom_nodes/DramaBox\n"
                "Then install its dependencies:\n"
                "  pip install -r <your_comfyui>/custom_nodes/DramaBox/requirements.txt"
            )

        # Auto-calculate duration from word count when gen_duration is not set
        if gen_duration and gen_duration > 0:
            effective_duration = float(gen_duration)
            logger.info(f"[DramaBox] Using manual gen_duration={effective_duration:.1f}s")
        else:
            effective_duration = _estimate_gen_duration(prompt, pace_wpm)
            logger.info(f"[DramaBox] Auto gen_duration={effective_duration:.1f}s (pace={pace_wpm:.0f} wpm)")

        server = _get_server()
        server._num_steps = num_steps

        # Write ComfyUI AUDIO voice reference to a temp WAV for DramaBox
        tmp_ref_path = None
        voice_ref_path = None
        if voice_ref is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            tmp_ref_path = tmp.name
            wave = voice_ref["waveform"]
            sr   = voice_ref["sample_rate"]
            if wave.ndim == 3:
                wave = wave[0]
            torchaudio.save(tmp_ref_path, wave.cpu().float(), sr)
            voice_ref_path = tmp_ref_path

        try:
            waveform, sample_rate = server.generate(
                prompt=prompt,
                voice_ref=voice_ref_path,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                duration_multiplier=1.0,
                seed=int(seed),
                gen_duration=effective_duration,
                ref_duration=float(ref_duration),
            )
        except Exception:
            if free_memory_after_generate:
                _free_all_servers()
            raise
        finally:
            if tmp_ref_path and os.path.exists(tmp_ref_path):
                try:
                    os.unlink(tmp_ref_path)
                except OSError:
                    pass

        wav_cpu = waveform.cpu().float()

        if free_memory_after_generate:
            _free_all_servers()

        return ({"waveform": wav_cpu.unsqueeze(0), "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


# ---------------------------------------------------------------------------
# Standalone unload node
# ---------------------------------------------------------------------------

class DramaBoxUnloadNode:
    """Frees all DramaBox models from VRAM when executed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Queue a prompt with this true to unload DramaBox from VRAM.",
                }),
            },
            "optional": {
                "passthrough_audio": ("AUDIO", {
                    "tooltip": "Pass audio through so this node can sit inline in a chain.",
                }),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    OUTPUT_NODE   = True
    FUNCTION      = "unload"
    CATEGORY      = "audio/tts"
    DESCRIPTION   = "Unloads all DramaBox models from VRAM."

    def unload(self, unload: bool, passthrough_audio=None):
        if unload:
            _free_all_servers()
        return {"ui": {"audio": []}, "result": (passthrough_audio,)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DramaBox":       DramaBoxNode,
    "DramaBoxUnload": DramaBoxUnloadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBox":       "DramaBox TTS - Nimble Wrapper",
    "DramaBoxUnload": "DramaBox Unload",
}
