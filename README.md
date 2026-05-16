# lingua_display

Real-time on-device translation kiosk — two people, two languages, one screen.

## Concept

A physical kiosk with a double-sided transparent OLED display. Two people stand on opposite sides and speak their own language. Each sees the other's words translated in real time as text — no internet, no cloud, no TTS.
Person A (speaks FR) ──► Mic A ──► VAD ──► DenoiseNR ──► Whisper STT ──► NLLB NMT ──► Display B (EN)
Person B (speaks EN) ──► Mic B ──► VAD ──► DenoiseNR ──► Whisper STT ──► NLLB NMT ──► Display A (FR)
## Key Design Decisions

- **100% on-device, fully offline** — no API calls, no network dependency
- **Visual output only** — no TTS, text display only
- **12 languages** via NLLB-200
- **Noise suppression** via noisereduce (filters background voices and ambient noise)
- **Target hardware** — ARM RK3588 SoC running Debian ARM64 (Radxa Rock 5B)
- **Target latency** — < 1s end-to-end

## Pipeline

| Stage | Component | Role |
|-------|-----------|------|
| Audio capture | Microphone (USB) | One per speaker side |
| Voice detection | Silero VAD | Detects speech segments, discards silence |
| Noise suppression | noisereduce | Filters background voices and ambient noise |
| Speech-to-text | faster-whisper medium | Transcribes + auto-detects language |
| Translation | NLLB-200 distilled-600M ONNX | Neural machine translation |
| Display | Web interface (FastAPI + WebSocket) | Shows translated text in real time |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Development OS | Ubuntu 25.04 native |
| Production OS | Debian ARM64 on Radxa Rock 5B (RK3588) |
| Python version | **3.12** (3.13+ not compatible) |
| STT engine | faster-whisper medium (GPU via CUDA, beam_size=5) |
| Noise suppression | noisereduce (stationary=False) |
| Translation model | facebook/nllb-200-distilled-600M → ONNX (GPU via CUDAExecutionProvider) |
| VAD | silero-vad (PyTorch) |
| Web server | FastAPI + uvicorn + WebSocket |
| Inference runtime | ONNX Runtime GPU (dev) → ONNX Runtime CPU / RKNN SDK (Radxa) |

## Hardware (dev machine)

- CPU: AMD Ryzen 7 7800X3D
- GPU: RTX 5070 Ti
- RAM: 32GB
- OS: Ubuntu 25.04
- Microphone: FiFine K669 (USB, device index 10)

## System Requirements

```bash
# CUDA Toolkit 13.0 (required for GCC 15 + RTX 5070 Ti / sm_120)
sudo apt install -y cuda-toolkit-13-0
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# cuDNN 9
sudo apt install -y libcudnn9-cuda-12

# System packages
sudo apt install -y git cmake build-essential ffmpeg curl python3.12 python3.12-venv

# Rust (required to build some Python packages)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

> ⚠️ Ubuntu 25.04 ships Python 3.14 by default. Python 3.12 must be installed via deadsnakes PPA.
> ⚠️ CUDA 12.8 is incompatible with GCC 15. Use CUDA 13.0.
> ⚠️ DeepFilterNet 0.5.6 is incompatible with torchaudio 2.9+. Use noisereduce instead.

## Project Structure
lingua_display/
├── venv/                              # Python 3.12 virtual environment
├── models/
│   └── nllb-600M-onnx-int8/          # ONNX export of NLLB-200
│       ├── encoder_model.onnx
│       ├── decoder_model.onnx
│       ├── decoder_with_past_model.onnx
│       └── sentencepiece.bpe.model
├── audio_samples/                     # WAV test files (16kHz mono)
│   ├── test_fr_16k.wav
│   └── test_fr_complexe.wav
└── src/
├── export_nllb_onnx.py            # One-time ONNX export script
├── test_vad.py                    # Silero VAD isolated test
├── test_nmt.py                    # NLLB translation isolated test
├── pipeline.py                    # Full pipeline (file-based, no mic)
├── realtime.py                    # Real-time pipeline (terminal output)
└── server.py                      # Web interface ← main entry point
## Installation (from scratch)

### 1. System setup

```bash
# Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y git cmake build-essential ffmpeg curl python3.12 python3.12-venv

# CUDA 13.0
sudo apt install -y cuda-toolkit-13-0
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# cuDNN 9
sudo apt install -y libcudnn9-cuda-12

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

### 2. Project setup

```bash
mkdir -p ~/projects/lingua_display
cd ~/projects/lingua_display
mkdir -p models/nllb-600M-onnx-int8 audio_samples src

python3.12 -m venv venv
source venv/bin/activate
```

### 3. Python dependencies

```bash
# PyTorch with CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Audio & VAD
pip install numpy soundfile scipy silero-vad sounddevice resampy noisereduce

# STT
pip install faster-whisper

# NMT
pip install optimum[onnxruntime] transformers sentencepiece onnxruntime-gpu

# Web server
pip install fastapi uvicorn websockets
```

### 4. Export NLLB model (once, ~2GB download, ~10-20 min)

```bash
python3 src/export_nllb_onnx.py
```

### 5. Prepare test audio

```bash
ffmpeg -i your_recording.wav -ar 16000 -ac 1 audio_samples/test_fr_16k.wav -y
```

## Running

```bash
cd ~/projects/lingua_display
source venv/bin/activate

# Web interface (recommended)
python3 src/server.py
# Open http://localhost:8000

# Terminal only
python3 src/realtime.py

# File-based pipeline test
python3 src/pipeline.py
```

## Microphone Setup

Find the correct device index:

```python
import sounddevice as sd
print(sd.query_devices())
```

Look for the USB audio device with 1 input channel. Update `DEVICE_INDEX` in `server.py` and `realtime.py`.

The microphone captures at 48kHz natively. The pipeline resamples to 16kHz internally.

## Validated Performance (Ubuntu 25.04, RTX 5070 Ti)

| Component | Status | Latency |
|-----------|--------|---------|
| Silero VAD | ✅ | ~100ms |
| noisereduce | ✅ | ~50ms |
| faster-whisper medium (GPU, beam_size=5) | ✅ | ~200ms |
| NLLB-200 ONNX (GPU) | ✅ | ~60ms |
| **Full pipeline** | ✅ | **~450ms** |

## Supported Languages

| Code | Language | NLLB token |
|------|----------|-----------|
| `fr` | French | `fra_Latn` |
| `en` | English | `eng_Latn` |
| `es` | Spanish | `spa_Latn` |
| `de` | German | `deu_Latn` |
| `ar` | Arabic | `arb_Arab` |
| `zh` | Chinese (Simplified) | `zho_Hans` |
| `ja` | Japanese | `jpn_Jpan` |
| `pt` | Portuguese | `por_Latn` |
| `ru` | Russian | `rus_Cyrl` |
| `it` | Italian | `ita_Latn` |
| `hi` | Hindi | `hin_Deva` |

## Target Hardware (Production)

**Radxa Rock 5B**
- SoC: Rockchip RK3588 (4× Cortex-A76 + 4× Cortex-A55)
- NPU: 6 TOPS (INT8)
- RAM: 16GB LPDDR5
- Display: HDMI 2.1 + DisplayPort 1.4 → dual OLED
- OS: Debian Bookworm ARM64

### Radxa deployment

```python
# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("medium", device=device,
                              compute_type="float16" if device == "cuda" else "int8")
```

On Radxa, replace:
- `onnxruntime-gpu` → `onnxruntime`
- RKNN SDK for NPU acceleration (roadmap)

## Known Issues & Workarounds

| Issue | Cause | Workaround |
|-------|-------|-----------|
| DeepFilterNet incompatible | torchaudio.backend removed in 2.9+ | Use noisereduce instead |
| CUDA `compute_120a` error | CUDA 12.8 too old for RTX 5070 Ti | Use CUDA 13.0 |
| GCC 15 incompatible with CUDA 12.8 | CUDA 12.8 supports up to GCC 14 | Use CUDA 13.0 |
| Python 3.14 incompatible with optimum | Too recent | Use Python 3.12 |
| VAD misses speech in small chunks | Silero needs ≥1s of audio | Accumulate 1s before VAD |
| WebSocket not updating | asyncio.run() fails in threads | Use Queue + asyncio.sleep |
| Background voices transcribed | Single mic captures everything | Use directional mics on Radxa |

## Roadmap

- [x] Silero VAD — validated
- [x] faster-whisper STT (GPU, beam_size=5) — validated
- [x] NLLB-200 ONNX NMT (GPU) — validated
- [x] Noise suppression (noisereduce) — integrated
- [x] Full real-time pipeline — validated
- [x] Web interface with WebSocket — validated
- [x] Auto language detection — validated
- [x] Live target language switching — validated
- [ ] Deploy on Radxa Rock 5B (Debian ARM64)
- [ ] Optimize Whisper with RKNN SDK (NPU)
- [ ] Optimize NLLB with RKNN SDK
- [ ] Dual microphone capture (one per side)
- [ ] Dual OLED display output (HDMI + DP)
- [ ] Auto-start on boot (systemd service)
- [ ] Stress test — 8h continuous operation
