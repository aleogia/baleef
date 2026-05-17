# lingua_display

Real-time on-device translation kiosk — two people, two languages, one screen.

## Concept

A physical kiosk with a double-sided display. Two people stand on opposite sides and speak their own language. Each sees the other's words translated in real time as text — no internet, no cloud, no TTS.

```
Person A (speaks FR) ──► Mic A ──► VAD ──► DenoiseNR ──► Whisper STT ──► NLLB NMT ──► Display A (EN)
Person B (speaks EN) ──► Mic B ──► VAD ──► DenoiseNR ──► Whisper STT ──► NLLB NMT ──► Display B (FR)
```

## Key Design Decisions

- **100% on-device, fully offline** — no API calls, no network dependency
- **Visual output only** — no TTS, text display only
- **Side A language fixed to French** — `language="fr"` passed to Whisper, skips detection (~150ms saved)
- **Side B always outputs French** — `TARGET_LANG["B"]` locked to `fra_Latn`, UI dropdown removed
- **12 languages** selectable for side A target via UI
- **Noise suppression** via noisereduce (filters background voices and ambient noise)
- **USB mic auto-detection** — devices resolved by name at startup, no hardcoded index
- **Translation cache** — persisted to disk, O(1) lookup, near-zero latency on repeated phrases
- **Display sleep prevention** — `xset` called at server startup, system suspend masked via systemd
- **Target hardware** — ARM RK3588 SoC running Debian ARM64 (Radxa Rock 5B)
- **Target latency** — < 1s end-to-end

## Pipeline

| Stage | Component | Role |
|-------|-----------|------|
| Audio capture | 2× USB microphone (line array beamforming) | One per speaker side, auto-detected at startup |
| Voice detection | Silero VAD (0.4s chunks) | Detects speech segments, discards silence |
| Noise suppression | noisereduce | Filters background voices and ambient noise |
| Speech-to-text | faster-whisper medium | Side A forced to `fr`; side B auto-detects |
| Translation | NLLB-200 distilled-600M ONNX int8 | Neural machine translation |
| Translation cache | JSON on disk | O(1) lookup, saves every 30s + on shutdown |
| Display | FastAPI + WebSocket | Black background, white text, 4-phrase feed |

## Web Interface

| URL | Description |
|-----|-------------|
| `/` | Dual-sided view — both columns side by side |
| `/side/A` | Full-screen side A — for dedicated display |
| `/side/B` | Full-screen side B — for dedicated display |
| `/admin` | Backoffice — live log, cache stats, latency, pause/clear |

### Kiosk mode (Chromium, dual display)
```bash
chromium --kiosk --app=http://localhost:8000/side/A &
DISPLAY=:0.1 chromium --kiosk --app=http://localhost:8000/side/B
```

### Lightweight alternative (Radxa / ARM)
```bash
sudo apt install cog
cog http://localhost:8000/side/A
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Development OS | Ubuntu 25.04 native |
| Production OS | Debian ARM64 on Radxa Rock 5B (RK3588) |
| Python version | **3.12** (3.13+ not compatible) |
| STT engine | faster-whisper medium (GPU via CUDA, beam_size=2) |
| Noise suppression | noisereduce (stationary=False) |
| Translation model | facebook/nllb-200-distilled-600M → ONNX int8 (CUDAExecutionProvider) |
| VAD | silero-vad (PyTorch) |
| Web server | FastAPI + uvicorn + WebSocket |
| Inference runtime | ONNX Runtime GPU (dev) → ONNX Runtime CPU / RKNN SDK (Radxa) |

## Hardware (dev machine)

- CPU: AMD Ryzen 7 7800X3D
- GPU: RTX 5070 Ti
- RAM: 32GB
- OS: Ubuntu 25.04
- Microphones: 2× USB PnP Audio Device (auto-detected at startup)

## Hardware References (Production)

Detailed comparisons available in `hardware_comparison.csv`, `hardware_microphones.csv`, `hardware_screens.csv`.

### SOC / Carte principale

| # | Référence | SoC | NPU | Affichage | Refroidissement | Prix | Lien |
|---|-----------|-----|-----|-----------|-----------------|------|------|
| 1 | **Orange Pi 5 Plus** | RK3588 | 6 TOPS | 2× HDMI 2.1 | Actif (kit officiel) | ~100€ | [orangepi.org](https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5-plus.html) |
| 2 | **Beelink SER7** | Ryzen 9 7940HS | Radeon 780M | 2× HDMI + DP | Actif (intégré) | ~380€ | [bee-link.com](https://www.bee-link.com/products/beelink-ser7-7840hs) |

- **Orange Pi 5 Plus** — meilleur rapport pour un kiosk sobre, même NPU RK3588 que la Radxa Rock 5B, portage RKNN possible
- **Beelink SER7** — x86 natif, aucun changement de code depuis le dev machine, Whisper large-v3 viable en temps réel

### Microphone (line array beamforming)

| # | Référence | Capsules | Forme | DSP | Prix | Lien |
|---|-----------|----------|-------|-----|------|------|
| 1 | **Seeed ReSpeaker USB Mic Array v2.0** | 7 mics | Circulaire | XVF3000 on-board | ~60€ | [seeedstudio.com](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v2-0.html) |
| 2 | **miniDSP UMA-16** | 16 mics | **Linéaire** | ODAS software | ~350€ | [minidsp.com](https://www.minidsp.com/products/usb-audio-interface/uma-16-microphone-array) |

- **ReSpeaker v2.0** — plug & play Linux/ARM, beamforming géré par XVF3000 sans charge CPU, idéal prototype
- **miniDSP UMA-16** — vrai line array 16 capsules, rejet latéral optimal pour isoler les deux côtés du kiosk

### Écran 12 pouces tactile

| # | Référence | Taille | Résolution | Interface | Type | Prix | Lien |
|---|-----------|--------|------------|-----------|------|------|------|
| 1 | **Waveshare 12.3" HDMI LCD** | 12.3" | 1920×720 | HDMI + USB touch | IPS 600 cd/m² | ~110$ | [waveshare.com](https://www.waveshare.com/12.3inch-hdmi-lcd.htm) |
| 2 | **Faytech FT116TMCAPOFHBOB** | 11.6" | 1920×1080 | HDMI + DP + USB touch | IPS open-frame 1000 cd/m² | ~630$ | [faytech.us](https://www.faytech.us/product/ft116tmcapofhbob-11-6-inch-open-frame-capacitive-touch-monitor/) |

- **Waveshare 12.3"** — plug & play Linux, slim, sans bordure, idéal intégration boîtier custom
- **Faytech open-frame** — qualité industrielle, IP65 face avant, 1000 cd/m² (lisible en lumière ambiante forte), flush-mount natif

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

```
lingua_display/
├── venv/                              # Python 3.12 virtual environment
├── models/
│   └── nllb-600M-onnx-int8/          # ONNX export of NLLB-200
│       ├── encoder_model.onnx
│       ├── decoder_model.onnx
│       ├── decoder_with_past_model.onnx
│       └── sentencepiece.bpe.model
├── translation_cache.json             # Persistent translation cache (auto-generated)
├── hardware_comparison.csv            # Comparatif SOC / cartes principales
├── hardware_microphones.csv           # Comparatif micros beamforming
├── hardware_screens.csv               # Comparatif écrans 12" tactiles
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
```

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

### 5. Prevent system sleep (kiosk mode)

```bash
# Screen blanking (per session — also called at server startup automatically)
xset s off && xset -dpms && xset s noblank

# System suspend (persistent across reboots)
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## Running

```bash
cd ~/projects/lingua_display
source venv/bin/activate
python3 src/server.py
```

| URL | Usage |
|-----|-------|
| http://localhost:8000 | Vue duale |
| http://localhost:8000/side/A | Side A plein écran |
| http://localhost:8000/side/B | Side B plein écran |
| http://localhost:8000/admin | Backoffice / logs en temps réel |

## Microphone Setup

Le serveur détecte automatiquement les deux micros USB PnP à chaque démarrage par leur nom — aucune configuration manuelle d'index nécessaire.

Pour lister les périphériques disponibles :
```python
import sounddevice as sd
print(sd.query_devices())
```

Les micros capturent en 48kHz natif. Le pipeline rééchantillonne à 16kHz en interne.

## Whisper Tuning

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `beam_size` | 2 | Équilibre vitesse / précision |
| `language` | `"fr"` (side A) / `None` (side B) | Supprime la détection sur side A (~150ms économisés) |
| `initial_prompt` | `"Conversation en français."` (side A) | Guide le modèle vers le bon registre, réduit les erreurs d'accent |
| `condition_on_previous_text` | `False` | Évite la contamination d'hallucinations entre phrases |

Pour améliorer la précision sur du vocabulaire spécifique, enrichir `initial_prompt` avec des mots-clés du contexte métier.

## Admin Backoffice (`/admin`)

Interface de supervision en temps réel :
- **Log live** — chaque traduction avec heure, side, langues, latence, indicateur cache
- **Statistiques** — total, taux de cache hits, compteurs par side, latence moyenne
- **Contrôles** — Pause (fige l'affichage sans perdre les entrées) et Effacer
- **Replay** — les 200 derniers événements sont renvoyés à la connexion

## Validated Performance (Ubuntu 25.04, RTX 5070 Ti)

| Component | Status | Latency |
|-----------|--------|---------|
| Silero VAD | ✅ | ~40ms (chunks 0.4s) |
| noisereduce | ✅ | ~50ms |
| faster-whisper medium (GPU, beam_size=2) | ✅ | ~250ms |
| NLLB-200 ONNX (GPU) | ✅ | ~60ms |
| Translation cache hit | ✅ | <1ms |
| **Full pipeline (no cache)** | ✅ | **~400ms** |

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

**Radxa Rock 5B** (référence actuelle)
- SoC: Rockchip RK3588 (4× Cortex-A76 + 4× Cortex-A55)
- NPU: 6 TOPS (INT8)
- RAM: 16GB LPDDR5
- Display: HDMI 2.1 + DisplayPort 1.4 → dual OLED
- OS: Debian Bookworm ARM64
- Limite: refroidissement passif insuffisant en charge soutenue

### Modèles Whisper recommandés sur Radxa (CPU/NPU)

| Modèle | Params | Vitesse RK3588 | Recommandation |
|--------|--------|----------------|----------------|
| faster-whisper tiny | 39M | ~1-2s/phrase | Minimum viable |
| faster-whisper small | 244M | ~3-5s/phrase | **Recommandé** |
| faster-whisper medium | 769M | ~12-18s/phrase | Trop lent |
| sherpa-onnx RKNN NPU | — | ~1s/phrase | Optimal si NPU activé |

### NPU acceleration (roadmap)

**sherpa-onnx** (recommandé) — API Python, modèles RKNN pré-compilés, VAD intégré :
```python
import sherpa_onnx
recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder="whisper-small-encoder.rknn",
    decoder="whisper-small-decoder.rknn",
    tokens="tokens.txt",
)
```

**rknn_model_zoo** (direct) — conversion manuelle ONNX→RKNN via rknn-toolkit2, modèles Whisper inclus.

### Notes de déploiement Radxa

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("small", device=device, compute_type="int8")
```

Remplacer :
- `onnxruntime-gpu` → `onnxruntime`
- RKNN SDK pour accélération NPU (roadmap)

## Translation Cache

Les traductions sont cachées dans `translation_cache.json` (racine du projet) et rechargées au démarrage. Lookups en O(1).

**Risques si le cache devient très volumineux :**
- `json.dump()` sérialise le dict entier — peut prendre plusieurs secondes sur RK3588. Le thread de sauvegarde tourne toutes les 30s en arrière-plan (ne bloque pas le pipeline audio).
- Démarrage lent si le fichier dépasse 100MB+.

**En pratique**, un kiosk FR↔EN accumule quelques milliers d'entrées au plus — quelques Mo. Pas de risque réel.

**Si besoin** : LRU eviction ou remplacement du JSON par SQLite (lectures/écritures unitaires).

## Known Issues & Workarounds

| Issue | Cause | Workaround |
|-------|-------|-----------|
| DeepFilterNet incompatible | torchaudio.backend retiré en 2.9+ | Utiliser noisereduce |
| CUDA `compute_120a` error | CUDA 12.8 trop ancien pour RTX 5070 Ti | Utiliser CUDA 13.0 |
| GCC 15 incompatible avec CUDA 12.8 | CUDA 12.8 supporte GCC ≤14 | Utiliser CUDA 13.0 |
| Python 3.14 incompatible avec optimum | Trop récent | Utiliser Python 3.12 |
| WebSocket non mis à jour | asyncio.run() échoue dans les threads | Queue + asyncio.run_coroutine_threadsafe |
| Voix de fond transcrites | Micro unique capte tout | Micros directionnels / beamforming sur Radxa |
| Index USB mic change | PipeWire réassigne dynamiquement | Détection par nom au démarrage |
| Écran s'éteint en veille | Gestion d'énergie OS | `xset s off -dpms` + `systemctl mask sleep.target` |
| Thread pipeline crashe silencieusement | Exception non catchée dans worker | Pas de relance automatique — watchdog systemd à implémenter |
| Cache JSON corrompu | Arrêt brutal pendant écriture | Sauvegarde atomique à implémenter (write tmp + rename) |
| `/admin` accessible sans auth | Pas d'authentification | Ajouter HTTP Basic Auth ou restreindre par IP |

## Roadmap

- [x] Silero VAD — validé
- [x] faster-whisper STT (GPU) — validé
- [x] NLLB-200 ONNX NMT (GPU) — validé
- [x] Noise suppression (noisereduce) — intégré
- [x] Pipeline temps réel complet — validé
- [x] Interface web WebSocket — validée
- [x] Détection automatique langue (side B) — validée
- [x] Langue source fixe side A (français) — implémenté
- [x] Changement langue cible side A via UI — validé
- [x] Langue sortie side B fixée (français) — implémenté
- [x] Cache traduction avec persistance disque — implémenté
- [x] Auto-détection double micro USB — implémenté
- [x] Routes plein écran `/side/A` et `/side/B` — implémentées
- [x] Backoffice admin live logs `/admin` — implémenté
- [x] Prévention veille écran et système — implémenté
- [x] Thème fond noir texte blanc — implémenté
- [ ] Déploiement Radxa Rock 5B (Debian ARM64)
- [ ] Optimisation Whisper RKNN SDK (NPU) via sherpa-onnx
- [ ] Optimisation NLLB RKNN SDK
- [ ] Dual OLED display (HDMI + DP)
- [ ] Relance automatique workers (watchdog systemd)
- [ ] Sauvegarde cache atomique (write tmp + rename)
- [ ] Authentification `/admin`
- [ ] Démarrage automatique au boot (service systemd)
- [ ] Stress test — 8h opération continue
