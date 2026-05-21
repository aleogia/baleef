# baleef

Traduction en temps réel de l'audio système — offline, sur GPU.

## Lancer le projet
```bash
source venv/bin/activate
python src/server.py
# Ouvrir http://localhost:8001
```

## Pièges d'installation
- `torch` et `torchaudio` s'installent séparément (pas dans requirements.txt) :
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
- Requiert CUDA + PipeWire ou PulseAudio

## Structure
- Tout le code est dans `src/server.py` — un seul fichier
- `models/` — modèles NLLB pré-téléchargés, non commités, ne pas toucher

## Points clés
- Fonctionne 100% offline, pas d'appels réseau
- Capture le loopback système (monitor PipeWire), pas le micro
- Le mode pipeline (`vad` / `rms` / `hybrid`) est changeable à chaud via l'UI
