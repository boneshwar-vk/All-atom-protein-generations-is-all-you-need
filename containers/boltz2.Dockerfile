# Boltz-2 image (default open structure-prediction backend). (placeholder)
# Boltz is MIT with open weights that auto-download on first run.
FROM aapg/base:latest

# RUN pip install boltz  # weights (MIT) download to the cache on first `boltz predict`
# Mount or bake data/weights/boltz2 for offline/container runs.
