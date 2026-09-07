# BladeScope Streamlit Community Cloud deployment

Application v3 is prepared for deployment from a clean GitHub clone but has not been externally deployed.

## Deployment coordinates

- Repository: `Stukhori/Bade-defect-recognition`
- Branch: `main`
- Main file path: `app/app.py`
- Python: `3.11`
- Dependency declaration: `app/requirements.txt`
- Debian system-package declaration: `packages.txt`
- Streamlit configuration: `.streamlit/config.toml`
- Secrets: none

The entrypoint-local dependency file pins Streamlit `1.62.0`, streamlit-cropper `0.3.1`, Ultralytics `8.3.150`, PyTorch `2.13.0+cpu`, torchvision `0.28.0+cpu`, and the remaining validated application packages. The exact frozen crop-classifier checkpoint and detector proposal checkpoint are tracked. No runtime model download is required.

The root `packages.txt` declares only `libgl1` and `libglib2.0-0t64`. Streamlit Community Cloud installs `libgl1` so the pinned `opencv-python` runtime can resolve `libGL.so.1`; on its Debian Trixie environment, `libglib2.0-0t64` provides the required `libgthread-2.0.so.0`. These system libraries do not change Python packages or application inference behavior.

## Deploy

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with a GitHub account that can access the repository.
2. Select **Create app** and choose the existing repository option.
3. Use branch `main` and enter `app/app.py` as the main file path.
4. In **Advanced settings**, select Python 3.11. No secrets are required.
5. Deploy. Streamlit runs from the repository root and reads `.streamlit/config.toml`.

## Validate locally

```powershell
uv run python scripts/validate_deployment.py
uv run streamlit run app/app.py --server.address 127.0.0.1
```

The validator requires both checkpoints and all deployment inputs, including `packages.txt`, to be tracked. It permits exactly the normalized apt declarations `libgl1` and `libglib2.0-0t64` in that order, verifies checkpoint byte identities, loads the frozen classifier on CPU, and checks the pinned dependency and Streamlit configuration contract.

## Scope

The experimental detector runs on CPU with fixed image size 640, threshold `0.39`, NMS IoU `0.7`, class-agnostic NMS, and maximum 300 detections. Users cannot change detector controls and must review proposals before classification. Prepared-crop and manual-region workflows remain available.

Uploads, proposals, crops, session history, visualizations, and exports remain in process memory. The app disables telemetry and external trackers, performs no external API call or runtime artifact download, and writes no prediction output. Deployment would not alter scientific results, but deployment itself remains unperformed.
