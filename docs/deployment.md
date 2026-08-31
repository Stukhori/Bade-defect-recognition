# Streamlit Community Cloud deployment

Application v2 is prepared for deployment from a clean GitHub clone. The exact 6.2 MB frozen Phase 6 checkpoint and its metadata are versioned solely as the inference asset required by the app; their validated hashes are unchanged.

## Deployment coordinates

- Repository: `Stukhori/Bade-defect-recognition`
- Branch: `main`
- Entrypoint: `app/app.py`
- Python: `3.11`
- Dependency declaration: `app/requirements.txt`
- Streamlit configuration: `.streamlit/config.toml`
- Secrets: none

The app-local requirements file is intentional. Streamlit Community Cloud searches the entrypoint directory before the repository root, so it uses `app/requirements.txt` instead of the research environment's root `uv.lock`. The deployment file pins the validated application environment and selects CPU-only PyTorch wheels on Linux.

## Deploy

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with a GitHub account that can access the repository.
2. Select **Create app** and choose the existing app/repository option.
3. Enter the repository, branch, and entrypoint shown above.
4. Open **Advanced settings** and select Python 3.11. No secrets are required.
5. Deploy. Streamlit runs from the repository root and reads the root `.streamlit/config.toml`.

## Validate locally

```powershell
uv run python scripts/validate_deployment.py
uv run streamlit run app/app.py --server.address 127.0.0.1
```

The deployment validator checks that all required files are tracked, verifies the checkpoint file SHA-256 and decoded state identity, loads it in CPU evaluation mode, and confirms the pinned dependency/configuration contract.

## Scope

Deployment does not change the research claims or add a detector. Uploads, crops, session history, visualizations, and exports remain in process memory. The app uses no external API and performs no runtime artifact download. Automatic localization remains unavailable because Phase 11B detector training and evaluation have not been completed.
