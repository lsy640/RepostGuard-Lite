# AIGI Detect Demo

Vue 3 frontend and local FastAPI inference service for the train-v3 RepostGuard M2/M3 checkpoints.

## Run locally

```bash
cd /Users/liushiyuan/Downloads/AGI/TikTok_project_5/demo-frontend
# First run only:
./scripts/setup.sh
# Start or safely restart the local demo:
./scripts/run_demo.sh
```

Open `http://127.0.0.1:8000`. The production runner builds the frontend and serves the UI and API from one local process. No uploaded image is sent to a remote service.

`run_demo.sh` safely restarts an older AIGI Detect Demo already running from this directory, so a stale process cannot keep serving an old frontend. It will never terminate an unrelated process using port 8000; in that case, either stop that process or select another port:

```bash
AIGI_DEMO_PORT=8001 ./scripts/run_demo.sh
```

After the terminal prints `Uvicorn running on http://127.0.0.1:8000`, open that address. If the tab was already open before rebuilding, reload it once to fetch the new hashed frontend bundle.

For hot-reload development, use `./scripts/run_dev.sh`; Vite runs on `http://127.0.0.1:5173` and proxies `/api` to port 8000.

## Model contract

| Model | Checkpoint SHA-256 | Frozen threshold | Calibration temperature |
| --- | --- | ---: | ---: |
| M2 | `468d3a58603fdf8dfe1b234a24fd8e52a99c6e4881e921bef6bb0cea64bbac34` | `0.99658203125` | `14.50161361694336` |
| M3 | `c83f70641a9c8d7f6808e794cfc8c28c0e478feeca7506e489c772a512115b2f` | `0.9970703125` | `14.526344299316406` |

The service validates checkpoint SHA-256, embedded config digest, train-v3 manifest lineage and strict state-dict loading. Temperature calibration is enabled only when the calibration artifact records the exact active checkpoint SHA.

## Tests

```bash
.venv/bin/python -m pytest server/tests
npm test
npm run build
```

Batch downloads deliberately retain the evaluator schema: a sorted JSON array whose rows contain only `image_path` and raw `pred`.
