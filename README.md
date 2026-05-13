# MLOps Drift Mitigation Pipeline

Automated pipeline for detecting and mitigating data drift in production ML models.
Built as a university internship project using Python, scikit-learn, Evidently AI,
MLflow, Grafana, FastAPI, Docker and GitHub Actions.

---

## Architecture

```
mlops-drift-pipeline/
├── src/
│   ├── training/
│   │   └── train.py          # RF-06 — model training + MLflow logging
│   ├── serving/
│   │   └── app.py            # RF-09 — FastAPI prediction service
│   └── monitoring/
│       └── monitor.py        # RF-01–05, RF-08, RF-10 — drift monitor
├── tests/
│   ├── test_pipeline.py      # Unit tests
│   └── integration/
│       └── test_api_smoke.py # Integration smoke tests
├── config/
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.monitor
├── data/
│   ├── reference/            # Written by train.py
│   └── production/           # Written by monitor.py (or real data)
├── logs/                     # Persistent event logs (RF-10)
├── reports/                  # HTML reports (RF-03)
├── .github/workflows/
│   └── mlops.yml             # CI/CD pipeline (RF-07)
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo>
cd mlops-drift-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the full stack

```bash
docker compose up -d
```

| Service     | URL                       |
|-------------|---------------------------|
| FastAPI     | http://localhost:8000     |
| API Docs    | http://localhost:8000/docs |
| MLflow UI   | http://localhost:5000     |
| Prometheus  | http://localhost:9090     |
| Grafana     | http://localhost:3000     |

### 3. Train an initial model

```bash
# Inside the container or locally with MLFLOW_TRACKING_URI set
python src/training/train.py
```

### 4. Run the drift monitor once

```bash
MONITOR_MODE=once python src/monitoring/monitor.py
```

### 5. Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Functional Requirements Mapping

| RF   | Description                          | Implementation                      |
|------|--------------------------------------|-------------------------------------|
| RF-01 | Continuous monitoring every X min  | `schedule` loop in `monitor.py`     |
| RF-02 | KS / PSI / chi-squared tests        | `StatisticalTester` class           |
| RF-03 | Evidently HTML reports              | `EvidentlyReporter` class           |
| RF-04 | Alert system with thresholds        | `AlertManager.evaluate()`           |
| RF-05 | Auto-retrain on critical drift      | `AlertManager.trigger_retraining()` |
| RF-06 | MLflow Model Registry               | `MLflowLogger` class                |
| RF-07 | CI/CD with GitHub Actions           | `.github/workflows/mlops.yml`       |
| RF-08 | Prometheus metrics + Grafana        | `/metrics` endpoint + Pushgateway   |
| RF-09 | FastAPI prediction endpoint         | `app.py` `/predict`                 |
| RF-10 | Persistent event logs               | `drift_events.jsonl`                |

---

## Environment Variables

| Variable                  | Default                    | Description                         |
|---------------------------|----------------------------|-------------------------------------|
| `MLFLOW_TRACKING_URI`     | `http://mlflow:5000`       | MLflow server URL                   |
| `MODEL_NAME`              | `random-forest-classifier` | Registered model name               |
| `MODEL_ALIAS`             | `champion`                 | Model alias to load                 |
| `MONITOR_MODE`            | `once`                     | `once` or `scheduled`               |
| `MONITOR_INTERVAL_MINUTES`| `5`                        | Monitoring frequency in minutes     |
| `PUSHGATEWAY_URL`         | `http://pushgateway:9091`  | Prometheus Pushgateway URL          |
| `GRAFANA_PASSWORD`        | `admin123`                 | Grafana admin password              |
