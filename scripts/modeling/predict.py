import atexit
import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from pprint import pformat
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import warnings

from fastapi import FastAPI, Query, Request, Response
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp
import typer
import uvicorn

from scripts.config import (
    drift_baseline_stats_path,
    drift_default_days,
    drift_ks_pvalue_threshold,
    drift_mean_diff_threshold,
    experiment_name,
    export_dir,
    inference_log_path,
    mlflow_state_dir,
    artifact_path,
    model_name,
    model_path,
    model_results_path,
    port,
    project_root,
    test_data_dir,
    test_metadata_path,
    upstream_host,
    upstream_port,
)
from scripts.modeling.train import setup_mlflow_tracking

cli = typer.Typer()
api = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

# The MLflow scoring server subprocess, set by start_scoring_server()
_scoring_proc = None


def _setup():
    """Common setup: run from the project root, point MLflow at the local state."""
    warnings.filterwarnings("ignore")
    os.chdir(project_root)
    os.makedirs(artifact_path, exist_ok=True)
    os.makedirs(mlflow_state_dir, exist_ok=True)


# Model selection
def get_best_run(experiment_name=experiment_name):
    """Return the run of the experiment with the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_best = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    logger.info(f"Best run: {experiment_best['run_id']}")
    logger.info(f"Test F1-score: {experiment_best.get('metrics.test_f1_score')}")
    return experiment_best


def get_production_model(client, model_name=model_name):
    """Fetch the current production model, by alias or by stage.

    Returns (model_uri, version, run_id) or (None, None, None) if there is
    no model in production yet.
    """
    # First, try to get the model with the alias "Production"
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            logger.info(f"Production model: {model_name} v{model.version} (run {model.run_id})")
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    # Fall back to checking stage="Production"
    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get("current_stage") == "Production":
            logger.info(f"Production model: {model_name} v{model.version} (run {model.run_id})")
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    logger.info("No model in production")
    return None, None, None


def select_run_to_promote(experiment_best, production_model_uri, prod_model_run_id):
    """Compare the best trained model to production.

    Returns the run id to promote, or None if the production model should stay.
    """
    train_model_score = experiment_best.get("metrics.test_f1_score")
    if train_model_score is None:
        train_model_score = experiment_best.get("metrics.f1_score")

    if production_model_uri is None:
        logger.info("No model in production. The trained model will be promoted.")
        return experiment_best["run_id"]

    prod_metrics = mlflow.get_run(prod_model_run_id).data.metrics
    prod_model_score = prod_metrics.get("test_f1_score") or prod_metrics.get("f1_score")

    logger.info(
        f"Comparing models: current best (training)={train_model_score}, "
        f"production={prod_model_score}"
    )

    if train_model_score > prod_model_score:
        logger.info("New model is better. Promoting the new model.")
        return experiment_best["run_id"]

    logger.info("Current production model is better. Keeping it.")
    return None


# Promotion
def wait_for_deployment(client, model_name, version, stage="Staging", max_attempts=20):
    for _ in range(max_attempts):
        details = dict(client.get_model_version(name=model_name, version=version))
        if details["current_stage"] == stage:
            logger.info(f"Transition completed to {stage}")
            return True
        logger.info(f"Waiting for transition to {stage} (current: {details['current_stage']})")
        time.sleep(2)
    return False


def promote_model(client, run_id, model_name=model_name):
    """Promote the registered version of `run_id` to Production (always via Staging).

    Returns the promoted version.
    """
    # Find the registered version that belongs to the winning run
    versions = [
        mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id
    ]
    if not versions:
        logger.info("Run was not registered yet, registering it now")
        model_details = mlflow.register_model(
            f"runs:/{run_id}/{artifact_path}", name=model_name
        )
        version = model_details.version
    else:
        version = versions[0].version

    logger.info(f"Promoting {model_name} version {version}")

    # Always via Staging
    details = dict(client.get_model_version(name=model_name, version=version))
    if details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        wait_for_deployment(client, model_name, version, "Staging")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    wait_for_deployment(client, model_name, version, "Production")
    client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
    client.set_model_version_tag(model_name, version, "deployed_to", "production")
    logger.success(f"{model_name} version {version} is now in Production")
    return version


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def resolve_model_uri(client, model_name=model_name, experiment_name=experiment_name):
    """Resolve which model to serve: Production alias, else the newest
    registered version, else the latest run's artifact."""
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, f"{model_name} version {mv.version} (Production alias)"
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return (
                source,
                f"{model_name} version {mv.version} (newest registered; no Production alias)",
            )
    except Exception:
        pass

    runs = mlflow.search_runs(
        experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"]
    )
    if runs.empty:
        raise RuntimeError(
            f"No registered {model_name} model and no runs in experiment {experiment_name} "
            "- nothing to export"
        )
    return (
        f"runs:/{runs.iloc[0].run_id}/{artifact_path}",
        "latest run artifact (nothing registered)",
    )


def export_model(client, fallback_version=None, export_dir=export_dir, model_name=model_name):
    """Download the model to serve into `export_dir` and write version.json next to it."""
    model_uri, description = resolve_model_uri(client)
    logger.info(f"Exporting: {description}")
    logger.info(f"  Source: {model_uri}")

    export_dir = Path(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=export_dir.parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)
    logger.info(f"  Exported to: {export_dir.resolve()}")

    # Bake the version info into the export for the /version endpoint
    try:
        prod_mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        start_time = client.get_run(prod_mv.run_id).info.start_time
        trained_at = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).isoformat()
        version_info = {"model_version": str(prod_mv.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": str(fallback_version), "trained_at": None}

    with open(export_dir / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)
    logger.info(f"Version info: {version_info}")
    return version_info


# Drift detection
def _no_drift():
    return {
        "ks_statistic": 0.0,
        "ks_pvalue": 1.0,
        "drift_detected": False,
        "threshold_ks": drift_ks_pvalue_threshold,
    }


def load_inference_logs(log_path=inference_log_path):
    """Load the inference log JSON file (may not exist yet)."""
    if not Path(log_path).exists():
        logger.warning(f"Inference log not found at {log_path}")
        return []
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                return []
            return logs
    except Exception as e:
        logger.warning(f"Failed to load inference log: {e}")
        return []


def filter_logs_by_time(logs, days=drift_default_days):
    """Only keep log entries from the last N days."""
    if not logs:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for log in logs:
        try:
            timestamp_str = log.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= cutoff:
                    filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered


def load_baseline_stats(baseline_path=drift_baseline_stats_path):
    """Load the baseline statistics written during training."""
    if not Path(baseline_path).exists():
        logger.warning(f"Baseline stats not found at {baseline_path}")
        return None
    try:
        with open(baseline_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load baseline stats: {e}")
        return None


def calculate_confidence_metrics(confidences):
    confidences_array = np.array(confidences)
    if len(confidences_array) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(confidences_array)),
        "std": float(np.std(confidences_array)),
        "min": float(np.min(confidences_array)),
        "max": float(np.max(confidences_array)),
    }


def check_confidence_drift(current_confidences, baseline_stats):
    """KS test of the current confidence scores against the baseline."""
    if not baseline_stats or "confidence_statistics" not in baseline_stats:
        return _no_drift()

    baseline = baseline_stats["confidence_statistics"]
    current_confidences_array = np.array(current_confidences)

    if len(current_confidences_array) < 2:
        return _no_drift()

    baseline_mean = baseline["mean"]
    baseline_std = baseline["std"]
    current_mean = float(np.mean(current_confidences_array))

    baseline_samples = np.random.normal(
        baseline_mean, baseline_std, size=len(current_confidences_array)
    )
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current_confidences_array)

    mean_diff = abs(current_mean - baseline_mean)
    ks_drift = ks_pvalue < drift_ks_pvalue_threshold
    mean_drift = mean_diff > drift_mean_diff_threshold

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "current_mean": current_mean,
        "baseline_mean": baseline_mean,
        "drift_detected": bool(ks_drift or mean_drift),
        "threshold_ks": drift_ks_pvalue_threshold,
        "threshold_mean": drift_mean_diff_threshold,
    }


def check_class_distribution_drift(predictions, baseline_stats):
    """KS test of the predicted class distribution against the baseline."""
    if not baseline_stats or "class_distribution" not in baseline_stats:
        return _no_drift()

    baseline = baseline_stats["class_distribution"]
    baseline_total = baseline["total"]

    if len(predictions) == 0:
        return _no_drift()

    baseline_dist = np.array(
        [
            baseline["class_0_count"] / baseline_total,
            baseline["class_1_count"] / baseline_total,
        ]
    )

    baseline_samples = np.random.choice([0, 1], size=baseline_total, p=baseline_dist)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < drift_ks_pvalue_threshold),
        "threshold_ks": drift_ks_pvalue_threshold,
    }


def detect_drift(logs, days=drift_default_days, baseline_stats=None):
    """Main drift detection: confidences + class distribution."""
    filtered_logs = filter_logs_by_time(logs, days=days)

    if not filtered_logs:
        logger.warning(f"No inference logs found in the last {days} days")
        return {
            "overall_drift_detected": False,
            "num_samples": 0,
            "confidence_drift": None,
            "class_distribution_drift": None,
            "error": "No logs found",
        }

    confidences = []
    predictions = []
    for log in filtered_logs:
        if "confidence" in log and log["confidence"] is not None:
            confidences.append(float(log["confidence"]))
        if "prediction" in log and log["prediction"] is not None:
            predictions.append(int(log["prediction"]))

    confidence_results = check_confidence_drift(confidences, baseline_stats or {})
    class_dist_results = check_class_distribution_drift(predictions, baseline_stats or {})

    overall_drift = confidence_results.get("drift_detected", False) or class_dist_results.get(
        "drift_detected", False
    )

    return {
        "overall_drift_detected": overall_drift,
        "num_samples": len(filtered_logs),
        "confidence_drift": confidence_results,
        "class_distribution_drift": class_dist_results,
        "confidence_stats": calculate_confidence_metrics(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }



# Serving
# ---------------------------------------------------------------------------
def start_scoring_server(model_path=model_path):
    """Start the MLflow scoring server (the pyfunc model) on the internal port."""
    global _scoring_proc

    env = os.environ.copy()
    # The scoring server execs `uvicorn` through PATH; make sure the
    # interpreter's bin directory is on it
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "models", "serve",
            "-m", str(model_path),
            "-p", str(upstream_port),
            "--workers", "1",
            "--host", upstream_host,
            "--no-conda",
        ],
        env=env,
        # Own process group: `mlflow models serve` starts uvicorn as a child, and
        # terminating only the parent would leave uvicorn running on the port
        start_new_session=True,
    )  # fmt: skip
    atexit.register(stop_scoring_server)
    _scoring_proc = proc
    logger.info(f"Scoring server starting on port {upstream_port}")
    return proc


def stop_scoring_server():
    """Stop the scoring server and the uvicorn process it started."""
    if _scoring_proc is None or _scoring_proc.poll() is not None:
        return
    try:
        os.killpg(_scoring_proc.pid, signal.SIGTERM)
        _scoring_proc.wait(timeout=10)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(_scoring_proc.pid, signal.SIGKILL)
    logger.info("Scoring server stopped")


def _upstream_health():
    """Return the scoring server's /health status code, None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"http://{upstream_host}:{upstream_port}/health",
            timeout=1.0,
        ) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


@api.get("/health")
def health():
    if _scoring_proc is not None and _scoring_proc.poll() is not None:
        return Response(
            content='{"status": "error"}', status_code=503, media_type="application/json"
        )
    if _upstream_health() == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(
        content='{"status": "loading"}', status_code=503, media_type="application/json"
    )


@api.get("/version")
def version():
    """Model version of the served model (watched by the monitoring platform)."""
    version_file = Path(model_path) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}


@api.post("/invocations")
@api.post("/infer")
async def invocations(request: Request):
    """Forward the request unchanged to the scoring server."""
    body = await request.body()
    req = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(
                content=resp.read(),
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(
            content=e.read(),
            status_code=e.code,
            media_type=e.headers.get("content-type", "application/json"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(
            content=f'{{"status": "error", "detail": "{e}"}}',
            status_code=502,
            media_type="application/json",
        )


@api.get("/drift")
def drift(days: int = Query(default=drift_default_days, ge=1)):
    """Run drift detection over the requests logged by the scoring server."""
    logs = load_inference_logs(inference_log_path)
    baseline = load_baseline_stats()
    return detect_drift(logs, days=days, baseline_stats=baseline)


def wait_for_health(base_url, max_attempts=120):
    """Poll /health until the scoring server is up (loading torch takes a while)."""
    for attempt in range(max_attempts):
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            logger.info(f"[{attempt}] /health -> {r.status_code} {r.text.strip()}")
            if r.status_code == 200:
                logger.success("Model server is ready!")
                return True
        except requests.RequestException:
            logger.info(f"[{attempt}] waiting for the API server...")
        time.sleep(2)
    return False


# Commands
@cli.command()
def export():
    """Select the best run, promote it if it beats production, export it for serving."""
    _setup()
    setup_mlflow_tracking(experiment_name)
    client = MlflowClient()

    experiment_best = get_best_run()
    if Path(model_results_path).exists():
        with open(model_results_path) as f:
            logger.info(
                f"Classification report:\n{pformat(json.load(f)['classification_report'])}"
            )

    production_model_uri, _, prod_model_run_id = get_production_model(client)
    run_id = select_run_to_promote(experiment_best, production_model_uri, prod_model_run_id)
    logger.info(f"Selected run: {run_id}")

    promoted_version = None
    if run_id is not None:
        promoted_version = promote_model(client, run_id)

    export_model(client, fallback_version=promoted_version)


@cli.command()
def serve(host: str = "0.0.0.0", port: int = port):
    """Serve the exported model: scoring server subprocess + FastAPI wrapper (blocking)."""
    _setup()
    start_scoring_server()
    try:
        uvicorn.run(api, host=host, port=port, log_level="info")
    finally:
        stop_scoring_server()


@cli.command()
def smoke_test(port: int = port):
    """Serve the model locally, send one test image and run the drift check."""
    _setup()
    start_scoring_server()

    # uvicorn.run would block, so the API runs in a background thread
    api_server = uvicorn.Server(uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info"))
    threading.Thread(target=api_server.run, daemon=True).start()
    base_url = f"http://localhost:{port}"

    try:
        if not wait_for_health(base_url):
            raise RuntimeError("Model server did not become healthy")

        # Send one of the test images as base64 (the portable contract)
        test_metadata = pd.read_csv(test_metadata_path)
        sample_file = test_metadata.iloc[0]["filename"]
        sample_label = int(test_metadata.iloc[0]["class"])
        logger.info(f"Sample image: {sample_file} (label={sample_label})")

        with open(Path(test_data_dir) / sample_file, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        r = requests.post(f"{base_url}/infer", json={"instances": [img_b64]}, timeout=60)
        logger.info(f"POST /infer -> {r.status_code}\n{pformat(r.json())}")
        logger.info(f"GET /version -> {requests.get(f'{base_url}/version').json()}")
        logger.info(f"GET /health  -> {requests.get(f'{base_url}/health').json()}")

        # Drift check, directly and over HTTP
        logs = load_inference_logs(inference_log_path)
        logger.info(f"Logged requests: {len(logs)}")
        drift_results = detect_drift(
            logs, days=drift_default_days, baseline_stats=load_baseline_stats()
        )
        logger.info(f"Drift check results:\n{pformat(drift_results)}")
        if drift_results["overall_drift_detected"]:
            logger.warning("Drift detected - time to retrain and redeploy")
        else:
            logger.info("No drift detected")
        logger.info(f"GET /drift ->\n{pformat(requests.get(f'{base_url}/drift').json())}")
    finally:
        api_server.should_exit = True
        stop_scoring_server()
        logger.info("API server stopped")


if __name__ == "__main__":
    cli()
