"""
Data drift monitoring script for the drift mitigation pipeline.

Compares a reference dataset against a production snapshot using
Evidently AI and statistical tests (KS, PSI, Chi-Square).
Exports metrics to Prometheus and triggers retraining if critical
drift is detected. (RF-01 – RF-05, RF-08, RF-10).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import schedule
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnDrift,
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DriftThresholds:
    """Statistical thresholds that trigger drift alerts."""

    ks_p_value: float = 0.05
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    chi2_p_value: float = 0.05
    evidently_dataset_drift: float = 0.5


@dataclass
class MonitoringConfig:
    """Runtime configuration for the monitoring pipeline."""

    reference_path: str = "data/reference/reference_dataset.csv"
    production_path: str = "data/production/production_dataset.csv"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    metrics_dir: str = "logs/metrics"
    pushgateway_url: str = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    interval_minutes: int = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))
    project_name: str = "MLOps_Drift_Pipeline"
    feature_names: list[str] = field(default_factory=lambda: [
        f"feature_{i}" for i in range(10)
    ])
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class StatisticalTester:
    """Applies KS, PSI and chi-squared tests on feature columns."""

    def __init__(self, thresholds: DriftThresholds) -> None:
        self.thresholds = thresholds

    @staticmethod
    def _compute_psi(
            reference: np.ndarray,
            production: np.ndarray,
            bins: int = 10,
            epsilon: float = 1e-6,
    ) -> float:
        """Compute the Population Stability Index between two distributions."""
        combined = np.concatenate([reference, production])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        prod_hist, _ = np.histogram(production, bins=bin_edges)
        ref_pct = (ref_hist + epsilon) / (len(reference) + epsilon * bins)
        prod_pct = (prod_hist + epsilon) / (len(production) + epsilon * bins)
        return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

    def run_ks_test(self, reference: pd.Series, production: pd.Series) -> dict[str, Any]:
        """Run a two-sample Kolmogorov–Smirnov test."""
        stat, p_value = stats.ks_2samp(reference.values, production.values)
        return {
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(p_value < self.thresholds.ks_p_value),
        }

    def run_psi(self, reference: pd.Series, production: pd.Series) -> dict[str, Any]:
        """Compute PSI and classify drift severity."""
        psi = self._compute_psi(reference.values, production.values)
        severity = "none" if psi < self.thresholds.psi_warning else "warning" if psi < self.thresholds.psi_critical else "critical"
        return {
            "psi": round(psi, 6),
            "severity": severity,
            "drift_detected": bool(severity != "none"),
        }

    def run_chi2_test(self, reference: pd.Series, production: pd.Series, bins: int = 10) -> dict[str, Any]:
        """Run a chi-squared test for feature distributions."""
        combined_min, combined_max = min(reference.min(), production.min()), max(reference.max(), production.max())
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)
        ref_counts, _ = np.histogram(reference.values, bins=bin_edges)
        prod_counts, _ = np.histogram(production.values, bins=bin_edges)
        epsilon = 1e-8
        ref_freq = (ref_counts + epsilon) / (ref_counts + epsilon).sum()
        prod_freq = (prod_counts + epsilon) / (prod_counts + epsilon).sum()
        n = len(production)
        stat, p_value = stats.chisquare(prod_freq * n, f_exp=ref_freq * n)
        return {
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(p_value < self.thresholds.chi2_p_value),
        }

    def analyse_all_features(self, reference: pd.DataFrame, production: pd.DataFrame, feature_cols: list[str]) -> dict[
        str, dict[str, Any]]:
        """Run all statistical tests for each feature column."""
        return {
            col: {
                "ks": self.run_ks_test(reference[col], production[col]),
                "psi": self.run_psi(reference[col], production[col]),
                "chi2": self.run_chi2_test(reference[col], production[col]),
            } for col in feature_cols
        }


# ---------------------------------------------------------------------------
# Evidently report generator
# ---------------------------------------------------------------------------

class EvidentlyReporter:
    """Generates Evidently HTML drift reports and test suites (RF-03)."""

    @staticmethod
    def generate_report(reference: pd.DataFrame, production: pd.DataFrame, output_path: str) -> Report:
        """Build and save a full data-drift HTML report."""
        report = Report(metrics=[DatasetDriftMetric(), DataDriftTable(), DatasetMissingValuesMetric()])
        report.run(reference_data=reference, current_data=production)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        logger.info("Evidently HTML report saved to %s.", output_path)
        return report

    @staticmethod
    def run_test_suite(reference: pd.DataFrame, production: pd.DataFrame, feature_cols: list[str],
                       drift_share_threshold: float = 0.5) -> tuple[TestSuite, bool]:
        """Run an Evidently TestSuite and return pass/fail result."""
        tests = [
                    TestNumberOfDriftedColumns(lt=max(1, int(len(feature_cols) * drift_share_threshold))),
                    TestShareOfDriftedColumns(lt=drift_share_threshold),
                ] + [TestColumnDrift(column_name=col) for col in feature_cols[:5]]

        suite = TestSuite(tests=tests)
        suite.run(reference_data=reference, current_data=production)
        return suite, bool(suite.as_dict()["summary"]["all_passed"])


# ---------------------------------------------------------------------------
# Alert & retraining trigger
# ---------------------------------------------------------------------------

class AlertManager:
    """Evaluates drift results and emits structured alerts (RF-04, RF-05)."""

    def __init__(self, thresholds: DriftThresholds, logs_dir: str) -> None:
        self.thresholds, self.logs_dir = thresholds, logs_dir
        os.makedirs(logs_dir, exist_ok=True)

    def evaluate(self, statistical_results: dict[str, dict[str, Any]], evidently_passed: bool) -> dict[str, Any]:
        """Decide alert level based on aggregated drift evidence."""
        drifted_features = [col for col, tests in statistical_results.items() if
                            any(t["drift_detected"] for t in tests.values())]
        critical_psi = [col for col, tests in statistical_results.items() if tests["psi"]["severity"] == "critical"]
        drift_fraction = len(drifted_features) / max(len(statistical_results), 1)

        level = "CRITICAL" if (
                    critical_psi or drift_fraction > self.thresholds.evidently_dataset_drift) else "WARNING" if (
                    drifted_features or not evidently_passed) else "OK"

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "drifted_features": drifted_features,
            "drift_fraction": round(drift_fraction, 4),
            "retrain_required": level == "CRITICAL"
        }

    def log_alert(self, alert: dict[str, Any]) -> None:
        """Append the alert to the persistent event log (RF-10)."""
        with open(os.path.join(self.logs_dir, "drift_events.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert) + "\n")
        logger.info("Alert logged: level=%s | drifted=%s", alert["level"], alert["drifted_features"])

    def trigger_retraining(self, alert: dict[str, Any]) -> None:
        """Simulate automatic retraining trigger (RF-05)."""
        if not alert["retrain_required"]: return
        logger.warning("CRITICAL DRIFT DETECTED — triggering retraining pipeline.")
        with open(os.path.join(self.logs_dir, "retraining_triggers.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**alert, "action": "retraining_triggered"}) + "\n")


# ---------------------------------------------------------------------------
# Prometheus metrics exporter
# ---------------------------------------------------------------------------

class MetricsExporter:
    """Pushes drift metrics to the Prometheus Pushgateway (RF-08)."""

    def __init__(self, pushgateway_url: str, metrics_dir: str) -> None:
        self.pushgateway_url, self.metrics_dir = pushgateway_url, metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

    def export(self, alert: dict[str, Any], statistical_results: dict[str, dict[str, Any]]) -> None:
        """Push metrics to Pushgateway and write to disk."""
        registry = CollectorRegistry()
        drift_level = Gauge("drift_alert_level", "0=OK, 1=WARNING, 2=CRITICAL", registry=registry)
        drift_frac = Gauge("drift_fraction", "Fraction of features with drift", registry=registry)
        psi_g = Gauge("feature_psi", "PSI per feature", ["feature"], registry=registry)

        drift_level.set({"OK": 0, "WARNING": 1, "CRITICAL": 2}.get(alert["level"], 0))
        drift_frac.set(alert["drift_fraction"])
        for feat, tests in statistical_results.items():
            psi_g.labels(feature=feat).set(tests.get("psi", {}).get("psi", 0))

        try:
            push_to_gateway(self.pushgateway_url, job="drift_monitor", registry=registry)
            logger.info("Metrics pushed to Pushgateway.")
        except Exception as e:
            logger.warning("Pushgateway unavailable: %s", e)


# ---------------------------------------------------------------------------
# Monitoring pipeline
# ---------------------------------------------------------------------------

class MonitoringPipeline:
    """Orchestrates the full drift monitoring cycle (RF-01 – RF-05)."""

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.tester = StatisticalTester(config.thresholds)
        self.alert_manager = AlertManager(config.thresholds, config.logs_dir)
        self.exporter = MetricsExporter(config.pushgateway_url, config.metrics_dir)

    def _load_production_data(self) -> pd.DataFrame:
        """Load production data from disk. Returns empty DF if file missing."""
        path = self.config.production_path
        if not Path(path).exists():
            logger.info("Production dataset not found at %s. Waiting for API traffic.", path)
            return pd.DataFrame()
        return pd.read_csv(path)

    def run_once(self) -> dict[str, Any]:
        """Execute a single monitoring cycle based on current disk state."""
        logger.info("=" * 60)
        logger.info("Monitoring cycle started at %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 60)

        # 1. Load data
        ref = pd.read_csv(self.config.reference_path)
        prod = self._load_production_data()
        feature_cols = [c for c in ref.columns if c != "target"]

        # 2. Case: No production data yet
        if prod.empty:
            logger.info("No production data available. Reporting status: OK (0).")
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "OK",
                "drifted_features": [],
                "drift_fraction": 0.0,
                "retrain_required": False
            }
            # Export zero metrics to keep Grafana clean
            dummy_results = {col: {"psi": {"psi": 0}, "ks": {"statistic": 0}} for col in feature_cols}
            self.exporter.export(alert, dummy_results)
            return alert

        # 3. Calculation: Real data exists
        logger.info("Running statistical drift tests on %d samples...", len(prod))
        results = self.tester.analyse_all_features(ref, prod, feature_cols)

        # Professional naming for report
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = f"{self.config.project_name}_Analysis_{ts}.html"
        rpath = os.path.join(self.config.reports_dir, fname)

        EvidentlyReporter.generate_report(ref[feature_cols], prod[feature_cols], rpath)
        _, passed = EvidentlyReporter.run_test_suite(ref[feature_cols], prod[feature_cols], feature_cols)

        # 4. Alert & Export
        alert = self.alert_manager.evaluate(results, passed)
        self.alert_manager.log_alert(alert)
        self.exporter.export(alert, results)
        self.alert_manager.trigger_retraining(alert)

        logger.info("Monitoring cycle complete. Alert level: %s", alert["level"])
        return alert

    def run_scheduled(self) -> None:
        """Run the monitoring pipeline on a recurring schedule."""
        schedule.every(self.config.interval_minutes).minutes.do(self.run_once)
        self.run_once()  # Startup run
        while True:
            schedule.run_pending()
            time.sleep(30)


if __name__ == "__main__":
    pipeline = MonitoringPipeline(MonitoringConfig())
    if os.getenv("MONITOR_MODE") == "scheduled":
        pipeline.run_scheduled()
    else:
        pipeline.run_once()