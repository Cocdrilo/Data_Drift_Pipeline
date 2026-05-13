"""
Unit tests for the data drift mitigation pipeline.

Covers: DataGenerator, ModelTrainer, StatisticalTester, AlertManager.
MLflow calls are avoided as these tests focus on logic and data processing.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# GLOBAL PATH FIX (Soluciona el problema de importación de 'src')
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.train import DataConfig, DataGenerator, ModelConfig, ModelTrainer
from src.monitoring.monitor import DriftThresholds, StatisticalTester, AlertManager

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FEATURE_NAMES = [f"feature_{i}" for i in range(10)]

def make_reference(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((n, 10)), columns=FEATURE_NAMES)
    df["target"] = rng.integers(0, 2, size=n)
    return df

def make_production(n: int = 200, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        rng.standard_normal((n, 10)) + drift, columns=FEATURE_NAMES
    )
    return df


# ===========================================================================
# Training tests
# ===========================================================================

class TestDataGenerator:

    def test_generate_shape(self):
        gen = DataGenerator(DataConfig(n_samples=200))
        X, y = gen.generate()

        assert X.shape == (200, 10)
        assert len(y) == 200
        assert set(y.unique()).issubset({0, 1})

    def test_split_proportions(self):
        gen = DataGenerator(DataConfig(n_samples=1000, test_size=0.2))
        X, y = gen.generate()
        X_train, X_test, y_train, y_test = gen.split(X, y)

        assert len(X_test) == pytest.approx(200, abs=5)
        assert len(X_train) + len(X_test) == 1000

    def test_save_reference_dataset(self, tmp_path):
        gen = DataGenerator(DataConfig(n_samples=100))
        X, y = gen.generate()
        out = str(tmp_path / "ref" / "reference.csv")
        gen.save_reference_dataset(X, y, path=out)

        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert "target" in df.columns
        assert len(df) == 100


class TestModelTrainer:

    def test_build_returns_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        trainer = ModelTrainer(ModelConfig())
        model = trainer.build()
        assert isinstance(model, RandomForestClassifier)

    def test_fit_and_evaluate(self):
        """Test the full fit/evaluate cycle ensuring the scaler is fitted."""
        gen = DataGenerator(DataConfig(n_samples=400))
        X, y = gen.generate()
        X_train, X_test, y_train, y_test = gen.split(X, y)

        trainer = ModelTrainer(ModelConfig(n_estimators=10))

        # CORRECCIÓN: Llamamos a trainer.fit() en lugar de trainer.build().fit()
        # Esto asegura que el StandardScaler interno también se entrene.
        trainer.fit(X_train, y_train)

        metrics = trainer.evaluate(X_test, y_test)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score", "roc_auc"}

    def test_hyperparams_match_config(self):
        cfg = ModelConfig(n_estimators=77, max_depth=4)
        trainer = ModelTrainer(cfg)
        params = trainer.get_hyperparams()

        assert params["n_estimators"] == 77
        assert params["max_depth"] == 4


# ===========================================================================
# Monitoring tests
# ===========================================================================

class TestStatisticalTester:

    @pytest.fixture
    def tester(self):
        return StatisticalTester(DriftThresholds())

    def test_ks_no_drift(self, tester):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.standard_normal(500))
        prod = pd.Series(rng.standard_normal(500))
        result = tester.run_ks_test(ref, prod)
        assert "statistic" in result
        assert "p_value" in result
        assert isinstance(result["drift_detected"], bool)

    def test_ks_detects_drift(self, tester):
        rng = np.random.default_rng(1)
        ref = pd.Series(rng.standard_normal(500))
        prod = pd.Series(rng.standard_normal(500) + 5.0)
        result = tester.run_ks_test(ref, prod)
        assert result["drift_detected"] is True

    def test_psi_no_drift(self, tester):
        rng = np.random.default_rng(2)
        ref = pd.Series(rng.standard_normal(500))
        prod = pd.Series(rng.standard_normal(500))
        result = tester.run_psi(ref, prod)
        assert result["severity"] == "none"

    def test_psi_critical_drift(self, tester):
        rng = np.random.default_rng(3)
        ref = pd.Series(rng.standard_normal(500))
        prod = pd.Series(rng.standard_normal(500) + 10.0)
        result = tester.run_psi(ref, prod)
        assert result["severity"] == "critical"

    def test_chi2_returns_expected_keys(self, tester):
        rng = np.random.default_rng(4)
        ref = pd.Series(rng.standard_normal(300))
        prod = pd.Series(rng.standard_normal(300))
        result = tester.run_chi2_test(ref, prod)
        assert {"statistic", "p_value", "drift_detected"} == set(result.keys())

    def test_analyse_all_features(self, tester):
        reference = make_reference()
        production = make_production(n=200, drift=3.0)
        cols = FEATURE_NAMES
        results = tester.analyse_all_features(
            reference[cols], production[cols], cols
        )
        assert set(results.keys()) == set(cols)
        for col_results in results.values():
            assert {"ks", "psi", "chi2"} == set(col_results.keys())


class TestAlertManager:

    @pytest.fixture
    def manager(self, tmp_path):
        return AlertManager(DriftThresholds(), logs_dir=str(tmp_path))

    def test_ok_when_no_drift(self, manager):
        no_drift = {
            "feature_0": {
                "ks":   {"drift_detected": False},
                "psi":  {"drift_detected": False, "severity": "none"},
                "chi2": {"drift_detected": False},
            }
        }
        alert = manager.evaluate(no_drift, evidently_passed=True)
        assert alert["level"] == "OK"
        assert alert["retrain_required"] is False

    def test_warning_partial_drift(self, manager):
        partial = {
            "feature_0": {
                "ks":   {"drift_detected": True},
                "psi":  {"drift_detected": False, "severity": "none"},
                "chi2": {"drift_detected": False},
            },
            "feature_1": {
                "ks":   {"drift_detected": False},
                "psi":  {"drift_detected": False, "severity": "none"},
                "chi2": {"drift_detected": False},
            },
        }
        alert = manager.evaluate(partial, evidently_passed=True)
        assert alert["level"] in {"WARNING", "CRITICAL"}

    def test_critical_on_psi_critical(self, manager):
        critical_psi = {
            f"feature_{i}": {
                "ks":   {"drift_detected": True},
                "psi":  {"drift_detected": True, "severity": "critical"},
                "chi2": {"drift_detected": True},
            }
            for i in range(6)
        }
        alert = manager.evaluate(critical_psi, evidently_passed=False)
        assert alert["level"] == "CRITICAL"
        assert alert["retrain_required"] is True

    def test_log_alert_writes_jsonl(self, manager, tmp_path):
        # CORRECCIÓN: Estructura del alert actualizada según el monitor honesto
        alert = {
            "timestamp": "2026-05-13T10:00:00+00:00",
            "level": "WARNING",
            "drifted_features": ["feature_0"],
            "drift_fraction": 0.1,
            "retrain_required": False,
        }
        manager.log_alert(alert)
        log_file = Path(manager.logs_dir) / "drift_events.jsonl"
        assert log_file.exists()
        entry = json.loads(log_file.read_text().strip())
        assert entry["level"] == "WARNING"