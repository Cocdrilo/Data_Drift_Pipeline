"""
Training script for the data drift mitigation pipeline.

Trains a Random Forest classifier on synthetic data, logs hyperparameters,
metrics, and registers the model in the MLflow Model Registry.
Supports adaptive retraining by incorporating production data.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for synthetic dataset generation."""

    n_samples: int = 5_000
    n_features: int = 10
    n_informative: int = 6
    n_redundant: int = 2
    random_state: int = 42
    test_size: float = 0.2
    feature_names: list[str] = field(default_factory=lambda: [
        f"feature_{i}" for i in range(10)
    ])


@dataclass
class ModelConfig:
    """Hyperparameter configuration for Random Forest."""

    n_estimators: int = 100
    max_depth: int = 6
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class MLflowConfig:
    """MLflow tracking and registry configuration."""

    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_name: str = "drift-mitigation-pipeline"
    model_name: str = "random-forest-classifier"
    artifact_path: str = "model"
    registered_model_alias: str = "champion"


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class DataGenerator:
    """Generates and preprocesses synthetic classification data."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def generate(self) -> tuple[pd.DataFrame, pd.Series]:
        """Generate a synthetic classification dataset.

        Returns:
            A tuple of (features DataFrame, target Series).
        """
        logger.info("Generating synthetic dataset with %d samples.", self.config.n_samples)
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            random_state=self.config.random_state,
        )
        df = pd.DataFrame(X, columns=self.config.feature_names)
        target = pd.Series(y, name="target")
        return df, target

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train / test partitions.

        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

    def save_reference_dataset(self, X: pd.DataFrame, y: pd.Series, path: str) -> None:
        """Persist the training data as the drift reference dataset.

        Args:
            X: Feature matrix.
            y: Target vector.
            path: Destination file path (.csv).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        reference = X.copy()
        reference["target"] = y.values
        reference.to_csv(path, index=False)
        logger.info("Reference dataset saved to %s.", path)


# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Encapsulates model training and evaluation logic."""

    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()

    def build(self) -> RandomForestClassifier:
        """Instantiate and return the Random Forest estimator."""
        self.model = RandomForestClassifier(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            min_samples_split=self.model_config.min_samples_split,
            min_samples_leaf=self.model_config.min_samples_leaf,
            max_features=self.model_config.max_features,
            random_state=self.model_config.random_state,
            n_jobs=self.model_config.n_jobs,
        )
        return self.model

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "ModelTrainer":
        """Scale features and fit the model.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.

        Returns:
            Self for method chaining.
        """
        if self.model is None:
            self.build()

        logger.info("Fitting model on %d samples.", len(X_train))
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)  # type: ignore[union-attr]
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Compute classification metrics on held-out data.

        Args:
            X_test: Test feature matrix.
            y_test: True labels.

        Returns:
            Dictionary of metric name → value.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        for name, value in metrics.items():
            logger.info("  %s: %.4f", name, value)

        return metrics

    def get_hyperparams(self) -> dict[str, Any]:
        """Return a flat dictionary of tracked hyperparameters."""
        return {
            "n_estimators": self.model_config.n_estimators,
            "max_depth": self.model_config.max_depth,
            "min_samples_split": self.model_config.min_samples_split,
            "min_samples_leaf": self.model_config.min_samples_leaf,
            "max_features": self.model_config.max_features,
        }


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------

class MLflowLogger:
    """Handles all MLflow tracking and model-registry operations."""

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
        logger.info("MLflow tracking URI: %s", config.tracking_uri)

    def log_run(
        self,
        trainer: ModelTrainer,
        metrics: dict[str, float],
        X_sample: pd.DataFrame,
    ) -> str:
        """Execute a full MLflow run: params, metrics, model artifact.

        Args:
            trainer: Fitted ModelTrainer instance.
            metrics: Evaluation metrics to log.
            X_sample: Small sample used to infer model signature.

        Returns:
            The MLflow run ID.
        """
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("MLflow run started: %s", run_id)

            # --- Parameters ---
            mlflow.log_params(trainer.get_hyperparams())

            # --- Metrics ---
            mlflow.log_metrics(metrics)

            # --- Model artifact ---
            X_scaled_sample = trainer.scaler.transform(X_sample)
            signature = infer_signature(
                X_scaled_sample,
                trainer.model.predict(X_scaled_sample),  # type: ignore[union-attr]
            )
            mlflow.sklearn.log_model(
                sk_model=trainer.model,
                artifact_path=self.config.artifact_path,
                signature=signature,
                registered_model_name=self.config.model_name,
                input_example=X_sample.head(3),
            )
            logger.info("Model logged and registered as '%s'.", self.config.model_name)

        return run_id

    def promote_latest_version(self) -> None:
        """Assign the 'champion' alias to the most recently registered version."""
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{self.config.model_name}'")
        if not versions:
            logger.warning("No registered versions found for model '%s'.", self.config.model_name)
            return

        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(
            name=self.config.model_name,
            alias=self.config.registered_model_alias,
            version=latest.version,
        )
        logger.info(
            "Alias '%s' → version %s of model '%s'.",
            self.config.registered_model_alias,
            latest.version,
            self.config.model_name,
        )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """Orchestrates the end-to-end training workflow."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        mlflow_config: MLflowConfig,
    ) -> None:
        self.data_generator = DataGenerator(data_config)
        self.trainer = ModelTrainer(model_config)
        self.mlflow_logger = MLflowLogger(mlflow_config)
        self.prod_path = "data/production/production_dataset.csv"

    def _load_production_data(self) -> pd.DataFrame:
        """Load production data and simulate ground truth labeling.

        Returns:
            A DataFrame containing production features and simulated targets.
        """
        if not os.path.exists(self.prod_path):
            return pd.DataFrame()

        df = pd.read_csv(self.prod_path)
        if df.empty:
            return pd.DataFrame()

        # Simulated ground truth for TFG demonstration
        df["target"] = np.random.randint(0, 2, size=len(df))
        logger.info("Loaded %d production samples for retraining.", len(df))
        return df

    def _cleanup_production_data(self) -> None:
        """Delete the production dataset after successful retraining.

        This ensures the next monitoring cycle starts from a clean slate
        with the new reference baseline.
        """
        if os.path.exists(self.prod_path):
            try:
                os.remove(self.prod_path)
                logger.info("Production dataset cleaned up successfully.")
            except Exception as e:
                logger.error("Failed to delete production dataset: %s", e)

    def run(self) -> None:
        """Execute the full training pipeline incorporating production data."""
        logger.info("=" * 60)
        logger.info("Starting Adaptive Training Pipeline")
        logger.info("=" * 60)

        # 1. Load data
        X_base, y_base = self.data_generator.generate()
        df_base = X_base.copy()
        df_base["target"] = y_base.values

        # 2. Load and merge production data
        df_prod = self._load_production_data()
        if not df_prod.empty:
            df_final = pd.concat([df_base, df_prod], ignore_index=True)
            logger.info("Merged dataset size: %d samples.", len(df_final))
        else:
            df_final = df_base

        X = df_final.drop(columns=["target"])
        y = df_final["target"]

        X_train, X_test, y_train, y_test = self.data_generator.split(X, y)

        # 3. Save NEW reference dataset (mitigates drift)
        self.data_generator.save_reference_dataset(
            X_train, y_train, path="data/reference/reference_dataset.csv"
        )

        # 4. Train & evaluate
        self.trainer.build()
        self.trainer.fit(X_train, y_train)
        metrics = self.trainer.evaluate(X_test, y_test)

        # 5. Log to MLflow
        run_id = self.mlflow_logger.log_run(
            trainer=self.trainer,
            metrics=metrics,
            X_sample=X_test.head(50),
        )
        self.mlflow_logger.promote_latest_version()

        # 6. Cleanup (The final piece of the puzzle)
        self._cleanup_production_data()

        logger.info("Training pipeline completed. Run ID: %s", run_id)




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = TrainingPipeline(
        data_config=DataConfig(),
        model_config=ModelConfig(),
        mlflow_config=MLflowConfig(),
    )
    pipeline.run()