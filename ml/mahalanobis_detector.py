"""
Algorithmic Trading Anomaly Detection Pipeline
Using Mahalanobis Distance and Volume Spike Analysis.

Academic Context:
Course: Pattern Recognition and Anomaly Detection (Semester 5)
Topic: Algorithmic Trading Fraud Detection Using Mahalanobis Distance
Problem Statement: Build a Mahalanobis distance pipeline to flag irregular high-frequency
trades and volume spikes in simulated financial markets.

IMPORTANT ACADEMIC CLARIFICATION:
This pipeline is designed for simulated market data. Flagged records are classified as
'Potential Anomaly' / 'Unusual Trade' / 'Candidate Fraud/Anomaly' - not confirmed fraud.
An anomaly indicates statistical divergence from the learned market pattern, warranting
further investigation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy import stats


class MahalanobisTradingDetector:
    """
    Multivariate Anomaly Detection Pipeline based on Mahalanobis Distance
    and Statistical Volume Spike (Z-Score) analysis.
    """

    REQUIRED_COLUMNS = [
        "trade_id",
        "timestamp",
        "symbol",
        "price",
        "quantity",
        "volume",
        "price_change"
    ]

    FEATURE_COLUMNS = ["price", "quantity", "volume", "price_change"]

    def __init__(
        self,
        mahalanobis_threshold: float = 3.0,
        volume_z_threshold: float = 3.0,
        epsilon_regularization: float = 1e-6
    ):
        """
        Parameters
        ----------
        mahalanobis_threshold : float
            Distance cutoff above which an observation is flagged as an anomaly.
            Default is 3.0. In chi-squared distribution with p=4 df, sqrt(chi2_inv(0.99)) ~ 3.64.
        volume_z_threshold : float
            Z-score cutoff for unilateral high-volume trade bursts.
        epsilon_regularization : float
            Small ridge added to covariance diagonal to guarantee positive-definiteness.
        """
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.volume_z_threshold = float(volume_z_threshold)
        self.epsilon = float(epsilon_regularization)

        # Fitted parameters
        self.mean_vector: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        self.inv_covariance_matrix: Optional[np.ndarray] = None
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self.volume_mean: float = 0.0
        self.volume_std: float = 1.0
        self.is_fitted: bool = False

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validates schema, column existence, non-emptiness, and numerical integrity.
        """
        if df is None or len(df) == 0:
            return False, "Dataset is empty."

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Invalid CSV: missing required column(s): {', '.join(missing_cols)}"

        # Check numeric conversions
        for col in self.FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    pd.to_numeric(df[col])
                except Exception:
                    return False, f"Invalid CSV: column '{col}' contains non-numeric values."

        return True, "Data validation passed."

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data by handling nulls, type conversions, and sorting by timestamp.
        """
        clean_df = df.copy()

        # Handle null values: forward-fill or median fill for numeric features
        for col in self.FEATURE_COLUMNS:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
            if clean_df[col].isna().any():
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

        # Ensure trade_id and symbol are clean strings
        clean_df["trade_id"] = clean_df["trade_id"].astype(str)
        clean_df["symbol"] = clean_df["symbol"].astype(str)
        clean_df["timestamp"] = clean_df["timestamp"].astype(str)

        return clean_df

    def fit(self, df: pd.DataFrame) -> "MahalanobisTradingDetector":
        """
        Estimates the baseline multivariate pattern:
        1. Feature mean vector mu
        2. Feature covariance matrix Sigma
        3. Moore-Penrose pseudo-inverse of covariance Sigma^+ with ridge regularization
        4. Volume univariate baseline (mean and standard deviation)
        """
        clean_df = self.preprocess_data(df)
        X = clean_df[self.FEATURE_COLUMNS].values

        # 1. Mean vector (p,)
        self.mean_vector = np.mean(X, axis=0)
        for idx, col in enumerate(self.FEATURE_COLUMNS):
            self.feature_means[col] = float(self.mean_vector[idx])
            self.feature_stds[col] = float(np.std(X[:, idx]) + 1e-9)

        # 2. Covariance matrix (p, p)
        # Using rowvar=False because rows are observations, columns are features
        cov = np.cov(X, rowvar=False)

        # Add ridge regularization along the diagonal to prevent ill-conditioning
        cov_reg = cov + self.epsilon * np.eye(cov.shape[0])
        self.covariance_matrix = cov_reg

        # 3. Numerically stable Moore-Penrose pseudo-inverse
        # This prevents crashes even if columns are collinear or singular
        self.inv_covariance_matrix = np.linalg.pinv(cov_reg)

        # 4. Volume univariate statistics for Z-score spike detector
        self.volume_mean = float(clean_df["volume"].mean())
        self.volume_std = float(clean_df["volume"].std())
        if self.volume_std <= 0:
            self.volume_std = 1.0

        self.is_fitted = True
        return self

    def calculate_mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates Mahalanobis distance for each observation in matrix X.
        D_M(x) = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing Mahalanobis distance.")

        diff = X - self.mean_vector  # (N, p)
        # Vectorized calculation: left_term = diff @ inv_cov (N, p)
        # distance^2 = sum(left_term * diff, axis=1)
        left_term = np.dot(diff, self.inv_covariance_matrix)
        dist_sq = np.sum(left_term * diff, axis=1)

        # Clip numerical inaccuracies that may lead to small negative values
        dist_sq = np.clip(dist_sq, 0, None)
        return np.sqrt(dist_sq)

    def calculate_volume_zscore(self, volumes: np.ndarray) -> np.ndarray:
        """
        Computes the statistical Z-score for trade volume:
        Z = (volume - mean_volume) / std_volume
        """
        return (volumes - self.volume_mean) / self.volume_std

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes complete detection pipeline on the input dataset:
        - Calculates Mahalanobis Distance
        - Calculates Volume Z-Score
        - Applies configurable thresholds
        - Formulates diagnostic classification and reason breakdown
        """
        clean_df = self.preprocess_data(df)

        if not self.is_fitted:
            self.fit(clean_df)

        X = clean_df[self.FEATURE_COLUMNS].values
        volumes = clean_df["volume"].values

        # 1. Mahalanobis distance calculation
        md_scores = self.calculate_mahalanobis_distance(X)
        clean_df["mahalanobis_distance"] = np.round(md_scores, 4)

        # 2. Volume Z-Score calculation
        vz_scores = self.calculate_volume_zscore(volumes)
        clean_df["volume_zscore"] = np.round(vz_scores, 4)

        # 3. Individual flags based on thresholds
        clean_df["flag_mahalanobis"] = clean_df["mahalanobis_distance"] > self.mahalanobis_threshold
        clean_df["flag_volume"] = clean_df["volume_zscore"] > self.volume_z_threshold

        # 4. Combined anomaly classification
        clean_df["is_anomaly"] = clean_df["flag_mahalanobis"] | clean_df["flag_volume"]
        clean_df["status"] = clean_df["is_anomaly"].map(
            {True: "POTENTIAL ANOMALY", False: "NORMAL"}
        )

        # 5. Diagnostic reason explanation
        def determine_reason(row):
            if row["flag_mahalanobis"] and row["flag_volume"]:
                return "Both"
            elif row["flag_mahalanobis"]:
                return "Mahalanobis Distance"
            elif row["flag_volume"]:
                return "Volume Spike"
            else:
                return "None"

        clean_df["reason"] = clean_df.apply(determine_reason, axis=1)

        # 6. Detailed academic interpretation narrative
        def format_interpretation(row):
            if row["status"] == "POTENTIAL ANOMALY":
                if row["reason"] == "Both":
                    return (
                        f"Trade exhibits both an elevated Mahalanobis distance ({row['mahalanobis_distance']:.2f} > {self.mahalanobis_threshold}) "
                        f"and an extreme volume spike (Z={row['volume_zscore']:.2f} > {self.volume_z_threshold}). "
                        "This multivariate and volume divergence indicates potential irregular trading behavior."
                    )
                elif row["reason"] == "Mahalanobis Distance":
                    return (
                        f"Mahalanobis distance ({row['mahalanobis_distance']:.2f}) substantially exceeds the configured anomaly "
                        f"threshold ({self.mahalanobis_threshold}). Joint feature relationships (price, quantity, volume, price change) "
                        "deviate significantly from the normal covariance structure."
                    )
                else:
                    return (
                        f"Trade volume is {row['volume_zscore']:.2f} standard deviations above the market mean, "
                        f"exceeding the volume spike threshold of {self.volume_z_threshold}. Flagged as a potential volume spike."
                    )
            else:
                return (
                    f"Observation lies well within normal multivariate trading boundaries (Mahalanobis distance: {row['mahalanobis_distance']:.2f} <= {self.mahalanobis_threshold}, "
                    f"Volume Z-score: {row['volume_zscore']:.2f} <= {self.volume_z_threshold})."
                )

        clean_df["interpretation"] = clean_df.apply(format_interpretation, axis=1)

        return clean_df

    def get_summary_statistics(self, scored_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Dynamically computes all high-level statistical aggregates for the dashboard.
        NEVER hardcodes metrics.
        """
        total_records = int(len(scored_df))
        anomaly_records = int(scored_df["is_anomaly"].sum())
        normal_records = total_records - anomaly_records
        anomaly_pct = round((anomaly_records / total_records * 100) if total_records > 0 else 0.0, 2)

        avg_volume = round(float(scored_df["volume"].mean()), 2)
        max_volume = round(float(scored_df["volume"].max()), 2)
        avg_md = round(float(scored_df["mahalanobis_distance"].mean()), 4)
        max_md = round(float(scored_df["mahalanobis_distance"].max()), 4)

        # Flag breakdown
        reason_counts = scored_df["reason"].value_counts().to_dict()
        md_only_count = int(reason_counts.get("Mahalanobis Distance", 0))
        vol_only_count = int(reason_counts.get("Volume Spike", 0))
        both_count = int(reason_counts.get("Both", 0))

        # Symbol breakdown
        symbol_stats = {}
        for sym, group in scored_df.groupby("symbol"):
            sym_total = len(group)
            sym_anom = int(group["is_anomaly"].sum())
            symbol_stats[sym] = {
                "total": sym_total,
                "anomalies": sym_anom,
                "anomaly_rate": round((sym_anom / sym_total * 100) if sym_total > 0 else 0.0, 2),
                "avg_price": round(float(group["price"].mean()), 2),
                "avg_volume": round(float(group["volume"].mean()), 2)
            }

        # Evaluation metrics against synthetic ground truth (if present)
        evaluation = None
        if "synthetic_anomaly" in scored_df.columns:
            y_true = scored_df["synthetic_anomaly"].astype(int).values
            y_pred = scored_df["is_anomaly"].astype(int).values

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))

            precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0

            evaluation = {
                "has_ground_truth": True,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "disclaimer": (
                    "Because verified real-world fraud labels are unavailable, classification accuracy "
                    "cannot be interpreted as real-world fraud-detection accuracy. Evaluation is based "
                    "primarily on anomaly-distance distributions, volume-spike detection, and consistency "
                    "of the synthetic test scenarios."
                )
            }
        else:
            evaluation = {
                "has_ground_truth": False,
                "disclaimer": (
                    "Custom uploaded dataset does not contain synthetic ground-truth labels. Evaluation "
                    "is based strictly on unsupervised Mahalanobis distance and statistical volume distributions."
                )
            }

        return {
            "total_trades": total_records,
            "normal_trades": normal_records,
            "anomalous_trades": anomaly_records,
            "anomaly_percentage": anomaly_pct,
            "average_volume": avg_volume,
            "maximum_volume": max_volume,
            "average_mahalanobis": avg_md,
            "maximum_mahalanobis": max_md,
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "volume_z_threshold": self.volume_z_threshold,
            "reasons": {
                "mahalanobis_only": md_only_count,
                "volume_only": vol_only_count,
                "both": both_count
            },
            "symbols": symbol_stats,
            "evaluation": evaluation,
            "covariance_features": self.FEATURE_COLUMNS,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds
        }
