"""
Infrared Spectroscopy Modeling Module

This module defines the SpectIRmodeling class, which trains, evaluates, 
and manages a multi-output Random Forest classifier. The model predicts 
the presence of molecular functional groups from infrared spectra.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

# Local imports
from ir_processing import TrainingData
from smiles_processing import smiles_to_fg_set
from fg_smarts_map import FG_IR_MAP, FG_LABELS

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SpectIRmodeling:
    """
    Manages the training, serialization, and prediction of a Random Forest 
    classifier for identifying functional groups from IR spectra.
    """

    def __init__(
        self,
        training_data: Optional[TrainingData] = None,
        n_points: int = 16,
        min_pos_samples: int = 5,
        threshold: float = 0.4,
        model_path: str | Path = "../stored/rf_multioutput_model.joblib"
    ):
        """
        Initializes the model. If training data is provided, a new model is 
        trained and saved. Otherwise, it attempts to load an existing model.

        Args:
            training_data (Optional[TrainingData]): Dataset for training.
            n_points (int): Number of grid points per spectral region.
            min_pos_samples (int): Minimum positive samples required to train a label.
            threshold (float): Probability threshold for predicting positive class.
            model_path (str | Path): Disk path for saving/loading the model state.
        """
        self.n_points = n_points
        self.min_pos_samples = min_pos_samples
        self.threshold = threshold
        self.model_path = Path(model_path)

        # Case A: Training data is provided -> Retrain and Save
        if training_data is not None:
            logging.info("Training data provided. Initializing training pipeline...")
            self.training_data = training_data
            self.X, self.Y, self.labels = self._build_training_data()
            self.model = self._train_fg_classifier()
            self.metrics = self._calculate_model_metrics()
            self.confidence = self.metrics.set_index('FG Label')['Confidence'].to_dict()
            self.summary = self._summary()

            # Save state to disk
            self._save_model()

        # Case B: No training data provided -> Attempt to Load
        else:
            logging.info(f"No training data provided. Attempting to load from {self.model_path}...")
            self._load_model()

    def _save_model(self) -> None:
        """Serializes the fitted model and metadata as a state dictionary to disk."""
        # Ensure parent directories exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "metrics": self.metrics,
            "confidence": self.confidence,
            "summary": self.summary,
            "X": self.X,
            "Y": self.Y,
            "labels": self.labels,
            "n_points": self.n_points,
            "min_pos_samples": self.min_pos_samples,
            "threshold": self.threshold
        }
        joblib.dump(state, self.model_path)
        logging.info(f"Successfully saved model and metadata to '{self.model_path}'!")

    def _load_model(self) -> None:
        """Deserializes the model and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at '{self.model_path}' and "
                "no training data was provided to train a new one."
            )

        state = joblib.load(self.model_path)

        # Restore class attributes
        self.model = state["model"]
        self.metrics = state["metrics"]
        self.confidence = state["confidence"]
        self.summary = state["summary"]
        self.X = state["X"]
        self.Y = state["Y"]
        self.labels = state["labels"]
        self.n_points = state["n_points"]
        self.min_pos_samples = state["min_pos_samples"]
        self.threshold = state["threshold"]

        logging.info(f"Successfully loaded model from '{self.model_path}'!")

    def _summary(self) -> str:
        """Generates a text summary of the model's performance metrics."""
        # 1. Isolate the two overall rows using string matching
        df_samples = self.metrics[self.metrics['FG Label'].str.contains('Samples', case=False, na=False)]
        df_macro = self.metrics[self.metrics['FG Label'].str.contains('Macro', case=False, na=False)]

        # 2. Start building the summary string
        lines = [f"Final data shape: X={self.X.shape}, Y={self.Y.shape}"]

        # 3. Add the Samples row if it exists
        if not df_samples.empty:
            s = df_samples.iloc[0]
            lines.append(
                f"OOB Metrics (Per-Compound / Samples): "
                f"Precision={s['Precision']:.3f}, "
                f"Recall={s['Recall']:.3f}, "
                f"F1-Score={s['F1-Score']:.3f}"
            )

        # 4. Add the Macro row if it exists
        if not df_macro.empty:
            m = df_macro.iloc[0]
            lines.append(
                f"OOB Metrics (Per-Class / Macro):      "
                f"Precision={m['Precision']:.3f}, "
                f"Recall={m['Recall']:.3f}, "
                f"F1-Score={m['F1-Score']:.3f}"
            )

        return "\n".join(lines)

    def _spectrum_to_regional_features(self, wn: np.ndarray, absorbance: np.ndarray) -> np.ndarray:
        """
        Resamples absorbance in each functional group region to a fixed-length grid.
        Produces a consistent feature vector regardless of raw spectral resolution.

        Args:
            wn (np.ndarray): Wavenumber array.
            absorbance (np.ndarray): Baseline-corrected absorbance array.

        Returns:
            np.ndarray: Concatenated feature vector for the model.
        """
        spectrum_max = absorbance.max() + 1e-9
        features = []

        for label, _, ranges, _ in FG_IR_MAP:
            # Normalise to always be a list of ranges
            if isinstance(ranges[0], (int, float)):
                ranges = [ranges]  # single tuple (lo, hi) → wrap in list

            fg_features = []
            for (lo, hi) in ranges:
                mask = (wn >= lo) & (wn <= hi)
                reg_wn = wn[mask]
                reg_ab = absorbance[mask]

                if len(reg_wn) < 2:
                    fg_features.append(np.zeros(self.n_points))
                    continue

                grid = np.linspace(lo, hi, self.n_points)
                resampled = interp1d(
                    reg_wn, reg_ab,
                    bounds_error=False,
                    fill_value=0.0,
                )(grid)

                fg_features.append(resampled / spectrum_max)

            # Concatenate all ranges for this FG into one block
            features.append(np.concatenate(fg_features))

        return np.concatenate(features)

    def _build_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts features and builds the binary target matrix from the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: X (features), Y (targets), and filtered labels.
        """
        mlb = MultiLabelBinarizer(classes=FG_LABELS)
        X, Y = [], []

        for c in self.training_data.spectra:
            smarts_fgs, _ = smiles_to_fg_set(c.smiles)
            feats = self._spectrum_to_regional_features(c.wn, c.absorbance)
            X.append(feats)
            Y.append(list(smarts_fgs))

        X = np.array(X)
        Y = mlb.fit_transform(Y)

        # 1. Identify which functional groups have at least min positive samples
        active_fg_mask = Y.sum(axis=0) >= self.min_pos_samples

        # 2. Filter target matrix and label list
        y_filtered = Y[:, active_fg_mask]
        fg_labels_filtered = [label for idx, label in enumerate(FG_LABELS) if active_fg_mask[idx]]

        return X, y_filtered, fg_labels_filtered

    def _define_classifier(
        self,
        n_estimators: int = 200,
        max_depth: int = 24,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
        min_samples_leaf: int = 3,
        oob_score: bool = True
    ) -> MultiOutputClassifier:
        """
        Defines the MultiOutput Random Forest architecture.

        Returns:
            MultiOutputClassifier: The configured but unfitted model pipeline.
        """
        # Random forest handles multilabel imbalance reasonably well
        # class_weight='balanced' helps for rare FGs
        base = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1,  # Keep as 1 to allow MultiOutputClassifier to handle n_jobs
            min_samples_leaf=min_samples_leaf,
            oob_score=oob_score,
        )
        return MultiOutputClassifier(base, n_jobs=n_jobs)

    def _train_fg_classifier(self) -> MultiOutputClassifier:
        """Trains the configured multi-output classifier."""
        model = self._define_classifier()
        model.fit(self.X, self.Y)
        return model

    def _get_positive_prob(self, est: RandomForestClassifier, feats: np.ndarray) -> float:
        """Extracts the probability of the positive class from a single estimator."""
        proba = est.predict_proba(feats)[0]
        if len(proba) == 1:
            # Only one class seen in training
            # Check which class the estimator knows about
            return float(proba[0]) if est.classes_[0] == 1 else 0.0
        return float(proba[1])  # index 1 = P(present)

    def _predict_fg(self, feats: np.ndarray) -> Dict[str, float]:
        """
        Predicts functional groups based on feature vectors.

        Args:
            feats (np.ndarray): Extracted regional features (shape: 1, -1).

        Returns:
            Dict[str, float]: A dictionary mapping functional groups to their probability scores.
        """
        fg_scores = {}
        for est, fg in zip(self.model.estimators_, self.labels):
            prob = self._get_positive_prob(est, feats)
            if not np.isnan(prob) and prob >= self.threshold:
                fg_scores[fg] = float(prob)
        return fg_scores

    def predict_fg_from_spectra(self, wn: np.ndarray, ab: np.ndarray) -> Dict[str, float]:
        """
        Wraps feature extraction and prediction for a raw spectrum.

        Args:
            wn (np.ndarray): Wavenumber array.
            ab (np.ndarray): Baseline-corrected absorbance array.

        Returns:
            Dict[str, float]: A dictionary of detected functional groups and their probabilities.
        """
        feats = self._spectrum_to_regional_features(wn, ab)
        return self._predict_fg(feats.reshape(1, -1))

    def _calculate_model_metrics(self) -> pd.DataFrame:
        """
        Calculates out-of-bag (OOB) classification metrics.

        Returns:
            pd.DataFrame: Sorted DataFrame of precision, recall, and F1 metrics.
        """
        n_samples, n_outputs = self.Y.shape
        y_pred_oob = np.zeros((n_samples, n_outputs))

        # This loop is safe because every estimator has both classes (0 and 1)
        for i, estimator in enumerate(self.model.estimators_):
            oob_probs = estimator.oob_decision_function_
            y_pred_oob[:, i] = (oob_probs[:, 1] >= self.threshold).astype(int)

        # 1. Calculate metrics per functional group
        fg_metrics = []
        for i in range(n_outputs):
            support = int(self.Y[:, i].sum())
            prec = precision_score(self.Y[:, i], y_pred_oob[:, i], zero_division=0)
            rec = recall_score(self.Y[:, i], y_pred_oob[:, i], zero_division=0)
            f1 = f1_score(self.Y[:, i], y_pred_oob[:, i], zero_division=0)

            # Baseline random chance prevalence
            prevalence = support / float(n_samples) if n_samples > 0 else 0

            # Determine the confidence tier
            if support >= 30 and prec >= 0.90 and rec >= 0.80:
                confidence = "complete"
            elif support >= 40 and f1 >= 0.70:
                confidence = "high"
            elif support >= 15 and (f1 >= 0.40 or f1 > (prevalence * 5)):
                confidence = "medium"
            else:
                confidence = "low"

            fg_metrics.append({
                "FG Label": self.labels[i],
                "Support": support,
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1-Score": round(f1, 3),
                "Confidence": confidence,
                "type": 0,
            })

        # 2. Calculate the OVERALL Macro-Average across all valid functional groups
        overall_precision = precision_score(self.Y, y_pred_oob, average='samples', zero_division=0)
        overall_recall = recall_score(self.Y, y_pred_oob, average='samples', zero_division=0)
        overall_f1 = f1_score(self.Y, y_pred_oob, average='samples', zero_division=0)

        # 3. Append overall metrics
        total_support = int(self.Y.sum())
        fg_metrics.append({
            "FG Label": "OVERALL (Samples Avg)",
            "Support": total_support,
            "Precision": round(overall_precision, 3),
            "Recall": round(overall_recall, 3),
            "F1-Score": round(overall_f1, 3),
            "Confidence": "",
            "type": 1,
        })

        macro_precision = precision_score(self.Y, y_pred_oob, average='macro', zero_division=0)
        macro_recall = recall_score(self.Y, y_pred_oob, average='macro', zero_division=0)
        macro_f1 = f1_score(self.Y, y_pred_oob, average='macro', zero_division=0)

        fg_metrics.append({
            "FG Label": "OVERALL (Macro Avg)",
            "Support": total_support,
            "Precision": round(macro_precision, 3),
            "Recall": round(macro_recall, 3),
            "F1-Score": round(macro_f1, 3),
            "Confidence": "",
            "type": 1,
        })

        # 4. Create and display the final DataFrame
        df_final_results = pd.DataFrame(fg_metrics).sort_values(
            by=['type', 'F1-Score'], ascending=[True, False]
        )
        df_final_results.drop(columns=['type'], inplace=True)
        return df_final_results

