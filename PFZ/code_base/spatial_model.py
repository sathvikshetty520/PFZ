"""
Spatial Machine Learning Models (Random Forest & Gradient Boosting)
Project: AI-Based Dynamic Fishing Zone Prediction System (Sahyadri College / VTU)

Implements:
1. LightGBM Gradient Boosting Classifier (handles missing chlorophyll natively).
2. Random Forest Classifier & Regressor (Scikit-Learn).
3. Metric evaluations: Precision, Recall, F1, ROC-AUC, PR-AUC.
4. Feature importance ranking & visualization.
5. Model persistence (save/load).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
from .config import MODELS_DIR, ALL_MODEL_FEATURES, RANDOM_SEED

class SpatialPFZModel:
    def __init__(self, model_type: str = "lightgbm", random_state: int = RANDOM_SEED):
        """
        Initializes the spatial machine learning model.
        model_type: 'lightgbm' or 'random_forest'
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.feature_names = ALL_MODEL_FEATURES
        self.model = None
        self._init_model()

    def _init_model(self):
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                importance_type='gain'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Use 'lightgbm' or 'random_forest'.")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Trains the spatial model on environmental features.
        """
        print(f"\n[Spatial Model] Training {self.model_type.upper()} on {len(X_train):,} samples...")
        
        # Select active features available in dataframe
        available_features = [f for f in self.feature_names if f in X_train.columns]
        self.feature_names = available_features
        X_tr = X_train[self.feature_names]

        # Handle NaNs for Random Forest (impute median), LightGBM handles NaNs natively
        if self.model_type == "random_forest":
            X_tr = X_tr.fillna(X_tr.median())
            if X_val is not None:
                X_val_clean = X_val[self.feature_names].fillna(X_tr.median())
        else:
            X_val_clean = X_val[self.feature_names] if X_val is not None else None

        if self.model_type == "lightgbm" and X_val is not None and y_val is not None:
            self.model.fit(
                X_tr, y_train,
                eval_set=[(X_val_clean, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
        else:
            self.model.fit(X_tr, y_train)

        print(f"✓ {self.model_type.upper()} training complete.")
        return self.get_feature_importances()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts PFZ probability score (0.0 to 1.0) for each grid cell.
        """
        X_clean = X[self.feature_names].copy()
        if self.model_type == "random_forest":
            X_clean = X_clean.fillna(X_clean.median())
        probs = self.model.predict_proba(X_clean)[:, 1]
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Binary classification prediction with configurable decision threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics on out-of-sample test set.
        """
        y_prob = self.predict_proba(X_test)
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        print("\n" + "="*50)
        print(f"📊 {self.model_type.upper()} TEST EVALUATION (Year 2021 Holdout)")
        print("="*50)
        print(f"  ROC-AUC Score:      {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC Score:       {metrics['pr_auc']:.4f}")
        print(f"  F1-Score:           {metrics['f1_score']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall:             {metrics['recall']:.4f}")
        print("="*50)
        return metrics

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance sorted descending.
        """
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        sorted_pairs = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        total = sum(importances) if sum(importances) > 0 else 1.0
        return {feat: round(float(imp / total), 4) for feat, imp in sorted_pairs}

    def save(self, filepath: str = None):
        """Saves model to disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"spatial_{self.model_type}.joblib")
        joblib.dump({"model": self.model, "features": self.feature_names, "type": self.model_type}, filepath)
        print(f"✓ Saved {self.model_type} model to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads trained model from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.feature_names = data["features"]
        print(f"✓ Loaded {instance.model_type} model from {filepath}")
        return instance
