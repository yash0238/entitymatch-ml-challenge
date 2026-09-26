"""
Ensemble model combining XGBoost, LightGBM, and CatBoost
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import logging
from pathlib import Path
from typing import List, Tuple

from utils import calculate_f05_score

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble of three gradient boosting models for entity matching
    """
    
    def __init__(self):
        """Initialize ensemble"""
        self.xgb_model = None
        self.lgb_model = None
        self.cb_model = None
        self.feature_columns = None
        self.threshold = 0.5
        self.weights = {'xgb': 0.4, 'lgb': 0.3, 'cb': 0.3}  # Weighted voting
        
    def _init_models(self):
        """Initialize all three models"""
        # XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist'
        )
        
        # LightGBM
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # CatBoost
        self.cb_model = cb.CatBoostClassifier(
            iterations=200,
            depth=7,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
    
    def prepare_training_data(self, 
                             features_df: pd.DataFrame,
                             ground_truth_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with labels
        
        Args:
            features_df: Features for candidate pairs
            ground_truth_df: Ground truth matches
            
        Returns:
            Tuple of (X, y)
        """
        logger.info("Preparing training data for ensemble...")
        
        # Create ground truth mapping
        gt_map = {}
        for _, row in ground_truth_df.iterrows():
            s1_id = row['source1_entity_id']
            matched_ids = row['matched_entity_ids']
            if matched_ids and not pd.isna(matched_ids) and matched_ids != '':
                gt_map[s1_id] = set(str(matched_ids).split(','))
            else:
                gt_map[s1_id] = set()
        
        # Create labels
        labels = []
        for _, row in features_df.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            
            is_match = cand_id in gt_map.get(s1_id, set())
            labels.append(int(is_match))
        
        features_df['label'] = labels
        
        # Separate features and labels
        self.feature_columns = [col for col in features_df.columns 
                               if col not in ['source1_entity_id', 'candidate_entity_id', 'label']]
        
        X = features_df[self.feature_columns]
        y = features_df['label']
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")
        logger.info(f"Negative samples: {(1-y).sum()} ({(1-y).mean()*100:.2f}%)")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2):
        """
        Train all three models in ensemble
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction for validation
        """
        logger.info("="*60)
        logger.info("TRAINING ENSEMBLE (XGBoost + LightGBM + CatBoost)")
        logger.info("="*60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logger.info(f"\nTrain: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # Initialize models
        self._init_models()
        
        # Train XGBoost
        logger.info("\n[1/3] Training XGBoost...")
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict_proba(X_val)[:, 1]
        logger.info("  ✓ XGBoost trained")
        
        # Train LightGBM
        logger.info("\n[2/3] Training LightGBM...")
        self.lgb_model.fit(X_train, y_train)
        lgb_pred = self.lgb_model.predict_proba(X_val)[:, 1]
        logger.info("  ✓ LightGBM trained")
        
        # Train CatBoost
        logger.info("\n[3/3] Training CatBoost...")
        self.cb_model.fit(X_train, y_train)
        cb_pred = self.cb_model.predict_proba(X_val)[:, 1]
        logger.info("  ✓ CatBoost trained")
        
        # Ensemble predictions
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE EVALUATION")
        logger.info("="*60)
        
        # Individual model performance
        for name, pred in [('XGBoost', xgb_pred), ('LightGBM', lgb_pred), ('CatBoost', cb_pred)]:
            best_f05, best_thresh = self._find_best_threshold(y_val, pred)
            logger.info(f"\n{name}:")
            logger.info(f"  Best F0.5: {best_f05:.4f} @ threshold {best_thresh:.3f}")
        
        # Weighted ensemble
        ensemble_pred = (
            self.weights['xgb'] * xgb_pred +
            self.weights['lgb'] * lgb_pred +
            self.weights['cb'] * cb_pred
        )
        
        best_f05, best_thresh = self._find_best_threshold(y_val, ensemble_pred)
        self.threshold = best_thresh
        
        logger.info(f"\nWeighted Ensemble (XGB:{self.weights['xgb']}, LGB:{self.weights['lgb']}, CB:{self.weights['cb']}):")
        logger.info(f"  Best F0.5: {best_f05:.4f} @ threshold {best_thresh:.3f}")
        
        # Final metrics
        ensemble_pred_binary = (ensemble_pred >= self.threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_val, ensemble_pred_binary, zero_division=0)
        recall = recall_score(y_val, ensemble_pred_binary, zero_division=0)
        
        logger.info(f"\n  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F0.5 Score: {best_f05:.4f}")
        
        # Feature importance (from XGBoost)
        if hasattr(self.xgb_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 15 Important Features (from XGBoost):")
            for idx, row in importance_df.head(15).iterrows():
                logger.info(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    def _find_best_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for F0.5"""
        from sklearn.metrics import precision_score, recall_score
        
        best_f05 = 0.0
        best_threshold = 0.5
        
        for threshold in np.arange(0.3, 0.9, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if y_pred.sum() == 0:
                continue
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f05 = calculate_f05_score(precision, recall)
            
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = threshold
        
        return best_f05, best_threshold
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        cb_pred = self.cb_model.predict_proba(X)[:, 1]
        
        ensemble_pred = (
            self.weights['xgb'] * xgb_pred +
            self.weights['lgb'] * lgb_pred +
            self.weights['cb'] * cb_pred
        )
        
        predictions = (ensemble_pred >= self.threshold).astype(int)
        return predictions
    
    def save_model(self, filepath: str):
        """Save ensemble model"""
        logger.info(f"Saving ensemble model to {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'cb_model': self.cb_model,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'weights': self.weights
        }
        
        joblib.dump(model_data, filepath)
        logger.info("Ensemble model saved successfully")
    
    def load_model(self, filepath: str):
        """Load ensemble model"""
        logger.info(f"Loading ensemble model from {filepath}")
        
        model_data = joblib.load(filepath)
        self.xgb_model = model_data['xgb_model']
        self.lgb_model = model_data['lgb_model']
        self.cb_model = model_data['cb_model']
        self.feature_columns = model_data['feature_columns']
        self.threshold = model_data['threshold']
        self.weights = model_data['weights']
        
        logger.info("Ensemble model loaded successfully")
