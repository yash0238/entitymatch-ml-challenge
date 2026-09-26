"""
Machine learning model for pairwise entity matching
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import logging
from pathlib import Path

from utils import calculate_f05_score

logger = logging.getLogger(__name__)


class EntityMatchingModel:
    """
    Binary classification model for entity matching
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize matching model
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.threshold = 0.5
        
    def _init_model(self):
        """Initialize the ML model"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_training_data(self, 
                             features_df: pd.DataFrame,
                             ground_truth_df: pd.DataFrame) -> tuple:
        """
        Prepare training data with labels
        
        Args:
            features_df: Features for candidate pairs
            ground_truth_df: Ground truth matches
            
        Returns:
            Tuple of (X, y)
        """
        logger.info("Preparing training data...")
        
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
        Train the matching model
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction for validation
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Initialize and train
        self._init_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold for F0.5
        self.threshold = self._optimize_threshold(y_val, y_val_pred_proba)
        
        y_val_pred = (y_val_pred_proba >= self.threshold).astype(int)
        
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f05 = calculate_f05_score(precision, recall)
        
        logger.info(f"\nValidation Results:")
        logger.info(f"  Threshold: {self.threshold:.3f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F0.5 Score: {f05:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Important Features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def _optimize_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """
        Find optimal threshold for F0.5 score
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        logger.info("Optimizing threshold for F0.5...")
        
        best_threshold = 0.5
        best_f05 = 0.0
        
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
        
        logger.info(f"Best threshold: {best_threshold:.3f} with F0.5: {best_f05:.4f}")
        return best_threshold
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict matches for feature matrix
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)[:, 1]
        predictions = (proba >= self.threshold).astype(int)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        logger.info(f"Saving model to {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info("Model saved successfully")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        logger.info(f"Loading model from {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.threshold = model_data['threshold']
        self.model_type = model_data['model_type']
        
        logger.info("Model loaded successfully")


def main():
    """Example usage"""
    pass


if __name__ == '__main__':
    main()
