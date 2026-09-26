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
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        groups=None,
        entity_ids=None,
        ground_truth_df: pd.DataFrame = None,
    ):
        """
        Train the matching model
        
        Args:
            X: Feature matrix
            y: Labels
            validation_split: Fraction for validation
        """
        logger.info(f"Training {self.model_type} model...")
        
        if groups is None or entity_ids is None or ground_truth_df is None:
            raise ValueError(
                "Entity-level validation requires pair groups, all Source 1 IDs, "
                "and the ground-truth dataframe"
            )

        groups = np.asarray(groups)
        entity_ids = np.asarray(entity_ids)
        truth_counts = {
            row.source1_entity_id: (
                0 if pd.isna(row.matched_entity_ids) or not str(row.matched_entity_ids).strip()
                else len(str(row.matched_entity_ids).split(','))
            )
            for row in ground_truth_df.itertuples(index=False)
        }
        entity_has_matches = np.asarray([truth_counts.get(entity_id, 0) > 0 for entity_id in entity_ids])
        train_entities, validation_entities = train_test_split(
            entity_ids,
            test_size=validation_split,
            random_state=42,
            stratify=entity_has_matches,
        )
        train_mask = np.isin(groups, train_entities)
        validation_mask = np.isin(groups, validation_entities)
        if not train_mask.any() or not validation_mask.any():
            raise ValueError("Entity split produced an empty train or validation candidate set")

        X_train, y_train = X.iloc[train_mask], np.asarray(y)[train_mask]
        X_val, y_val = X.iloc[validation_mask], np.asarray(y)[validation_mask]
        validation_groups = groups[validation_mask]
        
        # Initialize and train
        self._init_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        f05, self.threshold = self._optimize_entity_threshold(
            y_val,
            y_val_pred_proba,
            validation_groups,
            validation_entities,
            truth_counts,
        )
        
        logger.info(f"\nValidation Results:")
        logger.info(f"  Threshold: {self.threshold:.3f}")
        logger.info(f"  Macro entity F0.5: {f05:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Important Features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def _optimize_entity_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        group_ids: np.ndarray,
        entity_ids: np.ndarray,
        truth_counts: dict,
    ):
        """
        Tune the threshold against macro F0.5 over Source 1 entities.
        
        Args:
            y_true: True pair labels
            y_pred_proba: Predicted probabilities
            group_ids: Source 1 ID for each candidate pair
            entity_ids: Every Source 1 ID in the validation fold
            truth_counts: Number of true matches for each Source 1 ID
        """
        entity_index = {entity_id: index for index, entity_id in enumerate(entity_ids)}
        group_indices = np.fromiter(
            (entity_index[group_id] for group_id in group_ids),
            dtype=np.int64,
            count=len(group_ids),
        )
        true_counts = np.asarray([truth_counts.get(entity_id, 0) for entity_id in entity_ids])
        thresholds = np.unique(np.quantile(y_pred_proba, np.linspace(0.0, 1.0, 101)))

        best_score = -1.0
        best_threshold = 1.0
        for threshold in thresholds:
            selected = y_pred_proba >= threshold
            predicted_counts = np.bincount(group_indices[selected], minlength=len(entity_ids))
            true_positive_counts = np.bincount(
                group_indices[selected], weights=y_true[selected], minlength=len(entity_ids)
            )
            denominator = 0.25 * predicted_counts + true_counts
            entity_scores = np.divide(
                1.25 * true_positive_counts,
                denominator,
                out=np.zeros(len(entity_ids), dtype=float),
                where=denominator > 0,
            )
            entity_scores[(predicted_counts == 0) & (true_counts == 0)] = 1.0
            score = float(entity_scores.mean())
            if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
                best_score = score
                best_threshold = float(threshold)

        logger.info("Best entity-level threshold: %.4f (macro F0.5: %.4f)", best_threshold, best_score)
        return best_score, best_threshold
    
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
