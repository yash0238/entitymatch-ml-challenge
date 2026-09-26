"""
End-to-end pipeline for entity resolution
"""
import argparse
import logging
from pathlib import Path
import pandas as pd

from utils import load_tsv, save_tsv, load_all_sources, format_entity_list
from preprocessing import preprocess_dataframe, create_combined_text
from blocking import BlockingStrategy
from feature_engineering import FeatureExtractor
from matching_model import EntityMatchingModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityResolutionPipeline:
    """
    Complete pipeline for business entity resolution
    """
    
    def __init__(self, data_dir: str, output_dir: str = './output', model_dir: str = './models'):
        """
        Initialize pipeline
        
        Args:
            data_dir: Directory containing train/test data
            output_dir: Directory for output files
            model_dir: Directory for model artifacts
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.blocker = BlockingStrategy(
            phonetic_blocking=True,
            token_blocking=True,
            tfidf_blocking=True,
            country_blocking=True
        )
        self.feature_extractor = FeatureExtractor()
        self.model = EntityMatchingModel(model_type='xgboost')
    
    def train_pipeline(self):
        """
        Train the complete pipeline on training data
        """
        logger.info("="*60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Generate candidates
        logger.info("\n[1/5] Generating candidate pairs...")
        candidates_df = self.blocker.generate_candidates(str(self.data_dir), 'train')
        candidates_path = self.output_dir / 'candidate_pairs_train.tsv'
        save_tsv(candidates_df, str(candidates_path))
        
        # Step 2: Load and preprocess data
        logger.info("\n[2/5] Loading and preprocessing data...")
        sources = load_all_sources(str(self.data_dir), 'train')
        
        s1_df = preprocess_dataframe(sources['source1'])
        s2_df = preprocess_dataframe(sources['source2'])
        s3_df = preprocess_dataframe(sources['source3'])
        
        s1_df = create_combined_text(s1_df)
        s2_df = create_combined_text(s2_df)
        s3_df = create_combined_text(s3_df)
        
        # Step 3: Extract features
        logger.info("\n[3/5] Extracting features...")
        features_df = self.feature_extractor.extract_features_for_candidates(
            s1_df, s2_df, s3_df, candidates_df
        )
        
        # Save features
        features_path = self.model_dir / 'train_features.parquet'
        features_df.to_parquet(str(features_path))
        logger.info(f"Saved features to {features_path}")
        
        # Step 4: Prepare training data
        logger.info("\n[4/5] Preparing training data...")
        ground_truth = load_tsv(str(self.data_dir / 'train' / 'train_ground_truth.tsv'))
        X, y = self.model.prepare_training_data(features_df, ground_truth)
        
        # Step 5: Train model
        logger.info("\n[5/5] Training matching model...")
        self.model.train(
            X,
            y,
            validation_split=0.2,
            groups=features_df['source1_entity_id'],
            entity_ids=s1_df['entity_id'],
            ground_truth_df=ground_truth,
        )
        
        # Save model
        model_path = self.model_dir / 'matching_model.joblib'
        self.model.save_model(str(model_path))
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("="*60)
    
    def predict_pipeline(self):
        """
        Generate predictions on test data
        """
        logger.info("="*60)
        logger.info("STARTING PREDICTION PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load model
        logger.info("\n[1/5] Loading trained model...")
        model_path = self.model_dir / 'matching_model.joblib'
        self.model.load_model(str(model_path))
        
        # Step 2: Generate candidates
        logger.info("\n[2/5] Generating candidate pairs for test set...")
        candidates_df = self.blocker.generate_candidates(str(self.data_dir), 'test')
        candidates_path = self.output_dir / 'candidate_pairs.tsv'
        save_tsv(candidates_df, str(candidates_path))
        
        # Step 3: Load and preprocess test data
        logger.info("\n[3/5] Loading and preprocessing test data...")
        sources = load_all_sources(str(self.data_dir), 'test')
        
        s1_df = preprocess_dataframe(sources['source1'])
        s2_df = preprocess_dataframe(sources['source2'])
        s3_df = preprocess_dataframe(sources['source3'])
        
        s1_df = create_combined_text(s1_df)
        s2_df = create_combined_text(s2_df)
        s3_df = create_combined_text(s3_df)
        
        # Step 4: Extract features
        logger.info("\n[4/5] Extracting features...")
        features_df = self.feature_extractor.extract_features_for_candidates(
            s1_df, s2_df, s3_df, candidates_df
        )
        
        # Step 5: Predict matches
        logger.info("\n[5/5] Predicting matches...")
        
        if len(features_df) > 0:
            X = features_df[self.model.feature_columns]
            predictions = self.model.predict(X)
            features_df['prediction'] = predictions
            
            # Aggregate predictions per S1 entity
            results = {}
            for _, row in features_df.iterrows():
                if row['prediction'] == 1:
                    s1_id = row['source1_entity_id']
                    cand_id = row['candidate_entity_id']
                    
                    if s1_id not in results:
                        results[s1_id] = []
                    results[s1_id].append(cand_id)
        else:
            results = {}
        
        # Create submission dataframe
        submission_df = pd.DataFrame([
            {
                'source1_entity_id': s1_id,
                'matched_entity_ids': format_entity_list(sorted(results.get(s1_id, [])))
            }
            for s1_id in s1_df['entity_id']
        ])
        
        # Save submission
        submission_path = self.output_dir / 'matching_results.tsv'
        save_tsv(submission_df, str(submission_path))
        
        # Statistics
        total_matches = sum(len(v) for v in results.values())
        entities_with_matches = sum(1 for v in results.values() if len(v) > 0)
        
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Total S1 entities: {len(submission_df)}")
        logger.info(f"  Entities with matches: {entities_with_matches}")
        logger.info(f"  Total predicted matches: {total_matches}")
        logger.info(f"  Avg matches per entity: {total_matches/len(submission_df):.2f}")
        
        logger.info("\n" + "="*60)
        logger.info("PREDICTION PIPELINE COMPLETE")
        logger.info(f"Results saved to: {submission_path}")
        logger.info(f"Candidates saved to: {candidates_path}")
        logger.info("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], 
                       default='both', help='Pipeline mode')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory')
    parser.add_argument('--model-dir', default='models',
                       help='Model directory')
    
    args = parser.parse_args()
    
    pipeline = EntityResolutionPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir
    )
    
    if args.mode in ['train', 'both']:
        pipeline.train_pipeline()
    
    if args.mode in ['predict', 'both']:
        pipeline.predict_pipeline()


if __name__ == '__main__':
    main()
