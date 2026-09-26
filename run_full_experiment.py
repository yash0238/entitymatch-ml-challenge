"""
Full experiment runner with analysis and visualization
"""
import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_tsv, save_tsv, format_entity_list, parse_entity_list, evaluate_entity_match
from preprocessing import preprocess_dataframe, create_combined_text
from blocking import BlockingStrategy
from feature_engineering import AdvancedFeatureExtractor
from ensemble_model import EnsembleModel
from matching_model import EntityMatchingModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/experiment_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Run complete experiment with analysis
    """
    
    def __init__(self, sample_size=20000, use_ensemble=True):
        """
        Initialize experiment
        
        Args:
            sample_size: Number of S1 entities to process
            use_ensemble: Use ensemble model vs single XGBoost
        """
        self.sample_size = sample_size
        self.use_ensemble = use_ensemble
        self.results = {}
        
        # Create directories
        for dir_name in ['experiments', 'models', 'output']:
            Path(dir_name).mkdir(exist_ok=True)
    
    def run_full_experiment(self):
        """Run complete experiment"""
        logger.info("="*80)
        logger.info(f"EXPERIMENT: Entity Resolution Pipeline (n={self.sample_size})")
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info(f"Model: {'Ensemble (XGB+LGB+CB)' if self.use_ensemble else 'XGBoost'}")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Stage 1: Load Data
        logger.info("\n" + "="*80)
        logger.info("[STAGE 1/6] LOADING DATA")
        logger.info("="*80)
        s1_df, s2_df, s3_df, gt_df = self._load_data()
        
        # Stage 2: Preprocess
        logger.info("\n" + "="*80)
        logger.info("[STAGE 2/6] PREPROCESSING")
        logger.info("="*80)
        s1_df, s2_df, s3_df = self._preprocess_data(s1_df, s2_df, s3_df)
        
        # Stage 3: Blocking
        logger.info("\n" + "="*80)
        logger.info("[STAGE 3/6] BLOCKING (CANDIDATE GENERATION)")
        logger.info("="*80)
        candidates_df = self._run_blocking(s1_df, s2_df, s3_df)
        
        # Stage 4: Feature Extraction
        logger.info("\n" + "="*80)
        logger.info("[STAGE 4/6] FEATURE EXTRACTION")
        logger.info("="*80)
        features_df = self._extract_features(s1_df, s2_df, s3_df, candidates_df)
        
        # Stage 5: Train Model
        logger.info("\n" + "="*80)
        logger.info("[STAGE 5/6] MODEL TRAINING")
        logger.info("="*80)
        model = self._train_model(features_df, gt_df)
        
        # Stage 6: Evaluation
        logger.info("\n" + "="*80)
        logger.info("[STAGE 6/6] EVALUATION & ANALYSIS")
        logger.info("="*80)
        self._evaluate_and_analyze(model, features_df, gt_df, s1_df)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total Time: {total_time/60:.2f} minutes")
        logger.info(f"Results saved to: experiments/")
        logger.info(f"Model saved to: models/")
        
        # Save summary
        self._save_summary(total_time)
    
    def _load_data(self):
        """Load data"""
        logger.info(f"Loading {self.sample_size} S1 entities...")
        
        s1_df = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=self.sample_size)
        s2_df = pd.read_csv('data/train/train_source2.tsv', sep='\t', nrows=self.sample_size*3)
        s3_df = pd.read_csv('data/train/train_source3.tsv', sep='\t', nrows=self.sample_size*3)
        gt_df = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', nrows=self.sample_size)
        
        logger.info(f"  S1: {len(s1_df):,} records")
        logger.info(f"  S2: {len(s2_df):,} records")
        logger.info(f"  S3: {len(s3_df):,} records")
        logger.info(f"  GT: {len(gt_df):,} records")
        
        self.results['data_size'] = {
            's1': len(s1_df),
            's2': len(s2_df),
            's3': len(s3_df)
        }
        
        return s1_df, s2_df, s3_df, gt_df
    
    def _preprocess_data(self, s1_df, s2_df, s3_df):
        """Preprocess all data"""
        logger.info("Preprocessing data...")
        
        s1_df = preprocess_dataframe(s1_df)
        s2_df = preprocess_dataframe(s2_df)
        s3_df = preprocess_dataframe(s3_df)
        
        s1_df = create_combined_text(s1_df)
        s2_df = create_combined_text(s2_df)
        s3_df = create_combined_text(s3_df)
        
        logger.info("  ✓ Preprocessing complete")
        return s1_df, s2_df, s3_df
    
    def _run_blocking(self, s1_df, s2_df, s3_df):
        """Run blocking with multiple strategies"""
        from collections import defaultdict
        import jellyfish
        
        candidates = defaultdict(set)
        
        # Strategy 1: Token-based
        logger.info("  [1/3] Token-based blocking...")
        s2_index = defaultdict(list)
        s3_index = defaultdict(list)
        
        for _, row in s2_df.iterrows():
            for token in str(row['name_normalized']).split()[:4]:
                if len(token) > 2:
                    s2_index[token].append(row['entity_id'])
        
        for _, row in s3_df.iterrows():
            for token in str(row['name_normalized']).split()[:4]:
                if len(token) > 2:
                    s3_index[token].append(row['entity_id'])
        
        for _, s1_row in s1_df.iterrows():
            s1_id = s1_row['entity_id']
            for token in str(s1_row['name_normalized']).split()[:4]:
                if len(token) > 2:
                    candidates[s1_id].update(s2_index.get(token, []))
                    candidates[s1_id].update(s3_index.get(token, []))
        
        token_count = sum(len(v) for v in candidates.values())
        logger.info(f"    Generated {token_count:,} candidates")
        
        # Strategy 2: Phonetic
        logger.info("  [2/3] Phonetic blocking...")
        s2_soundex = defaultdict(list)
        s3_soundex = defaultdict(list)
        
        for _, row in s2_df.iterrows():
            if row['name_normalized']:
                soundex = jellyfish.soundex(row['name_normalized'])
                s2_soundex[soundex].append(row['entity_id'])
        
        for _, row in s3_df.iterrows():
            if row['name_normalized']:
                soundex = jellyfish.soundex(row['name_normalized'])
                s3_soundex[soundex].append(row['entity_id'])
        
        for _, s1_row in s1_df.iterrows():
            s1_id = s1_row['entity_id']
            if s1_row['name_normalized']:
                soundex = jellyfish.soundex(s1_row['name_normalized'])
                candidates[s1_id].update(s2_soundex.get(soundex, []))
                candidates[s1_id].update(s3_soundex.get(soundex, []))
        
        phonetic_count = sum(len(v) for v in candidates.values())
        logger.info(f"    Generated {phonetic_count:,} total candidates")
        
        # Strategy 3: Country filter
        logger.info("  [3/3] Country filtering...")
        s2_country = dict(zip(s2_df['entity_id'], s2_df['country_normalized']))
        s3_country = dict(zip(s3_df['entity_id'], s3_df['country_normalized']))
        combined_country = {**s2_country, **s3_country}
        s1_country = dict(zip(s1_df['entity_id'], s1_df['country_normalized']))
        
        filtered = {}
        for s1_id, cand_set in candidates.items():
            country = s1_country[s1_id]
            filtered[s1_id] = {c for c in cand_set if combined_country.get(c) == country}
        
        for s1_id in s1_df['entity_id']:
            if s1_id not in filtered:
                filtered[s1_id] = set()
        
        final_count = sum(len(v) for v in filtered.values())
        logger.info(f"    Final candidates: {final_count:,} ({final_count/len(s1_df):.1f} per entity)")
        
        self.results['blocking'] = {
            'total_candidates': final_count,
            'avg_per_entity': final_count/len(s1_df),
            'reduction_ratio': 1 - (final_count / (len(s1_df) * (len(s2_df) + len(s3_df))))
        }
        
        # Convert to DataFrame
        candidates_df = pd.DataFrame([
            {
                'source1_entity_id': s1_id,
                'candidate_entity_ids': format_entity_list(sorted(cands))
            }
            for s1_id, cands in filtered.items()
        ])
        
        # Save
        save_tsv(candidates_df, 'output/candidate_pairs.tsv')
        logger.info("  ✓ Saved to output/candidate_pairs.tsv")
        
        return candidates_df
    
    def _extract_features(self, s1_df, s2_df, s3_df, candidates_df):
        """Extract features"""
        logger.info("Extracting advanced features...")
        
        extractor = AdvancedFeatureExtractor()
        
        s1_map = {row['entity_id']: row for _, row in s1_df.iterrows()}
        s2_map = {row['entity_id']: row for _, row in s2_df.iterrows()}
        s3_map = {row['entity_id']: row for _, row in s3_df.iterrows()}
        combined_map = {**s2_map, **s3_map}
        
        feature_rows = []
        processed = 0
        max_pairs = 50000  # Limit for speed
        
        for _, cand_row in candidates_df.iterrows():
            if processed >= max_pairs:
                break
            
            s1_id = cand_row['source1_entity_id']
            candidate_ids = cand_row['candidate_entity_ids']
            
            if not candidate_ids or candidate_ids == '':
                continue
            
            s1_row = s1_map[s1_id]
            
            for cand_id in candidate_ids.split(',')[:15]:  # Max 15 per entity
                if processed >= max_pairs:
                    break
                
                cand_id = cand_id.strip()
                if cand_id not in combined_map:
                    continue
                
                cand_row_data = combined_map[cand_id]
                feats = extractor.extract_advanced_features(s1_row, cand_row_data)
                feats['source1_entity_id'] = s1_id
                feats['candidate_entity_id'] = cand_id
                feature_rows.append(feats)
                processed += 1
        
        features_df = pd.DataFrame(feature_rows)
        logger.info(f"  Extracted {len(features_df):,} feature vectors")
        logger.info(f"  Features per pair: {len(features_df.columns) - 2}")
        
        self.results['features'] = {
            'total_pairs': len(features_df),
            'num_features': len(features_df.columns) - 2
        }
        
        # Save
        features_df.to_parquet('experiments/features.parquet')
        logger.info("  ✓ Saved to experiments/features.parquet")
        
        return features_df
    
    def _train_model(self, features_df, gt_df):
        """Train model"""
        if self.use_ensemble:
            model = EnsembleModel()
        else:
            model = EntityMatchingModel(model_type='xgboost')
        
        X, y = model.prepare_training_data(features_df, gt_df)
        
        if y.sum() < 10:
            logger.warning("⚠️  Very few positive samples - results may be limited")
        
        model.train(X, y, validation_split=0.2)
        
        # Save model
        model_path = f"models/{'ensemble' if self.use_ensemble else 'xgboost'}_model.joblib"
        model.save_model(model_path)
        
        return model
    
    def _evaluate_and_analyze(self, model, features_df, gt_df, s1_df):
        """Evaluate and create analysis"""
        logger.info("\nGenerating evaluation metrics...")
        
        # Get predictions
        X = features_df[model.feature_columns]
        predictions = model.predict(X)
        features_df['prediction'] = predictions
        
        # Aggregate by S1 entity
        predicted_matches = {}
        for _, row in features_df[features_df['prediction'] == 1].iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            if s1_id not in predicted_matches:
                predicted_matches[s1_id] = []
            predicted_matches[s1_id].append(cand_id)
        
        # Ground truth mapping
        gt_map = {}
        for _, row in gt_df.iterrows():
            s1_id = row['source1_entity_id']
            matched_ids = row['matched_entity_ids']
            if matched_ids and not pd.isna(matched_ids) and matched_ids != '':
                gt_map[s1_id] = set(str(matched_ids).split(','))
            else:
                gt_map[s1_id] = set()
        
        # Compute metrics per entity
        entity_scores = []
        for s1_id in s1_df['entity_id']:
            pred = predicted_matches.get(s1_id, [])
            true = list(gt_map.get(s1_id, set()))
            
            precision, recall, f05 = evaluate_entity_match(pred, true)
            entity_scores.append({
                's1_id': s1_id,
                'num_predicted': len(pred),
                'num_true': len(true),
                'precision': precision,
                'recall': recall,
                'f05': f05
            })
        
        scores_df = pd.DataFrame(entity_scores)
        
        # Macro averages
        macro_precision = scores_df['precision'].mean()
        macro_recall = scores_df['recall'].mean()
        macro_f05 = scores_df['f05'].mean()
        
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"  Macro Precision: {macro_precision:.4f}")
        logger.info(f"  Macro Recall:    {macro_recall:.4f}")
        logger.info(f"  Macro F0.5:      {macro_f05:.4f}")
        logger.info("="*60)
        
        self.results['evaluation'] = {
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f05': float(macro_f05),
            'num_entities': len(scores_df)
        }
        
        # Save detailed scores
        scores_df.to_csv('experiments/entity_scores.csv', index=False)
        logger.info("\n  ✓ Saved entity scores to experiments/entity_scores.csv")
        
        # Create submission file
        submission_df = pd.DataFrame([
            {
                'source1_entity_id': s1_id,
                'matched_entity_ids': format_entity_list(sorted(predicted_matches.get(s1_id, [])))
            }
            for s1_id in s1_df['entity_id']
        ])
        save_tsv(submission_df, 'output/matching_results.tsv')
        logger.info("  ✓ Saved submission to output/matching_results.tsv")
        
        # Generate visualizations
        self._create_visualizations(scores_df)
    
    def _create_visualizations(self, scores_df):
        """Create analysis plots"""
        logger.info("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. F0.5 distribution
        axes[0, 0].hist(scores_df['f05'], bins=30, edgecolor='black')
        axes[0, 0].axvline(scores_df['f05'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {scores_df["f05"].mean():.3f}')
        axes[0, 0].set_xlabel('F0.5 Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('F0.5 Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Precision vs Recall
        axes[0, 1].scatter(scores_df['recall'], scores_df['precision'], alpha=0.5, s=20)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall (per entity)')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Predictions vs Ground Truth
        axes[1, 0].scatter(scores_df['num_true'], scores_df['num_predicted'], alpha=0.5, s=20)
        axes[1, 0].plot([0, scores_df['num_true'].max()], [0, scores_df['num_true'].max()], 
                       'r--', label='Perfect')
        axes[1, 0].set_xlabel('True Matches')
        axes[1, 0].set_ylabel('Predicted Matches')
        axes[1, 0].set_title('Predictions vs Ground Truth')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Performance by match count
        scores_df['match_bucket'] = pd.cut(scores_df['num_true'], bins=[0, 1, 3, 5, 100], 
                                           labels=['0-1', '2-3', '4-5', '6+'])
        bucket_stats = scores_df.groupby('match_bucket')['f05'].agg(['mean', 'count'])
        bucket_stats['mean'].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_xlabel('True Match Count')
        axes[1, 1].set_ylabel('Mean F0.5')
        axes[1, 1].set_title('Performance by Match Count')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/analysis_plots.png', dpi=150)
        logger.info("  ✓ Saved plots to experiments/analysis_plots.png")
        plt.close()
    
    def _save_summary(self, total_time):
        """Save experiment summary"""
        summary = {
            'experiment_config': {
                'sample_size': self.sample_size,
                'model_type': 'ensemble' if self.use_ensemble else 'xgboost',
                'timestamp': str(datetime.now()),
                'total_time_minutes': round(total_time/60, 2)
            },
            'results': self.results
        }
        
        with open('experiments/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n  ✓ Saved summary to experiments/experiment_summary.json")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full experiment')
    parser.add_argument('--sample-size', type=int, default=20000,
                       help='Number of S1 entities to process')
    parser.add_argument('--ensemble', action='store_true', default=True,
                       help='Use ensemble model')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        sample_size=args.sample_size,
        use_ensemble=args.ensemble
    )
    
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
