"""
Quick test script - Run on small sample to validate pipeline
"""
import sys
sys.path.append('src')

import pandas as pd
import logging
from pathlib import Path

from utils import load_tsv, save_tsv
from preprocessing import preprocess_dataframe, create_combined_text
from blocking import BlockingStrategy
from feature_engineering import AdvancedFeatureExtractor
from matching_model import EntityMatchingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sample_pipeline(sample_size=5000):
    """
    Run pipeline on small sample for testing
    
    Args:
        sample_size: Number of Source 1 records to process
    """
    logger.info("="*60)
    logger.info(f"RUNNING SAMPLE PIPELINE (n={sample_size})")
    logger.info("="*60)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    # Load sample data
    logger.info("\n[1/6] Loading sample data...")
    s1_df = pd.read_csv('data/train/train_source1.tsv', sep='\t', nrows=sample_size)
    s2_df = pd.read_csv('data/train/train_source2.tsv', sep='\t', nrows=sample_size*2)
    s3_df = pd.read_csv('data/train/train_source3.tsv', sep='\t', nrows=sample_size*2)
    gt_df = pd.read_csv('data/train/train_ground_truth.tsv', sep='\t', nrows=sample_size)
    
    logger.info(f"Loaded: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}")
    
    # Preprocess
    logger.info("\n[2/6] Preprocessing...")
    s1_df = preprocess_dataframe(s1_df)
    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)
    
    s1_df = create_combined_text(s1_df)
    s2_df = create_combined_text(s2_df)
    s3_df = create_combined_text(s3_df)
    
    # Blocking (simplified for speed)
    logger.info("\n[3/6] Generating candidates (simplified blocking)...")
    blocker = BlockingStrategy(
        phonetic_blocking=True,
        token_blocking=True,
        tfidf_blocking=False,  # Skip TF-IDF for speed
        country_blocking=True
    )
    
    # Manual simple blocking
    from collections import defaultdict
    import jellyfish
    
    candidates = defaultdict(set)
    
    # Create indices
    s2_index = defaultdict(list)
    s3_index = defaultdict(list)
    
    for _, row in s2_df.iterrows():
        # Token-based
        for token in str(row['name_normalized']).split()[:3]:  # First 3 tokens
            if len(token) > 2:
                s2_index[token].append(row['entity_id'])
    
    for _, row in s3_df.iterrows():
        for token in str(row['name_normalized']).split()[:3]:
            if len(token) > 2:
                s3_index[token].append(row['entity_id'])
    
    # Generate candidates
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row['entity_id']
        s1_country = s1_row['country_normalized']
        
        # Token matching
        for token in str(s1_row['name_normalized']).split()[:3]:
            if len(token) > 2:
                for cand_id in s2_index.get(token, []):
                    candidates[s1_id].add(cand_id)
                for cand_id in s3_index.get(token, []):
                    candidates[s1_id].add(cand_id)
    
    # Country filtering
    s2_country = dict(zip(s2_df['entity_id'], s2_df['country_normalized']))
    s3_country = dict(zip(s3_df['entity_id'], s3_df['country_normalized']))
    combined_country = {**s2_country, **s3_country}
    
    filtered_candidates = {}
    for s1_id, cand_set in candidates.items():
        s1_country = dict(zip(s1_df['entity_id'], s1_df['country_normalized']))[s1_id]
        filtered_candidates[s1_id] = {
            c for c in cand_set 
            if combined_country.get(c) == s1_country
        }
    
    # Ensure all S1 entities present
    for s1_id in s1_df['entity_id']:
        if s1_id not in filtered_candidates:
            filtered_candidates[s1_id] = set()
    
    total_cands = sum(len(v) for v in filtered_candidates.values())
    logger.info(f"Generated {total_cands} candidate pairs ({total_cands/len(s1_df):.1f} per entity)")
    
    # Feature extraction
    logger.info("\n[4/6] Extracting features...")
    extractor = AdvancedFeatureExtractor()
    
    s1_map = {row['entity_id']: row for _, row in s1_df.iterrows()}
    s2_map = {row['entity_id']: row for _, row in s2_df.iterrows()}
    s3_map = {row['entity_id']: row for _, row in s3_df.iterrows()}
    combined_map = {**s2_map, **s3_map}
    
    feature_rows = []
    for s1_id, cand_set in list(filtered_candidates.items())[:100]:  # First 100 for speed
        if not cand_set:
            continue
        s1_row = s1_map[s1_id]
        for cand_id in list(cand_set)[:10]:  # Max 10 candidates per entity
            if cand_id not in combined_map:
                continue
            cand_row = combined_map[cand_id]
            feats = extractor.extract_advanced_features(s1_row, cand_row)
            feats['source1_entity_id'] = s1_id
            feats['candidate_entity_id'] = cand_id
            feature_rows.append(feats)
    
    features_df = pd.DataFrame(feature_rows)
    logger.info(f"Extracted features for {len(features_df)} pairs")
    logger.info(f"Total features: {len(features_df.columns) - 2}")
    
    # Training
    logger.info("\n[5/6] Training model...")
    model = EntityMatchingModel(model_type='xgboost')
    X, y = model.prepare_training_data(features_df, gt_df)
    
    if len(X) > 100:
        model.train(X, y, validation_split=0.2)
        
        # Save model
        model.save_model('models/sample_model.joblib')
        logger.info("Model saved to models/sample_model.joblib")
    else:
        logger.warning("Not enough samples to train model")
    
    logger.info("\n" + "="*60)
    logger.info("SAMPLE PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Review feature importance in output")
    logger.info("2. Run full pipeline: python src/pipeline.py")
    logger.info("3. Experiment with notebooks/01_quick_eda.ipynb")

if __name__ == '__main__':
    run_sample_pipeline(sample_size=5000)
