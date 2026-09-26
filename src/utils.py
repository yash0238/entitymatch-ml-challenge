"""
Utility functions for entity resolution pipeline
"""
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tsv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load TSV file with proper tab separator
    
    Args:
        filepath: Path to TSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading TSV file: {filepath}")
    df = pd.read_csv(filepath, sep='\t', **kwargs)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def save_tsv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save DataFrame as TSV file
    
    Args:
        df: DataFrame to save
        filepath: Output path
        **kwargs: Additional arguments for df.to_csv
    """
    logger.info(f"Saving TSV file: {filepath}")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, sep='\t', index=False, **kwargs)
    logger.info(f"Saved {len(df)} rows to {filepath}")


def parse_entity_list(id_string: str) -> List[str]:
    """
    Parse comma-separated entity ID list
    
    Args:
        id_string: Comma-separated IDs or empty string
        
    Returns:
        List of entity IDs
    """
    if pd.isna(id_string) or id_string == '':
        return []
    return [x.strip() for x in str(id_string).split(',')]


def format_entity_list(id_list: List[str]) -> str:
    """
    Format list of entity IDs as comma-separated string
    
    Args:
        id_list: List of entity IDs
        
    Returns:
        Comma-separated string or empty string
    """
    if not id_list:
        return ''
    return ','.join(id_list)


def extract_source_prefix(entity_id: str) -> str:
    """
    Extract source prefix from entity ID
    
    Args:
        entity_id: Entity ID like 'S1-12345'
        
    Returns:
        Source prefix like 'S1'
    """
    match = re.match(r'^(S[123])-', entity_id)
    return match.group(1) if match else None


def validate_entity_id(entity_id: str, allowed_sources: List[str] = ['S1', 'S2', 'S3']) -> bool:
    """
    Validate entity ID format
    
    Args:
        entity_id: Entity ID to validate
        allowed_sources: List of allowed source prefixes
        
    Returns:
        True if valid, False otherwise
    """
    source = extract_source_prefix(entity_id)
    return source in allowed_sources


def load_all_sources(data_dir: str, split: str = 'train') -> Dict[str, pd.DataFrame]:
    """
    Load all three source files
    
    Args:
        data_dir: Directory containing dataset
        split: 'train' or 'test'
        
    Returns:
        Dictionary with keys 'source1', 'source2', 'source3'
    """
    data_path = Path(data_dir)
    sources = {}
    
    for i in [1, 2, 3]:
        filepath = data_path / split / f"{split}_source{i}.tsv"  # Fixed path with split subdirectory
        sources[f'source{i}'] = load_tsv(str(filepath))
    
    logger.info(f"Loaded all {split} sources")
    return sources


def calculate_f05_score(precision: float, recall: float) -> float:
    """
    Calculate F0.5 score (precision-heavy metric)
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F0.5 score
    """
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_entity_match(predicted: List[str], ground_truth: List[str]) -> Tuple[float, float, float]:
    """
    Evaluate single entity match
    
    Args:
        predicted: List of predicted entity IDs
        ground_truth: List of true entity IDs
        
    Returns:
        Tuple of (precision, recall, f0.5)
    """
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    
    if len(predicted_set) == 0 and len(ground_truth_set) == 0:
        # Correctly predicted singleton
        return 1.0, 1.0, 1.0
    
    if len(predicted_set) == 0 or len(ground_truth_set) == 0:
        # One is empty, other is not
        return 0.0, 0.0, 0.0
    
    true_positives = len(predicted_set & ground_truth_set)
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(ground_truth_set)
    
    f05 = calculate_f05_score(precision, recall)
    
    return precision, recall, f05


def create_entity_index(df: pd.DataFrame, id_col: str = 'entity_id') -> Dict[str, int]:
    """
    Create mapping from entity_id to DataFrame index
    
    Args:
        df: DataFrame with entity IDs
        id_col: Name of entity ID column
        
    Returns:
        Dictionary mapping entity_id to row index
    """
    return {entity_id: idx for idx, entity_id in enumerate(df[id_col])}


class DataStats:
    """Helper class for tracking data statistics"""
    
    @staticmethod
    def print_source_stats(sources: Dict[str, pd.DataFrame]) -> None:
        """Print statistics for all sources"""
        for name, df in sources.items():
            logger.info(f"\n{name.upper()} Statistics:")
            logger.info(f"  Total records: {len(df)}")
            logger.info(f"  Countries: {df['country'].unique().tolist()}")
            logger.info(f"  Country distribution:\n{df['country'].value_counts()}")
            logger.info(f"  Missing business_name: {df['business_name'].isna().sum()}")
            logger.info(f"  Missing business_address: {df['business_address'].isna().sum()}")
    
    @staticmethod
    def analyze_ground_truth(gt_df: pd.DataFrame) -> None:
        """Analyze ground truth statistics"""
        gt_df['match_list'] = gt_df['matched_entity_ids'].apply(parse_entity_list)
        gt_df['num_matches'] = gt_df['match_list'].apply(len)
        
        logger.info("\nGround Truth Statistics:")
        logger.info(f"  Total Source1 entities: {len(gt_df)}")
        logger.info(f"  Singletons (no matches): {(gt_df['num_matches'] == 0).sum()}")
        logger.info(f"  Entities with matches: {(gt_df['num_matches'] > 0).sum()}")
        logger.info(f"  Match distribution:\n{gt_df['num_matches'].value_counts().sort_index()}")
        logger.info(f"  Avg matches per entity: {gt_df['num_matches'].mean():.2f}")
        logger.info(f"  Max matches: {gt_df['num_matches'].max()}")
