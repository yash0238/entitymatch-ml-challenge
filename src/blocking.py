"""
Blocking / Candidate generation strategies for entity resolution
Goal: High recall (95%+) with low candidate rate (<1%)
"""
import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import jellyfish
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_all_sources, save_tsv, format_entity_list, parse_entity_list
from preprocessing import preprocess_dataframe, create_combined_text

logger = logging.getLogger(__name__)


class BlockingStrategy:
    """
    Multi-view blocking for candidate pair generation
    """
    
    def __init__(self, 
                 phonetic_blocking: bool = True,
                 token_blocking: bool = True,
                 tfidf_blocking: bool = True,
                 country_blocking: bool = True):
        """
        Initialize blocking strategy
        
        Args:
            phonetic_blocking: Use phonetic encoding
            token_blocking: Use token-based blocking
            tfidf_blocking: Use TF-IDF similarity blocking
            country_blocking: Filter by country match
        """
        self.phonetic_blocking = phonetic_blocking
        self.token_blocking = token_blocking
        self.tfidf_blocking = tfidf_blocking
        self.country_blocking = country_blocking
        
        self.candidate_pairs = defaultdict(set)  # S1 entity -> set of S2/S3 candidates
    
    def _phonetic_block(self, 
                        s1_df: pd.DataFrame, 
                        s2_df: pd.DataFrame, 
                        s3_df: pd.DataFrame) -> None:
        """
        Phonetic encoding based blocking (Soundex, Metaphone)
        """
        logger.info("Applying phonetic blocking...")
        
        # Generate phonetic codes
        for df in [s1_df, s2_df, s3_df]:
            df['soundex'] = df['name_normalized'].apply(
                lambda x: jellyfish.soundex(x) if x else ''
            )
            df['metaphone'] = df['name_normalized'].apply(
                lambda x: jellyfish.metaphone(x) if x else ''
            )
        
        # Create phonetic index for S2 and S3
        s2_soundex_index = defaultdict(list)
        s3_soundex_index = defaultdict(list)
        
        for _, row in s2_df.iterrows():
            s2_soundex_index[row['soundex']].append(row['entity_id'])
            s2_soundex_index[row['metaphone']].append(row['entity_id'])
        
        for _, row in s3_df.iterrows():
            s3_soundex_index[row['soundex']].append(row['entity_id'])
            s3_soundex_index[row['metaphone']].append(row['entity_id'])
        
        # Generate candidate pairs
        for _, s1_row in s1_df.iterrows():
            s1_id = s1_row['entity_id']
            
            # Match by soundex
            candidates = set()
            candidates.update(s2_soundex_index.get(s1_row['soundex'], []))
            candidates.update(s3_soundex_index.get(s1_row['soundex'], []))
            
            # Match by metaphone
            candidates.update(s2_soundex_index.get(s1_row['metaphone'], []))
            candidates.update(s3_soundex_index.get(s1_row['metaphone'], []))
            
            self.candidate_pairs[s1_id].update(candidates)
        
        logger.info(f"Phonetic blocking generated candidates for {len(self.candidate_pairs)} S1 entities")
    
    def _token_block(self, 
                     s1_df: pd.DataFrame, 
                     s2_df: pd.DataFrame, 
                     s3_df: pd.DataFrame) -> None:
        """
        Token-based blocking using name and address tokens
        """
        logger.info("Applying token-based blocking...")
        
        # Create token index for S2 and S3
        s2_token_index = defaultdict(list)
        s3_token_index = defaultdict(list)
        
        # Index by name tokens
        for _, row in s2_df.iterrows():
            for token in row['name_tokens']:
                if len(token) > 2:  # Only meaningful tokens
                    s2_token_index[token].append(row['entity_id'])
        
        for _, row in s3_df.iterrows():
            for token in row['name_tokens']:
                if len(token) > 2:
                    s3_token_index[token].append(row['entity_id'])
        
        # Generate candidates by token overlap
        for _, s1_row in s1_df.iterrows():
            s1_id = s1_row['entity_id']
            candidates = set()
            
            for token in s1_row['name_tokens']:
                if len(token) > 2:
                    candidates.update(s2_token_index.get(token, []))
                    candidates.update(s3_token_index.get(token, []))
            
            self.candidate_pairs[s1_id].update(candidates)
        
        logger.info(f"Token blocking updated candidates for {len(self.candidate_pairs)} S1 entities")
    
    def _tfidf_block(self, 
                     s1_df: pd.DataFrame, 
                     s2_df: pd.DataFrame, 
                     s3_df: pd.DataFrame,
                     top_k: int = 50,
                     threshold: float = 0.3) -> None:
        """
        TF-IDF based blocking using combined text
        """
        logger.info("Applying TF-IDF blocking...")
        
        # Combine all text
        all_df = pd.concat([s1_df, s2_df, s3_df], ignore_index=True)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            analyzer='char_wb'
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_df['combined_text'])
        
        # Split matrices
        n_s1 = len(s1_df)
        n_s2 = len(s2_df)
        
        s1_matrix = tfidf_matrix[:n_s1]
        s2_matrix = tfidf_matrix[n_s1:n_s1+n_s2]
        s3_matrix = tfidf_matrix[n_s1+n_s2:]
        
        # Compute similarities
        logger.info("Computing S1-S2 similarity...")
        s1_s2_sim = cosine_similarity(s1_matrix, s2_matrix)
        
        logger.info("Computing S1-S3 similarity...")
        s1_s3_sim = cosine_similarity(s1_matrix, s3_matrix)
        
        # Generate candidates based on top-k and threshold
        for i, s1_id in enumerate(s1_df['entity_id']):
            # S2 candidates
            s2_scores = s1_s2_sim[i]
            s2_top_indices = np.argsort(s2_scores)[-top_k:]
            s2_candidates = s2_df.iloc[s2_top_indices[s2_scores[s2_top_indices] >= threshold]]['entity_id'].tolist()
            
            # S3 candidates
            s3_scores = s1_s3_sim[i]
            s3_top_indices = np.argsort(s3_scores)[-top_k:]
            s3_candidates = s3_df.iloc[s3_top_indices[s3_scores[s3_top_indices] >= threshold]]['entity_id'].tolist()
            
            self.candidate_pairs[s1_id].update(s2_candidates + s3_candidates)
        
        logger.info(f"TF-IDF blocking updated candidates for {len(self.candidate_pairs)} S1 entities")
    
    def _country_filter(self, 
                        s1_df: pd.DataFrame, 
                        s2_df: pd.DataFrame, 
                        s3_df: pd.DataFrame) -> None:
        """
        Filter candidates by country match
        """
        logger.info("Applying country filtering...")
        
        # Create country mapping
        s2_country_map = dict(zip(s2_df['entity_id'], s2_df['country_normalized']))
        s3_country_map = dict(zip(s3_df['entity_id'], s3_df['country_normalized']))
        combined_country_map = {**s2_country_map, **s3_country_map}
        
        s1_country_map = dict(zip(s1_df['entity_id'], s1_df['country_normalized']))
        
        # Filter candidates
        filtered_pairs = defaultdict(set)
        total_before = sum(len(v) for v in self.candidate_pairs.values())
        
        for s1_id, candidates in self.candidate_pairs.items():
            s1_country = s1_country_map[s1_id]
            
            for candidate_id in candidates:
                candidate_country = combined_country_map.get(candidate_id)
                if candidate_country == s1_country:
                    filtered_pairs[s1_id].add(candidate_id)
        
        total_after = sum(len(v) for v in filtered_pairs.values())
        logger.info(f"Country filtering: {total_before} -> {total_after} candidates")
        
        self.candidate_pairs = filtered_pairs
    
    def generate_candidates(self, 
                           data_dir: str, 
                           split: str = 'train') -> pd.DataFrame:
        """
        Generate candidate pairs using all blocking strategies
        
        Args:
            data_dir: Directory containing data
            split: 'train' or 'test'
            
        Returns:
            DataFrame with candidate pairs
        """
        logger.info(f"Generating candidates for {split} split...")
        
        # Load and preprocess data
        sources = load_all_sources(data_dir, split)
        
        s1_df = preprocess_dataframe(sources['source1'])
        s2_df = preprocess_dataframe(sources['source2'])
        s3_df = preprocess_dataframe(sources['source3'])
        
        # Add combined text for TF-IDF
        s1_df = create_combined_text(s1_df)
        s2_df = create_combined_text(s2_df)
        s3_df = create_combined_text(s3_df)
        
        # Apply blocking strategies
        if self.phonetic_blocking:
            self._phonetic_block(s1_df, s2_df, s3_df)
        
        if self.token_blocking:
            self._token_block(s1_df, s2_df, s3_df)
        
        if self.tfidf_blocking:
            self._tfidf_block(s1_df, s2_df, s3_df)
        
        # Country filtering (if enabled)
        if self.country_blocking:
            self._country_filter(s1_df, s2_df, s3_df)
        
        # Ensure all S1 entities are present (even with empty candidates)
        for s1_id in s1_df['entity_id']:
            if s1_id not in self.candidate_pairs:
                self.candidate_pairs[s1_id] = set()
        
        # Convert to DataFrame
        result_df = pd.DataFrame([
            {
                'source1_entity_id': s1_id,
                'candidate_entity_ids': format_entity_list(sorted(candidates))
            }
            for s1_id, candidates in self.candidate_pairs.items()
        ])
        
        # Statistics
        total_candidates = sum(len(v) for v in self.candidate_pairs.values())
        avg_candidates = total_candidates / len(self.candidate_pairs) if self.candidate_pairs else 0
        
        logger.info(f"\nBlocking Statistics:")
        logger.info(f"  Total S1 entities: {len(result_df)}")
        logger.info(f"  Total candidate pairs: {total_candidates}")
        logger.info(f"  Avg candidates per entity: {avg_candidates:.2f}")
        logger.info(f"  Entities with no candidates: {(result_df['candidate_entity_ids'] == '').sum()}")
        
        return result_df


def main():
    """Main function for standalone execution"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data'  # Changed from ../data
    
    blocker = BlockingStrategy(
        phonetic_blocking=True,
        token_blocking=True,
        tfidf_blocking=True,
        country_blocking=True
    )
    
    candidates_df = blocker.generate_candidates(data_dir, 'train')
    
    output_path = '../output/candidate_pairs_train.tsv'
    save_tsv(candidates_df, output_path)
    logger.info(f"Saved candidate pairs to {output_path}")


if __name__ == '__main__':
    main()
