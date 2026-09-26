"""
Feature engineering for pairwise entity matching
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import jellyfish
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract similarity features for entity pairs
    """
    
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """Normalized Levenshtein distance"""
        if not s1 or not s2:
            return 0.0
        distance = jellyfish.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 1.0
    
    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str) -> float:
        """Jaro-Winkler similarity"""
        if not s1 or not s2:
            return 0.0
        return jellyfish.jaro_winkler_similarity(s1, s2)
    
    @staticmethod
    def token_jaccard_similarity(tokens1: set, tokens2: set) -> float:
        """Jaccard similarity on token sets"""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def fuzzy_ratio(s1: str, s2: str) -> float:
        """Fuzzy string matching ratio"""
        if not s1 or not s2:
            return 0.0
        return fuzz.ratio(s1, s2) / 100.0
    
    @staticmethod
    def fuzzy_partial_ratio(s1: str, s2: str) -> float:
        """Fuzzy partial ratio"""
        if not s1 or not s2:
            return 0.0
        return fuzz.partial_ratio(s1, s2) / 100.0
    
    @staticmethod
    def fuzzy_token_sort_ratio(s1: str, s2: str) -> float:
        """Fuzzy token sort ratio"""
        if not s1 or not s2:
            return 0.0
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    
    @staticmethod
    def numeric_overlap(nums1: list, nums2: list) -> float:
        """Overlap of numeric components"""
        if not nums1 or not nums2:
            return 0.0
        set1 = set(nums1)
        set2 = set(nums2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def phonetic_match(s1: str, s2: str) -> int:
        """Binary phonetic match (Soundex)"""
        if not s1 or not s2:
            return 0
        try:
            return int(jellyfish.soundex(s1) == jellyfish.soundex(s2))
        except:
            return 0
    
    def extract_pair_features(self, 
                             row1: pd.Series, 
                             row2: pd.Series) -> Dict[str, float]:
        """
        Extract all features for a pair of records
        
        Args:
            row1: First record (from source 1)
            row2: Second record (from source 2 or 3)
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Name features
        name1 = row1.get('name_normalized', '')
        name2 = row2.get('name_normalized', '')
        
        features['name_levenshtein'] = self.levenshtein_similarity(name1, name2)
        features['name_jaro_winkler'] = self.jaro_winkler_similarity(name1, name2)
        features['name_fuzzy_ratio'] = self.fuzzy_ratio(name1, name2)
        features['name_fuzzy_partial'] = self.fuzzy_partial_ratio(name1, name2)
        features['name_fuzzy_token_sort'] = self.fuzzy_token_sort_ratio(name1, name2)
        features['name_phonetic_match'] = self.phonetic_match(name1, name2)
        
        # Name token features
        tokens1 = row1.get('name_tokens', set())
        tokens2 = row2.get('name_tokens', set())
        features['name_token_jaccard'] = self.token_jaccard_similarity(tokens1, tokens2)
        
        # Address features
        addr1 = row1.get('address_normalized', '')
        addr2 = row2.get('address_normalized', '')
        
        features['addr_levenshtein'] = self.levenshtein_similarity(addr1, addr2)
        features['addr_jaro_winkler'] = self.jaro_winkler_similarity(addr1, addr2)
        features['addr_fuzzy_ratio'] = self.fuzzy_ratio(addr1, addr2)
        features['addr_fuzzy_partial'] = self.fuzzy_partial_ratio(addr1, addr2)
        features['addr_fuzzy_token_sort'] = self.fuzzy_token_sort_ratio(addr1, addr2)
        
        # Address token features
        addr_tokens1 = row1.get('address_tokens', set())
        addr_tokens2 = row2.get('address_tokens', set())
        features['addr_token_jaccard'] = self.token_jaccard_similarity(addr_tokens1, addr_tokens2)
        
        # Numeric overlap in addresses
        nums1 = row1.get('address_numbers', [])
        nums2 = row2.get('address_numbers', [])
        features['addr_numeric_overlap'] = self.numeric_overlap(nums1, nums2)
        
        # Country match
        country1 = row1.get('country_normalized', '')
        country2 = row2.get('country_normalized', '')
        features['country_match'] = int(country1 == country2)
        
        # Combined features
        features['name_length_ratio'] = min(len(name1), len(name2)) / max(len(name1), len(name2)) if name1 and name2 else 0.0
        features['addr_length_ratio'] = min(len(addr1), len(addr2)) / max(len(addr1), len(addr2)) if addr1 and addr2 else 0.0
        
        # Average similarities
        features['name_avg_similarity'] = np.mean([
            features['name_levenshtein'],
            features['name_jaro_winkler'],
            features['name_fuzzy_ratio'],
            features['name_token_jaccard']
        ])
        
        features['addr_avg_similarity'] = np.mean([
            features['addr_levenshtein'],
            features['addr_jaro_winkler'],
            features['addr_fuzzy_ratio'],
            features['addr_token_jaccard']
        ])
        
        features['overall_avg_similarity'] = np.mean([
            features['name_avg_similarity'],
            features['addr_avg_similarity']
        ])
        
        return features
    
    def extract_features_for_candidates(self,
                                       s1_df: pd.DataFrame,
                                       s2_df: pd.DataFrame,
                                       s3_df: pd.DataFrame,
                                       candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for all candidate pairs
        
        Args:
            s1_df: Source 1 DataFrame (preprocessed)
            s2_df: Source 2 DataFrame (preprocessed)
            s3_df: Source 3 DataFrame (preprocessed)
            candidates_df: Candidate pairs DataFrame
            
        Returns:
            DataFrame with features for each pair
        """
        logger.info("Extracting features for candidate pairs...")
        
        # Create entity ID to row mapping
        s1_map = {row['entity_id']: row for _, row in s1_df.iterrows()}
        s2_map = {row['entity_id']: row for _, row in s2_df.iterrows()}
        s3_map = {row['entity_id']: row for _, row in s3_df.iterrows()}
        combined_map = {**s2_map, **s3_map}
        
        feature_rows = []
        
        for _, cand_row in candidates_df.iterrows():
            s1_id = cand_row['source1_entity_id']
            candidate_ids = cand_row['candidate_entity_ids']
            
            if not candidate_ids or candidate_ids == '':
                continue
            
            s1_row = s1_map[s1_id]
            
            for cand_id in candidate_ids.split(','):
                cand_id = cand_id.strip()
                if cand_id not in combined_map:
                    continue
                
                cand_row_data = combined_map[cand_id]
                
                # Extract features
                features = self.extract_pair_features(s1_row, cand_row_data)
                features['source1_entity_id'] = s1_id
                features['candidate_entity_id'] = cand_id
                
                feature_rows.append(features)
        
        features_df = pd.DataFrame(feature_rows)
        logger.info(f"Extracted features for {len(features_df)} candidate pairs")
        
        return features_df


def main():
    """Example usage"""
    # This would be called from the main pipeline
    pass


if __name__ == '__main__':
    main()
