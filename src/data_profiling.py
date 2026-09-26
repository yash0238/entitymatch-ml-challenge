"""
Data profiling and exploratory data analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import logging

from utils import load_all_sources, load_tsv, parse_entity_list, DataStats
from preprocessing import TextNormalizer

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Comprehensive data profiling for entity resolution
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize profiler
        
        Args:
            data_dir: Path to dataset directory
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        self.test_dir = self.data_dir / 'test'
        
        # Load data
        logger.info("Loading training data...")
        self.train_sources = load_all_sources(str(self.data_dir), 'train')
        self.ground_truth = load_tsv(str(self.train_dir / 'train_ground_truth.tsv'))
        
    def analyze_basic_stats(self) -> dict:
        """
        Analyze basic dataset statistics
        
        Returns:
            Dictionary with statistics
        """
        logger.info("\n=== BASIC STATISTICS ===")
        
        stats = {}
        
        # Source statistics
        DataStats.print_source_stats(self.train_sources)
        
        # Ground truth statistics
        DataStats.analyze_ground_truth(self.ground_truth)
        
        return stats
    
    def analyze_noise_patterns(self) -> None:
        """
        Analyze noise patterns in business names and addresses
        """
        logger.info("\n=== NOISE PATTERN ANALYSIS ===")
        
        # Analyze name patterns
        all_names = pd.concat([
            self.train_sources['source1']['business_name'],
            self.train_sources['source2']['business_name'],
            self.train_sources['source3']['business_name']
        ])
        
        logger.info("\nBusiness Name Patterns:")
        logger.info(f"  Total names: {len(all_names)}")
        logger.info(f"  Unique names: {all_names.nunique()}")
        logger.info(f"  Avg name length: {all_names.str.len().mean():.1f} chars")
        
        # Common suffixes
        suffixes = []
        for name in all_names.dropna():
            tokens = str(name).lower().split()
            if tokens:
                last_token = tokens[-1].strip('.,')
                suffixes.append(last_token)
        
        suffix_counts = Counter(suffixes)
        logger.info("\n  Most common last tokens (potential suffixes):")
        for suffix, count in suffix_counts.most_common(20):
            logger.info(f"    {suffix}: {count}")
        
        # Analyze address patterns
        all_addresses = pd.concat([
            self.train_sources['source1']['business_address'],
            self.train_sources['source2']['business_address'],
            self.train_sources['source3']['business_address']
        ])
        
        logger.info("\nBusiness Address Patterns:")
        logger.info(f"  Total addresses: {len(all_addresses)}")
        logger.info(f"  Unique addresses: {all_addresses.nunique()}")
        logger.info(f"  Avg address length: {all_addresses.str.len().mean():.1f} chars")
        
        # Component analysis
        has_numbers = all_addresses.str.contains(r'\d', na=False).sum()
        has_commas = all_addresses.str.contains(',', na=False).sum()
        
        logger.info(f"  Addresses with numbers: {has_numbers} ({has_numbers/len(all_addresses)*100:.1f}%)")
        logger.info(f"  Addresses with commas: {has_commas} ({has_commas/len(all_addresses)*100:.1f}%)")
    
    def analyze_matching_patterns(self) -> None:
        """
        Analyze matching patterns in ground truth
        """
        logger.info("\n=== MATCHING PATTERN ANALYSIS ===")
        
        gt = self.ground_truth.copy()
        gt['match_list'] = gt['matched_entity_ids'].apply(parse_entity_list)
        gt['num_matches'] = gt['match_list'].apply(len)
        
        # Source distribution of matches
        s2_matches = 0
        s3_matches = 0
        both_matches = 0
        
        for match_list in gt['match_list']:
            has_s2 = any(eid.startswith('S2-') for eid in match_list)
            has_s3 = any(eid.startswith('S3-') for eid in match_list)
            
            if has_s2 and has_s3:
                both_matches += 1
            elif has_s2:
                s2_matches += 1
            elif has_s3:
                s3_matches += 1
        
        logger.info("\nMatch Source Distribution:")
        logger.info(f"  S1 entities with only S2 matches: {s2_matches}")
        logger.info(f"  S1 entities with only S3 matches: {s3_matches}")
        logger.info(f"  S1 entities with both S2 and S3 matches: {both_matches}")
        
        # Singleton analysis
        singletons = gt[gt['num_matches'] == 0]
        logger.info(f"\nSingleton Analysis:")
        logger.info(f"  Total singletons: {len(singletons)} ({len(singletons)/len(gt)*100:.1f}%)")
        logger.info(f"  Singleton impact: Correctly predicting adds {len(singletons)} perfect scores")
    
    def analyze_country_distribution(self) -> None:
        """
        Analyze country distribution and patterns
        """
        logger.info("\n=== COUNTRY DISTRIBUTION ===")
        
        for source_name, df in self.train_sources.items():
            logger.info(f"\n{source_name.upper()}:")
            country_counts = df['country'].value_counts()
            for country, count in country_counts.items():
                logger.info(f"  {country}: {count} ({count/len(df)*100:.1f}%)")
    
    def run_full_profiling(self) -> None:
        """
        Run complete data profiling pipeline
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE DATA PROFILING")
        logger.info("=" * 60)
        
        self.analyze_basic_stats()
        self.analyze_noise_patterns()
        self.analyze_matching_patterns()
        self.analyze_country_distribution()
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA PROFILING COMPLETE")
        logger.info("=" * 60)


def main():
    """Main function for standalone execution"""
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'data'  # Changed from ../data
    
    profiler = DataProfiler(data_dir)
    profiler.run_full_profiling()


if __name__ == '__main__':
    main()
