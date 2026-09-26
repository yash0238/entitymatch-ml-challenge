"""
Text preprocessing and normalization for business names and addresses
"""
import re
import regex as unicode_re
import unicodedata
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalize business names and addresses to handle variations
    """
    
    # Common business suffixes and their normalized forms
    BUSINESS_SUFFIXES = {
        r'\b(corporation|corp\.?)\b': 'corp',
        r'\b(incorporated|inc\.?)\b': 'inc',
        r'\b(limited|ltd\.?)\b': 'ltd',
        r'\b(company|co\.?)\b': 'co',
        r'\b(private|pvt\.?)\b': 'pvt',
        r'\b(public|pub\.?)\b': 'pub',
        r'\b(llc|l\.l\.c\.?)\b': 'llc',
        r'\b(llp|l\.l\.p\.?)\b': 'llp',
        r'\b(limited liability company)\b': 'llc',
        r'\b(limited liability partnership)\b': 'llp',
        r'\bलिमिटेड\b': 'ltd',
    }
    
    # Address component abbreviations
    ADDRESS_ABBREVIATIONS = {
        r'\b(street|str\.?)\b': 'st',
        r'\b(road|rd\.?)\b': 'rd',
        r'\b(avenue|ave\.?)\b': 'ave',
        r'\b(boulevard|blvd\.?)\b': 'blvd',
        r'\b(drive|dr\.?)\b': 'dr',
        r'\b(lane|ln\.?)\b': 'ln',
        r'\b(court|ct\.?)\b': 'ct',
        r'\b(place|pl\.?)\b': 'pl',
        r'\b(apartment|apt\.?|apt )\b': 'apt',
        r'\b(suite|ste\.?|ste )\b': 'ste',
        r'\b(unit|un\.?)\b': 'unit',
        r'\b(floor|fl\.?|flr\.?)\b': 'fl',
        r'\b(building|bldg\.?)\b': 'bldg',
        r'\b(north|n\.?)\b': 'n',
        r'\b(south|s\.?)\b': 's',
        r'\b(east|e\.?)\b': 'e',
        r'\b(west|w\.?)\b': 'w',
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ''
        
        text = unicodedata.normalize('NFKC', str(text)).lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and common punctuation
        text = unicode_re.sub(r'[^\p{L}\p{N}\p{M}\s\-.,&]', '', text)
        
        return text.strip()
    
    @classmethod
    def normalize_business_name(cls, name: str) -> str:
        """
        Normalize business name
        
        Args:
            name: Business name
            
        Returns:
            Normalized name
        """
        name = cls.clean_text(name)
        
        # Normalize business suffixes
        for pattern, replacement in cls.BUSINESS_SUFFIXES.items():
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        
        # Remove punctuation
        name = re.sub(r'[.,&\-]', ' ', name)
        
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    @classmethod
    def normalize_address(cls, address: str) -> str:
        """
        Normalize address
        
        Args:
            address: Business address
            
        Returns:
            Normalized address
        """
        address = cls.clean_text(address)
        
        # Normalize abbreviations
        for pattern, replacement in cls.ADDRESS_ABBREVIATIONS.items():
            address = re.sub(pattern, replacement, address, flags=re.IGNORECASE)
        
        # Remove extra punctuation
        address = re.sub(r'[.,]', ' ', address)
        
        # Remove extra spaces
        address = re.sub(r'\s+', ' ', address).strip()
        
        return address
    
    @staticmethod
    def extract_tokens(text: str) -> set:
        """
        Extract unique tokens from text
        
        Args:
            text: Input text
            
        Returns:
            Set of tokens
        """
        tokens = text.lower().split()
        # Filter out very short tokens (likely noise)
        return {t for t in tokens if len(t) > 1}
    
    @staticmethod
    def extract_numeric_components(text: str) -> list:
        """
        Extract numeric components (ZIP codes, house numbers, etc.)
        
        Args:
            text: Input text
            
        Returns:
            List of numeric strings
        """
        if pd.isna(text):
            return []
        return re.findall(r'\d+', str(text))


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess entire DataFrame with normalized fields
    
    Args:
        df: DataFrame with business_name and business_address
        
    Returns:
        DataFrame with additional normalized columns
    """
    logger.info("Preprocessing DataFrame...")
    
    df = df.copy()
    
    # Normalize business name
    df['name_normalized'] = df['business_name'].apply(
        TextNormalizer.normalize_business_name
    )
    
    # Normalize address
    df['address_normalized'] = df['business_address'].apply(
        TextNormalizer.normalize_address
    )
    
    # Extract tokens
    df['name_tokens'] = df['name_normalized'].apply(
        TextNormalizer.extract_tokens
    )
    df['address_tokens'] = df['address_normalized'].apply(
        TextNormalizer.extract_tokens
    )
    
    # Extract numeric components from address
    df['address_numbers'] = df['business_address'].apply(
        TextNormalizer.extract_numeric_components
    )
    
    # Country to lowercase
    df['country_normalized'] = df['country'].str.lower().str.strip()
    
    logger.info("Preprocessing complete")
    
    return df


def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined text field for TF-IDF and other analyses
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with combined_text column
    """
    df = df.copy()
    df['combined_text'] = (
        df['name_normalized'].fillna('') + ' ' + 
        df['address_normalized'].fillna('') + ' ' + 
        df['country_normalized'].fillna('')
    )
    df['combined_text'] = df['combined_text'].str.strip()
    
    return df
