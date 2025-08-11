#!/usr/bin/env python3
"""
Data cleaning script for Property Friends ML project.

This script cleans the train.csv and test.csv files to remove data quality issues
that may prevent model training. It handles common issues like:
- Missing values
- Non-positive areas, rooms, bathrooms, or prices
- Invalid property types
- Inconsistent data formats

Usage:
    python scripts/clean_data.py
    
Output:
    - train_clean.csv: Cleaned training data
    - test_clean.csv: Cleaned test data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logger import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)


def diagnose_data(file_path: str) -> pd.DataFrame:
    """Diagnose data quality issues in a CSV file."""
    logger.info(f"analyzing_data_file", file_path=file_path)
    
    try:
        df = pd.read_csv(file_path)
        logger.info("data_loaded", shape=df.shape, columns=list(df.columns))
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_total = missing_counts.sum()
        if missing_total > 0:
            logger.warning("missing_values_found", missing_by_column=missing_counts[missing_counts > 0].to_dict())
        
        # Check numerical columns for issues
        numerical_cols = ['net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'price']
        for col in numerical_cols:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    logger.warning(
                        "non_positive_values_found",
                        column=col,
                        count=negative_count,
                        min_value=df[col].min(),
                        max_value=df[col].max()
                    )
        
        # Check categorical columns
        if 'type' in df.columns:
            valid_types = ['casa', 'departamento']
            invalid_types = df[~df['type'].str.lower().isin(valid_types)]['type'].unique()
            if len(invalid_types) > 0:
                logger.warning("invalid_property_types_found", invalid_types=invalid_types.tolist())
        
        return df
        
    except Exception as e:
        logger.error("data_loading_failed", file_path=file_path, error=str(e))
        raise


def clean_data(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Clean the data by removing/fixing problematic rows."""
    logger.info("cleaning_data_started", original_rows=len(df))
    
    original_count = len(df)
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove rows with null values in critical columns
    critical_cols = ['type', 'sector', 'net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'price']
    available_cols = [col for col in critical_cols if col in df_clean.columns]
    
    if available_cols:
        df_clean = df_clean.dropna(subset=available_cols)
        logger.info("removed_null_rows", removed_count=original_count - len(df_clean))
    
    # Fix property types - convert to lowercase and keep only valid ones
    if 'type' in df_clean.columns:
        df_clean['type'] = df_clean['type'].str.lower().str.strip()
        valid_types = ['casa', 'departamento']
        before_count = len(df_clean)
        df_clean = df_clean[df_clean['type'].isin(valid_types)]
        logger.info("removed_invalid_property_types", removed_count=before_count - len(df_clean))
    
    # Remove rows with non-positive numerical values
    numerical_constraints = {
        'net_usable_area': lambda x: x > 0,
        'net_area': lambda x: x > 0,
        'n_rooms': lambda x: x >= 1,
        'n_bathroom': lambda x: x >= 1,
        'price': lambda x: x > 0
    }
    
    for col, constraint in numerical_constraints.items():
        if col in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean[constraint(df_clean[col])]
            removed = before_count - len(df_clean)
            if removed > 0:
                logger.info("removed_invalid_numerical_values", column=col, removed_count=removed)
    
    # Ensure net_area >= net_usable_area (logical constraint)
    if 'net_area' in df_clean.columns and 'net_usable_area' in df_clean.columns:
        before_count = len(df_clean)
        df_clean = df_clean[df_clean['net_area'] >= df_clean['net_usable_area']]
        removed = before_count - len(df_clean)
        if removed > 0:
            logger.info("removed_illogical_area_relationships", removed_count=removed)
    
    # Remove extreme outliers (properties with unrealistic values)
    if 'price' in df_clean.columns:
        # Remove properties with extremely high prices (likely data errors)
        price_99th = df_clean['price'].quantile(0.99)
        before_count = len(df_clean)
        df_clean = df_clean[df_clean['price'] <= price_99th * 5]  # Allow up to 5x the 99th percentile
        removed = before_count - len(df_clean)
        if removed > 0:
            logger.info("removed_extreme_price_outliers", removed_count=removed, price_threshold=price_99th * 5)
    
    final_count = len(df_clean)
    total_removed = original_count - final_count
    
    logger.info(
        "data_cleaning_completed",
        original_rows=original_count,
        final_rows=final_count,
        total_removed=total_removed,
        removal_percentage=round(total_removed / original_count * 100, 2)
    )
    
    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    logger.info("cleaned_data_saved", output_path=output_path)
    
    return df_clean


def main():
    """Main data cleaning process."""
    logger.info("data_cleaning_process_started")
    
    # Check if input files exist
    train_file = "train.csv"
    test_file = "test.csv"
    
    if not Path(train_file).exists():
        logger.error("training_file_not_found", file_path=train_file)
        print(f"❌ Error: {train_file} not found in current directory")
        print("Please ensure train.csv is in the project root directory")
        return False
    
    if not Path(test_file).exists():
        logger.error("test_file_not_found", file_path=test_file)
        print(f"❌ Error: {test_file} not found in current directory")
        print("Please ensure test.csv is in the project root directory")
        return False
    
    try:
        print("🔍 Analyzing and cleaning data files...")
        
        # Process training data
        print(f"\n📊 Processing {train_file}...")
        train_df = diagnose_data(train_file)
        train_clean = clean_data(train_df, "train_clean.csv")
        
        # Process test data
        print(f"\n📊 Processing {test_file}...")
        test_df = diagnose_data(test_file)
        test_clean = clean_data(test_df, "test_clean.csv")
        
        print("\n✅ Data cleaning completed successfully!")
        print(f"📁 Cleaned files created:")
        print(f"   - train_clean.csv ({len(train_clean)} rows)")
        print(f"   - test_clean.csv ({len(test_clean)} rows)")
        print("\n🚀 Next steps:")
        print("   1. Review the cleaned data: head train_clean.csv")
        print("   2. Train with cleaned data: python scripts/train_model.py --train-data train_clean.csv --test-data test_clean.csv")
        
        return True
        
    except Exception as e:
        logger.error("data_cleaning_failed", error=str(e))
        print(f"❌ Data cleaning failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
