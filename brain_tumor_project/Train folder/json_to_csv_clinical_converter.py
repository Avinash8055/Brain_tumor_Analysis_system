import json
import pandas as pd
import os
import logging
import argparse
from pathlib import Path
import math
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='clinical_conversion.log'
)
logger = logging.getLogger(__name__)

def create_output_directory(output_dir):
    """Create directory for processed clinical data."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

def convert_time_metrics(days, metric_type):
    """Convert days to appropriate time units."""
    try:
        days = float(days)
        if metric_type == "survival":
            return round(days / 365.25, 2)  # Convert to years
        elif metric_type == "followup":
            return round(days / 30.44, 2)   # Convert to months
        return None
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting time metric: {e}")
        return None

def extract_patient_data(patient_data):
    """Extract relevant patient information from JSON data."""
    try:
        # Extract basic information
        data = {
            'patient_id': patient_data.get('patient_id', None),
            'age': patient_data.get('age_at_diagnosis', None),
            'gender': patient_data.get('gender', None),
            'survival_months': convert_time_metrics(
                patient_data.get('days_to_death', None), 
                "survival"
            ),
            'followup_months': convert_time_metrics(
                patient_data.get('days_to_last_followup', None),
                "followup"
            )
        }
        
        # Add additional clinical features as needed
        return data
        
    except Exception as e:
        logger.error(f"Error extracting patient data: {e}")
        return None

def process_clinical_data(input_file, output_file):
    """Process clinical JSON data and save as CSV."""
    try:
        # Read JSON file
        with open(input_file, 'r') as f:
            clinical_data = json.load(f)
        
        logger.info(f"Loaded clinical data from {input_file}")
        
        # Process each patient
        processed_data = []
        for patient in clinical_data:
            data = extract_patient_data(patient)
            if data:
                processed_data.append(data)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(processed_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Processed {len(processed_data)} patients")
        logger.info(f"Saved processed data to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing clinical data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Convert brain tumor clinical data from JSON to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python json_to_csv_clinical_converter.py \\
    --lgg_input path/to/lgg_clinical.json \\
    --hgg_input path/to/gbm_clinical.json \\
    --output_dir path/to/output

Expected input format (JSON):
{
    "patient_id": "TCGA-XX-XXXX",
    "age_at_diagnosis": 45,
    "gender": "M/F",
    "days_to_death": 365,
    "days_to_last_followup": 730,
    ...
}

Output format (CSV):
patient_id,age,gender,survival_months,followup_months,...
        """
    )
    
    parser.add_argument(
        '--lgg_input',
        required=True,
        help='Path to the LGG clinical JSON file'
    )
    parser.add_argument(
        '--hgg_input',
        required=True,
        help='Path to the HGG/GBM clinical JSON file'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where processed CSV files will be saved'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    for input_file in [args.lgg_input, args.hgg_input]:
        if not os.path.exists(input_file):
            logger.error(f"Input file does not exist: {input_file}")
            print(f"Error: Input file does not exist: {input_file}")
            return
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Process LGG data
    lgg_output = os.path.join(args.output_dir, 'lgg_clinical_data.csv')
    if process_clinical_data(args.lgg_input, lgg_output):
        print(f"Successfully processed LGG clinical data: {lgg_output}")
    
    # Process HGG data
    hgg_output = os.path.join(args.output_dir, 'hgg_clinical_data.csv')
    if process_clinical_data(args.hgg_input, hgg_output):
        print(f"Successfully processed HGG clinical data: {hgg_output}")
    
    print(f"Conversion completed. Check {os.path.abspath('clinical_conversion.log')} for detailed logs.")

if __name__ == "__main__":
    main() 