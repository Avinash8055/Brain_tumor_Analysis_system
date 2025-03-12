import os
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mri_preprocessing.log'
)
logger = logging.getLogger(__name__)

def normalize_scan(scan):
    """Normalize MRI scan to zero mean and unit variance."""
    try:
        # Reshape to 2D for StandardScaler
        original_shape = scan.shape
        scan_2d = scan.reshape(-1, 1)
        
        # Normalize
        scaler = StandardScaler()
        scan_normalized = scaler.fit_transform(scan_2d)
        
        # Reshape back to original shape
        scan_normalized = scan_normalized.reshape(original_shape)
        
        return scan_normalized
        
    except Exception as e:
        logger.error(f"Error normalizing scan: {e}")
        return None

def standardize_dimensions(scan, target_shape=(240, 240, 155)):
    """Standardize scan dimensions through padding or cropping."""
    try:
        current_shape = scan.shape
        standardized = np.zeros(target_shape, dtype=scan.dtype)
        
        # Calculate padding/cropping for each dimension
        slices = tuple(slice(0, min(current_shape[i], target_shape[i])) 
                      for i in range(3))
        
        # Copy data
        standardized[slices] = scan[slices]
        return standardized
        
    except Exception as e:
        logger.error(f"Error standardizing dimensions: {e}")
        return None

def process_patient_scans(patient_dir, output_dir):
    """Process all MRI modalities for a single patient."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        modalities = ['flair', 't1', 't1ce', 't2']
        processed_count = 0
        
        for modality in modalities:
            input_file = os.path.join(patient_dir, f"{modality}.npy")
            output_file = os.path.join(output_dir, f"{modality}.npy")
            
            if not os.path.exists(input_file):
                logger.warning(f"Missing {modality} scan for {patient_dir}")
                continue
                
            try:
                # Load scan
                scan = np.load(input_file)
                
                # Standardize dimensions
                scan = standardize_dimensions(scan)
                if scan is None:
                    continue
                    
                # Normalize intensities
                scan = normalize_scan(scan)
                if scan is None:
                    continue
                    
                # Save processed scan
                np.save(output_file, scan)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
                continue
                
        return processed_count
        
    except Exception as e:
        logger.error(f"Error processing patient directory {patient_dir}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description='Standardize and normalize brain tumor MRI scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python standardize_mri_scans.py \\
    --input_dir path/to/organized/mri/data \\
    --output_dir path/to/preprocessed/output

Expected input structure:
  input_dir/
  ├── lgg/
  │   └── patient1/
  │       ├── flair.npy
  │       ├── t1.npy
  │       ├── t1ce.npy
  │       └── t2.npy
  └── hgg/
      └── [similar structure]

Output structure:
  output_dir/
  ├── lgg/
  │   └── patient1/
  │       ├── flair.npy  # Normalized and standardized
  │       ├── t1.npy
  │       ├── t1ce.npy
  │       └── t2.npy
  └── hgg/
      └── [similar structure]

Processing steps:
1. Standardize dimensions to (240, 240, 155)
2. Normalize intensities to zero mean and unit variance
        """
    )
    
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Base directory containing organized LGG and HGG patient folders'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Base directory where preprocessed scans will be saved'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    total_processed = 0
    
    # Process LGG and HGG directories
    for tumor_type in ['lgg', 'hgg']:
        input_path = os.path.join(args.input_dir, tumor_type)
        output_path = os.path.join(args.output_dir, tumor_type)
        
        if not os.path.exists(input_path):
            logger.warning(f"Directory not found: {input_path}")
            continue
            
        logger.info(f"Processing {tumor_type.upper()} scans")
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(input_path) 
                       if os.path.isdir(os.path.join(input_path, d))]
        
        if not patient_dirs:
            logger.warning(f"No patient directories found in {input_path}")
            continue
        
        # Process each patient's scans
        for patient_dir in tqdm(patient_dirs, desc=f"Processing {tumor_type.upper()} patients"):
            input_patient_dir = os.path.join(input_path, patient_dir)
            output_patient_dir = os.path.join(output_path, patient_dir)
            
            processed = process_patient_scans(input_patient_dir, output_patient_dir)
            total_processed += processed
            
            if processed == 0:
                logger.warning(f"Failed to process any scans for {patient_dir}")
    
    print(f"Total scans processed: {total_processed}")
    print(f"Preprocessing completed. Check {os.path.abspath('mri_preprocessing.log')} for detailed logs.")

if __name__ == "__main__":
    main() 