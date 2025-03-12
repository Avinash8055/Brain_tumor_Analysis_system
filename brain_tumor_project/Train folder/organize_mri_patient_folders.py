import os
import shutil
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='organize_mri.log'
)
logger = logging.getLogger(__name__)

def extract_patient_id(filename):
    """Extract patient ID and modality from filename."""
    try:
        # Remove file extension
        base_name = filename.lower().replace('.npy', '')
        
        # Split into components
        parts = base_name.split('_')
        
        # Extract patient ID and modality
        if len(parts) >= 2:
            patient_id = parts[0]
            modality = parts[-1]  # Last part should be the modality
            
            # Validate modality
            if modality not in ['flair', 't1', 't1ce', 't2', 'seg']:
                logger.warning(f"Unknown modality in file: {filename}")
                return None, None
                
            return patient_id, modality
            
        logger.warning(f"Invalid filename format: {filename}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error processing filename {filename}: {e}")
        return None, None

def organize_patient_files(input_dir, output_dir):
    """Organize MRI files into patient-specific directories."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all .npy files
        npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        
        if not npy_files:
            logger.warning(f"No .npy files found in {input_dir}")
            return 0
            
        organized_count = 0
        for filename in tqdm(npy_files, desc="Organizing files"):
            # Extract patient ID and modality
            patient_id, modality = extract_patient_id(filename)
            
            if not patient_id or not modality:
                continue
                
            # Create patient directory
            patient_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Move file to patient directory with standardized name
            src = os.path.join(input_dir, filename)
            dst = os.path.join(patient_dir, f"{modality}.npy")
            
            try:
                shutil.copy2(src, dst)
                organized_count += 1
            except Exception as e:
                logger.error(f"Error copying file {filename}: {e}")
                continue
                
        return organized_count
        
    except Exception as e:
        logger.error(f"Error organizing directory {input_dir}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description='Organize brain tumor MRI files into patient-specific directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python organize_mri_patient_folders.py \\
    --input_dir path/to/input \\
    --output_dir path/to/output

Expected input structure:
  input_dir/
  ├── lgg/
  │   ├── patient1_flair.npy
  │   ├── patient1_t1.npy
  │   ├── patient1_t1ce.npy
  │   └── patient1_t2.npy
  └── hgg/
      └── [similar structure]

Output structure:
  output_dir/
  ├── lgg/
  │   └── patient1/
  │       ├── flair.npy
  │       ├── t1.npy
  │       ├── t1ce.npy
  │       └── t2.npy
  └── hgg/
      └── [similar structure]
        """
    )
    
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Base directory containing LGG and HGG folders with .npy files'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Base directory where organized patient folders will be created'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    total_organized = 0
    
    # Process LGG and HGG directories
    for tumor_type in ['lgg', 'hgg']:
        input_path = os.path.join(args.input_dir, tumor_type)
        output_path = os.path.join(args.output_dir, tumor_type)
        
        if not os.path.exists(input_path):
            logger.warning(f"Directory not found: {input_path}")
            continue
            
        logger.info(f"Processing {tumor_type.upper()} data")
        organized = organize_patient_files(input_path, output_path)
        total_organized += organized
        
        print(f"Organized {organized} files for {tumor_type.upper()}")
    
    print(f"Total files organized: {total_organized}")
    print(f"Organization completed. Check {os.path.abspath('organize_mri.log')} for detailed logs.")

if __name__ == "__main__":
    main() 