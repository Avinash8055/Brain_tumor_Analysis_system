import os
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='nifti_conversion.log'  # Added log file
)
logger = logging.getLogger(__name__)

def setup_directories(output_base):
    """Create output directories for LGG and HGG data."""
    directories = {
        'lgg': Path(output_base) / 'lgg',
        'hgg': Path(output_base) / 'hgg'
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return directories

def load_nifti(file_path):
    """Load a NIfTI file (.nii or .nii.gz) and return the data array."""
    try:
        # Load the NIfTI file
        nifti_img = nib.load(file_path)
        
        # Get the data array
        data = nifti_img.get_fdata()
        
        # Convert to float32 to save memory
        data = data.astype(np.float32)
        
        logger.info(f"Loaded {file_path}")
        logger.info(f"Shape: {data.shape}, Data type: {data.dtype}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def process_patient_directory(input_dir, output_dir):
    """Process all NIfTI files in a patient directory."""
    try:
        # Look for both .nii and .nii.gz files
        nifti_files = list(Path(input_dir).glob('*.nii')) + list(Path(input_dir).glob('*.nii.gz'))
        
        if not nifti_files:
            logger.warning(f"No NIfTI files found in {input_dir}")
            return False
            
        # Process each NIfTI file
        for nifti_file in nifti_files:
            # Determine modality from filename
            filename = nifti_file.name.lower()
            
            # Extract modality
            if 'flair' in filename:
                modality = 'flair'
            elif 't1ce' in filename:
                modality = 't1ce'
            elif 't1' in filename:
                modality = 't1'
            elif 't2' in filename:
                modality = 't2'
            elif 'seg' in filename:
                modality = 'seg'
            else:
                logger.warning(f"Unknown modality in file: {nifti_file}")
                continue
            
            # Load and convert the file
            data = load_nifti(nifti_file)
            if data is None:
                continue
                
            # Create output filename
            output_file = Path(output_dir) / f"{modality}.npy"
            
            # Save as .npy
            np.save(output_file, data)
            logger.info(f"Saved {output_file}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing directory {input_dir}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Convert brain tumor NIfTI files (.nii or .nii.gz) to NumPy (.npy) format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example directory structure:
Input:
  input_directory/
  ├── lgg/
  │   ├── patient1/
  │   │   ├── flair.nii.gz
  │   │   ├── t1.nii.gz
  │   │   ├── t1ce.nii.gz
  │   │   └── t2.nii.gz
  └── hgg/
      └── [similar structure]

Output:
  output_directory/
  ├── lgg/
  │   ├── patient1/
  │   │   ├── flair.npy
  │   │   ├── t1.npy
  │   │   ├── t1ce.npy
  │   │   └── t2.npy
  └── hgg/
      └── [similar structure]
        """
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Specify the input directory containing LGG and HGG folders with NIfTI files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Specify the output directory where NPY files will be saved'
    )
    args = parser.parse_args()
    
    logger.info("Starting NIfTI to NPY conversion")
    
    # Validate input directory
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        print(f"Error: Input directory does not exist: {args.input}")
        return
    
    # Setup output directories
    output_dirs = setup_directories(args.output)
    
    # Process LGG and HGG directories
    for tumor_type in ['lgg', 'hgg']:
        input_dir = Path(args.input) / tumor_type
        if not input_dir.exists():
            logger.warning(f"Directory not found: {input_dir}")
            continue
            
        logger.info(f"Processing {tumor_type.upper()} data")
        
        # Get all patient directories
        patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not patient_dirs:
            logger.warning(f"No patient directories found in {input_dir}")
            continue
        
        # Process each patient directory with progress bar
        for patient_dir in tqdm(patient_dirs, desc=f"Processing {tumor_type.upper()} patients"):
            # Create output directory for this patient
            patient_output_dir = output_dirs[tumor_type] / patient_dir.name
            patient_output_dir.mkdir(exist_ok=True)
            
            # Process the patient directory
            success = process_patient_directory(patient_dir, patient_output_dir)
            if not success:
                logger.warning(f"Failed to process {patient_dir}")
    
    logger.info("Conversion completed")
    print(f"Conversion completed. Check {os.path.abspath('nifti_conversion.log')} for detailed logs.")

if __name__ == "__main__":
    main() 