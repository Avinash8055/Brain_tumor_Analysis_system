import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import torch
import sys
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print environment information
logger.info(f"Python version: {sys.version}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"PyTorch version: {torch.__version__}")

# Check for CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# MRI modalities to load
MRI_MODALITIES = ['flair', 't1', 't1ce', 't2', 'seg']

def create_output_directory():
    """Create output directory for combined data and clean up old files."""
    output_dir = Path('preprocessed_data/combined')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing files
    logger.info("Cleaning up old files...")
    for file in output_dir.glob("*"):
        try:
            file.unlink()  # Delete file
            logger.info(f"Deleted: {file}")
        except Exception as e:
            logger.error(f"Error deleting {file}: {e}")
    
    return output_dir

def normalize_scan(scan):
    """Normalize MRI scan to 0-1 range while preserving 3D structure."""
    try:
        # Ensure input is numpy array with correct dtype
        if not isinstance(scan, np.ndarray):
            scan = np.array(scan)
        
        # Convert to float32 explicitly
        scan = scan.astype(np.float32)
        logger.info(f"Initial scan shape: {scan.shape}, dtype: {scan.dtype}")
        
        # Check for invalid values
        if np.any(np.isnan(scan)) or np.any(np.isinf(scan)):
            logger.error("Input contains NaN or Inf values")
            return None
            
        # Handle zero arrays
        if np.all(scan == 0):
            logger.info("All-zero array detected")
            return scan
            
        try:
            # CPU-based normalization
            non_zero_mask = scan != 0
            if np.any(non_zero_mask):
                non_zero_values = scan[non_zero_mask]
                p1, p99 = np.percentile(non_zero_values, [1, 99])
                scan = np.clip(scan, p1, p99)
            
            # Min-max normalization
            scan_min = np.min(scan)
            scan_max = np.max(scan)
            if scan_max > scan_min:
                scan = (scan - scan_min) / (scan_max - scan_min)
            
            logger.info(f"Normalized shape: {scan.shape}, range: [{np.min(scan):.3f}, {np.max(scan):.3f}]")
            return scan
            
        except Exception as e:
            logger.error(f"CPU normalization failed: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error in normalize_scan: {str(e)}")
        if isinstance(scan, np.ndarray):
            logger.error(f"Input array info - Shape: {scan.shape}, dtype: {scan.dtype}")
        else:
            logger.error(f"Input is not a numpy array: {type(scan)}")
        return None

def get_all_patient_folders(base_path):
    """Get list of all patient folders in the directory."""
    try:
        if not os.path.exists(base_path):
            logger.error(f"Directory does not exist: {base_path}")
            return []
            
        folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        logger.info(f"Found folders in {base_path}: {folders}")
        return folders
    except Exception as e:
        logger.error(f"Error reading directory {base_path}: {str(e)}")
        return []

def load_mri_data(patient_folder, base_path):
    """Load all MRI modalities for a patient."""
    try:
        mri_data = {}
        patient_dir = Path(base_path) / patient_folder
        
        # Load each modality
        for modality in MRI_MODALITIES:
            file_path = patient_dir / f"{modality}.npy"
            
            if not file_path.exists():
                logger.warning(f"Missing {modality} scan for patient folder {patient_folder}")
                return None
            
            try:
                # Load scan
                scan = np.load(str(file_path))
                logger.info(f"Loaded {modality} scan with shape: {scan.shape}, dtype: {scan.dtype}")
                
                if np.any(np.isnan(scan)) or np.any(np.isinf(scan)):
                    logger.error(f"{modality} scan contains NaN or Inf values")
                    return None
                
                # Normalize non-segmentation scans
                if modality != 'seg':
                    logger.info(f"Normalizing {modality} scan for patient {patient_folder}")
                    scan = normalize_scan(scan)
                    if scan is None:
                        logger.error(f"Normalization failed for {modality}")
                        return None
                else:
                    # For segmentation, just flatten
                    scan = scan.flatten()
                
                mri_data[modality] = scan
                
            except Exception as e:
                logger.error(f"Error loading {modality} scan: {str(e)}")
                return None
        
        return mri_data
    
    except Exception as e:
        logger.error(f"Error loading MRI data from folder {patient_folder}: {str(e)}")
        return None

def process_dataset(clinical_csv, mri_base_path, output_file, tumor_type):
    """Process and merge clinical and MRI data while preserving 3D structure."""
    logger.info(f"Processing {tumor_type} dataset...")
    
    try:
        # Check if input paths exist
        if not os.path.exists(clinical_csv):
            logger.error(f"Clinical data file not found: {clinical_csv}")
            return
        if not os.path.exists(mri_base_path):
            logger.error(f"MRI data directory not found: {mri_base_path}")
            return
            
        # Load clinical data
        clinical_df = pd.read_csv(clinical_csv)
        logger.info(f"Loaded clinical data: {len(clinical_df)} records")
        
        # Get all patient folders
        patient_folders = get_all_patient_folders(mri_base_path)
        if not patient_folders:
            logger.error(f"No patient folders found in {mri_base_path}")
            return
        logger.info(f"Found {len(patient_folders)} MRI patient folders")
        
        # Initialize lists to store data
        mri_data_list = []  # Will store 3D MRI scans
        clinical_features_list = []  # Will store clinical features
        labels = []
        patient_ids = []
        original_folders = []
        
        total_patients = len(patient_folders)
        logger.info(f"Starting processing of {total_patients} patients...")
        
        # Process each patient folder
        for idx, folder in enumerate(sorted(patient_folders)):
            try:
                logger.info(f"Processing patient {idx + 1}/{total_patients}: {folder}")
                
                # Initialize MRI data dictionary for this patient
                patient_mri = {}
                
                # Load and process MRI data
                for modality in MRI_MODALITIES:
                    file_path = Path(mri_base_path) / folder / f"{modality}.npy"
                    if not file_path.exists():
                        raise FileNotFoundError(f"Missing {modality} scan")
                    
                    # Load scan
                    scan = np.load(str(file_path))
                    scan = scan.astype(np.float32)
                    
                    if modality == 'seg':
                        # For segmentation, keep as is
                        patient_mri[modality] = scan
                    else:
                        # Normalize while preserving 3D structure
                        normalized_scan = normalize_scan(scan)
                        if normalized_scan is None:
                            raise ValueError(f"Failed to normalize {modality} scan")
                        patient_mri[modality] = normalized_scan
                
                # Process clinical features
                clinical_features = []
                if idx < len(clinical_df):
                    row = clinical_df.iloc[idx]
                    for col in clinical_df.columns:
                        if col != 'Patient ID':
                            try:
                                val = float(row[col])
                            except (ValueError, TypeError):
                                val = 0.0
                            clinical_features.append(val)
                else:
                    # If no clinical data, add zeros
                    clinical_features_count = len(clinical_df.columns) - 1
                    clinical_features = [0.0] * clinical_features_count
                
                # Store all data
                mri_data_list.append(patient_mri)
                clinical_features_list.append(clinical_features)
                labels.append(1 if tumor_type == 'HGG' else 0)
                patient_ids.append(f"{tumor_type}_{idx+1:03d}")
                original_folders.append(folder)
                
                # Log progress
                if (idx + 1) % 10 == 0 or (idx + 1) == total_patients:
                    logger.info(f"Processed {idx + 1}/{total_patients} patients")
                    logger.info(f"MRI modalities: {list(patient_mri.keys())}")
                    logger.info(f"Clinical features: {len(clinical_features)}")
                
            except Exception as e:
                logger.error(f"Error processing patient {folder}: {str(e)}")
                continue
        
        if not mri_data_list:
            logger.error(f"No valid data processed for {tumor_type}")
            return
        
        # Convert lists to numpy arrays
        clinical_features = np.array(clinical_features_list)
        labels = np.array(labels)
        
        # Save combined dataset
        combined_data = {
            'mri_data': mri_data_list,  # List of dictionaries containing 3D MRI scans
            'clinical_features': clinical_features,  # 2D array of clinical features
            'labels': labels,
            'patient_ids': np.array(patient_ids),
            'original_folders': np.array(original_folders),
            'feature_names': {
                'mri_modalities': MRI_MODALITIES,
                'clinical': [col for col in clinical_df.columns if col != 'Patient ID']
            }
        }
        
        np.save(output_file, combined_data)
        logger.info(f"Saved combined dataset with {len(mri_data_list)} patients to {output_file}")
        
        # Save mapping file
        mapping_file = output_file.parent / f"{tumor_type.lower()}_mapping.csv"
        mapping_df = pd.DataFrame({
            'Generated_ID': patient_ids,
            'Original_Folder': original_folders,
            'Label': labels
        })
        mapping_df.to_csv(mapping_file, index=False)
        logger.info(f"Saved ID mapping to {mapping_file}")
        
        # Log dataset statistics
        logger.info(f"\nDataset Statistics for {tumor_type}:")
        logger.info(f"Total patients processed: {len(mri_data_list)}")
        logger.info(f"Clinical features per patient: {clinical_features.shape[1]}")
        logger.info(f"MRI modalities: {MRI_MODALITIES}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
    except Exception as e:
        logger.error(f"Error processing {tumor_type} dataset: {str(e)}")
        raise

def main():
    """Main function to merge MRI and clinical data."""
    logger.info("Starting data merger process")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Process LGG dataset
    process_dataset(
        clinical_csv='preprocessed_data/clinical/lgg_clinical_data.csv',
        mri_base_path='preprocessed_data/mri/lgg',
        output_file=output_dir / 'lgg_combined.npy',
        tumor_type='LGG'
    )
    
    # Process HGG dataset
    process_dataset(
        clinical_csv='preprocessed_data/clinical/hgg_clinical_data.csv',
        mri_base_path='preprocessed_data/mri/hgg',
        output_file=output_dir / 'hgg_combined.npy',
        tumor_type='HGG'
    )
    
    logger.info("Data merger process completed")

if __name__ == "__main__":
    main() 