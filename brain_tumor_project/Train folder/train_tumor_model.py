import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='model_training.log'
)
logger = logging.getLogger(__name__)

class BrainTumorDataset(Dataset):
    """Dataset class for brain tumor data."""
    def __init__(self, mri_data, clinical_features, labels):
        self.mri_data = torch.FloatTensor(mri_data)
        self.clinical_features = torch.FloatTensor(clinical_features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'mri': self.mri_data[idx],
            'clinical': self.clinical_features[idx],
            'label': self.labels[idx]
        }

class BrainTumorModel(nn.Module):
    """Neural network model for brain tumor classification."""
    def __init__(self, clinical_dim=3, mri_channels=4, fusion_dim=128):
        super().__init__()
        
        # MRI processing branch
        self.mri_conv = nn.Sequential(
            nn.Conv3d(mri_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Clinical data processing branch
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(256, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 2)
        )
        
    def forward(self, mri, clinical):
        # Process MRI data
        mri_features = self.mri_conv(mri)
        mri_features = mri_features.view(mri_features.size(0), -1)
        
        # Process clinical data
        clinical_features = self.clinical_fc(clinical)
        
        # Combine features
        combined = torch.cat([mri_features, clinical_features], dim=1)
        
        # Final classification
        return self.fusion(combined)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move data to device
        mri = batch['mri'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(mri, clinical)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            mri = batch['mri'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(mri, clinical)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(
        description='Train brain tumor classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python train_tumor_classifier.py \\
    --data_dir path/to/combined/data \\
    --output_dir path/to/model/output \\
    --epochs 100 \\
    --batch_size 8 \\
    --learning_rate 0.001

Input files expected in data_dir:
- lgg_combined.npz
- hgg_combined.npz

Each .npz file should contain:
- mri_data: (n_patients, 4, height, width, depth)
- clinical_features: (n_patients, n_features)
- labels: (n_patients,)

Output:
- Trained model saved as 'brain_tumor_model.pt'
- Training logs saved to 'model_training.log'
        """
    )
    
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directory containing combined data files'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where model will be saved'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    try:
        # Load LGG data
        lgg_data = np.load(os.path.join(args.data_dir, 'lgg_combined.npz'))
        lgg_mri = lgg_data['mri_data']
        lgg_clinical = lgg_data['clinical_features']
        lgg_labels = lgg_data['labels']
        
        # Load HGG data
        hgg_data = np.load(os.path.join(args.data_dir, 'hgg_combined.npz'))
        hgg_mri = hgg_data['mri_data']
        hgg_clinical = hgg_data['clinical_features']
        hgg_labels = hgg_data['labels']
        
        # Combine data
        mri_data = np.concatenate([lgg_mri, hgg_mri])
        clinical_features = np.concatenate([lgg_clinical, hgg_clinical])
        labels = np.concatenate([lgg_labels, hgg_labels])
        
        logger.info(f"Loaded {len(labels)} total samples")
        logger.info(f"MRI shape: {mri_data.shape}")
        logger.info(f"Clinical features shape: {clinical_features.shape}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        return
    
    # Split data
    indices = np.random.permutation(len(labels))
    split = int(len(indices) * (1 - args.val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    
    # Create datasets
    train_dataset = BrainTumorDataset(
        mri_data[train_idx],
        clinical_features[train_idx],
        labels[train_idx]
    )
    val_dataset = BrainTumorDataset(
        mri_data[val_idx],
        clinical_features[val_idx],
        labels[val_idx]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    model = BrainTumorModel(
        clinical_dim=clinical_features.shape[1],
        mri_channels=mri_data.shape[1]
    ).to(device)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(args.output_dir, 'brain_tumor_model.pt')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping triggered")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.output_dir, 'brain_tumor_model.pt')}")
    print(f"Check {os.path.abspath('model_training.log')} for detailed logs.")

if __name__ == "__main__":
    main() 