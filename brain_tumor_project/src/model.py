import torch
import torch.nn as nn
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_model(model_path='models/brain_tumor_model.pt'):
    """Load the trained brain tumor model."""
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error("Available files in models directory:")
            models_dir = os.path.dirname(model_path)
            if os.path.exists(models_dir):
                logger.error("\n".join(os.listdir(models_dir)))
            else:
                logger.error(f"Models directory not found: {models_dir}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        model = BrainTumorModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        return None

def predict(mri_data, clinical_data):
    """Make prediction using the brain tumor model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = load_model()
        if model is None:
            raise ValueError("Failed to load model")
        
        # Prepare input data
        mri_tensor = torch.FloatTensor(mri_data).unsqueeze(0).to(device)
        
        # Extract clinical features
        clinical_features = np.array([
            clinical_data.get('age', 0),
            clinical_data.get('gender', 0),
            clinical_data.get('kps_score', 0)
        ], dtype=np.float32)
        clinical_tensor = torch.FloatTensor(clinical_features).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(mri_tensor, clinical_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction[0]].item()
        
        # Convert prediction to tumor type
        tumor_type = "HGG" if prediction.item() == 1 else "LGG"
        
        return {
            "tumor_type": tumor_type,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy().tolist()
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    logger.info("Created models directory")
    
    # Create and save a model
    try:
        logger.info("Creating brain tumor model...")
        model = BrainTumorModel()
        model.eval()
        
        # Save the model
        model_path = 'models/brain_tumor_model.pt'
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc()) 