import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

# Get the number of GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Get the name of the current GPU
print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    
