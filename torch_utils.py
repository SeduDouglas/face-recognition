import torch

def get_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Check number of GPUs
        gpu_count = torch.cuda.device_count()
        
        # Detailed GPU information
        print("CUDA is available!")
        print(f"Number of CUDA devices: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Select the first GPU by default
        return torch.device("cuda:0")
    
    # Check for MPS (Metal Performance Shaders) on Mac
    elif torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders)")
        return torch.device("mps")
    
    # Fallback to CPU
    else:
        print("No GPU available. Using CPU.")
        return torch.device("cpu")