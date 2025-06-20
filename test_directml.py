import torch
import torch_directml

def test_directml():
    try:
        # Initialize DirectML
        dml = torch_directml.device()
        print(f"DirectML device: {dml}")
        
        # Create a test tensor
        x = torch.randn(1000, 1000).to(dml)
        y = torch.randn(1000, 1000).to(dml)
        
        # Perform a matrix multiplication
        z = torch.matmul(x, y)
        
        # Move result back to CPU and verify
        z_cpu = z.cpu()
        print("Matrix multiplication successful!")
        print(f"Result shape: {z_cpu.shape}")
        print(f"Device used: {z.device}")
        
        return True
    except Exception as e:
        print(f"Error testing DirectML: {str(e)}")
        return False

if __name__ == "__main__":
    test_directml() 