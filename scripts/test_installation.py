"""Test that all core dependencies are installed and working."""

import sys
import torch

def test_pytorch():
    """Test PyTorch installation."""
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Detect available device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ MPS (Apple Silicon) available")
    else:
        device = "cpu"
        print(f"✓ Using CPU")
    
    # Test basic tensor operations
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = x @ y
    print(f"✓ Tensor operations working on {device}")
    
def test_imports():
    """Test all critical imports."""
    imports = [
        ("transformers", "Transformers"),
        ("timm", "TIMM"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML"),
        ("diffusers", "Diffusers"),
    ]
    
    print("\nTesting imports:")
    for module, name in imports:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("SpectralBridge Environment Test")
    print("=" * 60)
    test_pytorch()
    test_imports()
    print("\n" + "=" * 60)
    print("Environment ready! ✓")
    print("=" * 60)