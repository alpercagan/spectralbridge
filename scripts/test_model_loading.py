"""Test loading BEATs and DINOv2 models locally."""

import torch
import timm
from pathlib import Path

def test_dinov2():
    """Load DINOv2 and test on dummy image."""
    print("\n" + "="*60)
    print("Testing DINOv2...")
    print("="*60)
    
    # Use MPS if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading DINOv2-base model...")
    model = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        num_classes=0  # Remove classification head, just get features
    )
    model = model.to(device)
    model.eval()
    
    # Test with dummy input (224x224 image)
    print("Testing forward pass...")
    dummy_image = torch.randn(1, 3, 518, 518).to(device)
    
    with torch.no_grad():
        features = model(dummy_image)
    
    print(f"✓ DINOv2 output shape: {features.shape}")
    print(f"  Expected: torch.Size([1, 768])")
    
    # Memory info
    if device.type == "mps":
        print(f"✓ Model successfully running on Apple Silicon (MPS)")
    
    return model

def test_beats():
    """Check BEATs checkpoint availability."""
    print("\n" + "="*60)
    print("Testing BEATs...")
    print("="*60)
    
    print("Note: BEATs requires manual checkpoint download")
    print("We'll implement the full loader in Week 1")
    print("For now, just checking we have the right structure...")
    
    # Check if we have a checkpoints directory ready
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Checkpoint directory ready: {checkpoint_dir}")
    print("\nBEATs download instructions:")
    print("1. Go to: https://github.com/microsoft/unilm/tree/master/beats")
    print("2. Download: BEATs_iter3_plus_AS2M.pt")
    print("3. Place in: outputs/checkpoints/")
    print("\nWe'll do this in Week 1, Day 1")
    
def main():
    print("="*60)
    print("Model Loading Test - M1 Mac")
    print("="*60)
    
    # Test DINOv2 
    dinov2_model = test_dinov2()
    
    # Test BEATs
    test_beats()
    
    print("\n" + "="*60)
    print("Model loading test complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Download BEATs checkpoint (see instructions above)")
    print("2. Set up VGGSound data pipeline")
    print("3. Create feature extraction pipeline")

if __name__ == "__main__":
    main()