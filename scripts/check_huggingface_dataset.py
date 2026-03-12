"""Check what's in the Loie/VGGSound HuggingFace dataset."""

from datasets import load_dataset
import sys

def check_vggsound_hf():
    """
    Investigate the Loie/VGGSound dataset structure.
    """
    print("="*60)
    print("Checking Loie/VGGSound on HuggingFace...")
    print("="*60)
    
    try:
        # Try loading without schema validation
        print("\nAttempting to load dataset (this may take a moment)...")
        
        # Load the dataset - it appears to be in webdataset format
        dataset = load_dataset("Loie/VGGSound", split="train", streaming=True)
        
        print("✓ Dataset loaded successfully!")
        
        # Get first sample
        print("\nFetching first sample...")
        sample = next(iter(dataset))
        
        print("\n" + "="*60)
        print("SAMPLE STRUCTURE")
        print("="*60)
        
        for key, value in sample.items():
            value_type = type(value).__name__
            
            if isinstance(value, bytes):
                print(f"  {key}: bytes (size: {len(value)} bytes = {len(value)/1024:.1f} KB)")
            elif isinstance(value, str):
                print(f"  {key}: {value_type} = '{value[:100]}...' " if len(value) > 100 else f"  {key}: {value_type} = '{value}'")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value_type} = {value}")
            elif value is None:
                print(f"  {key}: None")
            else:
                print(f"  {key}: {value_type}")
        
        print("\n" + "="*60)
        print("DATASET ASSESSMENT")
        print("="*60)
        
        has_mp4 = 'mp4' in sample
        has_audio = 'audio' in sample or 'wav' in sample
        has_image = 'image' in sample or 'jpg' in sample or 'png' in sample
        
        if has_mp4:
            print("✅ Has MP4 video files (binary data)")
            print("   → We can extract audio and frames from these!")
        
        if has_audio:
            print("✅ Has pre-extracted audio")
        
        if has_image:
            print("✅ Has pre-extracted images")
        
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        
        if has_mp4:
            print("🎉 This dataset has actual VIDEO FILES!")
            print("\nNext steps:")
            print("1. Download the dataset (it has pre-downloaded videos)")
            print("2. Extract audio from MP4 files")
            print("3. Extract middle frame from MP4 files")
            print("4. This saves us from YouTube downloading!")
            print("\nEstimated time saved: 2-3 days")
        else:
            print("⚠️  Dataset structure unclear.")
            print("   May need to download from YouTube ourselves.")
        
        # Try to get more info
        print("\n" + "="*60)
        print("Checking a few more samples...")
        print("="*60)
        
        for i, sample in enumerate(dataset):
            if i >= 3:  # Check first 3 samples
                break
            print(f"\nSample {i+1}:")
            print(f"  __key__: {sample.get('__key__', 'N/A')}")
            print(f"  has mp4: {'mp4' in sample}")
            if 'mp4' in sample and sample['mp4']:
                print(f"  mp4 size: {len(sample['mp4'])/1024:.1f} KB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTrying alternative loading method...")
        
        try:
            # Try loading non-streaming
            print("\nAttempting non-streaming load...")
            dataset = load_dataset("Loie/VGGSound", split="train")
            print(f"✓ Dataset has {len(dataset)} samples")
            print(f"✓ Columns: {dataset.column_names}")
            
            sample = dataset[0]
            print("\nFirst sample keys:", list(sample.keys()))
            
        except Exception as e2:
            print(f"❌ Also failed: {e2}")
            print("\nThis dataset may require special handling.")

if __name__ == "__main__":
    check_vggsound_hf()