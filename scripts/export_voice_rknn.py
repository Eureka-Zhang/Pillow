import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import model_vo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from Sensor.Voice.training_voice_model.model_vo import CNN_TCN_MTL, NUM_POSTURE_CLASSES
except ImportError:
    # Try alternate path if running from root
    sys.path.append(os.path.join(os.getcwd(), 'Sensor', 'Voice', 'training_voice_model'))
    from model_vo import CNN_TCN_MTL, NUM_POSTURE_CLASSES

def export_onnx(pth_path, onnx_path):
    print(f"Loading PyTorch model from {pth_path}...")
    model = CNN_TCN_MTL(snore_classes=2, posture_classes=NUM_POSTURE_CLASSES)
    
    # Load weights
    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return False
        
    model.eval()
    
    # Dummy input: [1, 1, 80, 301]
    # N_MELS=80, WINDOW_DURATION=3.0, SR=16000, HOP_LENGTH=160 -> 301 frames
    dummy_input = torch.randn(1, 1, 80, 301)
    
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["snore_logits", "posture_logits"], # Crucial: 2 outputs
        opset_version=11
    )
    print("ONNX export successful.")
    return True

def export_rknn(onnx_path, rknn_path, target_platform='rk3588'):
    try:
        from rknn.api import RKNN
    except ImportError:
        print("Error: rknn-toolkit2 not found. Please install it to export RKNN models.")
        print("Note: rknn-toolkit2 usually runs on x86 Linux PCs, not on the RK3588 board itself.")
        return False

    rknn = RKNN()
    rknn.config(
        target_platform=target_platform,
        quantized_dtype='asymmetric_quantized-8', # or 'w8a8' depending on version
        # optimization_level=3
    )
    
    print(f"Loading ONNX model: {onnx_path}")
    # Load ONNX with correct input shape
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=['input'],
        input_size_list=[[1, 1, 80, 301]] 
    )
    if ret != 0:
        print("Load ONNX failed!")
        return False
        
    print("Building RKNN model...")
    # Ideally use a dataset for quantization, but for now we might skip or use dummy if no dataset provided
    # To get best accuracy, use 'dataset' argument with a txt file pointing to .npy files or images
    # For now, we will use do_quantization=False to ensure it works first, or warn about it.
    # If the user wants quantization, they need to provide a dataset.
    
    # We'll default to fp16 (no quantization) if no dataset is provided, to be safe.
    # Or strict quantization requires a dataset.
    
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build RKNN failed!")
        return False
        
    print(f"Exporting RKNN model to {rknn_path}")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("Export RKNN failed!")
        return False
        
    rknn.release()
    print("RKNN export successful.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export Voice Model to ONNX/RKNN")
    parser.add_argument("pth_path", help="Path to input .pth model file")
    parser.add_argument("--output", help="Path to output .rknn file", default="voice_model.rknn")
    parser.add_argument("--onnx_only", action="store_true", help="Only export ONNX")
    
    args = parser.parse_args()
    
    onnx_path = args.pth_path.replace(".pth", ".onnx")
    
    if not export_onnx(args.pth_path, onnx_path):
        sys.exit(1)
        
    if args.onnx_only:
        print(f"Done. ONNX file: {onnx_path}")
        return

    if export_rknn(onnx_path, args.output):
        print(f"Done. RKNN file: {args.output}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
