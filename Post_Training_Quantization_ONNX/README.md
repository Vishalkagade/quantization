# Post-Training Quantization with PyTorch and ONNX Export

This project demonstrates post-training quantization (PTQ) techniques using PyTorch on the CIFAR-10 dataset, including model export to ONNX format and performance benchmarking.

## 🎯 Project Overview

- **Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)
- **Model**: ResNet-18 (pretrained on ImageNet, fine-tuned for CIFAR-10)
- **Quantization**: 8-bit post-training quantization using PyTorch's FX Graph Mode
- **Export**: ONNX format for cross-platform deployment
- **Benchmarking**: Speed and accuracy comparison between PyTorch and ONNX Runtime

## 🚀 Features

- ✅ Model training and evaluation on CIFAR-10
- ✅ Post-training quantization with calibration
- ✅ Model size reduction analysis
- ✅ Accuracy preservation assessment
- ✅ ONNX export with dynamic batch size support
- ✅ Performance benchmarking (PyTorch vs ONNX Runtime)
- ✅ Output difference analysis

## 📋 Requirements

```bash
pip install torch torchvision tqdm matplotlib onnx onnxruntime
```

## 🏃‍♂️ Quick Start

1. **Clone and navigate to the project directory**
2. **Install dependencies** (see requirements above)
3. **Run the notebook**: `post_training_quantization_CIFAR.ipynb`

## 📊 Results Summary

The quantization process typically achieves:
- **Model Size**: ~75% reduction (from float32 to int8)
- **Speed**: 1.5-3x faster inference (depending on hardware)
- **Accuracy**: <1% degradation with proper calibration

## 🔧 Key Components

### Quantization Process
1. **Model Preparation**: Add observers to track activation ranges
2. **Calibration**: Run representative data through the model
3. **Conversion**: Convert to quantized format with scale/zero-point parameters

### ONNX Export
- Opset version 13 for quantized model support
- Dynamic batch size for flexible deployment
- Cross-platform compatibility

### Benchmarking
- Inference speed comparison
- Output accuracy verification
- Statistical analysis of performance gains

## 📝 Usage Example

```python
# Load and quantize model
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# Prepare for quantization
model_prepared = prepare_fx(model, qconfig_dict, example_input)

# Calibrate with representative data
with torch.inference_mode():
    for data in calibration_loader:
        model_prepared(data)

# Convert to quantized model
quantized_model = convert_fx(model_prepared)

# Export to ONNX
torch.onnx.export(quantized_model, dummy_input, "model.onnx", opset_version=13)
```

## 🎛️ Configuration

- **Quantization Backend**: FBGEMM (CPU optimized)
- **Calibration**: 20+ batches of training data
- **ONNX Opset**: Version 13 (required for quantized models)

## 📈 Performance Metrics

The benchmark analyzes:
- **Inference Time**: Average and standard deviation
- **Speedup Factor**: ONNX vs PyTorch performance ratio
- **Output Difference**: Max and mean absolute differences
- **Numerical Accuracy**: Close output verification

## 🔍 File Structure

```
├── post_training_quantization_CIFAR.ipynb  # Main notebook
├── data/                                   # CIFAR-10 dataset (auto-downloaded)
├── cifar.pt                               # Trained model weights
├── model_quantized.pth                    # Quantized model weights
├── model_quantized.onnx                   # Exported ONNX model
└── README.md                              # This file
```

## 🛠️ Troubleshooting

**ONNX Export Issues**: Ensure you're using opset version 13+ for quantized models
**Import Errors**: Install missing packages with pip
**Device Issues**: Code automatically detects CUDA availability

## 📚 Learn More

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [Model Optimization Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---
*Built with PyTorch, ONNX Runtime, and ❤️*