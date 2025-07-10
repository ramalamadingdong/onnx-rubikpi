# ONNX RubikPI XNN - Language Model Inference

This project provides optimized ONNX Runtime inference implementations for running language models (Gemma 3 1B and Llama 3.2 1B) on RUBIK Pi.

## Features

- **Optimized ONNX Runtime**: Uses custom RUBIK-Pi ONNX Runtime wheel for enhanced performance
- **Multiple Inference Modes**: CPU-optimized, XNNPACK-accelerated, and standard implementations
- **KV Cache Management**: Efficient memory management for text generation
- **Flexible Model Support**: Supports both Gemma 3 1B and Llama 3.2 1B models
- **Quantization Support**: Works with INT8, Q4, and other quantized model variants

## Installation

### 1. Install Dependencies

```bash
# Install from requirements.txt (includes custom wheel)
pip install -r requirements.txt
```

### 2. Install Hugging Face CLI (Required for Model Downloads)

```bash
# Install Hugging Face CLI
pip install huggingface_hub
```

### 3. Model Licensing and Access Requirements

#### Licensed Models (Llama)

**Llama 3.2 models require license acceptance:**

1. **Request Access:**
   - Visit [Hugging Face Llama 3.2 page](https://huggingface.co/onnx-community/Llama-3.2-1B)
   - Click "Request Access" and fill out the form
   - Wait for approval (usually within a few hours to days)

2. **Authenticate with Hugging Face:**
   ```bash
   # Login to Hugging Face (required for licensed models)
   huggingface-cli login
   # Enter your Hugging Face token when prompted
   ```

3. **Get Your Access Token:**
   - Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions
   - Copy the token for use in the login command

#### Open Models (Gemma)

**Gemma models are open access** but authentication is still recommended for reliable downloads.

### 4. Download Models

**Important:** All models should be downloaded into the same directory as the Python scripts.

#### Gemma 3 1B Model

```bash
# Navigate to your project directory
cd onnx-rubikpi

# Download Gemma model (no authentication required)
huggingface-cli download onnx-community/gemma-3-1b-it-ONNX-GQA --local-dir ./gemma-3-1b-it-ONNX-GQA

# Verify download
ls -la gemma-3-1b-it-ONNX-GQA/onnx/
```

#### Llama 3.2 1B Model

**Prerequisites:** Must have completed license acceptance and authentication steps above.

```bash
# Navigate to your project directory
cd onnx-rubikpi

# Verify you're authenticated
huggingface-cli whoami

# Download Llama model (requires authentication)
huggingface-cli download onnx-community/Llama-3.2-1B --local-dir ./Llama-3.2-1b

# Verify download
ls -la Llama-3.2-1b/onnx/
```


### Expected Directory Structure

After downloading both models, your project should look like this:

```
onnx-rubikpi/
├── gemma_onnx_rt_cpu_optimized.py
├── gemma_onnx_rt_xnn.py
├── llama_onnx_rt_cpu.py
├── requirements.txt
├── README.md
├── gemma-3-1b-it-ONNX-GQA/
│   ├── onnx/
│   │   ├── model_int8.onnx      # ~1-2GB
│   │   ├── model_q4.onnx        # ~500MB-1GB
│   │   └── model_fp16.onnx      # ~2-4GB
│   ├── tokenizer.json
│   └── config.json
└── Llama-3.2-1b/
    ├── onnx/
    │   └── model_int8.onnx      # ~1-2GB
    ├── tokenizer.json
    └── config.json
```


## Usage

### Gemma 3 1B - XNNPACK Accelerated (Recommended)

```bash
python gemma_onnx_rt_xnn.py
```

Features:
- XNNPACK execution provider for ARM optimization
- Fast math optimizations
- Specifically tuned for ARM64 devices
- ~3 Tokens per Sec


### Gemma 3 1B - CPU Optimized

```bash
python gemma_onnx_rt_cpu_optimized.py
```

Features:
- Multi-threaded inference
- Automatic CPU core detection
- Memory optimizations
- Supports multiple model variants (INT8, Q4, FP16)
- Fallback to transformers if ONNX unavailable
- ~2 Tokens per Sec

### Llama 3.2 1B - CPU Inference

```bash
python llama_onnx_rt_cpu.py
```

Features:
- CPU-only inference
- Supports INT8 quantized models
- Optimized for memory efficiency
- ~1 Tokens per Sec


## Performance Tips

1. **Use INT8 Models**: Significantly faster and use less memory
2. **Optimize Thread Count**: The CPU-optimized version automatically detects cores
3. **Enable XNNPACK**: Use the XNNPACK version for ARM64 optimization
4. **Batch Processing**: For multiple prompts, use the batch generation features

## License and Legal Considerations

### Project License
This project is licensed under the MIT License. See the LICENSE file for details.

### Model Licenses
**Important:** Each model has its own license terms that you must comply with:

- **Llama 3.2 Models:** Subject to the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE)
  - Commercial use allowed with restrictions
  - Must comply with acceptable use policies
  - May require additional agreements for large-scale commercial use

- **Gemma Models:** Subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
  - Generally permissive for research and commercial use
  - Must comply with Google's usage policies

**Your Responsibility:** You are responsible for ensuring your use of these models complies with their respective license terms. Please review each model's license before use.

## Contributing

Feel free to submit issues and pull requests to improve the inference implementations.

## Acknowledgments

- Hugging Face for providing the models and ONNX conversions
- ONNX Runtime team for the optimized runtime
- Google for the Gemma models
- Meta for the Llama models
