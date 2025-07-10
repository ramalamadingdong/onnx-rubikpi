import onnxruntime as ort
import os
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

def setup_optimized_model_session(model_path, num_threads=None, enable_profiling=False):
    """Setup ONNX Runtime session with optimized configuration for high performance."""
    session_options = ort.SessionOptions()

    # Performance optimizations
    if num_threads is None:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 4  # Default fallback
        num_threads = min(cpu_count, 8)  # Use available cores, max 8

    # Set number of threads for CPU execution
    session_options.intra_op_num_threads = num_threads
    session_options.inter_op_num_threads = num_threads

    # Enable all optimizations
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Memory optimizations
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.enable_mem_reuse = True

    # Execution optimizations
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options.enable_profiling = enable_profiling

    # Disable some features for better performance
    session_options.add_session_config_entry("session.disable_prepacking", "1")
    session_options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")

    # Use CPU provider with optimizations
    providers = [
        ("CPUExecutionProvider", {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cpu_architecture": "arm64",  # Adjust based on your CPU
            "cpu_features": "ARMV8",  # Enable advanced CPU features
        })
    ]

    print(f"Setting up optimized session with {num_threads} threads")
    session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    return session

def load_gemma_model_and_tokenizer_optimized(num_threads=None):
    """Load Gemma 3 1B model and tokenizer with optimized settings."""
    root_dir = Path(__file__).parent
    model_dir = root_dir / "gemma-3-1b-it-ONNX-GQA"
    tokenizer_path = model_dir / "tokenizer.model"
    config_path = model_dir / "config.json"

    print(f"Loading optimized model from: {model_dir}")

    # Check if ONNX model exists - try different model variants
    onnx_dir = model_dir / "onnx"
    session = None

    # Try different ONNX model variants in order of preference
    # Prioritize larger models that are self-contained
    model_variants = [
        "model_int8.onnx",  # INT8 quantized (self-contained)
        "model_q4.onnx",  # Q4 quantized (self-contained)
        "model_bnb4.onnx",  # BNB4 quantized (self-contained)
        "model_fp16.onnx",  # FP16 model (self-contained)
        "model.onnx",  # Original model (may need external data)
    ]

    for model_name in model_variants:
        onnx_model_path = onnx_dir / model_name
        if onnx_model_path.exists():
            print(f"Trying ONNX model: {onnx_model_path}")
            try:
                session = setup_optimized_model_session(str(onnx_model_path), num_threads)
                print(f"Successfully loaded: {model_name}")
                break
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                session = None
                continue

    if session is None:
        print("Could not load any ONNX model variants.")
        print("Available ONNX models:", list(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else "No onnx directory found")
        print("Available files in model directory:", list(model_dir.iterdir()) if model_dir.exists() else "Directory not found")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except ImportError:
        print("Transformers library not available. Please install it with: pip install transformers")
        return None, None, None

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    return session, tokenizer, config

def initialize_optimized_kv_cache(config, batch_size=1, max_seq_len=2048):
    """Initialize optimized KV cache with pre-allocated memory."""
    num_layers = config.get("num_hidden_layers", 26)
    num_heads = config.get("num_attention_heads", 4)
    num_kv_heads = config.get("num_key_value_heads", 1)
    hidden_size = config.get("hidden_size", 1152)
    head_dim = config.get("head_dim", 256)

    print(f"Initializing optimized KV cache: {num_layers} layers, {num_kv_heads} KV heads")

    # Pre-allocate KV cache with optimal memory layout
    past_key_values = {}
    for layer_idx in range(num_layers):
        # Use float32 for compatibility with ONNX model
        dtype = np.float32

        past_key_values[f"past_key_values.{layer_idx}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=dtype
        )
        past_key_values[f"past_key_values.{layer_idx}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=dtype
        )

    return past_key_values, num_layers, num_heads, num_kv_heads, head_dim

def optimized_tokenize_input(tokenizer, prompt):
    """Optimized tokenization with error handling."""
    try:
        # Try transformers-style tokenization first
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].numpy()
    except Exception:
        # Fallback to tokenizers library
        if hasattr(tokenizer, 'encode'):
            encoding = tokenizer.encode(prompt)
            if hasattr(encoding, 'ids'):
                input_ids = np.array([encoding.ids], dtype=np.int64)
            else:
                input_ids = np.array([encoding], dtype=np.int64)
        else:
            raise ValueError("Tokenizer does not support expected methods")

    return input_ids

def batch_generate_text(session, tokenizer, config, prompts, max_length=50, temperature=0.7, do_sample=True):
    """Generate text for multiple prompts in batch for better throughput."""
    results = []

    # Process prompts in parallel if multiple
    if len(prompts) > 1:
        with ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as executor:
            futures = [
                executor.submit(generate_text_optimized, session, tokenizer, config, prompt, max_length, temperature, do_sample)
                for prompt in prompts
            ]
            results = [future.result() for future in futures]
    else:
        results = [generate_text_optimized(session, tokenizer, config, prompts[0], max_length, temperature, do_sample)]

    return results

def generate_text_optimized(session, tokenizer, config, prompt, max_length=50, temperature=0.7, do_sample=True):
    """Optimized text generation with improved performance."""
    start_time = time.time()

    # Optimized tokenization
    input_ids = optimized_tokenize_input(tokenizer, prompt)
    batch_size, seq_len = input_ids.shape

    # Pre-allocate arrays for better performance
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    # Initialize optimized KV cache
    past_key_values, num_layers, num_heads, num_kv_heads, head_dim = initialize_optimized_kv_cache(config, batch_size)

    if session is not None:
        # ONNX inference with optimizations
        model_inputs_info = session.get_inputs()

        # Pre-allocate model inputs dictionary
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

        # Add KV cache inputs efficiently
        expected_inputs = {input_info.name for input_info in model_inputs_info}
        for layer_idx in range(num_layers):
            key_name = f"past_key_values.{layer_idx}.key"
            value_name = f"past_key_values.{layer_idx}.value"

            if key_name in expected_inputs:
                model_inputs[key_name] = past_key_values[key_name]
            if value_name in expected_inputs:
                model_inputs[value_name] = past_key_values[value_name]

        try:
            # Initial inference
            outputs = session.run(None, model_inputs)
            logits = outputs[0]

            # Update KV cache efficiently
            if len(outputs) > 1:
                new_key_values = outputs[1:]
                past_key_values = update_kv_cache_optimized(past_key_values, new_key_values, num_layers, num_kv_heads)

            # Sample first token
            if do_sample and temperature > 0:
                next_token_logits = logits[0, -1, :] / temperature
                probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                next_token_id = np.random.choice(len(probs), p=probs)
            else:
                next_token_id = int(np.argmax(logits[0, -1, :]))

            generated_ids = [next_token_id]
            current_seq_len = seq_len

            # Optimized generation loop
            for step in range(max_length - 1):
                # Prepare next token input efficiently
                next_input_ids = np.array([[next_token_id]], dtype=np.int64)
                next_attention_mask = np.ones((batch_size, current_seq_len + 1), dtype=np.int64)
                next_position_ids = np.array([[current_seq_len]], dtype=np.int64)

                # Update model inputs efficiently
                model_inputs["input_ids"] = next_input_ids
                model_inputs["attention_mask"] = next_attention_mask
                model_inputs["position_ids"] = next_position_ids

                # Update KV cache inputs
                for layer_idx in range(num_layers):
                    key_name = f"past_key_values.{layer_idx}.key"
                    value_name = f"past_key_values.{layer_idx}.value"

                    if key_name in expected_inputs:
                        model_inputs[key_name] = past_key_values[key_name]
                    if value_name in expected_inputs:
                        model_inputs[value_name] = past_key_values[value_name]

                try:
                    outputs = session.run(None, model_inputs)
                    logits = outputs[0]

                    # Update KV cache
                    if len(outputs) > 1:
                        new_key_values = outputs[1:]
                        past_key_values = update_kv_cache_optimized(past_key_values, new_key_values, num_layers, num_kv_heads)

                except Exception as e:
                    print(f"Error during generation step {step}: {e}")
                    break

                # Sample next token
                if do_sample and temperature > 0:
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
                    next_token_id = np.random.choice(len(probs), p=probs)
                else:
                    next_token_id = int(np.argmax(logits[0, -1, :]))

                generated_ids.append(next_token_id)
                current_seq_len += 1

                # Check for end of sequence
                if hasattr(tokenizer, 'token_to_id'):
                    eos_token_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>")
                    if eos_token_id and next_token_id == eos_token_id:
                        break

            # Decode generated text
            generated_text = tokenizer.decode(generated_ids)
            full_response = prompt + generated_text

            end_time = time.time()
            elapsed = end_time - start_time
            tokens_generated = len(generated_ids)

            print(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.2f} tokens/s)")

            return full_response

        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            return generate_text_with_transformers_optimized(tokenizer, prompt, max_length, temperature, do_sample)

    else:
        return generate_text_with_transformers_optimized(tokenizer, prompt, max_length, temperature, do_sample)

def update_kv_cache_optimized(past_key_values, new_key_values, num_layers, num_kv_heads=1):
    """Optimized KV cache update with minimal memory allocation."""
    updated_kv = {}

    for layer_idx in range(num_layers):
        new_key = new_key_values[layer_idx * 2]
        new_value = new_key_values[layer_idx * 2 + 1]

        updated_kv[f"past_key_values.{layer_idx}.key"] = new_key
        updated_kv[f"past_key_values.{layer_idx}.value"] = new_value

    return updated_kv

def generate_text_with_transformers_optimized(tokenizer, prompt, max_length=50, temperature=0.7, do_sample=True):
    """Optimized transformers fallback."""
    try:
        from transformers import AutoModelForCausalLM
        import torch

        # Try to find the model in the expected directory
        model_path = Path(__file__).parent / "gemma-3-1b-it-ONNX-GQA"

        # Check if the model directory exists and contains model files
        if not model_path.exists():
            print(f"Model directory not found: {model_path}")
            return prompt

        # Look for common model file patterns
        model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
        if not model_files:
            print(f"No transformers model files (*.bin, *.safetensors) found in: {model_path}")
            print("Available files:", list(model_path.iterdir()))
            print("This directory only contains ONNX models. Please use ONNX inference or download the transformers model.")
            return prompt

        model = AutoModelForCausalLM.from_pretrained(str(model_path))
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length + len(inputs["input_ids"][0]),
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

    except Exception as e:
        print(f"Error during transformers generation: {e}")
        return prompt

def main_optimized():
    """Main function with performance optimizations."""
    print("="*70)
    print("GEMMA 3 1B ONNX INFERENCE - OPTIMIZED")
    print("="*70)

    try:
        # Load model with optimized settings
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 4  # Default fallback
        num_threads = min(cpu_count, 8)  # Use available cores
        session, tokenizer, config = load_gemma_model_and_tokenizer_optimized(num_threads)

        if tokenizer is None:
            print("Could not load tokenizer. Exiting.")
            return

        if session is None:
            print("ONNX session not available.")
            # Check if transformers model files exist
            model_path = Path(__file__).parent / "gemma-3-1b-it-ONNX-GQA"
            model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
            if not model_files:
                print("No transformers model files found either.")
                print("Please ensure either ONNX models or transformers model files are available.")
                return
            else:
                print("Will use transformers fallback.")

        print(f"\nUsing {num_threads} threads for inference")
        print("="*70)
        print("OPTIMIZED GENERATION INTERFACE")
        print("="*70)

        while True:
            user_prompt = input("\nEnter your prompt/question (or 'quit' to exit): ")
            if user_prompt.lower() == 'quit':
                break
            if not user_prompt.strip():
                continue

            try:
                # Clear memory before generation
                gc.collect()

                generated_text = generate_text_optimized(
                    session=session,
                    tokenizer=tokenizer,
                    config=config,
                    prompt=user_prompt,
                    max_length=50,  # Increased for better results
                    temperature=0.7,
                    do_sample=True
                )

                print(f"\nGenerated response: {generated_text}")
                print("-" * 50)

            except Exception as e:
                print(f"Error generating text: {e}")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("\nMake sure you have the required files in: gemma-3-1b-it-ONNX-GQA/")
        print("Required files:")
        print("- gemma-3-1b-it-ONNX-GQA/onnx/model_int8.onnx (or other .onnx files)")
        print("- gemma-3-1b-it-ONNX-GQA/tokenizer.json")
        print("- gemma-3-1b-it-ONNX-GQA/config.json")
        print("\nTo download Gemma models:")
        print("huggingface-cli download onnx-community/gemma-3-1b-it-ONNX-GQA --local-dir ./gemma-3-1b-it-ONNX-GQA")
        print("\nRequired dependencies:")
        print("pip install onnxruntime tokenizers numpy transformers huggingface_hub")
        print("\nSee README.md for complete setup instructions including licensing requirements.")

if __name__ == "__main__":
    main_optimized()
