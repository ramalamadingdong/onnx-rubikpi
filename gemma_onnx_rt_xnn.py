import sys
import os
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
import time

# Ensure we use the system onnxruntime, not the local one
if 'onnxruntime' in sys.modules:
    del sys.modules['onnxruntime']
import onnxruntime as ort

def setup_model_session(model_path):
    """Setup ONNX Runtime session with XNNPACK provider optimized for speed."""
    session_options = ort.SessionOptions()

    # Optimize for speed
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    session_options.enable_mem_reuse = True

    # Set intra-op parallelism to use all available cores
    session_options.intra_op_num_threads = 0  # 0 means use all available cores
    session_options.inter_op_num_threads = 0  # 0 means use all available cores

    # XNNPACK-specific optimizations
    xnnpack_options = {
        'intra_op_num_threads': 0,  # Use all available cores
        'enable_fast_math': True,   # Enable fast math for better performance
    }

    providers = [
        ("XnnpackExecutionProvider", xnnpack_options),
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    return session

def load_gemma_model_and_tokenizer():
    """Load Gemma 3 1B INT8 model and tokenizer."""
    # Set up paths
    root_dir = Path(__file__).parent / "gemma-3-1b-it-ONNX-GQA"
    model_path = root_dir / "onnx" / "model_int8.onnx"
    tokenizer_path = root_dir / "tokenizer.json"
    config_path = root_dir / "config.json"

    print(f"Loading model from: {model_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")

    # Load model session
    session = setup_model_session(str(model_path))

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    return session, tokenizer, config


    print("\n" + "="*50)
    print("MODEL INPUT/OUTPUT INSPECTION")
    print("="*50)

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"\nNumber of inputs: {len(inputs)}")
    for i, input_info in enumerate(inputs):
        print(f"Input {i}: {input_info.name}")
        print(f"  Shape: {input_info.shape}")
        print(f"  Type: {input_info.type}")

    print(f"\nNumber of outputs: {len(outputs)}")
    for i, output_info in enumerate(outputs):
        print(f"Output {i}: {output_info.name}")
        print(f"  Shape: {output_info.shape}")
        print(f"  Type: {output_info.type}")

    return inputs, outputs

def initialize_kv_cache(config, batch_size=1, max_seq_len=2048):
    """Initialize KV cache based on model configuration."""
    num_layers = config.get("num_hidden_layers", 26)
    num_heads = config.get("num_attention_heads", 4)
    num_kv_heads = config.get("num_key_value_heads", 1)
    hidden_size = config.get("hidden_size", 1152)
    head_dim = config.get("head_dim", 256)

    print(f"Initializing KV cache for {num_layers} layers, {num_heads} heads, {num_kv_heads} kv_heads, head_dim={head_dim}")

    # Initialize empty past key values
    past_key_values = {}
    for layer_idx in range(num_layers):
        # Format: (batch_size, num_kv_heads, seq_len, head_dim)
        past_key_values[f"past_key_values.{layer_idx}.key"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )
        past_key_values[f"past_key_values.{layer_idx}.value"] = np.zeros(
            (batch_size, num_kv_heads, 0, head_dim), dtype=np.float32
        )

    return past_key_values, num_layers, num_heads, num_kv_heads, head_dim

def prepare_model_inputs(input_ids, attention_mask, position_ids, past_key_values):
    """Prepare inputs for the model in the format expected by the ONNX model."""
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }

    # Add past key values
    model_inputs.update(past_key_values)

    return model_inputs

def update_kv_cache(past_key_values, new_key_values, num_layers):
    """Update the KV cache with new key-value pairs."""
    updated_kv = {}

    for layer_idx in range(num_layers):
        # Get new key and value tensors
        new_key = new_key_values[layer_idx * 2]
        new_value = new_key_values[layer_idx * 2 + 1]

        # Update cache
        updated_kv[f"past_key_values.{layer_idx}.key"] = new_key
        updated_kv[f"past_key_values.{layer_idx}.value"] = new_value

    return updated_kv

def generate_text(session, tokenizer, config, prompt, max_length=50, temperature=0.7, do_sample=True):
    """Generate text using the Gemma model with XNNPACK provider"""
    print(f"\nGenerating text for prompt: '{prompt}'")
    print(f"Max length: {max_length}, Temperature: {temperature}")

    # Tokenize input
    encoding = tokenizer.encode(prompt)
    input_ids = np.array([encoding.ids], dtype=np.int64)

    print(f"Token IDs: {encoding.ids[:10]}..." if len(encoding.ids) > 10 else f"Token IDs: {encoding.ids}")
    print(f"Tokens: {encoding.tokens[:10]}..." if len(encoding.tokens) > 10 else f"Tokens: {encoding.tokens}")

    batch_size, seq_len = input_ids.shape

    # Initialize attention mask and position IDs
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    # Initialize KV cache
    past_key_values, num_layers, num_heads, num_kv_heads, head_dim = initialize_kv_cache(config, batch_size)

    print(f"Model configuration: {num_layers} layers, {num_heads} heads, {num_kv_heads} kv_heads, head_dim={head_dim}")

    # First inference (prompt processing)
    print(f"\nProcessing initial prompt of length {seq_len}")

    model_inputs = prepare_model_inputs(input_ids, attention_mask, position_ids, past_key_values)

    try:
        outputs = session.run(None, model_inputs)
        logits = outputs[0]  # First output should be logits

        # Update KV cache from outputs
        if len(outputs) > 1:
            new_key_values = outputs[1:]  # Rest are key-value pairs
            past_key_values = update_kv_cache(past_key_values, new_key_values, num_layers)

    except Exception as e:
        print(f"Error during initial inference: {e}")
        return prompt

    # Sample first token
    if do_sample and temperature > 0:
        next_token_logits = logits[0, -1, :] / temperature
        probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
        next_token_id = np.random.choice(len(probs), p=probs)
    else:
        next_token_id = int(np.argmax(logits[0, -1, :]))

    generated_ids = [next_token_id]

    print(f"First generated token: {tokenizer.decode([next_token_id])}")

    # Generate remaining tokens
    current_seq_len = seq_len

    # Start timing after prompt processing
    start_time = time.time()
    tokens_generated = 1  # Already generated one token

    for step in range(max_length - 1):
        if step % 10 == 0:
            print(f"Generating token {step + 2}/{max_length}")

        # Prepare next token input
        next_input_ids = np.array([[next_token_id]], dtype=np.int64)
        next_attention_mask = np.ones((batch_size, current_seq_len + 1), dtype=np.int64)
        next_position_ids = np.array([[current_seq_len]], dtype=np.int64)

        model_inputs = prepare_model_inputs(
            next_input_ids,
            next_attention_mask,
            next_position_ids,
            past_key_values
        )

        try:
            outputs = session.run(None, model_inputs)
            logits = outputs[0]

            # Update KV cache
            if len(outputs) > 1:
                new_key_values = outputs[1:]
                past_key_values = update_kv_cache(past_key_values, new_key_values, num_layers)

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
        tokens_generated += 1
        current_seq_len += 1

        # Check for end of sequence
        if hasattr(tokenizer, 'token_to_id'):
            eos_token_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>")
            if eos_token_id and next_token_id == eos_token_id:
                print(f"End of sequence reached at step {step}")
                break

    end_time = time.time()
    elapsed = end_time - start_time
    if elapsed > 0:
        print(f"\nTokens generated: {tokens_generated}")
        print(f"Time taken for generation: {elapsed:.2f} seconds")
        print(f"Tokens per second: {tokens_generated / elapsed:.2f}")
    else:
        print("\nTiming error: elapsed time is zero.")

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids)
    full_response = prompt + generated_text

    return full_response

def main():
    """Main function demonstrating Gemma 3 1B INT8 inference following."""
    print("="*70)
    print("GEMMA 3 1B INT8 ONNX INFERENCE ")
    print("="*70)

    try:
        # Load model and tokenizer
        session, tokenizer, config = load_gemma_model_and_tokenizer()

        print("\n" + "="*70)
        print("GENERATION INTERFACE")
        print("="*70)

        user_prompt = input("\nEnter your prompt/question: ")
        if not user_prompt.strip():
            print("No prompt entered. Exiting.")
            return

        try:
            generated_text = generate_text(
                session=session,
                tokenizer=tokenizer,
                config=config,
                prompt=user_prompt,
                max_length=30,
                temperature=0.7,
                do_sample=True
            )

            print(f"\nOriginal prompt: {user_prompt}")
            print(f"Generated text: {generated_text}")
            print("-" * 50)

        except Exception as e:
            print(f"Error generating text: {e}")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("\nMake sure you have the required files:")
        print("- gemma-3-1b-it-ONNX-GQA/onnx/model_int8.onnx")
        print("- gemma-3-1b-it-ONNX-GQA/tokenizer.json")
        print("- gemma-3-1b-it-ONNX-GQA/config.json")
        print("\nTo download Gemma models:")
        print("huggingface-cli download onnx-community/gemma-3-1b-it-ONNX-GQA --local-dir ./gemma-3-1b-it-ONNX-GQA")
        print("\nRequired dependencies:")
        print("pip install onnxruntime tokenizers numpy huggingface_hub")
        print("\nSee README.md for complete setup instructions.")

if __name__ == "__main__":
    main()
