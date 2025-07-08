import onnxruntime as ort
import os
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import json
import time

def setup_model_session(model_path):
    """Setup ONNX Runtime session with CPU-only configuration."""
    session_options = ort.SessionOptions()

    # Set session configuration for optimization
    session_options.add_session_config_entry("session.disable_prepacking", "1")
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
    return session

def load_gemma_model_and_tokenizer():
    """Load Gemma 3 1B model and tokenizer."""
    # Set up paths - fix to use the correct directory
    root_dir = Path(__file__).parent
    model_dir = root_dir / "gemma-3-1b-it-ONNX-GQA"  # Updated path
    tokenizer_path = model_dir / "tokenizer.model"
    config_path = model_dir / "config.json"

    print(f"Loading model from: {model_dir}")
    print(f"Loading tokenizer from: {tokenizer_path}")

    # Check if ONNX model exists, if not, provide guidance
    onnx_model_path = model_dir / "onnx" / "model_int8.onnx"  # Updated path
    if onnx_model_path.exists():
        print(f"Found ONNX model at: {onnx_model_path}")
        session = setup_model_session(str(onnx_model_path))
    else:
        print("ONNX model not found. You need to convert the Gemma model to ONNX format first.")
        print("Please run the conversion script or place the ONNX model at:")
        print(f"  {onnx_model_path}")
        print("\nFor now, we'll use the original model format with transformers library.")

        # Load using transformers for now
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForCausalLM.from_pretrained(str(model_dir))
            session = None  # We'll use the model directly instead of ONNX session
        except ImportError:
            print("Transformers library not available. Please install it with:")
            print("pip install transformers")
            return None, None, None

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except ImportError:
        print("Transformers library not available. Please install it with:")
        print("pip install transformers")
        return None, None, None

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    return session, tokenizer, config


def initialize_kv_cache(config, batch_size=1, max_seq_len=2048):
    """Initialize KV cache based on Gemma model configuration."""
    num_layers = config.get("num_hidden_layers", 26)  # Gemma 3 1B has 26 layers
    num_heads = config.get("num_attention_heads", 4)  # Gemma 3 1B has 4 heads
    num_kv_heads = config.get("num_key_value_heads", 1)  # Gemma 3 1B has 1 KV head
    hidden_size = config.get("hidden_size", 1152)  # Gemma 3 1B has 1152 hidden size
    head_dim = config.get("head_dim", 256)  # Gemma 3 1B has 256 head dimension

    print(f"Initializing KV cache for {num_layers} layers, {num_heads} heads, {num_kv_heads} KV heads, head_dim={head_dim}")

    # Initialize empty past key values - format depends on ONNX model structure
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

def update_kv_cache(past_key_values, new_key_values, num_layers, num_kv_heads=1):
    """Update the KV cache with new key-value pairs from ONNX model outputs."""
    updated_kv = {}

    for layer_idx in range(num_layers):
        # The ONNX model outputs 'present.X.key' and 'present.X.value'
        # We need to convert these to 'past_key_values.X.key' and 'past_key_values.X.value' for next iteration
        new_key = new_key_values[layer_idx * 2]  # present.X.key
        new_value = new_key_values[layer_idx * 2 + 1]  # present.X.value

        # Update cache for next iteration
        updated_kv[f"past_key_values.{layer_idx}.key"] = new_key
        updated_kv[f"past_key_values.{layer_idx}.value"] = new_value

    return updated_kv

def generate_text(session, tokenizer, config, prompt, max_length=50, temperature=0.7, do_sample=True):
    """Generate text using the Gemma model"""
    print(f"\nGenerating text for prompt: '{prompt}'")
    print(f"Max length: {max_length}, Temperature: {temperature}")

    # Tokenize input
    try:
        # Try transformers-style tokenization first
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].numpy()
    except Exception as e:
        # Fallback to tokenizers library if transformers method fails
        try:
            if hasattr(tokenizer, 'encode'):
                encoding = tokenizer.encode(prompt)
                # Check if encoding has .ids attribute (tokenizers library)
                if hasattr(encoding, 'ids'):
                    input_ids = np.array([encoding.ids], dtype=np.int64)
                else:
                    # encoding is already a list of ids
                    input_ids = np.array([encoding], dtype=np.int64)
            else:
                raise ValueError("Tokenizer does not support expected methods")
        except Exception as e2:
            print(f"Error in tokenization: {e2}")
            raise ValueError(f"Failed to tokenize input: {e2}")

    batch_size, seq_len = input_ids.shape

    # Initialize attention mask and position IDs
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    # Initialize KV cache
    past_key_values, num_layers, num_heads, num_kv_heads, head_dim = initialize_kv_cache(config, batch_size)

    print(f"Model configuration: {num_layers} layers, {num_heads} heads, {num_kv_heads} KV heads, head_dim={head_dim}")

    # Check if we're using ONNX or transformers
    if session is not None:
        # ONNX inference
        print(f"\nProcessing initial prompt of length {seq_len} using ONNX")

        # First, let's inspect the model inputs to understand the expected format
        model_inputs_info = session.get_inputs()

        # Prepare inputs based on what the model actually expects
        model_inputs = {}

        # Add basic inputs
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids

        # Add KV cache inputs if expected
        for layer_idx in range(num_layers):
            key_name = f"past_key_values.{layer_idx}.key"
            value_name = f"past_key_values.{layer_idx}.value"

            # Check if these inputs are expected by the model
            expected_inputs = [input_info.name for input_info in model_inputs_info]
            if key_name in expected_inputs:
                model_inputs[key_name] = past_key_values[key_name]
            if value_name in expected_inputs:
                model_inputs[value_name] = past_key_values[value_name]

        try:
            outputs = session.run(None, model_inputs)
            print(f"ONNX model returned {len(outputs)} outputs")

            # The first output should be logits
            logits = outputs[0]
            print(f"Logits shape: {logits.shape}")

            # Update KV cache from outputs if available
            if len(outputs) > 1:
                print(f"Updating KV cache with {len(outputs) - 1} additional outputs")
                new_key_values = outputs[1:]  # Rest are present key-value pairs
                past_key_values = update_kv_cache(past_key_values, new_key_values, num_layers, num_kv_heads)

        except Exception as e:
            print(f"Error during initial inference: {e}")
            print("Falling back to transformers library")
            return generate_text_with_transformers(tokenizer, prompt, max_length, temperature, do_sample)

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

            # Prepare model inputs for next token
            model_inputs = {}
            model_inputs["input_ids"] = next_input_ids
            model_inputs["attention_mask"] = next_attention_mask
            model_inputs["position_ids"] = next_position_ids

            # Add updated KV cache
            for layer_idx in range(num_layers):
                key_name = f"past_key_values.{layer_idx}.key"
                value_name = f"past_key_values.{layer_idx}.value"

                expected_inputs = [input_info.name for input_info in model_inputs_info]
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
                    past_key_values = update_kv_cache(past_key_values, new_key_values, num_layers, num_kv_heads)

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

    else:
        # Transformers inference
        return generate_text_with_transformers(tokenizer, prompt, max_length, temperature, do_sample)

def generate_text_with_transformers(tokenizer, prompt, max_length=50, temperature=0.7, do_sample=True):
    """Generate text using transformers library as fallback."""
    print(f"\nProcessing with Transformers library")

    try:
        from transformers import AutoModelForCausalLM
        import torch

        # Load model if not already loaded
        model = AutoModelForCausalLM.from_pretrained(str(Path(__file__).parent / "gemma-3-1b-it-ONNX-GQA"))
        model.eval()

        # Generate using transformers
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

            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

    except Exception as e:
        print(f"Error during transformers generation: {e}")
        return prompt

def main():
    """Main function demonstrating Gemma 3 1B inference"""
    print("="*70)
    print("GEMMA 3 1B ONNX INFERENCE")
    print("="*70)

    try:
        # Load model and tokenizer
        session, tokenizer, config = load_gemma_model_and_tokenizer()

        if session is None and tokenizer is None:
            print("Could not load model session or tokenizer. Exiting.")
            return


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
        print("- gemma-3-1b-it-ONNX-GQA/tokenizer.model")
        print("- gemma-3-1b-it-ONNX-GQA/config.json")
        print("- gemma-3-1b-it-ONNX-GQA/onnx/model_int8.onnx (or other ONNX model)")
        print("\nRequired dependencies:")
        print("pip install onnxruntime tokenizers numpy transformers torch")

if __name__ == "__main__":
    main()
