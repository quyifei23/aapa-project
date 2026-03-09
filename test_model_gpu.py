#!/usr/bin/env python3
"""
Test script to verify if Llama-2-7b-chat-hf model can run on GPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model details
MODEL_PATH = "/root/autodl-tmp/model/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"

def test_model_gpu():
    """
    Test if the model can run on GPU
    """
    logger.info(f"Testing model at: {MODEL_PATH}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        logger.info("⚠️  GPU is not available, using CPU instead")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        logger.info(f"Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        logger.info("Model loaded successfully")
        
        # Test inference
        logger.info("Testing model inference...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model response: {response}")
        
        # Check GPU memory usage
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory used: {memory_used:.2f} GB")
            logger.info(f"GPU memory cached: {memory_cached:.2f} GB")
        
        logger.info("✅ Model test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return False

def main():
    """
    Main function
    """
    print("🚀 Llama-2-7b-chat-hf GPU Test")
    print("=" * 50)
    print(f"Testing model at: {MODEL_PATH}")
    print()
    
    success = test_model_gpu()
    
    if success:
        print("\n🎉 Test passed! Model can run on GPU.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
