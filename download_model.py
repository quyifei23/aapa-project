#!/usr/bin/env python3
"""
Download Llama-2-7b-chat-hf model using HuggingFace access token
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model details
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# Use autodl-tmp for model storage
CACHE_DIR = "/root/autodl-tmp/model"
# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def download_model(token):
    """
    Download the model using the provided HuggingFace access token
    
    Args:
        token: HuggingFace access token
    """
    logger.info(f"Starting download of model: {MODEL_NAME}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=token,
            cache_dir=CACHE_DIR
        )
        logger.info("Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model... (this may take a while)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=token,
            cache_dir=CACHE_DIR,
            torch_dtype="auto",
            device_map="auto"
        )
        logger.info("Model downloaded successfully")
        
        # Verify the model
        logger.info("Verifying model...")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model test output: {test_output}")
        
        logger.info("✅ Model download and verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def main():
    """
    Main function to handle user input and download process
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Llama-2-7b-chat-hf model using HuggingFace access token')
    parser.add_argument('--token', type=str, help='HuggingFace access token')
    args = parser.parse_args()
    
    print("🚀 Llama-2-7b-chat-hf Model Downloader")
    print("=" * 50)
    print("This script will help you download the meta-llama/Llama-2-7b-chat-hf model")
    print("using your HuggingFace access token.")
    print()
    
    # Get HuggingFace access token
    if args.token:
        token = args.token
        print("✅ Using token from command line argument")
    else:
        # Get HuggingFace access token from user
        token = input("Enter your HuggingFace access token: ").strip()
    
    if not token:
        print("❌ Error: Access token cannot be empty")
        sys.exit(1)
    
    print()
    print("📥 Starting download...")
    print("Note: This may take several minutes depending on your internet speed")
    print("=" * 50)
    
    success = download_model(token)
    
    if success:
        print("\n🎉 Download completed successfully!")
        print(f"Model is stored in: {CACHE_DIR}")
    else:
        print("\n❌ Download failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
