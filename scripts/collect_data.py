#!/usr/bin/env python3
"""
AAPA Data Collection Script

Collects attention data from agent tasks for pattern analysis.
"""

import os
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attention_recorder import AttentionRecorder
from src.data_loader import get_dataloader

def main():
    parser = argparse.ArgumentParser(description="Collect attention data from agent tasks")
    parser.add_argument("--dataset", type=str, default="swe-bench", 
                        choices=["swe-bench", "bfcl-v4"],
                        help="Dataset name")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--output-dir", type=str, default="data/attention_logs", help="Output directory")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K attention scores")
    parser.add_argument("--data-path", type=str, default="data/raw", help="Raw data path")
    args = parser.parse_args()
    
    # Initialize recorder
    print(f"Initializing AttentionRecorder with {args.model_name}...")
    recorder = AttentionRecorder(
        model_name=args.model_name,
        top_k=args.top_k
    )
    
    # Load tasks
    print(f"Loading {args.num_tasks} tasks from {args.dataset}...")
    data_path = Path(args.data_path)
    
    if args.dataset == "swe-bench":
        data_path = data_path / "SWE-bench"
    elif args.dataset == "bfcl-v4":
        data_path = data_path / "BFCL-v4"
    
    loader = get_dataloader(args.dataset, str(data_path))
    tasks = loader.load(num_tasks=args.num_tasks)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process tasks
    print(f"Processing {len(tasks)} tasks...")
    for task_idx, task in enumerate(tasks):
        print(f"Task {task_idx+1}/{len(tasks)}: {task['task_id']}")
        
        for round_idx, round_data in enumerate(task.get('rounds', [])):
            try:
                record = recorder.record(
                    prompt=round_data['prompt'],
                    response=round_data['response'],
                    tool_calls=round_data.get('tool_calls', []),
                    task_id=task['task_id'],
                    round_id=round_idx
                )
                
                # Save to file
                output_file = output_dir / f"{task['task_id']}_round{round_idx}.json"
                recorder.save_to_json(record, str(output_file))
                print(f"  ✓ Saved to {output_file}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
    
    print(f"\n✅ Data collection complete!")
    print(f"Saved to: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.json')))}")

if __name__ == "__main__":
    main()

