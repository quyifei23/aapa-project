#!/usr/bin/env python3
"""
Data Loaders for AAPA Framework

Load and preprocess agent task datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_from_disk, load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SWEbenchDataLoader:
    """Load and preprocess SWE-Bench tasks"""
    
    def __init__(self, data_path: str):
        """
        Initialize SWE-Bench data loader.
        
        Args:
            data_path: Path to SWE-Bench dataset directory
        """
        self.data_path = Path(data_path)
        self.tasks = []
    
    def load(self, num_tasks: Optional[int] = None) -> List[Dict]:
        """
        Load SWE-Bench tasks.
        
        Args:
            num_tasks: Number of tasks to load (None = all)
        
        Returns:
            List of task dictionaries
        """
        logger.info(f"Loading SWE-Bench tasks from {self.data_path}...")
        
        # Try to load from disk first
        if (self.data_path / "swe-bench-test").exists():
            dataset = load_from_disk(self.data_path / "swe-bench-test")
        elif self.data_path.exists():
            # Load from HuggingFace
            try:
                dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
            except Exception as e:
                logger.warning(f"Failed to load from HuggingFace: {e}")
                logger.info("Loading from local files...")
                dataset = self._load_from_files()
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Convert to our format
        self.tasks = []
        for i, item in enumerate(dataset):
            if num_tasks and len(self.tasks) >= num_tasks:
                break
            
            task = self._convert_item(item)
            if task:
                self.tasks.append(task)
        
        logger.info(f"Loaded {len(self.tasks)} tasks")
        return self.tasks
    
    def _load_from_files(self):
        """Load from local JSON files"""
        # TODO: Implement based on SWE-Bench file structure
        pass
    
    def _convert_item(self, item) -> Optional[Dict]:
        """Convert SWE-Bench item to our format"""
        try:
            # Extract tool calls from the task
            tool_calls = []
            
            # SWE-Bench typically has file operations
            if 'file_contents' in item:
                tool_calls.append({
                    "tool": "file_read",
                    "arguments": {
                        "path": item.get('problem_statement', '')[:100]  # Extract path if available
                    }
                })
            
            task = {
                "task_id": f"swe-bench-{item.get('instance_id', 'unknown')}",
                "problem_statement": item.get('problem_statement', ''),
                "rounds": [
                    {
                        "prompt": item.get('problem_statement', '')[:2000],  # Truncate if too long
                        "response": f"[Solution for {item.get('instance_id', 'unknown')}]",
                        "tool_calls": tool_calls
                    }
                ]
            }
            
            return task
        
        except Exception as e:
            logger.warning(f"Failed to convert item: {e}")
            return None


class BFCLDataLoader:
    """Load and preprocess BFCL v4 tasks"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.tasks = []
    
    def load(self, num_tasks: Optional[int] = None) -> List[Dict]:
        """Load BFCL v4 tasks"""
        logger.info(f"Loading BFCL v4 tasks from {self.data_path}...")
        
        # TODO: Implement BFCL v4 loader
        # For now, return mock tasks
        self.tasks = []
        for i in range(num_tasks or 10):
                       self.tasks.append({
                "task_id": f"bfcl-{i:03d}",
                "rounds": [
                    {
                        "prompt": f"Call function with argument {i}",
                        "response": f"Function result for {i}",
                        "tool_calls": [
                            {"tool": "web_search", "arguments": {"query": f"query_{i}"}}
                        ]
                    }
                ]
            })
        
        logger.info(f"Loaded {len(self.tasks)} tasks")
        return self.tasks


def get_dataloader(dataset_name: str, data_path: str):
    """Get data loader for dataset"""
    loaders = {
        "swe-bench": SWEbenchDataLoader,
        "bfcl-v4": BFCLDataLoader,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name](data_path)


