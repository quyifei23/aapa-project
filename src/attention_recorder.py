"""
Attention Recorder for AAPA Framework

Records top-K attention scores during LLM inference for agentic workloads.
"""

import torch
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenMetadata:
    """Metadata for a single token"""
    idx: int
    text: str
    position_norm: float
    is_prompt_start: bool
    is_prompt_end: bool
    token_type: str
    is_tool_argument: bool = False
    is_tool_name: bool = False


@dataclass
class AttentionRecord:
    """Complete attention record for a single turn"""
    task_id: str
    round_id: int
    prompt: str
    response: str
    tool_calls: List[Dict]
    input_length: int
    output_length: int
    n_layers: int
    top_k_scores: List[Dict]
    token_metadata: List[Dict]


class AttentionRecorder:
    """
    Records attention scores during LLM inference.
    
    Features:
    - Captures top-K attention scores per token
    - Extracts token metadata (position, type, etc.)
    - Supports batch processing
    - Saves to JSON format
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        top_k: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize the attention recorder.
        
        Args:
            model_name: HuggingFace model name
            top_k: Number of top attention scores to record per token
            device: Device to run model on (default: auto)
        """
        self.model_name = model_name
        self.top_k = top_k
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully")
        
        # Store attention scores during forward pass
        self.attention_scores = []
        self._hooks = []
    
    def register_hooks(self):
        """Register hooks to all attention layers"""
        self._hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if len(output) > 1 and output[1] is not None:
                    attn = output[1][-1]  # Last layer attention
                    attn_avg = attn.mean(dim=1).detach().cpu().numpy()
                    self.attention_scores.append(attn_avg)
            return hook
        
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)
        
        logger.info(f"Registered {len(self._hooks)} attention hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def record(
        self,
        prompt: str,
        response: str,
        tool_calls: List[Dict],
        task_id: str,
        round_id: int
    ) -> AttentionRecord:
        """
        Record attention for a single turn.
        
        Args:
            prompt: Input prompt
            response: Generated response
            tool_calls: List of tool calls in the response
            task_id: Task identifier
            round_id: Round number within task
        
        Returns:
            AttentionRecord with all attention data
        """
        self.attention_scores = []
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate with attention recording
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                return_dict_in_generate=True,
                output_attentions=True
            )
        
        output_length = outputs.sequences.shape[1] - input_length
        
        # Process attention scores
        attention_data = self._process_attention_scores(
            self.attention_scores,
            input_length
        )
        
        # Extract token metadata
        token_metadata = self._extract_token_metadata(prompt, inputs, tool_calls)
        
        record = AttentionRecord(
            task_id=task_id,
            round_id=round_id,
            prompt=prompt,
            response=response,
            tool_calls=tool_calls,
            input_length=input_length,
            output_length=output_length,
            n_layers=len(self.attention_scores),
            top_k_scores=attention_data,
            token_metadata=[asdict(tm) for tm in token_metadata]
        )
        
        return record
    
    def _process_attention_scores(
        self,
        scores: List[np.ndarray],
        input_length: int
    ) -> List[Dict]:
        """
        Process and aggregate attention scores.
        
        Args:
            scores: List of attention arrays (one per layer)
            input_length: Length of input sequence
        
        Returns:
            List of top-K scores for each token
        """
        if not scores:
            return []
        
        # Stack all layers
        all_scores = np.stack(scores, axis=0)  # (n_layers, seq_len, seq_len)
        
        # Average across layers
        avg_scores = all_scores.mean(axis=0)  # (seq_len, seq_len)
        
        # Get top-K for each token
        top_k_scores = []
        for i in range(min(input_length, avg_scores.shape[0])):
            token_scores = avg_scores[i, :input_length]
            top_indices = np.argsort(token_scores)[-self.top_k:][::-1]
            top_k_scores.append({
                "token_idx": i,
                "top_k": [(int(idx), float(token_scores[idx])) for idx in top_indices]
            })
        
        return top_k_scores
    
    def _extract_token_metadata(
        self,
        prompt: str,
        inputs,
        tool_calls: List[Dict]
    ) -> List[TokenMetadata]:
        """
        Extract metadata for each token.
        
        Args:
            prompt: Input prompt
            inputs: Tokenized inputs
            tool_calls: List of tool calls
        
        Returns:
            List of TokenMetadata objects
        """
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        metadata = []
        
        # Extract tool argument positions
        tool_arg_positions = set()
        tool_name_positions = set()
        for tc in tool_calls:
            if 'arguments' in tc:
                # Find positions of argument tokens
                arg_text = str(tc['arguments'])
                for i, tok in enumerate(tokens):
                    if arg_text and tok in arg_text:
                        tool_arg_positions.add(i)
            if 'tool' in tc or 'name' in tc:
                tool_name = tc.get('tool', tc.get('name', ''))
                for i, tok in enumerate(tokens):
                    if tool_name and tok in tool_name:
                        tool_name_positions.add(i)
        
        n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            metadata.append(TokenMetadata(
                idx=i,
                text=token,
                position_norm=i / n_tokens,
                is_prompt_start=i < n_tokens * 0.1,
                is_prompt_end=i > n_tokens * 0.9,
                token_type=self._classify_token_type(token),
                is_tool_argument=i in tool_arg_positions,
                is_tool_name=i in tool_name_positions
            ))
        
        return metadata
    
    def _classify_token_type(self, token: str) -> str:
        """Classify token type"""
        if token.startswith('Ġ') or (len(token) > 0 and token[0].isupper()):
            return 'keyword'
        elif token in ['(', ')', '[', ']', '{', '}', ',', '.', ':', ';', '?', '!']:
            return 'punctuation'
        elif token.isdigit() or token.replace('.', '').isdigit():
            return 'number'
        elif token.startswith('/') or token.startswith('\\') or '/' in token:
            return 'path'
        elif token.startswith('"') or token.endswith('"'):
            return 'string'
        else:
            return 'other'
    
    def save_to_json(self, record: AttentionRecord, filepath: str):
        """Save record to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(record), f, indent=2)
        
        logger.info(f"Saved attention record to {filepath}")
    
    def batch_record(
        self,
        tasks: List[Dict],
        output_dir: str
    ):
        """
        Record attention for multiple tasks.
        
        Args:
            tasks: List of task dictionaries with keys:
                   - task_id
                   - rounds: list of {prompt, response, tool_calls}
            output_dir: Directory to save attention records
        """
        logger.info(f"Processing {len(tasks)} tasks...")
        
        for task_idx, task in enumerate(tasks):
            task_id = task['task_id']
            logger.info(f"Task {task_idx+1}/{len(tasks)}: {task_id}")
            
            for round_idx, round_data in enumerate(task.get('rounds', [])):
                record = self.record(
                    prompt=round_data['prompt'],
                    response=round_data['response'],
                    tool_calls=round_data.get('tool_calls', []),
                    task_id=task_id,
                    round_id=round_idx
                )
                
                filepath = os.path.join(
                    output_dir,
                    f"{task_id}_round{round_idx}.json"
                )
                self.save_to_json(record, filepath)
        
        logger.info(f"Batch processing complete. Saved to {output_dir}")
