# AAPA: Agent Attention Pattern Analysis

🔍 **Understanding KV Cache Importance in Agentic LLM Inference**

## Overview

AAPA is a framework for analyzing attention patterns in agentic LLM workloads to enable efficient KV cache management through sparse loading and multi-tier storage placement.

## Key Features

- **Attention Recording**: Capture top-K attention scores during inference
- **Feature Engineering**: Extract token-level and task-level features
- **Pattern Mining**: Discover attention patterns across tool types and agent roles
- **Mapping Generation**: Create TAAP+ABDS prediction tables
- **Validation**: Comprehensive ablation studies and benchmarks

## Installation

```bash
# Create conda environment
conda create -n aapa python=3.10 -y
conda activate aapa

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install jupyter jupyterlab

# Install in development mode
pip install -e .
```

## Project Structure

```
aapa-project/
├── src/                    # Source code
│   ├── attention_recorder.py
│   ├── feature_extractor.py
│   ├── pattern_analyzer.py
│   └── utils.py
├── data/
│   ├── raw/               # Raw datasets (SWE-Bench, BFCL, AgentBench)
│   ├── attention_logs/    # Recorded attention data
│   └── processed/         # Processed features
├── notebooks/             # Jupyter notebooks for analysis
├── experiments/           # Configuration files
├── results/               # Experimental results
└── scripts/               # Shell scripts for automation
```

## Quick Start

### 1. Data Collection

```bash
bash scripts/collect_data.sh
```

### 2. Feature Engineering

```bash
python src/feature_extractor.py --input-dir data/attention_logs --output-dir data/processed
```

### 3. Pattern Mining

```bash
python src/pattern_analyzer.py --input-dir data/processed --output-dir results
```

### 4. Generate Mapping Tables

```bash
python src/generate_mapping.py --input-dir results --output-dir results/mappings
```

### 5. Validation

```bash
python src/validation.py --input-dir results --output-dir results/validation
```

## Datasets

- **SWE-Bench**: Code agent tasks with file operations
- **BFCL v4**: Function calling benchmarks
- **AgentBench**: General agent scenarios

## Methods

### TAAP (Tool-Aware Attention Prediction)
Predict important tokens based on tool call arguments.

### ABDS (Agent-Behavior-Driven Sparsity)
Predict important tokens based on agent behavior patterns.

### IATP (Importance-Aware Tiered Placement)
Distribute KV cache across HBM/CPU/SSD based on predicted importance.

## Citation

```bibtex
@article{aapa2026,
  title={AAPA: Agent Attention Pattern Analysis for Efficient KV Cache Management},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License
