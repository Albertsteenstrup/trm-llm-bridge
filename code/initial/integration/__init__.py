"""
Integration module for connecting TRM Sudoku solver to Qwen3-1.7B.

This package will contain:
- text_to_grid.py: Convert natural language Sudoku descriptions to 9x9 grid format
- router.py: Tiny classifier that routes Sudoku inputs to TRM branch
- trm_bridge.py: Bridge between LLM hidden states and TRM input format
- pipeline.py: End-to-end LLM+TRM inference pipeline

These components will be developed after:
1. TRM is trained on Sudoku-Extreme (original procedure)
2. Qwen3-1.7B is loaded and verified on HPC
3. Synthetic NL dataset is generated via NVIDIA API
"""
