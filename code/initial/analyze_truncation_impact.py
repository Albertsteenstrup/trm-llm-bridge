#!/usr/bin/env python3
"""
analyze_truncation_impact.py

Evaluates bridge model accuracy separately on messy examples that fit within 512 tokens
versus those that were truncated, to assess the impact of sequence length limits.

Usage:
    python code/initial/analyze_truncation_impact.py \
        --checkpoint checkpoints/initial/translator_bridge/best_model.pt \
        --dataset data/initial/sudoku_synthetic/rule/train_translator/sudoku_nl_messy_10000_v1.json \
        --output logs/initial/truncation_impact_analysis.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


def load_dataset(path: str) -> List[Dict]:
    """Load Sudoku NL dataset."""
    with open(path) as f:
        return json.load(f)


def categorize_by_length(
    dataset: List[Dict],
    tokenizer,
    max_seq_len: int = 512
) -> Tuple[List[Dict], List[Dict]]:
    """
    Categorize examples into those that fit within max_seq_len and those that don't.
    
    Returns:
        (within_limit, exceeds_limit)
    """
    within_limit = []
    exceeds_limit = []
    
    for item in tqdm(dataset, desc="Categorizing by length"):
        nl_text = item.get('nl_description', '')
        tokens = tokenizer.encode(nl_text)
        token_len = len(tokens)
        
        item_with_len = {**item, 'token_length': token_len}
        
        if token_len <= max_seq_len:
            within_limit.append(item_with_len)
        else:
            exceeds_limit.append(item_with_len)
    
    return within_limit, exceeds_limit


def grid_to_string(grid: List[List[int]]) -> str:
    """Convert 9x9 grid to 81-char string."""
    chars = []
    for row in grid:
        for val in row:
            chars.append('.' if val == 0 else str(val))
    return ''.join(chars)


def evaluate_accuracy(
    examples: List[Dict],
    model,
    tokenizer,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate bridge model accuracy on a set of examples.
    
    Returns dict with:
        - exact_match: fraction of puzzles with all 81 cells correct
        - cell_accuracy: fraction of individual cells correct
        - given_accuracy: fraction of given cells correct
        - empty_accuracy: fraction of empty cells correct
    """
    if len(examples) == 0:
        return {
            'exact_match': 0.0,
            'cell_accuracy': 0.0,
            'given_accuracy': 0.0,
            'empty_accuracy': 0.0,
            'num_examples': 0
        }
    
    model.eval()
    exact_matches = 0
    total_cells = 0
    correct_cells = 0
    total_given = 0
    correct_given = 0
    total_empty = 0
    correct_empty = 0
    
    with torch.no_grad():
        for item in tqdm(examples, desc="Evaluating"):
            nl_text = item.get('nl_description', '')
            target_grid = item.get('grid', [])
            target_str = grid_to_string(target_grid)
            
            # Tokenize
            inputs = tokenizer(
                nl_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits  # (1, 81, 10)
            predictions = logits.argmax(dim=-1).squeeze(0)  # (81,)
            
            # Convert to string
            pred_str = ''.join([str(p.item()) if p.item() > 0 else '.' for p in predictions])
            
            # Exact match
            if pred_str == target_str:
                exact_matches += 1
            
            # Cell-level accuracy
            for i, (pred_char, target_char) in enumerate(zip(pred_str, target_str)):
                total_cells += 1
                if pred_char == target_char:
                    correct_cells += 1
                
                # Given vs empty
                if target_char != '.':
                    total_given += 1
                    if pred_char == target_char:
                        correct_given += 1
                else:
                    total_empty += 1
                    if pred_char == target_char:
                        correct_empty += 1
    
    return {
        'exact_match': exact_matches / len(examples),
        'cell_accuracy': correct_cells / total_cells,
        'given_accuracy': correct_given / total_given if total_given > 0 else 0.0,
        'empty_accuracy': correct_empty / total_empty if total_empty > 0 else 0.0,
        'num_examples': len(examples)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze truncation impact on bridge accuracy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained bridge checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to messy dataset JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save analysis results')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length used during training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    # Ensure output path is absolute or relative to current directory
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    
    print(f"Output will be saved to: {output_path}")
    
    print(f"Loading tokenizer...")
    # Import transformers here to avoid initialization issues
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")
    
    print(f"Categorizing by sequence length (max={args.max_seq_len})...")
    within_limit, exceeds_limit = categorize_by_length(dataset, tokenizer, args.max_seq_len)
    print(f"Within limit: {len(within_limit)} ({len(within_limit)/len(dataset)*100:.2f}%)")
    print(f"Exceeds limit: {len(exceeds_limit)} ({len(exceeds_limit)/len(dataset)*100:.2f}%)")
    
    print(f"\nLoading model from {args.checkpoint}...")
    # Import model classes - need to add path first
    import sys
    sys.path.insert(0, 'code/initial/integration')
    from train_translator_bridge import BridgeConfig, QFormerBridge
    
    # Load checkpoint first to get config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model with config from checkpoint or default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use default config with correct hidden size
        config = BridgeConfig()
        config.llm_hidden_size = 2048  # Qwen-1.7B hidden size
    
    model = QFormerBridge(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
    
    model.to(args.device)
    model.eval()
    
    print(f"\nEvaluating on examples within {args.max_seq_len} tokens...")
    within_results = evaluate_accuracy(within_limit, model, tokenizer, args.device)
    
    print(f"\nEvaluating on examples exceeding {args.max_seq_len} tokens...")
    exceeds_results = evaluate_accuracy(exceeds_limit, model, tokenizer, args.device)
    
    # Compute statistics
    results = {
        'max_seq_len': args.max_seq_len,
        'dataset': args.dataset,
        'checkpoint': args.checkpoint,
        'total_examples': len(dataset),
        'within_limit': {
            'count': len(within_limit),
            'percentage': len(within_limit) / len(dataset) * 100,
            'metrics': within_results
        },
        'exceeds_limit': {
            'count': len(exceeds_limit),
            'percentage': len(exceeds_limit) / len(dataset) * 100,
            'metrics': exceeds_results
        },
        'accuracy_gap': {
            'exact_match': within_results['exact_match'] - exceeds_results['exact_match'],
            'cell_accuracy': within_results['cell_accuracy'] - exceeds_results['cell_accuracy'],
            'given_accuracy': within_results['given_accuracy'] - exceeds_results['given_accuracy'],
            'empty_accuracy': within_results['empty_accuracy'] - exceeds_results['empty_accuracy']
        }
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nWithin {args.max_seq_len} tokens ({len(within_limit)} examples):")
    print(f"  Exact match:     {within_results['exact_match']*100:.2f}%")
    print(f"  Cell accuracy:   {within_results['cell_accuracy']*100:.2f}%")
    print(f"  Given accuracy:  {within_results['given_accuracy']*100:.2f}%")
    print(f"  Empty accuracy:  {within_results['empty_accuracy']*100:.2f}%")
    
    print(f"\nExceeds {args.max_seq_len} tokens ({len(exceeds_limit)} examples):")
    print(f"  Exact match:     {exceeds_results['exact_match']*100:.2f}%")
    print(f"  Cell accuracy:   {exceeds_results['cell_accuracy']*100:.2f}%")
    print(f"  Given accuracy:  {exceeds_results['given_accuracy']*100:.2f}%")
    print(f"  Empty accuracy:  {exceeds_results['empty_accuracy']*100:.2f}%")
    
    print(f"\nAccuracy gap (within - exceeds):")
    print(f"  Exact match:     {results['accuracy_gap']['exact_match']*100:+.2f}%")
    print(f"  Cell accuracy:   {results['accuracy_gap']['cell_accuracy']*100:+.2f}%")
    print(f"  Given accuracy:  {results['accuracy_gap']['given_accuracy']*100:+.2f}%")
    print(f"  Empty accuracy:  {results['accuracy_gap']['empty_accuracy']*100:+.2f}%")
    
    print(f"\nResults saved to {args.output}")
    
    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    gap = results['accuracy_gap']['exact_match']
    if abs(gap) < 0.05:
        print("The accuracy gap is small (<5%), suggesting truncation has minimal impact.")
        print("The model learns robust representations from partial information.")
    elif gap > 0.05:
        print(f"Truncated examples perform {gap*100:.1f}% worse, indicating sequence length is a bottleneck.")
        print("Consider increasing max_seq_len to 1024 for better coverage.")
    else:
        print(f"Truncated examples perform {-gap*100:.1f}% better (unexpected).")
        print("This may indicate overfitting to shorter examples or noise in longer ones.")


if __name__ == '__main__':
    main()
