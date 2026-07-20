#!/usr/bin/env python3
"""
Generate few-shot prompt templates from extracted dataset.
Run: python generate_few_shot_prompts.py --input ./datasets/sparx_all.jsonl --output ./prompts/
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def load_extractions(jsonl_path: Path) -> List[Dict]:
    """Load JSONL file into list of examples."""
    examples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def group_by_tier(examples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group examples by tier."""
    grouped = defaultdict(list)
    for ex in examples:
        tier = ex.get("metadata", {}).get("tier", "UNKNOWN")
        grouped[tier].append(ex)
    return grouped


def create_few_shot_prompt(
    cipher: str,
    variant: str,
    tier_examples: Dict[str, List[Dict]],
    target_tier: str,
    num_shots: int = 3
) -> str:
    """Create a few-shot prompt for a specific tier."""
    
    # System message
    prompt = f"""You are an expert in translating Python cryptographic implementations to Isabelle/HOL formal specifications.

Task: Translate {cipher}-{variant} {target_tier} components from Python to Isabelle/HOL.

Rules:
1. Preserve exact cryptographic semantics
2. Use Isabelle word library operations (word_rotl, word_rotr, push_bit, drop_bit)
3. Convert Python loops to recursive functions
4. Maintain 16-bit word boundaries
5. Output ONLY the Isabelle/HOL definition, no explanations

Examples:
"""
    
    # Collect examples from lower tiers (easier first)
    tier_order = ["T1", "T2", "T3", "T4"]
    examples_to_show = []
    
    for tier in tier_order:
        if tier == target_tier:
            break
        if tier in tier_examples and tier_examples[tier]:
            # Take 1-2 examples from each lower tier
            examples_to_show.extend(tier_examples[tier][:2])
    
    # Add examples from target tier
    if target_tier in tier_examples:
        examples_to_show.extend(tier_examples[target_tier][:num_shots])
    
    # Format examples
    for i, ex in enumerate(examples_to_show, 1):
        prompt += f"\n--- Example {i} ---\n"
        prompt += f"Python:\n{ex['input']}\n\n"
        prompt += f"Isabelle/HOL:\n{ex['output']}\n"
    
    # Add instruction for the actual request
    prompt += f"\n--- Now translate this component ---\n"
    prompt += "Python:\n{input_python}\n\n"
    prompt += "Isabelle/HOL:\n"
    
    return prompt


def create_chain_of_thought_prompt(
    cipher: str,
    variant: str,
    example: Dict
) -> str:
    """Create a Chain-of-Thought prompt with reasoning steps."""
    
    return f"""Translate this {cipher}-{variant} component from Python to Isabelle/HOL.

First, analyze the Python code:
1. What is the component type? {example.get('metadata', {}).get('component_type', 'unknown')}
2. Does it contain loops? {example.get('metadata', {}).get('semantics', {}).get('contains_loop', False)}
3. What bitwise operations are used? {example.get('metadata', {}).get('semantics', {}).get('operators', [])}

Step-by-step translation:
1. Identify the function signature and types
2. Convert each operation to Isabelle equivalent
3. Transform loops to recursion if needed
4. Add type annotations and bit-width constraints

Python code:
{example['input']}

Isabelle/HOL translation:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL file from extraction")
    parser.add_argument("--output-dir", default="./prompts", help="Output directory for prompts")
    parser.add_argument("--cipher", default="sparx", help="Cipher name")
    parser.add_argument("--variant", default="64_128", help="Variant")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    examples = load_extractions(input_path)
    grouped = group_by_tier(examples)
    
    print(f"Loaded {len(examples)} examples")
    for tier, exs in grouped.items():
        print(f"  {tier}: {len(exs)} examples")
    
    # Generate few-shot prompts for each tier
    for target_tier in ["T1", "T2", "T3", "T4"]:
        if target_tier not in grouped:
            continue
        
        prompt = create_few_shot_prompt(
            cipher=args.cipher,
            variant=args.variant,
            tier_examples=grouped,
            target_tier=target_tier,
            num_shots=3
        )
        
        output_file = output_dir / f"{args.cipher}_{args.variant}_{target_tier}_few_shot.txt"
        output_file.write_text(prompt)
        print(f"Created: {output_file}")
    
    # Generate Chain-of-Thought prompts for a few examples
    cot_dir = output_dir / "cot"
    cot_dir.mkdir(exist_ok=True)
    
    for tier, exs in grouped.items():
        for i, ex in enumerate(exs[:3]):  # 3 examples per tier
            prompt = create_chain_of_thought_prompt(args.cipher, args.variant, ex)
            output_file = cot_dir / f"{args.cipher}_{args.variant}_{tier}_cot_{i+1}.txt"
            output_file.write_text(prompt)
    
    print(f"\nChain-of-Thought prompts saved to: {cot_dir}")
    
    # Also create a JSON version for programmatic use
    prompts_json = {
        "cipher": args.cipher,
        "variant": args.variant,
        "total_examples": len(examples),
        "few_shot_prompts": {},
        "zero_shot_template": f"Translate this {args.cipher}-{args.variant} component from Python to Isabelle/HOL.\n\nPython:\n{{input_python}}\n\nIsabelle/HOL:\n",
    }
    
    for target_tier in ["T1", "T2", "T3", "T4"]:
        if target_tier in grouped:
            prompts_json["few_shot_prompts"][target_tier] = {
                "template": create_few_shot_prompt(
                    args.cipher, args.variant, grouped, target_tier, 3
                ),
                "examples": [ex for ex in grouped[target_tier][:5]]  # Store raw examples
            }
    
    json_output = output_dir / f"{args.cipher}_{args.variant}_prompts.json"
    json_output.write_text(json.dumps(prompts_json, indent=2))
    print(f"JSON prompts saved to: {json_output}")


if __name__ == "__main__":
    main()