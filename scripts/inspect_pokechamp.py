#!/usr/bin/env python3
"""Inspect the PokeChamp dataset format"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess
    subprocess.run(["pip", "install", "datasets"], check=True)
    from datasets import load_dataset

print("Loading PokeChamp dataset (streaming to avoid full download)...")
dataset = load_dataset("milkkarten/pokechamp", split="train", streaming=True)

# Get first few examples
print("\n" + "="*80)
print("SAMPLE RECORDS")
print("="*80)

for i, example in enumerate(dataset):
    if i >= 2:
        break
    print(f"\n--- Example {i} ---")
    print(f"Keys: {list(example.keys())}")
    for key, value in example.items():
        if key == "text":
            # Show first 2000 chars of battle log
            print(f"\n{key} (first 2000 chars):")
            print(value[:2000])
            print(f"\n... (total length: {len(value)} chars)")
        else:
            print(f"{key}: {value}")
