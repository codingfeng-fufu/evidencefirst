#!/usr/bin/env python3
"""
Parallel Dual-Path evaluation for 1000 questions
Splits work across multiple processes to speed up evaluation
"""

import json
import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import argparse

def run_batch(args):
    """Run evaluation on a batch of questions"""
    batch_id, start_idx, batch_size, data_file, kg_file, output_dir = args

    cache_tag = f"dual_path_batch_{batch_id}"

    cmd = [
        'python', 'evaluate.py',
        '--mode', 'full',
        '--variant', 'evidence_aug',
        '--data', data_file,
        '--kg', kg_file,
        '--start', str(start_idx),
        '--n', str(batch_size),
        '--out-dir', output_dir,
        '--cache-tag', cache_tag,
        '--save-pipeline-details'
    ]

    print(f"[Batch {batch_id}] Starting questions {start_idx}-{start_idx+batch_size-1}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per batch
        )

        if result.returncode == 0:
            print(f"[Batch {batch_id}] ✓ Completed questions {start_idx}-{start_idx+batch_size-1}")
            return (batch_id, True, None)
        else:
            print(f"[Batch {batch_id}] ✗ Failed: {result.stderr[:200]}")
            return (batch_id, False, result.stderr)

    except subprocess.TimeoutExpired:
        print(f"[Batch {batch_id}] ✗ Timeout after 2 hours")
        return (batch_id, False, "Timeout")
    except Exception as e:
        print(f"[Batch {batch_id}] ✗ Error: {str(e)}")
        return (batch_id, False, str(e))

def merge_caches(output_dir, n_batches, final_tag):
    """Merge all batch caches into one final cache"""
    print("\n" + "="*60)
    print("Merging batch results...")
    print("="*60)

    merged = {}

    for batch_id in range(n_batches):
        cache_file = Path(output_dir) / f"cache_dual_path_batch_{batch_id}_full.json"

        if cache_file.exists():
            print(f"  Loading batch {batch_id}: {cache_file.name}")
            with open(cache_file, 'r') as f:
                batch_data = json.load(f)
                merged.update(batch_data)
        else:
            print(f"  ⚠️  Batch {batch_id} cache not found: {cache_file.name}")

    # Save merged cache
    final_cache = Path(output_dir) / f"cache_{final_tag}_full.json"
    with open(final_cache, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n✓ Merged {len(merged)} questions into {final_cache.name}")
    return len(merged)

def main():
    parser = argparse.ArgumentParser(description='Run parallel Dual-Path evaluation')
    parser.add_argument('--n-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--data', default='data/hotpotqa_1000_test.json', help='Dataset file')
    parser.add_argument('--kg', default='comagraag/data/hotpotqa_1000_kgs_merged.pkl', help='KG file')
    parser.add_argument('--out-dir', default='results', help='Output directory')
    parser.add_argument('--final-tag', default='dual_path_1000_parallel', help='Final cache tag')
    parser.add_argument('--n-questions', type=int, default=1000, help='Total questions')

    args = parser.parse_args()

    print("="*60)
    print("PARALLEL DUAL-PATH EVALUATION")
    print("="*60)
    print(f"Workers: {args.n_workers}")
    print(f"Questions: {args.n_questions}")
    print(f"Dataset: {args.data}")
    print(f"KG: {args.kg}")
    print(f"Output: {args.out_dir}")
    print("="*60)

    # Calculate batch splits
    batch_size = args.n_questions // args.n_workers
    batches = []

    for i in range(args.n_workers):
        start_idx = i * batch_size
        if i == args.n_workers - 1:
            current_batch_size = args.n_questions - start_idx  # Last batch gets remainder
        else:
            current_batch_size = batch_size

        batches.append((i, start_idx, current_batch_size, args.data, args.kg, args.out_dir))

    print(f"\nBatch distribution:")
    for batch_id, start, size, _, _, _ in batches:
        print(f"  Batch {batch_id}: questions {start}-{start+size-1} ({size} questions)")

    print("\n" + "="*60)
    print("Starting parallel execution...")
    print("="*60 + "\n")

    start_time = time.time()

    # Run batches in parallel
    with mp.Pool(processes=args.n_workers) as pool:
        results = pool.map(run_batch, batches)

    elapsed = time.time() - start_time

    # Check results
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)

    successful = sum(1 for _, success, _ in results if success)
    failed = sum(1 for _, success, _ in results if not success)

    print(f"Successful batches: {successful}/{args.n_workers}")
    print(f"Failed batches: {failed}/{args.n_workers}")
    print(f"Total time: {elapsed/60:.1f} minutes")

    if failed > 0:
        print("\nFailed batches:")
        for batch_id, success, error in results:
            if not success:
                print(f"  Batch {batch_id}: {error}")

    # Merge caches
    if successful > 0:
        n_merged = merge_caches(args.out_dir, args.n_workers, args.final_tag)

        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total questions processed: {n_merged}")
        print(f"Expected: {args.n_questions}")
        print(f"Coverage: {n_merged/args.n_questions*100:.1f}%")
        print(f"Average time per question: {elapsed/n_merged:.1f}s")

        if n_merged == args.n_questions:
            print("\n✅ ALL QUESTIONS COMPLETED!")
        else:
            print(f"\n⚠️  Missing {args.n_questions - n_merged} questions")
    else:
        print("\n❌ No successful batches to merge")

    print("="*60)

if __name__ == '__main__':
    main()
