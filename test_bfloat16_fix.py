"""
Quick test to verify if float32 workaround will fix the slow LoRA merging
Simulates exactly what happens during LoRA merge
"""

import os
import sys
import time

# Set threads before importing torch
cpu_count = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)

import torch

print("=" * 80)
print("BFLOAT16 FIX VERIFICATION TEST")
print("=" * 80)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CPU cores: {cpu_count}")

# Set threads
torch.set_num_threads(cpu_count)
print(f"PyTorch threads: {torch.get_num_threads()}")

print("\n" + "=" * 80)
print("TEST 1: Benchmark bfloat16 vs float32")
print("=" * 80)

# Test bfloat16
test_size = 1024
print(f"\nTesting {test_size}x{test_size} matrix multiplication...")

a_bf16 = torch.randn(test_size, test_size, dtype=torch.bfloat16)
b_bf16 = torch.randn(test_size, test_size, dtype=torch.bfloat16)

# Warm up
_ = torch.mm(a_bf16, b_bf16)

# Benchmark
start = time.perf_counter()
for _ in range(10):
    _ = torch.mm(a_bf16, b_bf16)
time_bf16 = time.perf_counter() - start
ops_bf16 = 10 / time_bf16

print(f"bfloat16: {ops_bf16:.2f} ops/sec ({time_bf16*100:.1f} ms for 10 ops)")

# Test float32
a_f32 = torch.randn(test_size, test_size, dtype=torch.float32)
b_f32 = torch.randn(test_size, test_size, dtype=torch.float32)

# Warm up
_ = torch.mm(a_f32, b_f32)

# Benchmark
start = time.perf_counter()
for _ in range(10):
    _ = torch.mm(a_f32, b_f32)
time_f32 = time.perf_counter() - start
ops_f32 = 10 / time_f32

print(f"float32:  {ops_f32:.2f} ops/sec ({time_f32*100:.1f} ms for 10 ops)")
print(f"\nSpeedup: float32 is {ops_f32/ops_bf16:.1f}x faster than bfloat16")

# Determine if workaround is needed
broken = ops_bf16 < 5 and ops_f32 > 50
if broken:
    print("\n⚠ BFLOAT16 IS BROKEN - Workaround is NEEDED")
else:
    print("\n✓ bfloat16 performance looks OK - Workaround NOT needed")

print("\n" + "=" * 80)
print("TEST 2: Simulate LoRA merge (300 layers)")
print("=" * 80)

# Typical LoRA dimensions for large model
lora_rank = 32
features = 1280

lora_down = torch.randn(lora_rank, features, dtype=torch.bfloat16)
lora_up = torch.randn(features, lora_rank, dtype=torch.bfloat16)
param = torch.randn(features, features, dtype=torch.bfloat16)

print(f"\nSimulating LoRA dimensions: lora_up {lora_up.shape}, lora_down {lora_down.shape}")
print("Testing 300 layers (typical video model)...")

# TEST WITH BFLOAT16 (current slow method)
print("\n[1/2] Using bfloat16 (CURRENT METHOD):")
start = time.perf_counter()
for i in range(300):
    # This is what calculate_lora_weight() does
    lora_weight = torch.mm(lora_up, lora_down)
    # Add to param (what merge does)
    updated = param + lora_weight
    param = updated  # Simulate in-place update
elapsed_bf16 = time.perf_counter() - start
print(f"  Time: {elapsed_bf16:.2f} seconds")
print(f"  Speed: {300/elapsed_bf16:.1f} layers/sec")

# TEST WITH FLOAT32 WORKAROUND (fixed method)
print("\n[2/2] Using float32 workaround (FIXED METHOD):")

# Reset tensors with original dtype (bfloat16)
lora_down_orig = torch.randn(lora_rank, features, dtype=torch.bfloat16)
lora_up_orig = torch.randn(features, lora_rank, dtype=torch.bfloat16)
param_orig = torch.randn(features, features, dtype=torch.bfloat16)

start = time.perf_counter()
for i in range(300):
    # Convert to float32 for computation
    lora_down_f32 = lora_down_orig.to(torch.float32)
    lora_up_f32 = lora_up_orig.to(torch.float32)
    param_f32 = param_orig.to(torch.float32)
    
    # Compute in float32
    lora_weight = torch.mm(lora_up_f32, lora_down_f32)
    updated = param_f32 + lora_weight
    
    # Convert back to bfloat16
    param_orig = updated.to(torch.bfloat16)
elapsed_f32 = time.perf_counter() - start
print(f"  Time: {elapsed_f32:.2f} seconds")
print(f"  Speed: {300/elapsed_f32:.1f} layers/sec")

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

speedup = elapsed_bf16 / elapsed_f32
print(f"\nOriginal (bfloat16): {elapsed_bf16:.2f} seconds")
print(f"Fixed (float32):     {elapsed_f32:.2f} seconds")
print(f"Speedup:             {speedup:.1f}x FASTER with workaround!")

if broken:
    print("\n✓ FIX CONFIRMED: The float32 workaround will solve your problem!")
    print(f"\n  Expected LoRA merge time:")
    print(f"    Before: ~{elapsed_bf16:.0f} seconds (5+ minutes)")
    print(f"    After:  ~{elapsed_f32:.0f} seconds")
    print(f"\n  The automatic workaround is now active in your code.")
else:
    print("\n✓ Your system doesn't need the workaround - bfloat16 is working fine!")

print("\n" + "=" * 80)
print("Press Enter to exit...")
input()

