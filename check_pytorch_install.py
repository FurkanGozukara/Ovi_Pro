"""
Deep diagnostic check of PyTorch installation
This will definitively show if PyTorch/MKL is broken
"""

import os
import sys

# Set env vars first
cpu_count = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_NUM_THREADS'] = str(cpu_count)

import torch
import numpy as np
import time

print("=" * 80)
print("DEEP PYTORCH INSTALLATION CHECK")
print("=" * 80)

print("\n1. PYTORCH BUILD INFO:")
print("-" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch file location: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (build): {torch.version.cuda if torch.version.cuda else 'N/A'}")

print("\n2. PYTORCH BUILD CONFIGURATION:")
print("-" * 80)
try:
    import torch.__config__
    config = torch.__config__.show()
    print(config)
except:
    print("Unable to show config")

print("\n3. MKL LIBRARY CHECK:")
print("-" * 80)
# Check if MKL is actually linked
try:
    import torch.backends.mkl as mkl
    print(f"MKL available: {mkl.is_available()}")
    if mkl.is_available():
        print(f"MKL version: {mkl.get_mkl_version()}")
except Exception as e:
    print(f"MKL check failed: {e}")

print("\n4. BLAS/LAPACK LIBRARIES:")
print("-" * 80)
# Check which BLAS is being used
try:
    print("Checking numpy BLAS config (shares libraries with PyTorch):")
    np.show_config()
except Exception as e:
    print(f"Cannot show numpy config: {e}")

print("\n5. THREAD SETTINGS:")
print("-" * 80)
print(f"Environment OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"Environment MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
print(f"PyTorch get_num_threads(): {torch.get_num_threads()}")
print(f"PyTorch get_num_interop_threads(): {torch.get_num_interop_threads()}")

# Try to set threads
torch.set_num_threads(cpu_count)
print(f"After torch.set_num_threads({cpu_count}): {torch.get_num_threads()}")

print("\n6. SIMPLE MATMUL TEST (Pure PyTorch):")
print("-" * 80)
size = 1024
a = torch.randn(size, size, dtype=torch.float32)
b = torch.randn(size, size, dtype=torch.float32)

# Warm up
for _ in range(3):
    _ = torch.mm(a, b)

# Time it
start = time.perf_counter()
for _ in range(10):
    c = torch.mm(a, b)
elapsed = time.perf_counter() - start

ops_per_sec = 10 / elapsed
print(f"1024x1024 float32 matmul: {ops_per_sec:.2f} ops/sec ({elapsed*100:.1f}ms per op)")
print(f"Expected: 20-50 ops/sec on working system")
print(f"Status: {'✓ OK' if ops_per_sec > 15 else '✗ BROKEN' if ops_per_sec < 5 else '⚠ SLOW'}")

print("\n7. SIMPLE MATMUL TEST (bfloat16 - used in LoRA):")
print("-" * 80)
a_bf16 = torch.randn(size, size, dtype=torch.bfloat16)
b_bf16 = torch.randn(size, size, dtype=torch.bfloat16)

# Warm up
for _ in range(3):
    _ = torch.mm(a_bf16, b_bf16)

# Time it
start = time.perf_counter()
for _ in range(10):
    c_bf16 = torch.mm(a_bf16, b_bf16)
elapsed_bf16 = time.perf_counter() - start

ops_per_sec_bf16 = 10 / elapsed_bf16
print(f"1024x1024 bfloat16 matmul: {ops_per_sec_bf16:.2f} ops/sec ({elapsed_bf16*100:.1f}ms per op)")
print(f"Expected: 15-40 ops/sec on working system")
print(f"Status: {'✓ OK' if ops_per_sec_bf16 > 10 else '✗ BROKEN' if ops_per_sec_bf16 < 3 else '⚠ SLOW'}")

print("\n8. NUMPY MATMUL TEST (Cross-check):")
print("-" * 80)
a_np = np.random.randn(size, size).astype(np.float32)
b_np = np.random.randn(size, size).astype(np.float32)

# Warm up
for _ in range(3):
    _ = np.dot(a_np, b_np)

# Time it
start = time.perf_counter()
for _ in range(10):
    c_np = np.dot(a_np, b_np)
elapsed_np = time.perf_counter() - start

ops_per_sec_np = 10 / elapsed_np
print(f"1024x1024 numpy.dot: {ops_per_sec_np:.2f} ops/sec ({elapsed_np*100:.1f}ms per op)")
print(f"Expected: 15-50 ops/sec on working system")
print(f"Status: {'✓ OK' if ops_per_sec_np > 10 else '✗ BROKEN' if ops_per_sec_np < 3 else '⚠ SLOW'}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

issues = []
if ops_per_sec_bf16 < 3:
    issues.append("PyTorch bfloat16 matmul is CRITICALLY slow")
if ops_per_sec < 5:
    issues.append("PyTorch float32 matmul is CRITICALLY slow")
if ops_per_sec_np < 3:
    issues.append("NumPy matmul is also slow - system-wide issue")
if torch.get_num_threads() < cpu_count * 0.5:
    issues.append(f"PyTorch using only {torch.get_num_threads()} threads instead of {cpu_count}")

if not issues:
    print("✓ PyTorch installation appears to be working correctly!")
    print("\nIf LoRA merging is still slow, the issue is elsewhere in the code.")
else:
    print("✗ PROBLEMS DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\nRECOMMENDED ACTIONS:")
    if ops_per_sec_np < 3:
        print("  → System-wide performance issue (not just PyTorch)")
        print("    Check: CPU thermal throttling, power settings, antivirus")
    else:
        print("  → PyTorch-specific issue")
        print("    1. Reinstall PyTorch:")
        print("       pip uninstall torch torchvision torchaudio -y")
        print("       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("    2. Check for conflicting installations (conda vs pip)")
        print("    3. Try a different PyTorch version")

print("\n" + "=" * 80)
print("Press Enter to exit...")
input()

