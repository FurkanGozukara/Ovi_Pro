"""
Standalone Performance Diagnostic Test for LoRA Merging
Tests CPU, RAM, and PyTorch performance to identify bottlenecks

Run this on both systems and compare results.
"""

import os
import sys

# CRITICAL: Set MKL/OpenMP threads BEFORE importing torch
# This must happen before any numpy/torch imports!
cpu_count = os.cpu_count()
if cpu_count:
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
    print(f"[INIT] Set thread environment variables to {cpu_count} BEFORE importing torch")

import torch
import time
import psutil
import platform

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_system_info():
    """Display system information"""
    print_section("SYSTEM INFORMATION")
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU info
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    print(f"\nCPU cores: {cpu_count_physical} physical, {cpu_count_logical} logical")
    
    # RAM info
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.total / (1024**3):.1f} GB total, {ram.available / (1024**3):.1f} GB available")
    
    # Environment variables check
    print("\nThread environment variables:")
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")
    
    # PyTorch threads
    print(f"\nPyTorch intra-op threads: {torch.get_num_threads()}")
    print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    
    # BLAS backend detection
    try:
        import torch.__config__ as torch_config
        config_str = torch_config.show()
        if "MKL" in config_str:
            blas_lib = "Intel MKL (optimized)"
        elif "OpenBLAS" in config_str:
            blas_lib = "OpenBLAS (optimized)"
        elif "BLAS" in config_str:
            blas_lib = "Generic BLAS (may be slow)"
        else:
            blas_lib = "None detected (VERY SLOW!)"
        print(f"BLAS backend: {blas_lib}")
    except Exception as e:
        print(f"BLAS backend: Unable to detect (error: {type(e).__name__})")

def test_torch_mm_performance():
    """Benchmark torch.mm() performance"""
    print_section("TORCH.MM() PERFORMANCE TEST")
    
    # Set threads to use all logical cores
    cpu_count = psutil.cpu_count(logical=True)
    torch.set_num_threads(cpu_count)
    print(f"Set PyTorch threads to: {cpu_count}")
    
    # Monitor CPU usage during test
    print("\n⚠ IMPORTANT: Watch Task Manager / Resource Monitor during this test!")
    print("   CPU usage should spike to 80-100% during matrix multiplication.")
    print("   If it stays at 10-20%, threads are NOT being used!\n")
    time.sleep(2)  # Give user time to open task manager
    
    # Test different matrix sizes
    test_configs = [
        (512, "Small matrices (512x512)"),
        (1024, "Medium matrices (1024x1024)"),
        (2048, "Large matrices (2048x2048)"),
        (4096, "Very large matrices (4096x4096)"),
    ]
    
    results = []
    
    for size, description in test_configs:
        print(f"\n{description}:")
        
        # Create test tensors
        a = torch.randn(size, size, dtype=torch.bfloat16)
        b = torch.randn(size, size, dtype=torch.bfloat16)
        
        # Warm-up (important for MKL)
        for _ in range(3):
            _ = torch.mm(a, b)
        
        # Benchmark
        num_iters = 20 if size <= 2048 else 10
        start = time.perf_counter()
        for _ in range(num_iters):
            result = torch.mm(a, b)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iters / elapsed
        ms_per_op = (elapsed * 1000) / num_iters
        
        print(f"  Operations/sec: {ops_per_sec:.2f}")
        print(f"  Time per op: {ms_per_op:.1f} ms")
        
        results.append((size, ops_per_sec, ms_per_op))
        
        del a, b, result
    
    # Performance assessment
    print("\n" + "-" * 80)
    print("PERFORMANCE ASSESSMENT:")
    _, ops_2048, _ = results[2]  # Use 2048x2048 as reference
    
    if ops_2048 >= 15:
        status = "✓ EXCELLENT - CPU performance is good"
    elif ops_2048 >= 8:
        status = "⚠ MODERATE - CPU performance is acceptable but not optimal"
    elif ops_2048 >= 3:
        status = "⚠ SLOW - CPU performance is degraded"
    else:
        status = "✗ CRITICAL - CPU performance is severely degraded!"
    
    print(f"{status}")
    print(f"Reference (2048x2048): {ops_2048:.2f} ops/sec")
    print("\nExpected values:")
    print("  High-end CPU (Ryzen 9, i9):     25-40 ops/sec")
    print("  Mid-range CPU (Ryzen 5, i5):    15-25 ops/sec")
    print("  Low-end CPU / thermal issues:   5-15 ops/sec")
    print("  Severe problems:                <5 ops/sec")

def test_tensor_operations():
    """Test various tensor operations that are used in LoRA merging"""
    print_section("TENSOR OPERATIONS TEST")
    
    # Simulate LoRA-like operations
    print("Testing LoRA-like operations (similar to actual merge)...")
    
    # Typical LoRA dimensions
    lora_down = torch.randn(32, 1280, dtype=torch.bfloat16)  # rank=32, features=1280
    lora_up = torch.randn(1280, 32, dtype=torch.bfloat16)
    param = torch.randn(1280, 1280, dtype=torch.bfloat16)
    
    # Warm-up
    for _ in range(3):
        delta = torch.mm(lora_up, lora_down)
        _ = param + delta
    
    # Benchmark 300 layers (typical for video model)
    num_layers = 300
    print(f"Simulating {num_layers} layer merges...")
    
    start = time.perf_counter()
    for i in range(num_layers):
        # Simulate what happens in calculate_lora_weight
        lora_weight = torch.mm(lora_up, lora_down)
        
        # Simulate adding to parameter
        updated_param = param + lora_weight
        
        # Simulate in-place update
        param = updated_param
    
    elapsed = time.perf_counter() - start
    
    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Time per layer: {elapsed*1000/num_layers:.1f} ms")
    print(f"Layers per second: {num_layers/elapsed:.1f}")
    
    print("\nExpected performance:")
    print("  Fast system:    < 5 seconds  (>60 layers/sec)")
    print("  Normal system:  5-20 seconds  (15-60 layers/sec)")
    print("  Slow system:    20-60 seconds  (5-15 layers/sec)")
    print("  Very slow:      > 60 seconds  (<5 layers/sec)")
    
    if elapsed < 5:
        status = "✓ EXCELLENT"
    elif elapsed < 20:
        status = "✓ GOOD"
    elif elapsed < 60:
        status = "⚠ ACCEPTABLE"
    else:
        status = "✗ POOR"
    
    print(f"\nStatus: {status}")

def test_memory_contiguity():
    """Test impact of non-contiguous tensors"""
    print_section("MEMORY CONTIGUITY TEST")
    
    size = 2048
    print(f"Testing {size}x{size} matrices...")
    
    # Test 1: Contiguous tensors (normal)
    a_cont = torch.randn(size, size, dtype=torch.bfloat16)
    b_cont = torch.randn(size, size, dtype=torch.bfloat16)
    
    # Warm-up
    _ = torch.mm(a_cont, b_cont)
    
    # Benchmark contiguous
    start = time.perf_counter()
    for _ in range(10):
        _ = torch.mm(a_cont, b_cont)
    time_cont = time.perf_counter() - start
    
    # Test 2: Non-contiguous tensors (transpose creates non-contiguous view)
    a_noncont = torch.randn(size, size, dtype=torch.bfloat16).t()
    b_noncont = torch.randn(size, size, dtype=torch.bfloat16).t()
    
    print(f"a_noncont.is_contiguous(): {a_noncont.is_contiguous()}")
    print(f"b_noncont.is_contiguous(): {b_noncont.is_contiguous()}")
    
    # Warm-up
    _ = torch.mm(a_noncont, b_noncont)
    
    # Benchmark non-contiguous
    start = time.perf_counter()
    for _ in range(10):
        _ = torch.mm(a_noncont, b_noncont)
    time_noncont = time.perf_counter() - start
    
    # Test 3: Non-contiguous made contiguous
    a_made_cont = a_noncont.contiguous()
    b_made_cont = b_noncont.contiguous()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        _ = torch.mm(a_made_cont, b_made_cont)
    time_made_cont = time.perf_counter() - start
    
    print(f"\nResults (10 iterations):")
    print(f"  Contiguous:                {time_cont*1000:.1f} ms  (baseline)")
    print(f"  Non-contiguous:            {time_noncont*1000:.1f} ms  ({time_noncont/time_cont:.1f}x slower)")
    print(f"  Non-contiguous->contiguous:{time_made_cont*1000:.1f} ms  ({time_made_cont/time_cont:.1f}x slower)")
    
    if time_noncont / time_cont > 2.0:
        print("\n⚠ WARNING: Non-contiguous tensors are significantly slower on this system!")
    else:
        print("\n✓ Non-contiguous tensor overhead is minimal")

def test_cpu_frequency():
    """Check CPU frequency and thermal throttling"""
    print_section("CPU FREQUENCY CHECK")
    
    try:
        freq = psutil.cpu_freq()
        print(f"Current CPU frequency: {freq.current:.0f} MHz")
        print(f"Min frequency: {freq.min:.0f} MHz")
        print(f"Max frequency: {freq.max:.0f} MHz")
        
        if freq.current < freq.max * 0.7:
            print("\n⚠ WARNING: CPU is running significantly below max frequency!")
            print("  Possible causes:")
            print("  - Thermal throttling (check cooling)")
            print("  - Power saving mode enabled")
            print("  - BIOS settings limiting frequency")
        else:
            print("\n✓ CPU frequency looks normal")
    except Exception as e:
        print(f"Unable to read CPU frequency: {e}")
    
    # CPU usage
    print(f"\nCurrent CPU usage: {psutil.cpu_percent(interval=1)}%")

def main():
    print("=" * 80)
    print("  LoRA PERFORMANCE DIAGNOSTIC TEST")
    print("  Run this on both fast and slow systems to compare")
    print("=" * 80)
    
    try:
        test_system_info()
        test_cpu_frequency()
        test_torch_mm_performance()
        test_tensor_operations()
        test_memory_contiguity()
        
        print("\n" + "=" * 80)
        print("  TEST COMPLETE")
        print("=" * 80)
        print("\nPlease share the complete output from both systems for comparison.")
        print("Pay special attention to:")
        print("  1. BLAS backend")
        print("  2. torch.mm() operations/sec (2048x2048 test)")
        print("  3. Total time for 300 layer simulation")
        print("  4. CPU frequency")
        
    except Exception as e:
        print(f"\n\n✗ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()

