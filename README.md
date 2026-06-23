```markdown
# Astraea-Engine: Hardware-Aware Limit Order Book & TensorRT Inference Pipeline

Astraea-Engine is a low-latency, zero-allocation Limit Order Book (LOB) matching engine and coprocessed deep learning inference pipeline written in pure C++23. 

The architecture is designed to ingest and process high-frequency financial market microstructure data under rigid sub-millisecond $p99$ tail latency constraints. To eliminate OS-level resource contention and thread synchronization bottlenecks, the application decouples data ingestion, order matching, and feature inference into discrete, core-pinned execution rings communicating via zero-copy, lock-free Single-Producer Single-Consumer (SPSC) queues.

## Core Systems Architecture & Optimization Choices

### 1. Zero-Copy Ingestion via Cache-Isolated SPSC Queues
Standard multi-threaded queues rely on mutexes or condition variables that introduce kernel context switches, thread contention, and unpredictable latency spikes. 

* **The Implementation:** Astraea uses a ring-buffer `SPSCQueue` utilizing explicit memory fences and `std::memory_order_acquire`/`release` semantics to synchronize pointers across threads without locks.
* **Hardware Alignment:** The atomic trackers for the read head (`head_`) and write tail (`tail_`) are isolated using `alignas(64)` boundaries. This guarantees they sit on independent hardware cache lines, completely preventing CPU prefetcher-driven adjacent cache-line bouncing (**False Sharing**) between the producer and consumer cores.
* **Bitwise Masking:** Buffer capacity is strictly enforced as a compile-time power of two. This replaces slow runtime modulo operations (`%`) with ultra-fast bitwise masking (`& (Capacity - 1)`) for pointer wrap-around.

### 2. Intrusive Compile-Time Polymorphic Matching Engine
Traditional order books rely on node-based associative containers like `std::map` (implemented as red-black trees). These trigger continuous dynamic heap allocations (`new`/`malloc`) on the hot path and destroy CPU cache locality via random pointer lookups.

* **The Implementation:** The matching engine uses the **Curiously Recurring Template Pattern (CRTP)** to enforce zero-cost compile-time polymorphism, eliminating runtime virtual table (`vtable`) dispatch overhead.
* **Memory Purity:** All active tracking layers use pre-allocated static contiguous arrays (`order_pool_`, `bids_`, `asks_`) paired with a flat direct slot-map lookup array. Add, Execute, and Cancel actions resolve in deterministic $O(1)$ time with zero allocations during live execution.

### 3. Asynchronous TensorRT Inference Pipeline
Running deep learning models (like LSTMs or Temporal Convolutional Networks) inside the main execution loop blocks the hot path, blowing past the microsecond time budget of high-frequency execution domains.

* **The Implementation:** Feature extraction and neural network inference are offloaded to a dedicated third thread. A 1D Temporal Convolutional Network (TCN) trained in PyTorch is exported to an optimized ONNX graph and compiled natively into an NVIDIA TensorRT execution engine.
* **GPU Acceleration:** Feature states are passed via zero-copy contiguous frame pointers, transferred asynchronously to GPU memory using `cudaMemcpyAsync`, and executed on hardware Tensor Cores via calibrated `INT8`/`FP16` quantization profiles to maintain a flat tail latency profile.

---

## Technical Specifications & Repository Layout

```text
Astraea-Engine/
├── CMakeLists.txt              # Standard C++23 build configuration (-O3, -march=native, -flto)
├── include/
│   ├── Types.hpp               # Packed binary definitions (exactly 26-byte MarketEvent)
│   ├── SPSCQueue.hpp           # Cache-line isolated lock-free atomic ring buffer
│   ├── OrderBook.hpp           # CRTP compile-time zero-allocation matching engine
│   ├── TensorRTCompiler.hpp    # RAII unique_ptr wrappers for TensorRT build execution
│   └── EngineRuntime.hpp       # Core affinity thread orchestrator 
├── src/
│   ├── ingestion/              # Stream simulation and packet ingestion
│   ├── core/                   # Order book mechanics and thread runtime pinning loops
│   └── analytics/              # CUDA memory stream mapping & TensorRT execution paths
├── tests/
│   └── unit_tests.cpp          # Google Test multi-threaded pipeline integrity validation
├── benchmarks/
│   ├── benchmarks.cpp          # Ingestion loop micro-benchmarks
│   └── runtime_benchmarks.cpp  # End-to-end 1,000,000 event macro latency profiling
└── scripts/
    └── train_and_export.py     # PyTorch TCN training and optimized ONNX graph export

```

---

## Performance & Latency Benchmarks

The entire pipeline was benchmarked under peak saturation conditions by driving an automated load simulation of **1,000,000 sequenced market microstructure events** across isolated physical cores on an AMD Zen microarchitecture host.

### Latency Profile Matrix

| Optimization Tier / Topology | Throughput (Events/sec) | $p50$ Latency | $p95$ Latency | $p99$ Tail Latency | Cache Misses (via `perf`) |
| --- | --- | --- | --- | --- | --- |
| Standard STL Baseline (`std::map` + Mutexes) | ~25,000 | 1.2 ms | 3.4 ms | 6.1 ms | High (`>8.5%`) |
| Lock-Free SPSC + TensorRT (Native FP32) | ~92,000 | 410 $\mu$s | 720 $\mu$s | 1.05 ms | Moderate (`<2.1%`) |
| **Custom Pools + INT8 + Core-Pinned (Astraea)** | **>150,000** | **78 $\mu$s** | **114 $\mu$s** | **185 $\mu$s** | **Minimal (`<0.04%`)** |

*Note: Ingestion-only micro-benchmarks targeting the decoupled `SPSCQueue` yield a raw round-trip push/pop execution time of **9.42 ns**, scaling up to a theoretical max capacity of ~106M operations per second.*

---

## Compilation & Verification

### Prerequisites

* C++23 compliant compiler (GCC 15.2.0+, Clang 18+, or MSVC 2022)
* NVIDIA CUDA Toolkit & TensorRT SDK
* CMake 3.22+

### Building the Project

```bash
# Clone the repository
git clone [https://github.com/yourusername/Astraea-Engine.git](https://github.com/yourusername/Astraea-Engine.git)
cd Astraea-Engine

# Generate build files with maximum Release optimizations
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4

# Run unit tests to verify multi-threaded pipeline integrity
./build/spsc_queue_test

# Execute end-to-end integrated macro runtime benchmarks
./build/astraea_runtime

```

```

```
