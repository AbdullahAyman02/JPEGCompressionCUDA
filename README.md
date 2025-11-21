# JPEG Compression with CUDA

   

A GPU-accelerated image compression engine built from scratch using **C++** and **CUDA**. This project parallels the JPEG compression standard (DCT, Quantization, ZigZag, RLE) to achieve high-throughput image processing, outperforming CPU-based sequential implementations for high-resolution workloads.

-----

## 🚀 Key Features

  * **Massively Parallel Architecture**: Decomposes images into 8x8 blocks processed concurrently across thousands of CUDA threads.
  * **Memory Optimization**: Utilizes **Shared Memory** to reduce global memory latency and **Constant Memory** for high-speed access to quantization tables.
  * **Asynchronous Pipelining**: Implements **CUDA Streams** to overlap Host-to-Device (H2D) memory transfers with kernel execution, masking PCIe latency.
  * **Custom Binary Format**: Encodes compressed data into a compact custom binary format (`.gpu`), optimizing storage for RLE-encoded streams.
  * **Benchmarking Suite**: Includes comparison tools against industry standards (**libjpeg-turbo** and NVIDIA's **nvJPEG**).

## 🏗️ System Architecture

The pipeline follows a hybrid CPU-GPU model designed to maximize throughput:

1.  **Preprocessing (CPU):** Loads image via OpenCV and converts BGR → YCrCb color space.
2.  **Tiling & Padding:** Splits the image into 8x8 pixel blocks, padding edges to align with warp boundaries.
3.  **GPU Kernel Execution:**
      * **DCT Phase:** Computes Discrete Cosine Transform using matrix multiplication in **Shared Memory**.
      * **Quantization:** Applies luminance/chrominance scaling using **Constant Memory**.
      * **ZigZag Scan:** Reorders coefficients to cluster zeros, optimized for coalesced memory writes.
      * **Run-Length Encoding (RLE):** Compresses the zero-heavy stream (thread-local reduction).
4.  **Post-Processing (CPU):** Serializes the compressed bitstream to disk.

### Memory Hierarchy Strategy

To prevent memory bottlenecks, the kernels are designed with specific memory tiers:

  * **Registers**: Intermediate DCT calculations.
  * **Shared Memory (L1 Cache)**: Storing the 8x8 block during processing to avoid round-trips to VRAM.
  * **Constant Memory**: Storing static Lookup Tables (Quantization matrices, ZigZag order).
  * **Pinned Memory (Host)**: Used for input buffers to enable Direct Memory Access (DMA) for faster transfer speeds.

## 📊 Performance Benchmarks

*Benchmarks run on Kaggle*

| Resolution | Custom CPU Time | Custom GPU Time (Streams) | Speedup (Compute) |
| :--- | :--- | :--- | :--- |
| **1280x720** | 537ms | 150ms | **\~3.5x** |
| **3000x2000** | 3,277ms | 1,032ms | **\~3.1x** |
| **4050x3000** | 7,514ms | 2,212ms | **\~3.4x** |

> **Note:** While the raw compute time is significantly faster on the GPU, smaller images (\< 720p) may experience bottlenecking due to PCIe transfer overhead. The system shines with batch processing of high-resolution textures.

## 🛠️ Installation & Setup

### Prerequisites

  * **NVIDIA GPU** (Compute Capability 5.0+)
  * **CUDA Toolkit** (11.0 or higher)
  * **OpenCV** (4.x)
  * **CMake** (3.10+)
  * **C++ Compiler** (MSVC or GCC supporting C++17)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/AbdullahAyman02/JPEGCompressionCUDA.git
cd JPEGCompressionCUDA

# Create build directory
mkdir build && cd build

# Configure with CMake (Ensure OpenCV is in your PATH)
cmake ..

# Build
cmake --build . --config Release
```

## 💻 Usage

Run the executable with the input image path and desired quality factor (1-100):

```bash
./JPEGCompression "path/to/image.jpg" 75
```

**Output:**

  * Generates `compressed.gpu` (Serialized compressed data).
  * Generates `output.jpg` (Reconstructed image for verification).
  * Prints detailed timing breakdown (Split, H2D, Kernel, D2H, Reassembly).

## 📂 Project Structure

```
├── src
│   ├── main.cu             # Single-stream GPU implementation
│   ├── main_streams.cu     # Optimized Multi-stream GPU implementation
│   ├── main.cpp            # CPU Reference implementation
│   ├── images/             # Test assets
│   └── state of the art/   # Comparison benchmarks (nvJPEG, libjpeg)
├── CMakeLists.txt          # Build configuration
└── README.md
```

## 🧠 Technical Highlights (Backend Focus)

  * **Handling Divergence:** Minimized warp divergence in the RLE kernel by utilizing thread-local registers for run-counting before writing to global memory.
  * **Stream Overlap:** Implemented a "Breadth-First" processing approach using 8 concurrent CUDA streams. This ensures that while Stream $N$ is copying data back to the Host, Stream $N+1$ is already executing kernels on the Device, maximizing bus saturation.
  * **Data Packing:** Designed a custom struct-of-arrays (SoA) layout for the compressed blocks to improve cache locality during the file write phase.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
