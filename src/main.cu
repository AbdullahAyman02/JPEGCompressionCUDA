%%writefile 1200488.cu
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cooperative_groups.h>
#include <chrono>
#include <sys/stat.h>
#include <fstream>

using namespace std;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

// GPU Constants
__constant__ int d_LuminanceQuantTable[8][8];
__constant__ int d_ChrominanceQuantTable[8][8];
__constant__ float d_dctMatrix[8][8];
__constant__ float d_idctMatrix[8][8];
__constant__ int d_zigzagOrder[64];

// Compression Structures
struct GPURLEBlock
{
    int y_size, cb_size, cr_size;
    int16_t *y_data;
    int16_t *cb_data;
    int16_t *cr_data;
};

struct GPUCompressedData
{
    int width, height;
    int num_blocks;
    GPURLEBlock *blocks;
};

// Timing utilities
class GPUTimer
{
    cudaEvent_t start, stop;

public:
    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void record(cudaEvent_t event) { cudaEventRecord(event); }
    float elapsed(cudaEvent_t start, cudaEvent_t stop)
    {
        float ms;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// Host Quantization Tables and DCT Matrices
static const int h_LumTable[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}};

static const int h_ChromTable[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}};

static const float h_dctMatrix[8][8] = {
    {0.35355338f, 0.35355338f, 0.35355338f, 0.35355338f, 0.35355338f, 0.35355338f, 0.35355338f, 0.35355338f},
    {0.49039264f, 0.41573481f, 0.27778512f, 0.09754516f, -0.09754516f, -0.27778512f, -0.41573481f, -0.49039264f},
    {0.46193975f, 0.19134172f, -0.19134172f, -0.46193975f, -0.46193975f, -0.19134172f, 0.19134172f, 0.46193975f},
    {0.41573481f, -0.09754516f, -0.49039264f, -0.27778512f, 0.27778512f, 0.49039264f, 0.09754516f, -0.41573481f},
    {0.35355338f, -0.35355338f, -0.35355338f, 0.35355338f, 0.35355338f, -0.35355338f, -0.35355338f, 0.35355338f},
    {0.27778512f, -0.49039264f, 0.09754516f, 0.41573481f, -0.41573481f, -0.09754516f, 0.49039264f, -0.27778512f},
    {0.19134172f, -0.46193975f, 0.46193975f, -0.19134172f, -0.19134172f, 0.46193975f, -0.46193975f, 0.19134172f},
    {0.09754516f, -0.27778512f, 0.41573481f, -0.49039264f, 0.49039264f, -0.41573481f, 0.27778512f, -0.09754516f}};

static const float h_idctMatrix[8][8] = {
    {0.35355338f, 0.49039264f, 0.46193975f, 0.41573481f, 0.35355338f, 0.27778512f, 0.19134172f, 0.09754516f},
    {0.35355338f, 0.41573481f, 0.19134172f, -0.09754516f, -0.35355338f, -0.49039264f, -0.46193975f, -0.27778512f},
    {0.35355338f, 0.27778512f, -0.19134172f, -0.49039264f, -0.35355338f, 0.09754516f, 0.46193975f, 0.41573481f},
    {0.35355338f, 0.09754516f, -0.46193975f, -0.27778512f, 0.35355338f, 0.41573481f, -0.19134172f, -0.49039264f},
    {0.35355338f, -0.09754516f, -0.46193975f, 0.27778512f, 0.35355338f, -0.41573481f, -0.19134172f, 0.49039264f},
    {0.35355338f, -0.27778512f, -0.19134172f, 0.49039264f, -0.35355338f, -0.09754516f, 0.46193975f, -0.41573481f},
    {0.35355338f, -0.41573481f, 0.19134172f, 0.09754516f, -0.35355338f, 0.49039264f, -0.46193975f, 0.27778512f},
    {0.35355338f, -0.49039264f, 0.46193975f, -0.41573481f, 0.35355338f, -0.27778512f, 0.19134172f, -0.09754516f}};

static const int zigzagOrder[64] = {
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63};

// GPU Kernels
__device__ float determineScale(int quality) {
    quality = max(1, min(quality, 100)); // Clamp quality to [1, 100]
    
    if (quality < 50) {
        return 50.0f / quality; // More aggressive quantization
    } else {
        return 2.1f - (quality * 2.0f / 100.0f); // Less quantization
    }
}

__device__ void zigzagScan(const int16_t *channel, int16_t *output)
{
    for (int i = 0; i < 64; i++)
    {
        int row = d_zigzagOrder[i] / 8;
        int col = d_zigzagOrder[i] % 8;
        output[i] = channel[row * 8 + col];
    }
}

__device__ void rleEncode(const int16_t *zigzag, int16_t *output, int &size)
{
    int16_t current = zigzag[0];
    int count = 1;
    size = 0;

    for (int i = 1; i < 64; i++)
    {
        if (zigzag[i] == current)
        {
            count++;
        }
        else
        {
            output[size++] = count;
            output[size++] = current;
            current = zigzag[i];
            count = 1;
        }
    }
    output[size++] = count;
    output[size++] = current;
}

__global__ void gpuCompressKernel(float *input, GPUCompressedData output, int quality)
{
    cg::thread_block tile = cg::this_thread_block();
    __shared__ float smem[8][8][3];
    __shared__ int16_t dct_coeffs[8][8];
    __shared__ int16_t zigzag[64];
    __shared__ int16_t rle_buffer[128];

    int block_idx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load block data
    for (int c = 0; c < 3; c++)
    {
        int idx = block_idx * 192 + ty * 24 + tx * 3 + c;   // 192 = 8 * 8 * 3, 24 = 8 * 3
        smem[ty][tx][c] = input[idx];
    }
    tile.sync();

    float scale = determineScale(quality);

    if (tile.thread_rank() == 0)
    {
        for (int channel = 0; channel < 3; channel++)
        {
            // DCT
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < 8; k++)
                    {
                        for (int l = 0; l < 8; l++)
                        {
                            sum += d_dctMatrix[i][k] * smem[k][l][channel] * d_idctMatrix[l][j];
                        }
                    }
                    dct_coeffs[i][j] = static_cast<int16_t>(round(sum));
                }
            }

            // Quantization
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    float quant_step = (channel == 0 ? d_LuminanceQuantTable[i][j] : d_ChrominanceQuantTable[i][j]) * scale;
                    if (quant_step < 1.0f) quant_step = 1.0f;
                    dct_coeffs[i][j] = static_cast<int16_t>(round(dct_coeffs[i][j] / quant_step));
                }
            }

            // Zigzag + RLE + Store
            zigzagScan(&dct_coeffs[0][0], zigzag);
            if (channel == 0)
            {
                rleEncode(zigzag, rle_buffer, output.blocks[block_idx].y_size);
                for (int i = 0; i < output.blocks[block_idx].y_size; i++)
                    output.blocks[block_idx].y_data[i] = rle_buffer[i];
            }
            else if (channel == 1)
            {
                rleEncode(zigzag, rle_buffer, output.blocks[block_idx].cb_size);
                for (int i = 0; i < output.blocks[block_idx].cb_size; i++)
                    output.blocks[block_idx].cb_data[i] = rle_buffer[i];
            }
            else // channel == 2
            {
                rleEncode(zigzag, rle_buffer, output.blocks[block_idx].cr_size);
                for (int i = 0; i < output.blocks[block_idx].cr_size; i++)
                    output.blocks[block_idx].cr_data[i] = rle_buffer[i];
            }
        }
    }

    tile.sync();
}

__global__ void gpuDecompressKernel(float *output, GPUCompressedData input, int quality)
{
    cg::thread_block tile = cg::this_thread_block();
    __shared__ float smem[8][8][3];
    __shared__ int16_t zigzag[64];
    __shared__ float dct_coeffs[8][8];

    int block_idx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (tile.thread_rank() == 0)
    {
        // Process Y channel
        int idx = 0;
        for (int i = 0; i < input.blocks[block_idx].y_size; i += 2)
        {
            int count = input.blocks[block_idx].y_data[i];
            int16_t value = input.blocks[block_idx].y_data[i + 1];
            while (count--)
                zigzag[idx++] = value;
        }
        for (int i = 0; i < 64; i++)
        {
            int row = d_zigzagOrder[i] / 8;
            int col = d_zigzagOrder[i] % 8;
            dct_coeffs[row][col] = zigzag[i];
        }
        float scale = determineScale(quality);
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                float quant_step = d_LuminanceQuantTable[i][j] * scale;
                dct_coeffs[i][j] *= quant_step;
            }
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++)
                {
                    for (int l = 0; l < 8; l++)
                    {
                        sum += d_idctMatrix[i][k] * dct_coeffs[k][l] * d_dctMatrix[l][j];
                    }
                }
                smem[i][j][0] = sum;
            }
        }

        // Process Cb channel
        idx = 0;
        for (int i = 0; i < input.blocks[block_idx].cb_size; i += 2)
        {
            int count = input.blocks[block_idx].cb_data[i];
            int16_t value = input.blocks[block_idx].cb_data[i + 1];
            while (count--)
                zigzag[idx++] = value;
        }
        for (int i = 0; i < 64; i++)
        {
            int row = d_zigzagOrder[i] / 8;
            int col = d_zigzagOrder[i] % 8;
            dct_coeffs[row][col] = zigzag[i];
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                float quant_step = d_ChrominanceQuantTable[i][j] * scale;
                dct_coeffs[i][j] *= quant_step;
            }
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++)
                {
                    for (int l = 0; l < 8; l++)
                    {
                        sum += d_idctMatrix[i][k] * dct_coeffs[k][l] * d_dctMatrix[l][j];
                    }
                }
                smem[i][j][1] = sum;
            }
        }

        // Process Cr channel
        idx = 0;
        for (int i = 0; i < input.blocks[block_idx].cr_size; i += 2)
        {
            int count = input.blocks[block_idx].cr_data[i];
            int16_t value = input.blocks[block_idx].cr_data[i + 1];
            while (count--)
                zigzag[idx++] = value;
        }
        for (int i = 0; i < 64; i++)
        {
            int row = d_zigzagOrder[i] / 8;
            int col = d_zigzagOrder[i] % 8;
            dct_coeffs[row][col] = zigzag[i];
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                float quant_step = d_ChrominanceQuantTable[i][j] * scale;
                dct_coeffs[i][j] *= quant_step;
            }
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < 8; k++)
                {
                    for (int l = 0; l < 8; l++)
                    {
                        sum += d_idctMatrix[i][k] * dct_coeffs[k][l] * d_dctMatrix[l][j];
                    }
                }
                smem[i][j][2] = sum;
            }
        }
    }
    tile.sync();

    for (int c = 0; c < 3; c++)
    {
        int idx = block_idx * 192 + ty * 24 + tx * 3 + c;
        output[idx] = smem[ty][tx][c];
    }
}

// Host functions
void initializeGPUConstants()
{
    cudaMemcpyToSymbol(d_LuminanceQuantTable, h_LumTable, sizeof(h_LumTable));
    cudaMemcpyToSymbol(d_ChrominanceQuantTable, h_ChromTable, sizeof(h_ChromTable));
    cudaMemcpyToSymbol(d_dctMatrix, h_dctMatrix, sizeof(h_dctMatrix));
    cudaMemcpyToSymbol(d_idctMatrix, h_idctMatrix, sizeof(h_idctMatrix));
    cudaMemcpyToSymbol(d_zigzagOrder, zigzagOrder, sizeof(zigzagOrder));
}

size_t getFileSize(const string &filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : 0;
}

void saveCompressedData(const GPUCompressedData &data, const string &filename)
{
    ofstream file(filename, ios::binary);
    int width = data.width;
    int height = data.height;
    int num_blocks = data.num_blocks;

    file.write(reinterpret_cast<const char *>(&width), sizeof(int));
    file.write(reinterpret_cast<const char *>(&height), sizeof(int));
    file.write(reinterpret_cast<const char *>(&num_blocks), sizeof(int));

    GPURLEBlock *h_blocks = new GPURLEBlock[num_blocks];
    cudaMemcpy(h_blocks, data.blocks, num_blocks * sizeof(GPURLEBlock), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_blocks; i++)
    {
        GPURLEBlock &block = h_blocks[i];
        int16_t *h_y_data = new int16_t[block.y_size];
        int16_t *h_cb_data = new int16_t[block.cb_size];
        int16_t *h_cr_data = new int16_t[block.cr_size];

        cudaMemcpy(h_y_data, block.y_data, block.y_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cb_data, block.cb_data, block.cb_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cr_data, block.cr_data, block.cr_size * sizeof(int16_t), cudaMemcpyDeviceToHost);

        file.write(reinterpret_cast<const char *>(&block.y_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(h_y_data), block.y_size * sizeof(int16_t));
        file.write(reinterpret_cast<const char *>(&block.cb_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(h_cb_data), block.cb_size * sizeof(int16_t));
        file.write(reinterpret_cast<const char *>(&block.cr_size), sizeof(int));
        file.write(reinterpret_cast<const char *>(h_cr_data), block.cr_size * sizeof(int16_t));

        delete[] h_y_data;
        delete[] h_cb_data;
        delete[] h_cr_data;
    }
    delete[] h_blocks;
}

GPUCompressedData loadCompressedData(const string &filename)
{
    ifstream file(filename, ios::binary);
    GPUCompressedData data;

    file.read(reinterpret_cast<char *>(&data.width), sizeof(int));
    file.read(reinterpret_cast<char *>(&data.height), sizeof(int));
    file.read(reinterpret_cast<char *>(&data.num_blocks), sizeof(int));

    cudaMalloc(&data.blocks, data.num_blocks * sizeof(GPURLEBlock));
    GPURLEBlock *h_blocks = new GPURLEBlock[data.num_blocks];

    for (int i = 0; i < data.num_blocks; i++)
    {
        GPURLEBlock block;
        file.read(reinterpret_cast<char *>(&block.y_size), sizeof(int));
        file.read(reinterpret_cast<char *>(&block.cb_size), sizeof(int));
        file.read(reinterpret_cast<char *>(&block.cr_size), sizeof(int));

        int16_t *h_y_data = new int16_t[block.y_size];
        int16_t *h_cb_data = new int16_t[block.cb_size];
        int16_t *h_cr_data = new int16_t[block.cr_size];

        file.read(reinterpret_cast<char *>(h_y_data), block.y_size * sizeof(int16_t));
        file.read(reinterpret_cast<char *>(h_cb_data), block.cb_size * sizeof(int16_t));
        file.read(reinterpret_cast<char *>(h_cr_data), block.cr_size * sizeof(int16_t));

        cudaMalloc(&block.y_data, block.y_size * sizeof(int16_t));
        cudaMalloc(&block.cb_data, block.cb_size * sizeof(int16_t));
        cudaMalloc(&block.cr_data, block.cr_size * sizeof(int16_t));

        cudaMemcpy(block.y_data, h_y_data, block.y_size * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(block.cb_data, h_cb_data, block.cb_size * sizeof(int16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(block.cr_data, h_cr_data, block.cr_size * sizeof(int16_t), cudaMemcpyHostToDevice);

        delete[] h_y_data;
        delete[] h_cb_data;
        delete[] h_cr_data;

        h_blocks[i] = block;
    }
    cudaMemcpy(data.blocks, h_blocks, data.num_blocks * sizeof(GPURLEBlock), cudaMemcpyHostToDevice);
    delete[] h_blocks;
    return data;
}

void splitIntoBlocks(const cv::Mat &image, vector<cv::Mat> &blocks)
{
    int blockSize = 8;
    int width = image.cols;
    int height = image.rows;

    for (int y = 0; y < height; y += blockSize)
    {
        for (int x = 0; x < width; x += blockSize)
        {
            cv::Rect roi(x, y, blockSize, blockSize);
            if (roi.x + roi.width > width)
                roi.width = width - roi.x;
            if (roi.y + roi.height > height)
                roi.height = height - roi.y;
            cv::Mat block = image(roi).clone();
            if (block.rows < blockSize || block.cols < blockSize)
            {
                cv::copyMakeBorder(block, block, 0, blockSize - block.rows, 0, blockSize - block.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
            blocks.push_back(block);
        }
    }
}

void assembleBlocks(cv::Mat &image, const vector<cv::Mat> &blocks, cv::Size imageSize)
{
    image.create(imageSize, CV_32FC3);
    int blockSize = 8;
    int currentBlock = 0;
    for (int y = 0; y < imageSize.height; y += blockSize)
    {
        for (int x = 0; x < imageSize.width; x += blockSize)
        {
            if (currentBlock >= blocks.size())
                break;
            cv::Rect roi(x, y, blockSize, blockSize);
            cv::Mat block = blocks[currentBlock++];
            if (roi.x + roi.width > imageSize.width)
                roi.width = imageSize.width - roi.x;
            if (roi.y + roi.height > imageSize.height)
                roi.height = imageSize.height - roi.y;
            block(cv::Rect(0, 0, roi.width, roi.height)).copyTo(image(roi));
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        cerr << "Usage: " << argv[0] << " <image_path> [quality]" << endl;
        return -1;
    }
    const char *IMAGEPATH = argv[1];
    const int quality = argc == 3 ? stoi(argv[2]) : 75;

    initializeGPUConstants();
    GPUTimer timer;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cv::Mat image = cv::imread(IMAGEPATH, cv::IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }
    cv::Mat ycbcr;
    cv::cvtColor(image, ycbcr, cv::COLOR_BGR2YCrCb);
    ycbcr.convertTo(ycbcr, CV_32FC3);

    vector<cv::Mat> blocks;
    splitIntoBlocks(ycbcr, blocks);
    int num_blocks = blocks.size();

    float *d_input, *d_output;
    cudaMalloc(&d_input, num_blocks * 8 * 8 * 3 * sizeof(float));
    cudaMalloc(&d_output, num_blocks * 8 * 8 * 3 * sizeof(float));

    for (int i = 0; i < num_blocks; i++)
    {
        cudaMemcpy(d_input + i * 192, blocks[i].data, 192 * sizeof(float), cudaMemcpyHostToDevice);
    }

    GPUCompressedData compressed;
    compressed.width = image.cols;
    compressed.height = image.rows;
    compressed.num_blocks = num_blocks;
    cudaMalloc(&compressed.blocks, num_blocks * sizeof(GPURLEBlock));

    GPURLEBlock *h_blocks = new GPURLEBlock[num_blocks];
    for (int i = 0; i < num_blocks; i++)
    {
        cudaMalloc(&h_blocks[i].y_data, 128 * sizeof(int16_t));
        cudaMalloc(&h_blocks[i].cb_data, 128 * sizeof(int16_t));
        cudaMalloc(&h_blocks[i].cr_data, 128 * sizeof(int16_t));
    }
    cudaMemcpy(compressed.blocks, h_blocks, num_blocks * sizeof(GPURLEBlock), cudaMemcpyHostToDevice);
    delete[] h_blocks;

    cudaEventRecord(start);
    gpuCompressKernel<<<num_blocks, dim3(8, 8)>>>(d_input, compressed, quality);
    cudaEventRecord(stop);
    float compress_time = timer.elapsed(start, stop);

    saveCompressedData(compressed, "compressed.gpu");

    cudaEventRecord(start);
    gpuDecompressKernel<<<num_blocks, dim3(8, 8)>>>(d_output, compressed, quality);
    cudaEventRecord(stop);
    float decompress_time = timer.elapsed(start, stop);

    vector<cv::Mat> reconstructed(num_blocks);
    for (int i = 0; i < num_blocks; i++)
    {
        reconstructed[i].create(8, 8, CV_32FC3);
        cudaMemcpy(reconstructed[i].data, d_output + i * 192, 192 * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cv::Mat final_image;
    assembleBlocks(final_image, reconstructed, image.size());
    final_image.convertTo(final_image, CV_8UC3);
    cv::cvtColor(final_image, final_image, cv::COLOR_YCrCb2BGR);
    cv::imwrite("output.jpg", final_image);

    // Calculate stats PROPERLY
    size_t original_raw_size = image.total() * image.elemSize(); // W*H*3 bytes
    size_t compressed_size = getFileSize("compressed.gpu");

    cout << "=== Performance Results ===" << endl;
    cout << "Compression time: " << compress_time << " ms" << endl;
    cout << "Decompression time: " << decompress_time << " ms" << endl;
    cout << "Original RAW size: " << original_raw_size << " bytes" << endl;
    cout << "Compressed size: " << compressed_size << " bytes" << endl;
    cout << "Compression ratio: " << (float)original_raw_size / compressed_size << ":1" << endl;
    cout << "Space saving: "
         << (1.0 - (double)compressed_size / original_raw_size) * 100.0 << "%" << endl;

    cudaFree(d_input);
    cudaFree(d_output);
    // Additional cleanup for compressed data
    GPURLEBlock *h_cleanup = new GPURLEBlock[num_blocks];
    cudaMemcpy(h_cleanup, compressed.blocks, num_blocks * sizeof(GPURLEBlock), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_blocks; i++)
    {
        cudaFree(h_cleanup[i].y_data);
        cudaFree(h_cleanup[i].cb_data);
        cudaFree(h_cleanup[i].cr_data);
    }
    delete[] h_cleanup;
    cudaFree(compressed.blocks);

    return 0;
}