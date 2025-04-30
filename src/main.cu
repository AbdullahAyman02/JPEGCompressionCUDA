#include <iostream>
#include <vector>
#include <cuda.h>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;

// Define the quantization tables for YCbCr color space
// First the Chrominance Quantization Table, which is used to multiply the CbCr channel of the YCbCr image
__constant__ int ChrominanceQuantizationTable[8][8];
int h_ChromTable[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}};

// Then the Luminance Quantization Table, which is used to multiply the Y channel of the YCbCr image
__constant__ int LuminanceQuantizationTable[8][8];
int h_LumTable[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}};

// DCT matrix T obtained from matlab dctmtx(8)
__constant__ float dctMatrix[8][8];
float h_dctMatrix[8][8] = {
    {0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536},
    {0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904},
    {0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619},
    {0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157},
    {0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536},
    {0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778},
    {0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913},
    {0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975}};

// transposed DCT matrix T' obtained from matlab dctmtx(8) with a transpose
__constant__ float idctMatrix[8][8];
float h_idctMatrix[8][8] = {
    {0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1913, 0.0975},
    {0.3536, 0.4157, 0.1913, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778},
    {0.3536, 0.2778, -0.1913, -0.4904, -0.3536, 0.0975, 0.4619, 0.4157},
    {0.3536, 0.0975, -0.4619, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904},
    {0.3536, -0.0975, -0.4619, 0.2778, 0.3536, -0.4157, -0.1913, 0.4904},
    {0.3536, -0.2778, -0.1913, 0.4904, -0.3536, -0.0975, 0.4619, -0.4157},
    {0.3536, -0.4157, 0.1913, 0.0975, -0.3536, 0.4904, -0.4619, 0.2778},
    {0.3536, -0.4904, 0.4619, -0.4157, 0.3536, -0.2778, 0.1913, -0.0975}};

__host__ __device__ float determineScale(int quality)
{
    // Ensure quality is in valid range
    if (quality < 1)
        quality = 1;
    if (quality > 100)
        quality = 100;

    float scale;
    if (quality < 50)
    {
        scale = 50.0f / quality;
    }
    else
    {
        scale = 2.000001f - (quality * 2.0f / 100.0f);
    }

    return scale;
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
            exit(code);
    }
}

__host__ void splitIntoBlocks(const cv::Mat &image, std::vector<cv::Mat> &blocks)
{
    // TODO : Implement the function to split the image into 8x8 blocks
    // and store them in the blocks vector.
    // The blocks should be of type CV_32F and should be of size 8x8.
    // Keep in mind that if the image size is not divisible into 8x8 blocks, we pad with zeroes for now until I can figure out how to handle that.

    // Step 1: Pad the image to make it divisible by 8
    int pad_rows = (image.rows % 8 == 0) ? 0 : (8 - image.rows % 8);
    int pad_cols = (image.cols % 8 == 0) ? 0 : (8 - image.cols % 8);

    cv::Mat padded_image;
    cv::copyMakeBorder(image, padded_image, 0, pad_rows, 0, pad_cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cout << "Padding image: " << pad_rows << " rows and " << pad_cols << " columns." << endl;
    cout << "Padded image size: " << padded_image.size() << endl;

    // Convert the image to CV_32F type for DCT processing
    padded_image.convertTo(padded_image, CV_32FC3);

    // Step 2: Split the padded image into 8x8 blocks
    for (int row = 0; row < padded_image.rows; row += 8)
    {
        for (int col = 0; col < padded_image.cols; col += 8)
        {
            cv::Mat block = padded_image(cv::Rect(col, row, 8, 8)).clone(); // Clone to avoid modifying the original image
            blocks.push_back(block);
        }
    }
}

__host__ void processBlock(cv::Mat &block, const int quality)
{
    // TODO : Implement the function to apply DCT to the block and quantize it using the given quantization matrices.

    // Step 1: Split the block into channels
    vector<cv::Mat> channels(3);
    cv::split(block, channels); // Split the block into Y, Cr, and Cb channels

    // Step 2: Determine the scale factor based on the quality factor
    float scale = determineScale(quality);

    // Step 3: Apply DCT to each channel separately and quantize
    for (int c = 0; c < 3; ++c)
    {
        cv::dct(channels[c], channels[c]); // Apply DCT to each channel separately

        // Quantize the DCT coefficients
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                // Use the luminance quantization table for the Y channel (0) and the chrominance quantization table for Cb (1) and Cr (2) channels
                int quantTable = (c == 0) ? LuminanceQuantizationTable[i][j] : ChrominanceQuantizationTable[i][j];
                channels[c].at<float>(i, j) = round(channels[c].at<float>(i, j) / (quantTable * scale));
            }
        }
    }

    // Step 4: Merge the channels back into a single block
    cv::merge(channels, block);
}

__host__ void deprocessBlock(cv::Mat &block, const int quality)
{
    // TODO : Implement the function to dequantize the block and apply inverse DCT to reconstruct the image.

    // Calculate the same scale factor used during quantization
    float scale = determineScale(quality);

    // Step 1: Split the block into channels
    vector<cv::Mat> channels(3);
    cv::split(block, channels);

    // Step 2: Dequantize and apply inverse DCT to each channel
    for (int c = 0; c < 3; ++c)
    {
        // Dequantize
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                int quantTable = (c == 0) ? LuminanceQuantizationTable[i][j] : ChrominanceQuantizationTable[i][j];
                channels[c].at<float>(i, j) *= (quantTable * scale); // Apply the same scale factor used in quantization
            }
        }

        // Apply inverse DCT
        cv::idct(channels[c], channels[c]);
    }

    // Step 3: Merge the channels back
    cv::merge(channels, block);
}

__host__ void assembleBlocks(cv::Mat &image, const std::vector<cv::Mat> &blocks, const cv::Size &image_size)
{
    // TODO : Implement the function to reassemble the image from the blocks.
    // The image should be of the same size as the original image, hence why we pass the image_size parameter.

    // Calculate the padded dimensions that were used when splitting
    int padded_height = image_size.height + (image_size.height % 8 == 0 ? 0 : 8 - image_size.height % 8);
    int padded_width = image_size.width + (image_size.width % 8 == 0 ? 0 : 8 - image_size.width % 8);
    cv::Size padded_size(padded_width, padded_height);

    cout << "Original image size: " << image_size << endl;
    cout << "Padded image size for reassembly: " << padded_size << endl;

    // First create a padded image to match the dimensions used during splitting
    cv::Mat padded_result(padded_size, CV_32FC3, cv::Scalar(0, 0, 0));

    // Place blocks into the padded image
    int index = 0;
    for (int row = 0; row < padded_size.height; row += 8)
    {
        for (int col = 0; col < padded_size.width; col += 8)
        {
            if (index < blocks.size())
            {
                blocks[index++].copyTo(padded_result(cv::Rect(col, row, 8, 8)));
            }
        }
    }

    // Now crop the padded result to get the original image size
    image = padded_result(cv::Rect(0, 0, image_size.width, image_size.height)).clone();
}

__host__ void compareImages(const char *originalImagePath, const char *reconstructedImagePath)
{
    // Get file sizes
    FILE *f_orig = fopen(originalImagePath, "rb");
    FILE *f_recon = fopen(reconstructedImagePath, "rb");

    if (f_orig && f_recon)
    {
        // Get original file size
        fseek(f_orig, 0, SEEK_END);
        long orig_size = ftell(f_orig);

        // Get reconstructed file size
        fseek(f_recon, 0, SEEK_END);
        long recon_size = ftell(f_recon);

        // Close files
        fclose(f_orig);
        fclose(f_recon);

        // Calculate compression ratio
        double compression_ratio = static_cast<double>(orig_size) / recon_size;

        // Print results
        std::cout << "Original file size: " << orig_size << " bytes" << std::endl;
        std::cout << "Reconstructed file size: " << recon_size << " bytes" << std::endl;
        std::cout << "Compression ratio: " << compression_ratio << ":1" << std::endl;
        std::cout << "Space saving: " << (1.0 - 1.0 / compression_ratio) * 100.0 << "%" << std::endl;
    }
    else
    {
        std::cerr << "Error opening files for size comparison" << std::endl;
    }
}

__global__ void processBlocksKernel(float *d_input_blocks, float *d_output_blocks, int quality)
{
    // I first need to create a shared memory for the block to be processed and fill it with the data needed
    __shared__ float block[8][8][3]; // 8x8 block with 3 channels (Y, Cb, Cr)

    // Index to determine which block to process
    // int inputBlockIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int inputBlockIndex = blockIdx.x;

    // Each thread will load one pixel from the input blocks into shared memory
    // Keep in mind that one pixel -> 3 channels, not only one channel.
    for (int c = 0; c < 3; c++)
    {
        // Determine index of pixel inside the block
        // First go to that specific block in the input blocks (inputBlockIndex * 8 * 8 * 3)
        // Then go to the specific row (threadIdx.y * 8 * 3) and column (threadIdx.x * 3) of the block
        // Finally go to the specific channel (c) of the pixel
        int pixelIndex = inputBlockIndex * 8 * 8 * 3 + threadIdx.y * 8 * 3 + threadIdx.x * 3 + c;
        // Check boundary condition
        if (pixelIndex < 8 * 8 * 3 * gridDim.x) // Ensure we don't go out of bounds
        {
            // Load the pixel into shared memory
            block[threadIdx.y][threadIdx.x][c] = d_input_blocks[pixelIndex];
        }
    }

    __syncthreads(); // Synchronize threads to ensure all data is loaded into shared memory

    // Now apply DCT and quantization to the block in shared memory
    // Each thread is responsible for only one DCT coefficient, then quantize that coefficient and return the result to global memory

    // First determine the scale that will be used for quantization
    float scale = determineScale(quality);

    // Then apply DCT using the equation dctCoeff = D * block * D'
    for (int c = 0; c < 3; c++)
    {
        float dctCoeff = 0.0f;

        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                dctCoeff += dctMatrix[threadIdx.y][row] * block[row][col][c] * idctMatrix[col][threadIdx.x];
            }
        }

        // Quantize the DCT coefficient using the appropriate quantization table
        int quantTable = (c == 0) ? LuminanceQuantizationTable[threadIdx.y][threadIdx.x] : ChrominanceQuantizationTable[threadIdx.y][threadIdx.x];
        dctCoeff = round(dctCoeff / (quantTable * scale)); // Quantization step

        // Store the result in global memory
        // Calculate the index in the output blocks
        int outputIndex = inputBlockIndex * 8 * 8 * 3 + threadIdx.y * 8 * 3 + threadIdx.x * 3 + c;
        // Check boundary condition
        if (outputIndex < 8 * 8 * 3 * gridDim.x) // Ensure we don't go out of bounds
        {
            d_output_blocks[outputIndex] = dctCoeff; // Store the quantized DCT coefficient in global memory
        }
    }
}

__global__ void deprocessBlocksKernel(float *d_input_blocks, float *d_output_blocks, int quality)
{
    // Create shared memory for the block to be processed
    __shared__ float block[8][8][3]; // 8x8 block with 3 channels (Y, Cb, Cr)

    // Index to determine which block to process
    int blockIndex = blockIdx.x;

    // Thread coordinates within the 8x8 block
    int tx = threadIdx.x; // Column index (0-7)
    int ty = threadIdx.y; // Row index (0-7)

    // Load quantized DCT coefficients into shared memory
    for (int c = 0; c < 3; c++)
    {
        // Calculate the index in the input array
        int pixelIndex = blockIndex * 8 * 8 * 3 + (ty * 8 + tx) * 3 + c;
        // Load the coefficient into shared memory
        block[ty][tx][c] = d_input_blocks[pixelIndex];
    }

    __syncthreads(); // Ensure all threads have loaded their data

    // Calculate scale factor for dequantization
    float scale = determineScale(quality);

    // Each thread computes one pixel value for each channel
    for (int c = 0; c < 3; c++)
    {
        // First dequantize the DCT coefficient
        int quantTable = (c == 0) ? LuminanceQuantizationTable[ty][tx] : ChrominanceQuantizationTable[ty][tx];
        block[ty][tx][c] *= (quantTable * scale); // Dequantization step
    }

    __syncthreads(); // Ensure all coefficients are dequantized

    // Now apply IDCT to get the pixel values
    for (int c = 0; c < 3; c++)
    {
        // Compute pixel value using IDCT: pixel = T^T * DCT * T
        float pixelValue = 0.0f;
        
        for (int u = 0; u < 8; u++)
        {
            for (int v = 0; v < 8; v++)
            {
                // For IDCT, we use idctMatrix (T^T) first, then dctMatrix (T)
                // Note: In matrix multiplication for IDCT, the order is reversed compared to DCT
                pixelValue += idctMatrix[ty][u] * block[u][v][c] * dctMatrix[v][tx];
            }
        }
        
        // Store the reconstructed pixel value in the output
        int outputIndex = blockIndex * 8 * 8 * 3 + ty * 8 * 3 + tx * 3 + c;
        // Check boundary condition
        if (outputIndex < 8 * 8 * 3 * gridDim.x) // Ensure we don't go out of bounds
        {
            d_output_blocks[outputIndex] = pixelValue;
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
    const int quality = argc == 3 ? stoi(argv[2]) : 75; // Quality factor for quantization

    // Move necessary data into constant memory
    cudaMemcpyToSymbol(LuminanceQuantizationTable, h_LumTable, sizeof(LuminanceQuantizationTable));
    cudaMemcpyToSymbol(ChrominanceQuantizationTable, h_ChromTable, sizeof(ChrominanceQuantizationTable));
    cudaMemcpyToSymbol(dctMatrix, h_dctMatrix, sizeof(dctMatrix));
    cudaMemcpyToSymbol(idctMatrix, h_idctMatrix, sizeof(idctMatrix));
    gpuErrchk(cudaPeekAtLastError());

    // Step 1: Read the image in RGB format
    cv::Mat image = cv::imread(IMAGEPATH, cv::IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }
    cout << "Image size: " << image.size() << endl;
    cout << "Image type: " << image.type() << endl;

    // Step 2: Convert the image to YCbCr color space
    cv::Mat ycbcr_image;
    cv::cvtColor(image, ycbcr_image, cv::COLOR_BGR2YCrCb);

    // Step 3: Divide the image into 8*8 blocks
    std::vector<cv::Mat> blocks;
    splitIntoBlocks(ycbcr_image, blocks);

    // Step 4: Apply DCT to each block and quantize it using the given quantization matrix
    // for (auto &block : blocks)
    // processBlock(block, quality);
    // ******************************************************************************
    // Parallelize the processing of blocks using CUDA
    // ******************************************************************************
    // Allocate device memory for blocks
    const int numOfBlocks = blocks.size();
    const int blockSize = 8 * 8 * 3; // 8x8 blocks with 3 channels (Y, Cb, Cr)
    float *d_input_blocks, *d_output_blocks;
    cudaMalloc((void **)&d_input_blocks, numOfBlocks * blockSize * sizeof(float));
    gpuErrchk(cudaPeekAtLastError());
    cudaMalloc((void **)&d_output_blocks, numOfBlocks * blockSize * sizeof(float));
    gpuErrchk(cudaPeekAtLastError());

    // Copy blocks to device memory
    for (int i = 0; i < numOfBlocks; ++i)
    {
        cudaMemcpy(d_input_blocks + i * blockSize, (float *)blocks[i].data, blockSize * sizeof(float), cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }

    // Launch the kernel to process blocks in parallel
    dim3 dimBlock(8, 8);
    dim3 dimGrid(numOfBlocks);
    processBlocksKernel<<<dimGrid, dimBlock>>>(d_input_blocks, d_output_blocks, quality);
    gpuErrchk(cudaGetLastError());
    cudaDeviceSynchronize();

    // Now launch the inverse DCT kernel
    // Note: We're using d_output_blocks as input and d_input_blocks as output
    // This is just to reuse memory.
    deprocessBlocksKernel<<<dimGrid, dimBlock>>>(d_output_blocks, d_input_blocks, quality);
    gpuErrchk(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy the results back to host
    for (int i = 0; i < numOfBlocks; ++i)
    {
        cudaMemcpy((float *)blocks[i].data, d_input_blocks + i * blockSize, blockSize * sizeof(float), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaGetLastError());
    }
    
    // Free device memory
    cudaFree(d_input_blocks);
    cudaFree(d_output_blocks);

    // *****************************************************************************
    // End CUDA processing
    // *****************************************************************************


    // // Step 6: Reverse step 4: Dequantize the blocks and apply inverse DCT to reconstruct the image
    // for (auto &block : blocks)
    //     deprocessBlock(block, quality); // 75 is the quality factor for dequantization

    // Step 7: Reassemble the image from the blocks
    cv::Mat reconstructed_image;
    assembleBlocks(reconstructed_image, blocks, image.size());

    // Show the original and reconstructed images
    // cv::imshow("Original Image", image);
    // Convert the reconstructed image back to BGR before displaying
    cv::Mat reconstructed_bgr;
    reconstructed_image.convertTo(reconstructed_bgr, CV_8UC3);
    cv::cvtColor(reconstructed_bgr, reconstructed_bgr, cv::COLOR_YCrCb2BGR);
    // cv::imshow("Reconstructed Image", reconstructed_bgr);
    // cv::waitKey(0);

    // Save both images to disk
    cv::imwrite("original.jpg", image);
    cv::imwrite("reconstructed.jpg", reconstructed_bgr);

    cout << "Original and reconstructed images saved." << endl;

    // Step 8: Perform comparisons
    compareImages("original.jpg", "reconstructed.jpg");
    cout << "Comparison completed." << endl;
    return 0;
}