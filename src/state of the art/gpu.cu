%%writefile gpu_jpeg.cu
#include <nvjpeg.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

#define CHECK_NVJPEG(call) { \
    nvjpegStatus_t status = call; \
    if (status != NVJPEG_STATUS_SUCCESS) { \
        std::cerr << "nvJPEG error at line " << __LINE__ << ": " << status << std::endl; \
        exit(1); \
    } \
}

int main() {
    nvjpegHandle_t handle;
    nvjpegJpegState_t state;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encoder_params;
    cudaStream_t stream;
    
    cudaStreamCreate(&stream);
    CHECK_NVJPEG(nvjpegCreate(NVJPEG_BACKEND_GPU_HYBRID, nullptr, &handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &encoder_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, &encoder_params, stream));

    // Read input
    FILE* fp = fopen("../input/images/pexels.png", "rb");
    if (!fp) {
        std::cerr << "Failed to open input.jpg" << std::endl;
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* jpg_data = new unsigned char[length];
    fread(jpg_data, 1, length, fp);
    fclose(fp);

    // Get image info
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegChromaSubsampling_t subsampling;
    int nComponents;
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, jpg_data, length, &nComponents, &subsampling, widths, heights));
    std::cout << "nComponents: " << nComponents << std::endl;
    std::cout << "Subsampling: " << subsampling << std::endl;

    // Debug print
    std::cout << "Image dimensions: " << widths[0] << "x" << heights[0] << std::endl;
    
    // Allocate single RGB buffer with proper pitch
    nvjpegImage_t imgdesc{};
    for (int c = 0; c < 3; ++c) {
        size_t pitch = widths[0];
        size_t size = pitch * heights[0];
        cudaError_t cudaErr = cudaHostAlloc(&imgdesc.channel[c], size, cudaHostAllocMapped);
        if (cudaErr != cudaSuccess) {
            std::cerr << "CUDA host alloc failed: " << cudaGetErrorString(cudaErr) << std::endl;
            exit(1);
        }
        imgdesc.pitch[c] = pitch;
    }
    for (int c = 3; c < NVJPEG_MAX_COMPONENT; ++c) {
        imgdesc.channel[c] = nullptr;
        imgdesc.pitch[c] = 0;
    }
    CHECK_NVJPEG(nvjpegDecode(handle, state, jpg_data, length, NVJPEG_OUTPUT_YUV, &imgdesc, stream));
    cudaStreamSynchronize(stream);
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        std::cerr << "CUDA error after decode: " << cudaGetErrorString(cudaErr) << std::endl;
        return 1;
    }
    
    // Encode
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encoder_params, 75, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encoder_params, NVJPEG_CSS_444, stream));
    CHECK_NVJPEG(nvjpegEncodeImage(handle, encoder_state, encoder_params, &imgdesc, 
        NVJPEG_INPUT_RGB, widths[0], heights[0], stream));

    cudaStreamSynchronize(stream);

    // Retrieve compressed data
    size_t compressed_size = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, nullptr, &compressed_size, stream));
    unsigned char* jpg_output = new unsigned char[compressed_size];
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, encoder_state, jpg_output, &compressed_size, stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Write output
    FILE* out = fopen("gpu_compressed.jpg", "wb");
    fwrite(jpg_output, 1, compressed_size, out);
    fclose(out);

    std::cout << "GPU Compression: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() 
              << " μs\n";
    size_t raw_size = widths[0] * heights[0] * 3;
    std::cout << "Original RAW size: " << raw_size << " bytes\n";
    std::cout << "Compressed size: " << compressed_size << " bytes\n";
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
            << (double)raw_size / compressed_size << ":1\n";
    std::cout << "Space saving: "
            << (1.0 - (double)compressed_size / raw_size) * 100.0 << "%\n";

    // Cleanup
    cudaFreeHost(imgdesc.channel[0]);
    delete[] jpg_data;
    delete[] jpg_output;
    nvjpegDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}