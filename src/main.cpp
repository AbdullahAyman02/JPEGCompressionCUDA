#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;

const string IMAGEPATH = "../../src/images/slax.jfif";
const int quality = 50; // Quality factor for quantization

// Define the quantization tables for YCbCr color space
// First the Chrominance Quantization Table, which is used to multiply the CbCr channel of the YCbCr image
const int ChrominanceQuantizationTable[8][8] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}};

// Then the Luminance Quantization Table, which is used to multiply the Y channel of the YCbCr image
const int LuminanceQuantizationTable[8][8] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}};

// Precomputed Zigzag scan pattern for 8x8 block
const int zigzagOrder[64] = {
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63};

// Structure to hold RLE encoded data
struct RLEPair
{
    int16_t run_length;  // 2 bytes
    int16_t value;       // 2 bytes (quantized coefficients are integers!)
};

// Structure to hold processed block data
struct ProcessedBlock
{
    std::vector<RLEPair> y_rle;
    std::vector<RLEPair> cb_rle;
    std::vector<RLEPair> cr_rle;
};

double measureExecutionTime(std::function<void()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count(); // Returns time in milliseconds
}

void splitIntoBlocks(const cv::Mat &image, std::vector<cv::Mat> &blocks)
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

float determineScale(int quality) {
    quality = std::clamp(quality, 1, 100);
    
    if (quality < 50) {
        return 50.0f / quality; // More aggressive quantization
    } else {
        return 2.1f - (quality * 2.0f / 100.0f); // Less quantization
    }
}

void myDCT(cv::Mat &block)
{
    // TODO : Implement the function to apply DCT to the block.
    // Note: Without using opencv's dct function.
    // You can use the DCT formula or any other method to compute the DCT.
    cv::Mat tempBlock = block.clone();

    // `block` will store the discrete cosine transform

    double ci, cj, dct1, sum;
    int m = 8, n = 8;
    double pi = 3.14159265358979323846;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {

            // ci and cj depends on frequency as well as
            // number of row and columns of specified matrix
            if (i == 0)
                ci = 1 / sqrt(m);
            else
                ci = sqrt(2) / sqrt(m);
            if (j == 0)
                cj = 1 / sqrt(n);
            else
                cj = sqrt(2) / sqrt(n);

            // sum will temporarily store the sum of
            // cosine signals
            sum = 0;
            for (int k = 0; k < m; k++)
            {
                for (int l = 0; l < n; l++)
                {
                    dct1 = tempBlock.at<float>(k, l) *
                           cos((2 * k + 1) * i * pi / (2 * m)) *
                           cos((2 * l + 1) * j * pi / (2 * n));
                    sum = sum + dct1;
                }
            }
            block.at<float>(i, j) = ci * cj * sum;
        }
    }
}

void myIDCT(cv::Mat &block)
{
    cv::Mat tempBlock = block.clone();
    double ci, cj;
    double pi = 3.14159265358979323846;
    int m = 8, n = 8;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // Calculate the pixel value at position (i,j)
            float sum = 0.0;

            for (int k = 0; k < m; k++)
            {
                for (int l = 0; l < n; l++)
                {
                    // Determine coefficients based on frequency
                    if (k == 0)
                        ci = 1 / sqrt(m);
                    else
                        ci = sqrt(2) / sqrt(m);

                    if (l == 0)
                        cj = 1 / sqrt(n);
                    else
                        cj = sqrt(2) / sqrt(n);

                    // Apply IDCT formula
                    sum += ci * cj * tempBlock.at<float>(k, l) *
                           cos((2 * i + 1) * k * pi / (2 * m)) *
                           cos((2 * j + 1) * l * pi / (2 * n));
                }
            }

            block.at<float>(i, j) = sum;
        }
    }
}

// Function to perform zigzag scan on a single channel
std::vector<int> zigzagScan(const cv::Mat &channel)
{
    std::vector<int> zigzag(64);
    for (int i = 0; i < 64; ++i)
    {
        int row = zigzagOrder[i] / 8;
        int col = zigzagOrder[i] % 8;
        zigzag[i] = channel.at<float>(row, col);
    }
    return zigzag;
}

// Function to perform inverse zigzag scan
cv::Mat inverseZigzag(const std::vector<float> &zigzag)
{
    cv::Mat channel(8, 8, CV_32F);
    for (int i = 0; i < 64; ++i)
    {
        int row = zigzagOrder[i] / 8;
        int col = zigzagOrder[i] % 8;
        channel.at<float>(row, col) = zigzag[i];
    }
    return channel;
}

// RLE encoding for a single channel
std::vector<RLEPair> rleEncode(const std::vector<int> &zigzag)
{
    std::vector<RLEPair> encoded;
    if (zigzag.empty())
        return encoded;

    int16_t current = zigzag[0];
    int16_t count = 1;

    for (size_t i = 1; i < zigzag.size(); ++i)
    {
        if (zigzag[i] == current)
        {
            count++;
        }
        else
        {
            encoded.push_back({count, current});
            current = zigzag[i];
            count = 1;
        }
    }
    encoded.push_back({count, current});

    return encoded;
}

// RLE decoding for a single channel
std::vector<float> rleDecode(const std::vector<RLEPair> &encoded)
{
    std::vector<float> decoded;
    for (const auto &pair : encoded)
    {
        decoded.insert(decoded.end(), pair.run_length, pair.value);
    }
    return decoded;
}

ProcessedBlock processBlockWithRLE(cv::Mat &block, const int quality)
{
    ProcessedBlock processed;

    // Split channels
    vector<cv::Mat> channels(3);
    cv::split(block, channels);

    // Determine scale factor
    float scale = determineScale(quality);

    // Process each channel
    for (int c = 0; c < 3; ++c)
    {
        // Apply DCT
        // myDCT(channels[c]);
        cv::dct(channels[c], channels[c]); // Using OpenCV's DCT for simplicity

        // Quantize
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                int quantTable = (c == 0) ? LuminanceQuantizationTable[i][j]
                                          : ChrominanceQuantizationTable[i][j];
                // channels[c].at<float>(i, j) = round(channels[c].at<float>(i, j) / (quantTable * scale));
                float quant_step = quantTable * scale;
        
                // Ensure minimum quantization step of 1
                if (quant_step < 1.0f) quant_step = 1.0f;
                
                channels[c].at<float>(i,j) = round(channels[c].at<float>(i,j) / quant_step);            }
        }

        // Zigzag scan and RLE encode
        auto zigzag = zigzagScan(channels[c]);
        auto rle = rleEncode(zigzag);

        // Store in the appropriate channel
        if (c == 0)
            processed.y_rle = rle;
        else if (c == 1)
            processed.cb_rle = rle;
        else
            processed.cr_rle = rle;
    }

    return processed;
}

cv::Mat deprocessBlockWithRLE(const ProcessedBlock &processed, const int quality)
{
    cv::Mat block(8, 8, CV_32FC3);
    vector<cv::Mat> channels(3);

    float scale = determineScale(quality);

    // Process Y channel
    auto y_decoded = rleDecode(processed.y_rle);
    channels[0] = inverseZigzag(y_decoded);

    // Process Cb channel
    auto cb_decoded = rleDecode(processed.cb_rle);
    channels[1] = inverseZigzag(cb_decoded);

    // Process Cr channel
    auto cr_decoded = rleDecode(processed.cr_rle);
    channels[2] = inverseZigzag(cr_decoded);

    // Dequantize and apply IDCT
    for (int c = 0; c < 3; ++c)
    {
        // Dequantize
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                int quantTable = (c == 0) ? LuminanceQuantizationTable[i][j]
                                          : ChrominanceQuantizationTable[i][j];
                channels[c].at<float>(i, j) *= (quantTable * scale);
            }
        }

        // Apply IDCT
        // myIDCT(channels[c]);
        cv::idct(channels[c], channels[c]); // Using OpenCV's IDCT for simplicity
    }

    // Merge channels
    cv::merge(channels, block);
    return block;
}

void processBlock(cv::Mat &block, const int quality)
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
        // myDCT(channels[c]); // Custom DCT function

        // Quantize the DCT coefficients
        for (int i = 0; i < 8; ++i)
        {
            for (int j = 0; j < 8; ++j)
            {
                // Use the luminance quantization table for the Y channel (0) and the chrominance quantization table for Cb (1) and Cr (2) channels
                int quantTable = (c == 0) ? LuminanceQuantizationTable[i][j] : ChrominanceQuantizationTable[i][j];
                // channels[c].at<float>(i, j) = round(channels[c].at<float>(i, j) / (quantTable * scale));
                float quant_step = quantTable * scale;
        
                // Ensure minimum quantization step of 1
                if (quant_step < 1.0f) quant_step = 1.0f;
                
                channels[c].at<float>(i,j) = round(channels[c].at<float>(i,j) / quant_step);            }
        }
    }

    // Step 4: Merge the channels back into a single block
    cv::merge(channels, block);
}

void benchmarks(const cv::Mat &block)
{
    // TODO : Implement the function to perform some calculations that could prove useful later.
    // Some examples include:
    // - Calculate the MSE (Mean Squared Error) between the original block and the processed block
    // - Calculate the PSNR (Peak Signal-to-Noise Ratio) between the original block and the processed block
    // - Calculate the SSIM (Structural Similarity Index) between the original block and the processed block
    // - Calculate the compression ratio between the original block and the processed block
}

void deprocessBlock(cv::Mat &block, const int quality)
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
        // myIDCT(channels[c]); // Custom IDCT function
    }

    // Step 3: Merge the channels back
    cv::merge(channels, block);
}

void assembleBlocks(cv::Mat &image, const std::vector<cv::Mat> &blocks, const cv::Size &image_size)
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

void saveCompressedData(const string &filename,
                        const vector<ProcessedBlock> &blocks,
                        const cv::Size &image_size)
{
    ofstream out(filename, ios::binary);

    // Write image dimensions
    int width = image_size.width;
    int height = image_size.height;
    out.write(reinterpret_cast<const char *>(&width), sizeof(width));
    out.write(reinterpret_cast<const char *>(&height), sizeof(height));

    // Write number of blocks
    int num_blocks = blocks.size();
    out.write(reinterpret_cast<const char *>(&num_blocks), sizeof(num_blocks));

    // Write each block's data
    for (const auto &block : blocks)
    {
        // Write Y channel
        uint32_t y_size = block.y_rle.size();
        out.write(reinterpret_cast<const char *>(&y_size), sizeof(y_size));
        out.write(reinterpret_cast<const char *>(block.y_rle.data()),
                  y_size * sizeof(RLEPair));

        // Write Cb channel
        uint32_t cb_size = block.cb_rle.size();
        out.write(reinterpret_cast<const char *>(&cb_size), sizeof(cb_size));
        out.write(reinterpret_cast<const char *>(block.cb_rle.data()),
                  cb_size * sizeof(RLEPair));

        // Write Cr channel
        uint32_t cr_size = block.cr_rle.size();
        out.write(reinterpret_cast<const char *>(&cr_size), sizeof(cr_size));
        out.write(reinterpret_cast<const char *>(block.cr_rle.data()),
                  cr_size * sizeof(RLEPair));
    }
}

vector<ProcessedBlock> loadCompressedData(const string &filename,
                                          cv::Size &image_size)
{
    ifstream in(filename, ios::binary);

    // Read image dimensions
    int width, height;
    in.read(reinterpret_cast<char *>(&width), sizeof(width));
    in.read(reinterpret_cast<char *>(&height), sizeof(height));
    image_size = cv::Size(width, height);

    // Read number of blocks
    int num_blocks;
    in.read(reinterpret_cast<char *>(&num_blocks), sizeof(num_blocks));

    vector<ProcessedBlock> blocks(num_blocks);

    for (int i = 0; i < num_blocks; ++i)
    {
        // Read Y channel
        uint32_t y_size;
        in.read(reinterpret_cast<char *>(&y_size), sizeof(y_size));
        blocks[i].y_rle.resize(y_size);
        in.read(reinterpret_cast<char *>(blocks[i].y_rle.data()),
                y_size * sizeof(RLEPair));

        // Read Cb channel
        uint32_t cb_size;
        in.read(reinterpret_cast<char *>(&cb_size), sizeof(cb_size));
        blocks[i].cb_rle.resize(cb_size);
        in.read(reinterpret_cast<char *>(blocks[i].cb_rle.data()),
                cb_size * sizeof(RLEPair));

        // Read Cr channel
        uint32_t cr_size;
        in.read(reinterpret_cast<char *>(&cr_size), sizeof(cr_size));
        blocks[i].cr_rle.resize(cr_size);
        in.read(reinterpret_cast<char *>(blocks[i].cr_rle.data()),
                cr_size * sizeof(RLEPair));
    }

    return blocks;
}

int main(int argc, char **argv)
{
    cv::Mat image = cv::imread(IMAGEPATH, cv::IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }
    cout << "Image size: " << image.size() << endl;
    cout << "Image type: " << image.type() << endl;

    cv::Mat ycbcr_image;
    cv::cvtColor(image, ycbcr_image, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> blocks;
    double splitTime = measureExecutionTime([&]()
                                            { splitIntoBlocks(ycbcr_image, blocks); });

    // ================== COMPRESSION PHASE ==================
    // Process blocks with RLE
    vector<ProcessedBlock> processedBlocks;
    double processTime = measureExecutionTime([&]()
                                              {
        for (auto &block : blocks) {
            processedBlocks.push_back(processBlockWithRLE(block, quality));
        } });

    // Save compressed data
    blocks.clear();
    const string compressed_filename = "compressed.shattah";
    saveCompressedData(compressed_filename, processedBlocks, image.size());
    
    // Get original file size
    uintmax_t original_raw_size = image.total() * image.elemSize(); 
    uintmax_t compressed_size = fs::file_size(compressed_filename);

    // ================== DECOMPRESSION PHASE ==================
    // Load compressed data
    cv::Size loaded_size;
    auto loadedBlocks = loadCompressedData(compressed_filename, loaded_size);
    
    // Reconstruct blocks from RLE
    double deprocessTime = measureExecutionTime([&]() {
        for (const auto &processed : loadedBlocks) {
            blocks.push_back(deprocessBlockWithRLE(processed, quality));
        }
    });

    cv::Mat reconstructed_image;
    // Measure reassembly time
    double assembleTime = measureExecutionTime([&]()
                                               { assembleBlocks(reconstructed_image, blocks, image.size()); });

    // Calculate total time
    double totalTime = splitTime + processTime + deprocessTime + assembleTime;

    // Print timing results
    std::cout << "\n===== CPU Timing Results =====\n";
    std::cout << "Split time: " << splitTime << " ms\n";
    std::cout << "DCT + Quantization + RLE time: " << processTime << " ms\n";
    std::cout << "IDCT + Dequantization + RLD time: " << deprocessTime << " ms\n";
    std::cout << "Reassembly time: " << assembleTime << " ms\n";
    std::cout << "Total time: " << totalTime << " ms\n";

    // Show the original and reconstructed images
    cv::imshow("Original Image", image);
    // Convert the reconstructed image back to BGR before displaying
    cv::Mat reconstructed_bgr;
    reconstructed_image.convertTo(reconstructed_bgr, CV_8UC3);
    cv::cvtColor(reconstructed_bgr, reconstructed_bgr, cv::COLOR_YCrCb2BGR);
    cv::imshow("Reconstructed Image", reconstructed_bgr);
    cv::waitKey(0);

    // ================== COMPRESSION METRICS ==================
    cout << "\n===== Compression Results =====" << endl;
    cout << "Original size: " << original_raw_size << " bytes" << endl;
    cout << "Compressed size: " << compressed_size << " bytes" << endl;
    cout << "Compression ratio: " 
         << fixed << setprecision(2) 
         << (double)original_raw_size/compressed_size << ":1" << endl;
    cout << "Space saving: " 
         << (1.0 - (double)compressed_size/original_raw_size)*100.0 << "%" << endl;

    return 0;
}