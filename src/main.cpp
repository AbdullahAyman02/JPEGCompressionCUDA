#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
const string IMAGEPATH = "../../src/images/tembo.jpg";
const int quality = 99; // Quality factor for quantization

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


double measureExecutionTime(std::function<void()> func) {
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

float determineScale(int quality)
{
    // Ensure quality is in valid range
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    
    float scale;
    if (quality < 50) {
        scale = 50.0f / quality;
    } else {
        scale = 2.000001f - (quality * 2.0f / 100.0f);
    }
    
    return scale;
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

void compareImages(const char *originalImagePath, const char *reconstructedImagePath)
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

int main(int argc, char **argv)
{
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
    // Measure splitting time
    double splitTime = measureExecutionTime([&]() {
        splitIntoBlocks(ycbcr_image, blocks);
    });

    // Step 4: Apply DCT to each block and quantize it using the given quantization matrix
    // Measure DCT + quantization time
    double processTime = measureExecutionTime([&]() {
        for (auto &block : blocks)
            processBlock(block, quality);
    });

    // Step 5: Perform some calculations that could prove useful later
    for (auto &block : blocks)
        benchmarks(block);

    // Step 6: Reverse step 4: Dequantize the blocks and apply inverse DCT to reconstruct the image
    // Measure IDCT + dequantization time
    double deprocessTime = measureExecutionTime([&]() {
        for (auto &block : blocks)
            deprocessBlock(block, quality);
    });

    // Step 7: Reassemble the image from the blocks
    cv::Mat reconstructed_image;
    // Measure reassembly time
    double assembleTime = measureExecutionTime([&]() {
        assembleBlocks(reconstructed_image, blocks, image.size());
    });

    // Calculate total time
    double totalTime = splitTime + processTime + deprocessTime + assembleTime;

    // Print timing results
    std::cout << "\n===== CPU Timing Results =====\n";
    std::cout << "Split time: " << splitTime << " ms\n";
    std::cout << "DCT + Quantization time: " << processTime << " ms\n";
    std::cout << "IDCT + Dequantization time: " << deprocessTime << " ms\n";
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

    // Save both images to disk
    cv::imwrite("original.jpg", image);
    cv::imwrite("reconstructed.jpg", reconstructed_bgr);

    cout << "Original and reconstructed images saved." << endl;

    // Step 8: Perform comparisons
    compareImages("original.jpg", "reconstructed.jpg");
    cout << "Comparison completed." << endl;
    return 0;
}