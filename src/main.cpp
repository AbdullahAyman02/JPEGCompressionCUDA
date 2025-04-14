#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
const string IMAGEPATH = "../../src/images/R.jpg";

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

void splitIntoBlocks(const cv::Mat &image, std::vector<cv::Mat> &blocks)
{
    // TODO : Implement the function to split the image into 8x8 blocks
    // and store them in the blocks vector.
    // The blocks should be of type CV_32F and should be of size 8x8.
    // Keep in mind that if the image size is not divisible into 8x8 blocks, we pad with zeroes for now until I can figure out how to handle that.
}

void processBlock(cv::Mat &block)
{
    // TODO : Implement the function to apply DCT to the block and quantize it using the given quantization matrices.

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

void deprocessBlock(cv::Mat &block)
{
    // TODO : Implement the function to dequantize the block and apply inverse DCT to reconstruct the image.
}

void assembleBlocks(cv::Mat &image, const std::vector<cv::Mat> &blocks, const cv::Size &image_size)
{
    // TODO : Implement the function to reassemble the image from the blocks.
    // The image should be of the same size as the original image, hence why we pass the image_size parameter.
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

    // Step 2: Convert the image to YCbCr color space
    cv::Mat ycbcr_image;
    cv::cvtColor(image, ycbcr_image, cv::COLOR_BGR2YCrCb); // OpenCV only has YCrCb, not YCbCr, so I have to keep that in mind when multiplicating using Quantization matrix

    // Step 3: Divide the image into 8*8 blocks
    std::vector<cv::Mat> blocks;
    splitIntoBlocks(ycbcr_image, blocks);

    // Step 4: Apply DCT to each block and quantize it using the given quantization matrix
    for (auto block : blocks)
        processBlock(block);

    // Step 5: Perform some calculations that could prove useful later
    for (auto block : blocks)
        benchmarks(block);

    // Step 6: Reverse step 4: Dequantize the blocks and apply inverse DCT to reconstruct the image
    for (auto block : blocks)
        deprocessBlock(block);

    // Step 7: Reassemble the image from the blocks
    cv::Mat reconstructed_image;
    assembleBlocks(reconstructed_image, blocks, image.size());

    // Show the original and reconstructed images
    cv::imshow("Original Image", image);
    // Convert the reconstructed image back to BGR before displaying
    cv::Mat reconstructed_bgr;
    cv::cvtColor(reconstructed_image, reconstructed_bgr, cv::COLOR_YCrCb2BGR);
    cv::imshow("Reconstructed Image", reconstructed_bgr);
    cv::waitKey(0);
    return 0;
}