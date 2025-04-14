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
    
    // Step 1: Pad the image to make it divisible by 8
    int pad_rows = (image.rows % 8 == 0) ? 0 : (8 - image.rows % 8);
    int pad_cols = (image.cols % 8 == 0) ? 0 : (8 - image.cols % 8);

    cv::Mat padded_image;
    cv::copyMakeBorder(image, padded_image, 0, pad_rows, 0, pad_cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Convert the block to CV_32F type for DCT processing
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

    // Step 1: Create an empty image of the same size as the original image
    image = cv::Mat(image_size, CV_32FC3, cv::Scalar(0, 0, 0)); // Initialize with zeros

    // Step 2: Iterate through the blocks and place them back into the image
    int index = 0;
    for (int row = 0; row < image.rows; row += 8)
    {
        for (int col = 0; col < image.cols; col += 8)
        {
            // Check if the index is within bounds of the blocks vector
            if (index < blocks.size())
            {
                cv::Mat block = blocks[index++];
                block.copyTo(image(cv::Rect(col, row, 8, 8)));
            }
        }
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

    // Step 2: Convert the image to YCbCr color space
    cv::Mat ycbcr_image;
    image.convertTo(ycbcr_image, cv::COLOR_BGR2YCrCb); // OpenCV only has YCrCb, not YCbCr, so I have to keep that in mind when multiplicating using Quantization matrix

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
    reconstructed_image.convertTo(reconstructed_bgr, CV_8UC3);
    cv::imshow("Reconstructed Image", reconstructed_bgr);
    cv::waitKey(0);
    return 0;
}