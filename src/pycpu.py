import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the standard luminance quantization table (used in JPEG)
# You'll need a chrominance table as well for color images
quant_table_lum = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

quant_table_chr = np.array([  # Add chrominance table
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

class JPEGCompression:
    def __init__(self, quality=50):
        # Initialize the quality parameter (0-100, higher is better quality)
        self.quality = quality

    def read_image(self, path):
        # Reads an image from the specified path
        image = cv2.imread(path, cv2.IMREAD_COLOR)  # Read in color (BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image

    def display_image(self, image, title="Image"):
        # Displays an image using matplotlib
        plt.axis('off')  # Turn off axis labels
        plt.title(title)  # Set the title of the plot
        plt.imshow(image)  # Display the image
        plt.show()  # Show the plot

    def quantize(self, block, is_lum=True):
        # Quantizes a block of DCT coefficients using the quantization table
        q_table = quant_table_lum if is_lum else quant_table_chr  # Select correct table

        # Calculate the scaling factor based on the quality parameter (from the research paper)
        S = 5000 / self.quality if self.quality < 50 else 200 - 2 * self.quality
        scaled_q_table = np.floor((q_table * S + 50) / 100)  # Scale the quantization table
        scaled_q_table[scaled_q_table == 0] = 1  # Prevent division by zero

        # Quantize the block by dividing by the scaled quantization table
        return np.round(block / scaled_q_table).astype(int)

    def dequantize(self, block, is_lum=True):
        # Dequantizes a block of quantized DCT coefficients
        q_table = quant_table_lum if is_lum else quant_table_chr  # Select correct table

        # Calculate the scaling factor (same as in quantize)
        S = 5000 / self.quality if self.quality < 50 else 200 - 2 * self.quality
        scaled_q_table = np.floor((q_table * S + 50) / 100)
        scaled_q_table[scaled_q_table == 0] = 1

        # Dequantize by multiplying by the scaled quantization table
        return block * scaled_q_table

    def compress_image(self, image):
        # Convert to YCbCr *BEFORE* processing blocks
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        rows, cols, channels = image_ycrcb.shape  # Get dimensions from YCbCr image
        compressed_img = np.zeros_like(image_ycrcb, dtype=np.float32)

        for c in range(channels):
            for i in range(0, rows, 8):
                for j in range(0, cols, 8):
                    block = image_ycrcb[i:i + 8, j:j + 8, c]  # Extract block from YCbCr image

                    if block.shape != (8, 8):
                        block = np.pad(block, ((0, 8 - block.shape[0]), (0, 8 - block.shape[1])), mode='constant')

                    is_lum = (c == 0)

                    block_f = np.float32(block)
                    dct_coeffs = cv2.dct(block_f)

                    quantized_block = self.quantize(dct_coeffs, is_lum=is_lum)
                    dequantized_block = self.dequantize(quantized_block, is_lum=is_lum)

                    idct_coeffs = cv2.idct(dequantized_block)
                    compressed_img[i:i + 8, j:j + 8, c] = idct_coeffs

        compressed_img = cv2.cvtColor(np.uint8(compressed_img), cv2.COLOR_YCrCb2RGB)
        return compressed_img

    def save_image(self, image, output_path):
        """Saves the image to the specified output path."""
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for saving
        cv2.imwrite(output_path, image)  # Save the image using OpenCV