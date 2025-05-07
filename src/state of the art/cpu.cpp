#include <stdio.h>
#include <jpeglib.h>
#include <chrono>
#include <sys/stat.h>
#include <math.h>
#include <cstring>

// Helper to get file size
size_t getFileSize(const char* filename) {
    struct stat st;
    if (stat(filename, &st) == 0)
        return st.st_size;
    return 0;
}

int main(int argc, char** argv) {
    const char* filename = "../input/images/pexels.png";
    const char* output = "cpu_compressed.jpg";
    int quality = 75;

    // Read image
    FILE* infile = fopen(filename, "rb");
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width * channels, 1);

    // Store original image for PSNR
    unsigned char* original_img = new unsigned char[width * height * channels];
    size_t row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(original_img + row * width * channels, buffer[0], width * channels);
        row++;
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    // Compression setup
    struct jpeg_compress_struct cinfo_out;
    struct jpeg_error_mgr jerr_out;
    FILE* outfile = fopen(output, "wb");
    
    cinfo_out.err = jpeg_std_error(&jerr_out);
    jpeg_create_compress(&cinfo_out);
    jpeg_stdio_dest(&cinfo_out, outfile);

    cinfo_out.image_width = width;
    cinfo_out.image_height = height;
    cinfo_out.input_components = channels;
    cinfo_out.in_color_space = JCS_RGB;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    jpeg_set_defaults(&cinfo_out);
    jpeg_set_quality(&cinfo_out, quality, TRUE);
    jpeg_start_compress(&cinfo_out, TRUE);

    // Rewind original image for compression
    row = 0;
    while (cinfo_out.next_scanline < cinfo_out.image_height) {
        JSAMPROW row_pointer[1];
        row_pointer[0] = &original_img[row * width * channels];
        jpeg_write_scanlines(&cinfo_out, row_pointer, 1);
        row++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    jpeg_finish_compress(&cinfo_out);
    jpeg_destroy_compress(&cinfo_out);
    fclose(outfile);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Get sizes
    size_t raw_size = width * height * channels;
    size_t compressed_size = getFileSize(output);

    printf("CPU Compression Time: %ld μs\n", duration.count());
    printf("Original RAW size: %zu bytes\n", raw_size);
    printf("Compressed size: %zu bytes\n", compressed_size);
    printf("Compression ratio: %.2f:1\n", (double)raw_size / compressed_size);
    printf("Space saving: %.2f%%\n", (1.0 - (double)compressed_size / raw_size) * 100.0);

    // Optional: Decompress compressed image and compute PSNR
    FILE* compfile = fopen(output, "rb");
    struct jpeg_decompress_struct cinfo2;
    struct jpeg_error_mgr jerr2;
    cinfo2.err = jpeg_std_error(&jerr2);
    jpeg_create_decompress(&cinfo2);
    jpeg_stdio_src(&cinfo2, compfile);
    jpeg_read_header(&cinfo2, TRUE);
    jpeg_start_decompress(&cinfo2);

    unsigned char* decompressed_img = new unsigned char[width * height * channels];
    row = 0;
    JSAMPARRAY buffer2 = (*cinfo2.mem->alloc_sarray)((j_common_ptr)&cinfo2, JPOOL_IMAGE, width * channels, 1);
    while (cinfo2.output_scanline < cinfo2.output_height) {
        jpeg_read_scanlines(&cinfo2, buffer2, 1);
        memcpy(decompressed_img + row * width * channels, buffer2[0], width * channels);
        row++;
    }
    jpeg_finish_decompress(&cinfo2);
    jpeg_destroy_decompress(&cinfo2);
    fclose(compfile);

    delete[] original_img;
    delete[] decompressed_img;

    return 0;
}