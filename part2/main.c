#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 8
#define PI 3.14159265358979323846

/* Standard JPEG luminance quantization table */
static const int quant_table[BLOCK_SIZE][BLOCK_SIZE] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

static float basis[BLOCK_SIZE][BLOCK_SIZE];
static float basis_transposed[BLOCK_SIZE][BLOCK_SIZE];

/*
 * Part 2 RVV kernel:
 * Process eight single-precision values in parallel,
 * then reduce the vector into one scalar sum.
 */
static float rvv_dot8(
    const float *values,
    const float *weights,
    float scale)
{
#if defined(__riscv) && defined(__riscv_v)
    float result = 0.0f;

    __asm__ volatile(
        /* Eight 32-bit floating-point elements. */
        "li t0, 8\n\t"
        "vsetvli zero, t0, e32, m1, ta, ma\n\t"

        /* Load eight input values and eight weights. */
        "vle32.v v1, (%[values])\n\t"
        "vle32.v v2, (%[weights])\n\t"

        /* Compute scale * weights[i] * values[i]. */
        "vfmul.vf v3, v2, %[scale]\n\t"
        "vfmul.vv v4, v3, v1\n\t"

        /* Use a zero vector as the reduction seed. */
        "vmv.v.i v0, 0\n\t"

        /* Reduce eight lanes into one scalar result. */
        "vfredusum.vs v5, v4, v0\n\t"

        /* Store lane 0 into the C variable result. */
        "vfmv.f.s ft0, v5\n\t"
        "fsw ft0, (%[result])\n\t"

        :
        : [values] "r"(values),
          [weights] "r"(weights),
          [scale] "f"(scale),
          [result] "r"(&result)
        : "t0", "ft0", "memory"
    );

    return result;
#else
    float result = 0.0f;

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        result += scale * weights[i] * values[i];
    }

    return result;
#endif
}

static void initialize_basis(void)
{
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        double alpha;

        if (u == 0) {
            alpha = sqrt(1.0 / BLOCK_SIZE);
        } else {
            alpha = sqrt(2.0 / BLOCK_SIZE);
        }

        for (int x = 0; x < BLOCK_SIZE; ++x) {
            double angle =
                ((2.0 * x + 1.0) * u * PI) /
                (2.0 * BLOCK_SIZE);

            basis[u][x] = (float)(alpha * cos(angle));
            basis_transposed[x][u] = basis[u][x];
        }
    }
}

static int read_token(FILE *file, char *buffer, size_t buffer_size)
{
    int character;
    size_t index = 0;

    do {
        character = fgetc(file);

        if (character == '#') {
            do {
                character = fgetc(file);
            } while (character != '\n' && character != EOF);
        }
    } while (isspace(character));

    if (character == EOF) {
        return 0;
    }

    while (character != EOF &&
           !isspace(character) &&
           character != '#') {
        if (index + 1 < buffer_size) {
            buffer[index++] = (char)character;
        }

        character = fgetc(file);
    }

    if (character == '#') {
        do {
            character = fgetc(file);
        } while (character != '\n' && character != EOF);
    }

    buffer[index] = '\0';
    return 1;
}

static uint8_t *read_pgm(
    const char *path,
    int *width,
    int *height)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        fprintf(
            stderr,
            "Cannot open %s: %s\n",
            path,
            strerror(errno));

        return NULL;
    }

    char token[64];

    if (!read_token(file, token, sizeof(token)) ||
        strcmp(token, "P5") != 0) {
        fprintf(stderr, "Input must be a binary PGM P5 file.\n");
        fclose(file);
        return NULL;
    }

    if (!read_token(file, token, sizeof(token))) {
        fclose(file);
        return NULL;
    }

    *width = atoi(token);

    if (!read_token(file, token, sizeof(token))) {
        fclose(file);
        return NULL;
    }

    *height = atoi(token);

    if (!read_token(file, token, sizeof(token))) {
        fclose(file);
        return NULL;
    }

    int maximum_value = atoi(token);

    if (*width <= 0 ||
        *height <= 0 ||
        maximum_value != 255) {
        fprintf(
            stderr,
            "Unsupported PGM: width=%d height=%d max=%d\n",
            *width,
            *height,
            maximum_value);

        fclose(file);
        return NULL;
    }

    size_t pixel_count =
        (size_t)(*width) * (size_t)(*height);

    uint8_t *image = malloc(pixel_count);

    if (image == NULL) {
        fprintf(
            stderr,
            "Unable to allocate %zu bytes.\n",
            pixel_count);

        fclose(file);
        return NULL;
    }

    if (fread(image, 1, pixel_count, file) != pixel_count) {
        fprintf(stderr, "Unexpected end of PGM image data.\n");
        free(image);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return image;
}

static int write_pgm(
    const char *path,
    const uint8_t *image,
    int width,
    int height)
{
    FILE *file = fopen(path, "wb");

    if (file == NULL) {
        fprintf(
            stderr,
            "Cannot write %s: %s\n",
            path,
            strerror(errno));

        return 0;
    }

    fprintf(file, "P5\n%d %d\n255\n", width, height);

    size_t pixel_count =
        (size_t)width * (size_t)height;

    int success =
        fwrite(image, 1, pixel_count, file) == pixel_count;

    fclose(file);
    return success;
}

static void process_block(
    const uint8_t *input,
    uint8_t *output,
    int width,
    int starting_row,
    int starting_column,
    uint64_t *quantized_checksum,
    uint64_t *nonzero_count)
{
    float shifted[BLOCK_SIZE][BLOCK_SIZE];
    float dequantized[BLOCK_SIZE][BLOCK_SIZE];

    /*
     * Convert the input pixels from [0, 255]
     * to approximately [-128, 127].
     */
    for (int x = 0; x < BLOCK_SIZE; ++x) {
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            int image_index =
                (starting_row + x) * width +
                (starting_column + y);

            shifted[x][y] =
                (float)input[image_index] - 128.0f;
        }
    }

    /*
     * Scalar direct two-dimensional DCT.
     *
     * F(u,v) =
     * sum_x sum_y
     * C(u,x) C(v,y) f(x,y)
     */
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        for (int v = 0; v < BLOCK_SIZE; ++v) {
            float sum = 0.0f;

            for (int x = 0; x < BLOCK_SIZE; ++x) {
                /*
                 * Vectorize y = 0...7.
                 * Eight products are computed in parallel and then
                 * reduced into the scalar accumulator.
                 */
                sum += rvv_dot8(
                    &shifted[x][0],
                    &basis[v][0],
                    basis[u][x]);
            }

            int quantized =
                (int)lroundf(
                    sum /
                    (float)quant_table[u][v]);

            if (quantized != 0) {
                ++(*nonzero_count);
            }

            /*
             * FNV-1a-style checksum used to compare
             * Part 1, Part 2 and Part 3 results.
             */
            *quantized_checksum ^= (uint32_t)quantized;
            *quantized_checksum *= UINT64_C(1099511628211);

            dequantized[u][v] =
                (float)(
                    quantized *
                    quant_table[u][v]);
        }
    }

    /*
     * Scalar inverse two-dimensional DCT.
     */
    for (int x = 0; x < BLOCK_SIZE; ++x) {
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            float sum = 0.0f;

            for (int u = 0; u < BLOCK_SIZE; ++u) {
                /*
                 * Vectorize v = 0...7.
                 * basis_transposed[y][v] equals basis[v][y],
                 * allowing one contiguous vector load.
                 */
                sum += rvv_dot8(
                    &dequantized[u][0],
                    &basis_transposed[y][0],
                    basis[u][x]);
            }

            int reconstructed =
                (int)lroundf(sum + 128.0f);

            if (reconstructed < 0) {
                reconstructed = 0;
            }

            if (reconstructed > 255) {
                reconstructed = 255;
            }

            int image_index =
                (starting_row + x) * width +
                (starting_column + y);

            output[image_index] =
                (uint8_t)reconstructed;
        }
    }
}

int main(int argc, char **argv)
{
    const char *input_path =
        argc >= 2
            ? argv[1]
            : "../data/input_gray.pgm";

    const char *output_path =
        argc >= 3
            ? argv[2]
            : "reconstructed.pgm";

    /*
     * Third argument:
     * 0 or omitted = process the complete image.
     * Positive value = process only that many blocks.
     */
    uint64_t requested_blocks =
        argc >= 4
            ? strtoull(argv[3], NULL, 10)
            : 0;

    int width = 0;
    int height = 0;

    uint8_t *input =
        read_pgm(input_path, &width, &height);

    if (input == NULL) {
        return EXIT_FAILURE;
    }

    if ((width % BLOCK_SIZE) != 0 ||
        (height % BLOCK_SIZE) != 0) {
        fprintf(
            stderr,
            "Image dimensions must be divisible by %d.\n",
            BLOCK_SIZE);

        free(input);
        return EXIT_FAILURE;
    }

    size_t pixel_count =
        (size_t)width * (size_t)height;

    uint8_t *output = malloc(pixel_count);

    if (output == NULL) {
        fprintf(stderr, "Unable to allocate output image.\n");
        free(input);
        return EXIT_FAILURE;
    }

    memcpy(output, input, pixel_count);

    initialize_basis();

    uint64_t total_blocks =
        (uint64_t)(width / BLOCK_SIZE) *
        (uint64_t)(height / BLOCK_SIZE);

    uint64_t blocks_to_process;

    if (requested_blocks == 0 ||
        requested_blocks > total_blocks) {
        blocks_to_process = total_blocks;
    } else {
        blocks_to_process = requested_blocks;
    }

    uint64_t quantized_checksum =
        UINT64_C(1469598103934665603);

    uint64_t nonzero_count = 0;
    uint64_t processed_blocks = 0;

    for (int row = 0;
         row < height &&
         processed_blocks < blocks_to_process;
         row += BLOCK_SIZE) {

        for (int column = 0;
             column < width &&
             processed_blocks < blocks_to_process;
             column += BLOCK_SIZE) {

            process_block(
                input,
                output,
                width,
                row,
                column,
                &quantized_checksum,
                &nonzero_count);

            ++processed_blocks;
        }
    }

    double squared_error = 0.0;
    uint64_t compared_pixels = 0;
    processed_blocks = 0;

    for (int row = 0;
         row < height &&
         processed_blocks < blocks_to_process;
         row += BLOCK_SIZE) {

        for (int column = 0;
             column < width &&
             processed_blocks < blocks_to_process;
             column += BLOCK_SIZE) {

            for (int x = 0; x < BLOCK_SIZE; ++x) {
                for (int y = 0; y < BLOCK_SIZE; ++y) {
                    size_t index =
                        (size_t)(row + x) *
                        (size_t)width +
                        (size_t)(column + y);

                    double difference =
                        (double)input[index] -
                        (double)output[index];

                    squared_error +=
                        difference * difference;

                    ++compared_pixels;
                }
            }

            ++processed_blocks;
        }
    }

    double mse =
        squared_error /
        (double)compared_pixels;

    double psnr;

    if (mse == 0.0) {
        psnr = INFINITY;
    } else {
        psnr =
            10.0 *
            log10(
                (255.0 * 255.0) /
                mse);
    }

    if (!write_pgm(
            output_path,
            output,
            width,
            height)) {

        fprintf(stderr, "Failed to write output image.\n");

        free(output);
        free(input);
        return EXIT_FAILURE;
    }

    printf("Image: %d x %d\n", width, height);

    printf(
        "Processed blocks: %" PRIu64 " / %" PRIu64 "\n",
        blocks_to_process,
        total_blocks);

    printf(
        "Quantized nonzero coefficients: %" PRIu64 "\n",
        nonzero_count);

    printf(
        "Quantized checksum: %" PRIu64 "\n",
        quantized_checksum);

    printf("MSE: %.6f\n", mse);
    printf("PSNR: %.6f dB\n", psnr);
    printf("Output: %s\n", output_path);

    free(output);
    free(input);

    return EXIT_SUCCESS;
}
