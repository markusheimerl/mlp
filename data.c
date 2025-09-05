#include "data.h"

void generate_data(float** X, float** y, int num_samples, int input_dim, int output_dim,
                   float range_min, float range_max) {
    int total_x = num_samples * input_dim;
    int total_y = num_samples * output_dim;
    
    *X = (float*)malloc(total_x * sizeof(float));
    *y = (float*)malloc(total_y * sizeof(float));
    
    // Generate input data in column-major format: [input_dim × num_samples]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int feature = 0; feature < input_dim; feature++) {
            (*X)[feature * num_samples + sample] = range_min + 
                ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
        }
    }
    
    // Create random transformation matrix W: [output_dim × input_dim]
    float* W = (float*)malloc(output_dim * input_dim * sizeof(float));
    float w_scale = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
    }
    
    // Transform input data: y = W * X
    // Using column-major BLAS: y[output_dim × num_samples] = W[output_dim × input_dim] * X[input_dim × num_samples]
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                output_dim, num_samples, input_dim,
                1.0f, W, output_dim,
                *X, input_dim,
                0.0f, *y, output_dim);
    
    // Add noise to outputs
    for (int i = 0; i < total_y; i++) {
        float noise = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
        (*y)[i] += noise;
    }
    
    // Free temporary transformation matrix
    free(W);
    
    printf("Generated regression data: %d samples, %d inputs, %d outputs (with %.3f noise)\n", 
           num_samples, input_dim, output_dim, w_scale);
}

void save_data(float* X, float* y, int num_samples, int input_dim, int output_dim,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) { 
        printf("Error: cannot write %s\n", filename); 
        return; 
    }
    
    // Header
    for (int i = 0; i < input_dim; i++) {
        fprintf(f, "x%d,", i);
    }
    for (int i = 0; i < output_dim; i++) {
        fprintf(f, "y%d%s", i, i == output_dim-1 ? "\n" : ",");
    }
    
    // Data
    for (int s = 0; s < num_samples; s++) {
        for (int i = 0; i < input_dim; i++) {
            fprintf(f, "%.6f,", X[i * num_samples + s]);
        }
        for (int i = 0; i < output_dim; i++) {
            fprintf(f, "%.6f%s", y[i * num_samples + s], i == output_dim-1 ? "\n" : ",");
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}