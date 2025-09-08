#include "data.h"

void generate_data(float** X, float** y, int num_samples, int input_dim, int output_dim,
                   float range_min, float range_max) {
    int total_x = num_samples * input_dim;
    int total_y = num_samples * output_dim;
    
    *X = (float*)malloc(total_x * sizeof(float));
    *y = (float*)malloc(total_y * sizeof(float));
    
    // Generate input data in row-major format: [num_samples × input_dim]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int feature = 0; feature < input_dim; feature++) {
            (*X)[sample * input_dim + feature] = range_min + 
                ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
        }
    }
    
    // Create random transformation matrix W: [input_dim × output_dim]
    float* W = (float*)malloc(input_dim * output_dim * sizeof(float));
    float w_scale = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < input_dim * output_dim; i++) {
        W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
    }
    
    // Transform input data: y = X * W
    // Using row-major BLAS: y[num_samples × output_dim] = X[num_samples × input_dim] * W[input_dim × output_dim]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                num_samples, output_dim, input_dim,
                1.0f, *X, input_dim,
                W, output_dim,
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
            fprintf(f, "%.6f,", X[s * input_dim + i]);
        }
        for (int i = 0; i < output_dim; i++) {
            fprintf(f, "%.6f%s", y[s * output_dim + i], i == output_dim-1 ? "\n" : ",");
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}