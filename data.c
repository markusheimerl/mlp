#include "data.h"

void generate_data(float** X, float** y, int num_samples, int input_dim, int output_dim,
                   float range_min, float range_max) {
    int total_x = num_samples * input_dim;
    int total_y = num_samples * output_dim;
    
    *X = (float*)malloc(total_x * sizeof(float));
    *y = (float*)malloc(total_y * sizeof(float));
    
    // Generate data in row-major format: [num_samples × input_dim]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int feature = 0; feature < input_dim; feature++) {
            (*X)[sample * input_dim + feature] = range_min + 
                ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
        }
    }
    
    // Generate output data in row-major format: [num_samples × output_dim]
    for (int sample = 0; sample < num_samples; sample++) {
        for (int out = 0; out < output_dim; out++) {
            (*y)[sample * output_dim + out] = range_min + 
                ((float)rand() / (float)RAND_MAX) * (range_max - range_min);
        }
    }
    
    printf("Generated random data: %d samples, %d inputs, %d outputs\n", 
           num_samples, input_dim, output_dim);
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