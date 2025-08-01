#include "data.h"

static float evaluate_synthetic_function(int num_terms, const float* coefficients, const int* operations,
                              const int* idx1, const int* idx2, const int* add_subtract, const float* x) {
    float result = 0.0f;
    
    for (int i = 0; i < num_terms; i++) {
        float coefficient = coefficients[i];
        int operation = operations[i];
        int input_idx1 = idx1[i];
        int input_idx2 = idx2[i];
        int add_sub = add_subtract[i];
        
        float term_value = 0.0f;
        
        switch (operation) {
            case 0: term_value = coefficient * sinf(x[input_idx1] * 2.0f); break;
            case 1: term_value = coefficient * cosf(x[input_idx1] * 1.5f); break;
            case 2: term_value = coefficient * tanhf(x[input_idx1] + x[input_idx2]); break;
            case 3: term_value = coefficient * expf(-powf(x[input_idx1], 2)); break;
            case 4: term_value = coefficient * logf(fabsf(x[input_idx1]) + 1.0f); break;
            case 5: term_value = coefficient * powf(x[input_idx1], 2) * x[input_idx2]; break;
            case 6: term_value = coefficient * sinhf(x[input_idx1] * x[input_idx2]); break;
            case 7: term_value = coefficient * x[input_idx1] * sinf(x[input_idx2] * M_PI); break;
        }
        
        if (add_sub == 0) {
            result += term_value;
        } else {
            result -= term_value;
        }
    }
    
    return result;
}

static void print_symbolic_function(int output_idx, int num_terms, const float* coefficients, 
                                  const int* operations, const int* idx1, const int* idx2, 
                                  const int* add_subtract) {
    printf("y%d = ", output_idx);
    
    for (int i = 0; i < num_terms; i++) {
        float coeff = coefficients[i];
        int op = operations[i];
        int in1 = idx1[i];
        int in2 = idx2[i];
        int add_sub = add_subtract[i];
        
        // Print sign
        if (i == 0) {
            if (add_sub == 1) printf("-");
        } else {
            printf(" %s ", (add_sub == 0) ? "+" : "-");
        }
        
        // Print coefficient if not 1.0
        if (fabsf(coeff - 1.0f) > 1e-6) {
            printf("%.3f*", coeff);
        }
        
        // Print operation
        switch (op) {
            case 0: printf("sin(2*x%d)", in1); break;
            case 1: printf("cos(1.5*x%d)", in1); break;
            case 2: printf("tanh(x%d + x%d)", in1, in2); break;
            case 3: printf("exp(-x%d^2)", in1); break;
            case 4: printf("log(|x%d| + 1)", in1); break;
            case 5: printf("x%d^2*x%d", in1, in2); break;
            case 6: printf("sinh(x%d*x%d)", in1, in2); break;
            case 7: printf("x%d*sin(π*x%d)", in1, in2); break;
        }
    }
    printf("\n");
}

void generate_synthetic_data(float** X, float** y, int num_samples, int input_dim, int output_dim, 
                           float input_min, float input_max) {
    // Allocate memory
    *X = (float*)malloc(num_samples * input_dim * sizeof(float));
    *y = (float*)malloc(num_samples * output_dim * sizeof(float));
    
    // Generate random input data
    for (int i = 0; i < num_samples * input_dim; i++) {
        float rand_val = (float)rand() / (float)RAND_MAX;
        (*X)[i] = input_min + rand_val * (input_max - input_min);
    }
    
    // Create function parameters for each output dimension
    int* num_terms_per_output = (int*)malloc(output_dim * sizeof(int));
    float** coefficients = (float**)malloc(output_dim * sizeof(float*));
    int** operations = (int**)malloc(output_dim * sizeof(int*));
    int** idx1 = (int**)malloc(output_dim * sizeof(int*));
    int** idx2 = (int**)malloc(output_dim * sizeof(int*));
    int** add_subtract = (int**)malloc(output_dim * sizeof(int*));
    
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        // Random number of terms between 6 and 12
        int num_terms = 6 + (rand() % 7);
        num_terms_per_output[output_idx] = num_terms;
        
        // Allocate arrays for this function's terms
        coefficients[output_idx] = (float*)malloc(num_terms * sizeof(float));
        operations[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx1[output_idx] = (int*)malloc(num_terms * sizeof(int));
        idx2[output_idx] = (int*)malloc(num_terms * sizeof(int));
        add_subtract[output_idx] = (int*)malloc(num_terms * sizeof(int));

        // Generate random terms
        for (int term = 0; term < num_terms; term++) {
            coefficients[output_idx][term] = 0.1f + 0.4f * ((float)rand() / (float)RAND_MAX);
            operations[output_idx][term] = rand() % 8;
            idx1[output_idx][term] = rand() % input_dim;
            idx2[output_idx][term] = rand() % input_dim;
            add_subtract[output_idx][term] = rand() % 2;
        }
    }
    
    // Print symbolic representation of generated functions
    printf("\nGenerated synthetic functions:\n");
    for (int output_idx = 0; output_idx < output_dim; output_idx++) {
        print_symbolic_function(output_idx, num_terms_per_output[output_idx], 
                              coefficients[output_idx], operations[output_idx], 
                              idx1[output_idx], idx2[output_idx], add_subtract[output_idx]);
    }
    printf("\n");
    
    // Generate output data by evaluating each function
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < output_dim; j++) {
            (*y)[i * output_dim + j] = evaluate_synthetic_function(num_terms_per_output[j], 
                                                        coefficients[j], operations[j], 
                                                        idx1[j], idx2[j], add_subtract[j], 
                                                        &(*X)[i * input_dim]);
        }
    }
    
    // Clean up
    for (int i = 0; i < output_dim; i++) {
        free(coefficients[i]);
        free(operations[i]);
        free(idx1[i]);
        free(idx2[i]);
        free(add_subtract[i]);
    }
    free(num_terms_per_output);
    free(coefficients);
    free(operations);
    free(idx1);
    free(idx2);
    free(add_subtract);
}

void save_data(float* X, float* y, int num_samples, int input_dim, int output_dim, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Write header
    for (int i = 0; i < input_dim; i++) {
        fprintf(file, "x%d,", i);
    }
    for (int i = 0; i < output_dim - 1; i++) {
        fprintf(file, "y%d,", i);
    }
    fprintf(file, "y%d\n", output_dim - 1);
    
    // Write data
    for (int i = 0; i < num_samples; i++) {
        // Input features
        for (int j = 0; j < input_dim; j++) {
            fprintf(file, "%.17f,", X[i * input_dim + j]);
        }
        // Output values
        for (int j = 0; j < output_dim - 1; j++) {
            fprintf(file, "%.17f,", y[i * output_dim + j]);
        }
        fprintf(file, "%.17f\n", y[i * output_dim + output_dim - 1]);
    }
    
    fclose(file);
    printf("Data saved to %s\n", filename);
}
