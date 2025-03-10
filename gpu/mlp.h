#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Device pointers for weights and gradients
    float* d_fc1_weight;     // hidden_dim x input_dim
    float* d_fc2_weight;     // output_dim x hidden_dim
    float* d_fc1_weight_grad; // hidden_dim x input_dim
    float* d_fc2_weight_grad; // output_dim x hidden_dim
    
    // Host copies of weights and error
    float* h_fc1_weight;
    float* h_fc2_weight;
    float* h_error;
    
    // Device pointers for Adam parameters
    float* d_fc1_m;  // First moment for fc1
    float* d_fc1_v;  // Second moment for fc1
    float* d_fc2_m;  // First moment for fc2
    float* d_fc2_v;  // Second moment for fc2
    float beta1;   // Exponential decay rate for first moment
    float beta2;   // Exponential decay rate for second moment
    float epsilon; // Small constant for numerical stability
    int t;         // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Device pointers for helper arrays
    float* d_layer1_output;   // batch_size x hidden_dim
    float* d_predictions;     // batch_size x output_dim
    float* d_error;          // batch_size x output_dim
    float* d_pre_activation; // batch_size x hidden_dim
    float* d_error_hidden;   // batch_size x hidden_dim
    float* d_X;              // batch_size x input_dim
    float* d_y;             // batch_size x output_dim

    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} MLP;

// Initialize the network with configurable dimensions
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    
    // Store dimensions
    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;
    mlp->batch_size = batch_size;
    
    // Initialize Adam parameters
    mlp->beta1 = 0.9f;
    mlp->beta2 = 0.999f;
    mlp->epsilon = 1e-8f;
    mlp->t = 0;
    mlp->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&mlp->cublas_handle));
    
    // Allocate host memory for weights and error
    mlp->h_fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    mlp->h_fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    mlp->h_error = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize weights on host
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        mlp->h_fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        mlp->h_fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&mlp->d_fc1_weight, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_fc2_weight, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_fc1_weight_grad, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_fc2_weight_grad, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&mlp->d_fc1_m, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_fc1_v, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_fc2_m, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_fc2_v, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&mlp->d_layer1_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_predictions, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_pre_activation, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_error_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_y, batch_size * output_dim * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(mlp->d_fc1_weight, mlp->h_fc1_weight, 
                         hidden_dim * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc2_weight, mlp->h_fc2_weight, 
                         output_dim * hidden_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(mlp->d_fc1_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc1_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc2_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc2_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    return mlp;
}

// Free network memory
void free_mlp(MLP* mlp) {
    // Free device memory
    cudaFree(mlp->d_fc1_weight);
    cudaFree(mlp->d_fc2_weight);
    cudaFree(mlp->d_fc1_weight_grad);
    cudaFree(mlp->d_fc2_weight_grad);
    cudaFree(mlp->d_fc1_m);
    cudaFree(mlp->d_fc1_v);
    cudaFree(mlp->d_fc2_m);
    cudaFree(mlp->d_fc2_v);
    cudaFree(mlp->d_layer1_output);
    cudaFree(mlp->d_predictions);
    cudaFree(mlp->d_error);
    cudaFree(mlp->d_pre_activation);
    cudaFree(mlp->d_error_hidden);
    cudaFree(mlp->d_X);
    cudaFree(mlp->d_y);
    
    // Free host memory
    free(mlp->h_fc1_weight);
    free(mlp->h_fc2_weight);
    free(mlp->h_error);
    
    // Destroy cuBLAS handle
    cublasDestroy(mlp->cublas_handle);
    
    free(mlp);
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_mlp(float* output, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = pre_activation[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_mlp(float* error_hidden, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = pre_activation[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        error_hidden[idx] *= sigmoid + x * sigmoid * (1.0f - sigmoid);
    }
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* X) {
    CHECK_CUDA(cudaMemcpy(mlp->d_X, X, mlp->batch_size * mlp->input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // First layer
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            mlp->hidden_dim,    // n
                            mlp->batch_size,    // m
                            mlp->input_dim,     // k
                            &alpha,
                            mlp->d_fc1_weight,  // A
                            mlp->hidden_dim,    // lda
                            mlp->d_X,           // B
                            mlp->input_dim,     // ldb
                            &beta,
                            mlp->d_layer1_output, // C
                            mlp->hidden_dim));    // ldc

    // Store pre-activation values
    CHECK_CUDA(cudaMemcpy(mlp->d_pre_activation, mlp->d_layer1_output,
                         mlp->batch_size * mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Apply Swish activation
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_layer1_output,
        mlp->d_pre_activation,
        mlp->batch_size * mlp->hidden_dim
    );

    // Second layer
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            mlp->output_dim,     // n
                            mlp->batch_size,     // m
                            mlp->hidden_dim,     // k
                            &alpha,
                            mlp->d_fc2_weight,   // A
                            mlp->output_dim,     // lda
                            mlp->d_layer1_output,// B
                            mlp->hidden_dim,     // ldb
                            &beta,
                            mlp->d_predictions,  // C
                            mlp->output_dim));   // ldc
}

// Custom kernel for calculating error and squared error
__global__ void calc_error_kernel_mlp(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* y) {
    CHECK_CUDA(cudaMemcpy(mlp->d_y, y, mlp->batch_size * mlp->output_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Calculate error (predictions - y)
    int size = mlp->batch_size * mlp->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;


    calc_error_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_error,
        mlp->d_predictions,
        mlp->d_y,
        size
    );

    // Calculate loss on CPU
    CHECK_CUDA(cudaMemcpy(mlp->h_error, mlp->d_error, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        loss += mlp->h_error[i] * mlp->h_error[i];
    }

    return loss / size;
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    CHECK_CUDA(cudaMemset(mlp->d_fc1_weight_grad, 0, 
                         mlp->hidden_dim * mlp->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc2_weight_grad, 0, 
                         mlp->output_dim * mlp->hidden_dim * sizeof(float)));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* X) {
    CHECK_CUDA(cudaMemcpy(mlp->d_X, X, mlp->batch_size * mlp->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Gradient of second layer
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            mlp->output_dim,     // n
                            mlp->hidden_dim,     // m
                            mlp->batch_size,     // k
                            &alpha,
                            mlp->d_error,        // A
                            mlp->output_dim,     // lda
                            mlp->d_layer1_output,// B
                            mlp->hidden_dim,     // ldb
                            &beta,
                            mlp->d_fc2_weight_grad, // C
                            mlp->output_dim));   // ldc

    // Backpropagate error through second layer
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            mlp->hidden_dim,     // n
                            mlp->batch_size,     // m
                            mlp->output_dim,     // k
                            &alpha,
                            mlp->d_fc2_weight,   // A
                            mlp->output_dim,     // lda
                            mlp->d_error,        // B
                            mlp->output_dim,     // ldb
                            &beta,
                            mlp->d_error_hidden, // C
                            mlp->hidden_dim));   // ldc

    // Apply Swish derivative
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_error_hidden,
        mlp->d_pre_activation,
        mlp->batch_size * mlp->hidden_dim
    );

    // Gradient of first layer
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            mlp->hidden_dim,     // n
                            mlp->input_dim,      // m
                            mlp->batch_size,     // k
                            &alpha,
                            mlp->d_error_hidden, // A
                            mlp->hidden_dim,     // lda
                            mlp->d_X,            // B
                            mlp->input_dim,      // ldb
                            &beta,
                            mlp->d_fc1_weight_grad, // C
                            mlp->hidden_dim));   // ldc
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_mlp(
    float* weight,
    float* grad,
    float* m,
    float* v,
    float beta1,
    float beta2,
    float epsilon,
    float learning_rate,
    float weight_decay,
    float alpha_t,
    int size,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_mlp(MLP* mlp, float learning_rate) {
    mlp->t++;
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update fc1 weights
    int fc1_size = mlp->hidden_dim * mlp->input_dim;
    int fc1_blocks = (fc1_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<fc1_blocks, block_size>>>(
        mlp->d_fc1_weight,
        mlp->d_fc1_weight_grad,
        mlp->d_fc1_m,
        mlp->d_fc1_v,
        mlp->beta1,
        mlp->beta2,
        mlp->epsilon,
        learning_rate,
        mlp->weight_decay,
        alpha_t,
        fc1_size,
        mlp->batch_size
    );
    
    // Update fc2 weights
    int fc2_size = mlp->output_dim * mlp->hidden_dim;
    int fc2_blocks = (fc2_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<fc2_blocks, block_size>>>(
        mlp->d_fc2_weight,
        mlp->d_fc2_weight_grad,
        mlp->d_fc2_m,
        mlp->d_fc2_v,
        mlp->beta1,
        mlp->beta2,
        mlp->epsilon,
        learning_rate,
        mlp->weight_decay,
        alpha_t,
        fc2_size,
        mlp->batch_size
    );
}

// Save model weights to binary file
void save_mlp(MLP* mlp, const char* filename) {
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(mlp->h_fc1_weight, mlp->d_fc1_weight,
                         mlp->hidden_dim * mlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(mlp->h_fc2_weight, mlp->d_fc2_weight,
                         mlp->output_dim * mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&mlp->input_dim, sizeof(int), 1, file);
    fwrite(&mlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&mlp->output_dim, sizeof(int), 1, file);
    fwrite(&mlp->batch_size, sizeof(int), 1, file);
    fwrite(mlp->h_fc1_weight, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->h_fc2_weight, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(&mlp->t, sizeof(int), 1, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
MLP* load_mlp(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    fread(mlp->h_fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->h_fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    fread(&mlp->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_fc1_weight, mlp->h_fc1_weight,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc2_weight, mlp->h_fc2_weight,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}

#endif