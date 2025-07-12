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
    
    // Allocate host memory for weights (local variables)
    float* fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Initialize weights on host
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale2;
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
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(mlp->d_fc1_weight, fc1_weight, 
                         hidden_dim * input_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc2_weight, fc2_weight, 
                         output_dim * hidden_dim * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(mlp->d_fc1_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc1_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc2_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_fc2_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    // Free local host memory
    free(fc1_weight);
    free(fc2_weight);
    
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

// CUDA kernel for calculating error
__global__ void calc_error_kernel_mlp(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Z = XW₁
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            mlp->hidden_dim,
                            mlp->batch_size,
                            mlp->input_dim,
                            &alpha,
                            mlp->d_fc1_weight,
                            mlp->input_dim,
                            d_X,
                            mlp->input_dim,
                            &beta,
                            mlp->d_layer1_output,
                            mlp->hidden_dim));

    // Store pre-activation values for backward pass
    CHECK_CUDA(cudaMemcpy(mlp->d_pre_activation, mlp->d_layer1_output,
                         mlp->batch_size * mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // A = Zσ(Z)
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_layer1_output,
        mlp->d_pre_activation,
        mlp->batch_size * mlp->hidden_dim
    );

    // Y = AW₂
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            mlp->output_dim,
                            mlp->batch_size,
                            mlp->hidden_dim,
                            &alpha,
                            mlp->d_fc2_weight,
                            mlp->hidden_dim,
                            mlp->d_layer1_output,
                            mlp->hidden_dim,
                            &beta,
                            mlp->d_predictions,
                            mlp->output_dim));
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    int size = mlp->batch_size * mlp->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    calc_error_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_error,
        mlp->d_predictions,
        d_y,
        size
    );

    float loss;
    CHECK_CUBLAS(cublasSdot(mlp->cublas_handle, size, mlp->d_error, 1, mlp->d_error, 1, &loss));

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
void backward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ∂L/∂W₂ = Aᵀ(∂L/∂Y)
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            mlp->hidden_dim,
                            mlp->output_dim,
                            mlp->batch_size,
                            &alpha,
                            mlp->d_layer1_output,
                            mlp->hidden_dim,
                            mlp->d_error,
                            mlp->output_dim,
                            &beta,
                            mlp->d_fc2_weight_grad,
                            mlp->hidden_dim));

    // ∂L/∂A = (∂L/∂Y)(W₂)ᵀ
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            mlp->hidden_dim,
                            mlp->batch_size,
                            mlp->output_dim,
                            &alpha,
                            mlp->d_fc2_weight,
                            mlp->hidden_dim,
                            mlp->d_error,
                            mlp->output_dim,
                            &beta,
                            mlp->d_error_hidden,
                            mlp->hidden_dim));

    // ∂L/∂Z = ∂L/∂A ⊙ [σ(Z) + Zσ(Z)(1-σ(Z))]
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_error_hidden,
        mlp->d_pre_activation,
        mlp->batch_size * mlp->hidden_dim
    );

    // ∂L/∂W₁ = Xᵀ(∂L/∂Z)
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_T,
                            mlp->input_dim,
                            mlp->hidden_dim,
                            mlp->batch_size,
                            &alpha,
                            d_X,
                            mlp->input_dim,
                            mlp->d_error_hidden,
                            mlp->hidden_dim,
                            &beta,
                            mlp->d_fc1_weight_grad,
                            mlp->input_dim));
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
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
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
    // Allocate temporary host memory for weights
    float* fc1_weight = (float*)malloc(mlp->hidden_dim * mlp->input_dim * sizeof(float));
    float* fc2_weight = (float*)malloc(mlp->output_dim * mlp->hidden_dim * sizeof(float));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(fc1_weight, mlp->d_fc1_weight,
                         mlp->hidden_dim * mlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fc2_weight, mlp->d_fc2_weight,
                         mlp->output_dim * mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(fc1_weight);
        free(fc2_weight);
        return;
    }
    
    fwrite(&mlp->input_dim, sizeof(int), 1, file);
    fwrite(&mlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&mlp->output_dim, sizeof(int), 1, file);
    fwrite(&mlp->batch_size, sizeof(int), 1, file);
    fwrite(fc1_weight, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(fc2_weight, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(&mlp->t, sizeof(int), 1, file);
    
    // Also save Adam state variables
    float* fc1_m = (float*)malloc(mlp->hidden_dim * mlp->input_dim * sizeof(float));
    float* fc1_v = (float*)malloc(mlp->hidden_dim * mlp->input_dim * sizeof(float));
    float* fc2_m = (float*)malloc(mlp->output_dim * mlp->hidden_dim * sizeof(float));
    float* fc2_v = (float*)malloc(mlp->output_dim * mlp->hidden_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(fc1_m, mlp->d_fc1_m, 
                         mlp->hidden_dim * mlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fc1_v, mlp->d_fc1_v, 
                         mlp->hidden_dim * mlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fc2_m, mlp->d_fc2_m, 
                         mlp->output_dim * mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fc2_v, mlp->d_fc2_v, 
                         mlp->output_dim * mlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    fwrite(fc1_m, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(fc1_v, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(fc2_m, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(fc2_v, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    
    // Free temporary host memory
    free(fc1_weight);
    free(fc2_weight);
    free(fc1_m);
    free(fc1_v);
    free(fc2_m);
    free(fc2_v);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
MLP* load_mlp(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int input_dim, hidden_dim, output_dim, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Allocate temporary host memory for weights
    float* fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    fread(fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    fread(&mlp->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_fc1_weight, fc1_weight,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc2_weight, fc2_weight,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Load Adam state variables
    float* fc1_m = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* fc1_v = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* fc2_m = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    float* fc2_v = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    fread(fc1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(fc1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(fc2_m, sizeof(float), output_dim * hidden_dim, file);
    fread(fc2_v, sizeof(float), output_dim * hidden_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(mlp->d_fc1_m, fc1_m,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc1_v, fc1_v,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc2_m, fc2_m,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_fc2_v, fc2_v,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(fc1_weight);
    free(fc2_weight);
    free(fc1_m);
    free(fc1_v);
    free(fc2_m);
    free(fc2_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}

#endif