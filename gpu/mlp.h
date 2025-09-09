#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublasLt.h>
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

// cuBLASLt Error checking macro
#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Weights and gradients
    float* d_W1;      // [input_dim x hidden_dim]
    float* d_W2;      // [hidden_dim x output_dim]
    float* d_W1_grad; // [input_dim x hidden_dim]
    float* d_W2_grad; // [hidden_dim x output_dim]
    
    // Adam parameters
    float* d_W1_m;    // First moment for W1
    float* d_W1_v;    // Second moment for W1
    float* d_W2_m;    // First moment for W2
    float* d_W2_v;    // Second moment for W2
    float beta1;      // Exponential decay rate for first moment
    float beta2;      // Exponential decay rate for second moment
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Layer outputs and working buffers
    float* d_layer_preact;  // [batch_size x hidden_dim]
    float* d_layer_postact; // [batch_size x hidden_dim]
    float* d_layer_output;  // [batch_size x output_dim]
    float* d_grad_hidden;   // [batch_size x hidden_dim]
    float* d_grad_output;   // [batch_size x output_dim]

    // Loss computation buffer
    float* d_loss_result;   // [1]

    // cuBLASLt handle
    cublasLtHandle_t cublaslt_handle;
    
    // cuBLASLt descriptors for forward pass
    cublasLtMatmulDesc_t forward_matmul_desc;
    cublasLtMatrixLayout_t W1_layout, X_layout, H_layout;
    cublasLtMatrixLayout_t W2_layout, S_layout, Y_layout;
    cublasLtMatrixLayout_t W1_grad_layout, W2_grad_layout;
    
    // cuBLASLt descriptors for backward pass
    cublasLtMatmulDesc_t backward_matmul_NT_desc;  // No transpose A, transpose B
    cublasLtMatmulDesc_t backward_matmul_TN_desc;  // Transpose A, no transpose B
    cublasLtMatrixLayout_t grad_output_layout;
    cublasLtMatrixLayout_t postact_layout;
    cublasLtMatrixLayout_t grad_hidden_layout;
    cublasLtMatrixLayout_t X_backward_layout;
    cublasLtMatrixLayout_t grad_X_layout;
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} MLP;

// Function prototypes
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, cublasLtHandle_t cublaslt_handle);
void free_mlp(MLP* mlp);
void forward_pass_mlp(MLP* mlp, float* d_X);
float calculate_loss_mlp(MLP* mlp, float* d_y);
void zero_gradients_mlp(MLP* mlp);
void backward_pass_mlp(MLP* mlp, float* d_X, float* d_grad_X);
void update_weights_mlp(MLP* mlp, float learning_rate);
void save_mlp(MLP* mlp, const char* filename);
MLP* load_mlp(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle);

#endif