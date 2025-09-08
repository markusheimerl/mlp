#include "mlp.h"

// Initialize the MLP
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
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
    
    // Initialize cuBLAS and cuBLASLt
    mlp->cublas_handle = cublas_handle;
    mlp->cublaslt_handle = cublaslt_handle;
    
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    int hidden_buffer_size = batch_size * hidden_dim;
    int output_buffer_size = batch_size * output_dim;
    
    // Allocate host memory for weight initialization
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    // Initialize weights on host
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    
    for (int i = 0; i < w1_size; i++) {
        h_W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < w2_size; i++) {
        h_W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&mlp->d_W1, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_grad, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_grad, w2_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_m, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_v, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_m, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_v, w2_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&mlp->d_layer_preact, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_layer_postact, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_layer_output, output_buffer_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&mlp->d_error_hidden, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_error_output, output_buffer_size * sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(mlp->d_W1_m, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_v, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_m, 0, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_v, 0, w2_size * sizeof(float)));
    
    // Create cuBLASLt matrix multiplication descriptors
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&mlp->forward_matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&mlp->backward_matmul_NT_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&mlp->backward_matmul_TN_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // Set transpose operations for backward pass descriptors
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(mlp->backward_matmul_NT_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(mlp->backward_matmul_NT_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    transA = CUBLAS_OP_T;
    transB = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(mlp->backward_matmul_TN_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(mlp->backward_matmul_TN_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors for forward pass
    // W1: [input_dim x hidden_dim], X: [batch_size x input_dim], H: [batch_size x hidden_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W1_layout, CUDA_R_32F, input_dim, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W1_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->X_layout, CUDA_R_32F, batch_size, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->X_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->H_layout, CUDA_R_32F, batch_size, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->H_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // W2: [hidden_dim x output_dim], S: [batch_size x hidden_dim], Y: [batch_size x output_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W2_layout, CUDA_R_32F, hidden_dim, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W2_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->S_layout, CUDA_R_32F, batch_size, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->S_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->Y_layout, CUDA_R_32F, batch_size, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->Y_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Gradient layouts
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W1_grad_layout, CUDA_R_32F, input_dim, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W1_grad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W2_grad_layout, CUDA_R_32F, hidden_dim, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W2_grad_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Create matrix layout descriptors for backward pass
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->error_output_layout, CUDA_R_32F, batch_size, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->error_output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->postact_layout, CUDA_R_32F, batch_size, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->postact_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->error_hidden_layout, CUDA_R_32F, batch_size, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->error_hidden_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->X_backward_layout, CUDA_R_32F, batch_size, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->X_backward_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->grad_X_layout, CUDA_R_32F, batch_size, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->grad_X_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_W1); free(h_W2);
    
    return mlp;
}

// Free MLP memory
void free_mlp(MLP* mlp) {
    // Destroy cuBLASLt descriptors
    cublasLtMatmulDescDestroy(mlp->forward_matmul_desc);
    cublasLtMatmulDescDestroy(mlp->backward_matmul_NT_desc);
    cublasLtMatmulDescDestroy(mlp->backward_matmul_TN_desc);
    
    // Destroy forward pass layouts
    cublasLtMatrixLayoutDestroy(mlp->W1_layout);
    cublasLtMatrixLayoutDestroy(mlp->X_layout);
    cublasLtMatrixLayoutDestroy(mlp->H_layout);
    cublasLtMatrixLayoutDestroy(mlp->W2_layout);
    cublasLtMatrixLayoutDestroy(mlp->S_layout);
    cublasLtMatrixLayoutDestroy(mlp->Y_layout);
    cublasLtMatrixLayoutDestroy(mlp->W1_grad_layout);
    cublasLtMatrixLayoutDestroy(mlp->W2_grad_layout);
    
    // Destroy backward pass layouts
    cublasLtMatrixLayoutDestroy(mlp->error_output_layout);
    cublasLtMatrixLayoutDestroy(mlp->postact_layout);
    cublasLtMatrixLayoutDestroy(mlp->error_hidden_layout);
    cublasLtMatrixLayoutDestroy(mlp->X_backward_layout);
    cublasLtMatrixLayoutDestroy(mlp->grad_X_layout);
    
    // Free device memory
    cudaFree(mlp->d_W1); cudaFree(mlp->d_W2);
    cudaFree(mlp->d_W1_grad); cudaFree(mlp->d_W2_grad);
    cudaFree(mlp->d_W1_m); cudaFree(mlp->d_W1_v);
    cudaFree(mlp->d_W2_m); cudaFree(mlp->d_W2_v);
    cudaFree(mlp->d_layer_preact); cudaFree(mlp->d_layer_postact);
    cudaFree(mlp->d_layer_output);
    cudaFree(mlp->d_error_hidden); cudaFree(mlp->d_error_output);
    free(mlp);
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_mlp(float* output, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = pre_activation[idx];
        output[idx] = h / (1.0f + expf(-h));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_mlp(float* error_hidden, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = pre_activation[idx];
        float sigmoid = 1.0f / (1.0f + expf(-h));
        error_hidden[idx] *= sigmoid + h * sigmoid * (1.0f - sigmoid);
    }
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // H = XW₁
    CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle,
                                  mlp->forward_matmul_desc,
                                  &alpha,
                                  d_X, mlp->X_layout,
                                  mlp->d_W1, mlp->W1_layout,
                                  &beta,
                                  mlp->d_layer_preact, mlp->H_layout,
                                  mlp->d_layer_preact, mlp->H_layout,
                                  NULL, NULL, 0, 0));

    // S = H⊙σ(H)
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_layer_postact,
        mlp->d_layer_preact,
        mlp->batch_size * mlp->hidden_dim
    );

    // Y = SW₂
    CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle,
                                  mlp->forward_matmul_desc,
                                  &alpha,
                                  mlp->d_layer_postact, mlp->S_layout,
                                  mlp->d_W2, mlp->W2_layout,
                                  &beta,
                                  mlp->d_layer_output, mlp->Y_layout,
                                  mlp->d_layer_output, mlp->Y_layout,
                                  NULL, NULL, 0, 0));
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(mlp->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            mlp->output_dim, mlp->batch_size,
                            &alpha, mlp->d_layer_output, mlp->output_dim,
                            &beta, d_y, mlp->output_dim,
                            mlp->d_error_output, mlp->output_dim));
    CHECK_CUBLAS(cublasSdot(mlp->cublas_handle, (mlp->batch_size * mlp->output_dim), 
                           mlp->d_error_output, 1, mlp->d_error_output, 1, &loss));
    
    return loss / (mlp->batch_size * mlp->output_dim);
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    CHECK_CUDA(cudaMemset(mlp->d_W1_grad, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_grad, 0, w2_size * sizeof(float)));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* d_X, float* d_grad_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ∂L/∂W₂ = S^T(∂L/∂Y)
    CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle,
                                  mlp->backward_matmul_TN_desc,
                                  &alpha,
                                  mlp->d_layer_postact, mlp->postact_layout,
                                  mlp->d_error_output, mlp->error_output_layout,
                                  &alpha,
                                  mlp->d_W2_grad, mlp->W2_grad_layout,
                                  mlp->d_W2_grad, mlp->W2_grad_layout,
                                  NULL, NULL, 0, 0));

    // ∂L/∂S = (∂L/∂Y)W₂^T
    CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle,
                                  mlp->backward_matmul_NT_desc,
                                  &alpha,
                                  mlp->d_error_output, mlp->error_output_layout,
                                  mlp->d_W2, mlp->W2_layout,
                                  &beta,
                                  mlp->d_error_hidden, mlp->error_hidden_layout,
                                  mlp->d_error_hidden, mlp->error_hidden_layout,
                                  NULL, NULL, 0, 0));

    // ∂L/∂H = ∂L/∂S⊙[σ(H)+H⊙σ(H)⊙(1-σ(H))]
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_error_hidden,
        mlp->d_layer_preact,
        mlp->batch_size * mlp->hidden_dim
    );

    // ∂L/∂W₁ = X^T(∂L/∂H)
    CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle,
                                  mlp->backward_matmul_TN_desc,
                                  &alpha,
                                  d_X, mlp->X_backward_layout,
                                  mlp->d_error_hidden, mlp->error_hidden_layout,
                                  &alpha,
                                  mlp->d_W1_grad, mlp->W1_grad_layout,
                                  mlp->d_W1_grad, mlp->W1_grad_layout,
                                  NULL, NULL, 0, 0));
    
    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂H)W₁^T
        CHECK_CUBLASLT(cublasLtMatmul(mlp->cublaslt_handle,
                                      mlp->backward_matmul_NT_desc,
                                      &alpha,
                                      mlp->d_error_hidden, mlp->error_hidden_layout,
                                      mlp->d_W1, mlp->W1_layout,
                                      &beta,
                                      d_grad_X, mlp->grad_X_layout,
                                      d_grad_X, mlp->grad_X_layout,
                                      NULL, NULL, 0, 0));
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_mlp(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
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
    
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Update W₁ weights
    int W1_blocks = (w1_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<W1_blocks, block_size>>>(
        mlp->d_W1, mlp->d_W1_grad, mlp->d_W1_m, mlp->d_W1_v,
        mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
        alpha_t, w1_size, mlp->batch_size
    );
    
    // Update W₂ weights
    int W2_blocks = (w2_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<W2_blocks, block_size>>>(
        mlp->d_W2, mlp->d_W2_grad, mlp->d_W2_m, mlp->d_W2_v,
        mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
        alpha_t, w2_size, mlp->batch_size
    );
}

// Save MLP weights to binary file
void save_mlp(MLP* mlp, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&mlp->input_dim, sizeof(int), 1, file);
    fwrite(&mlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&mlp->output_dim, sizeof(int), 1, file);
    fwrite(&mlp->batch_size, sizeof(int), 1, file);
    
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Allocate temporary host memory for weights
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(h_W1, mlp->d_W1, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, mlp->d_W2, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W1, sizeof(float), w1_size, file);
    fwrite(h_W2, sizeof(float), w2_size, file);
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_v = (float*)malloc(w1_size * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_v = (float*)malloc(w2_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_W1_m, mlp->d_W1_m, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W1_v, mlp->d_W1_v, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_m, mlp->d_W2_m, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_v, mlp->d_W2_v, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W1_m, sizeof(float), w1_size, file);
    fwrite(h_W1_v, sizeof(float), w1_size, file);
    fwrite(h_W2_m, sizeof(float), w2_size, file);
    fwrite(h_W2_v, sizeof(float), w2_size, file);
    
    // Free temporary host memory
    free(h_W1); free(h_W2);
    free(h_W1_m); free(h_W1_v);
    free(h_W2_m); free(h_W2_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load MLP weights from binary file
MLP* load_mlp(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublas_handle, cublaslt_handle);
    
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    
    // Load weights
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    fread(h_W1, sizeof(float), w1_size, file);
    fread(h_W2, sizeof(float), w2_size, file);
    
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_v = (float*)malloc(w1_size * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_v = (float*)malloc(w2_size * sizeof(float));
    
    fread(h_W1_m, sizeof(float), w1_size, file);
    fread(h_W1_v, sizeof(float), w1_size, file);
    fread(h_W2_m, sizeof(float), w2_size, file);
    fread(h_W2_v, sizeof(float), w2_size, file);
    
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_m, h_W1_m, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_v, h_W1_v, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_m, h_W2_m, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_v, h_W2_v, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_W1); free(h_W2);
    free(h_W1_m); free(h_W1_v);
    free(h_W2_m); free(h_W2_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}