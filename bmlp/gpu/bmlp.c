#include "bmlp.h"

// Initialize the network with configurable dimensions
BMLP* init_bmlp(int input_dim, int hidden_dim, int output_dim, int batch_size, cublasHandle_t cublas_handle) {
    BMLP* bmlp = (BMLP*)malloc(sizeof(BMLP));
    
    // Store dimensions
    bmlp->input_dim = input_dim;
    bmlp->hidden_dim = hidden_dim;
    bmlp->output_dim = output_dim;
    bmlp->batch_size = batch_size;
    
    // Initialize Adam parameters
    bmlp->beta1 = 0.9f;
    bmlp->beta2 = 0.999f;
    bmlp->epsilon = 1e-8f;
    bmlp->t = 0;
    bmlp->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    bmlp->cublas_handle = cublas_handle;
    
    // Allocate host memory for weights
    float* W1 = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* W2 = (float*)malloc(output_dim * hidden_dim * hidden_dim * sizeof(float));
    float* W3 = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Initialize weights on host
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim * hidden_dim);
    float scale_W3 = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim * hidden_dim; i++) {
        W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        W3[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W3;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&bmlp->d_W1, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W2, output_dim * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W3, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W1_grad, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W2_grad, output_dim * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W3_grad, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&bmlp->d_W1_m, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W1_v, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W2_m, output_dim * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W2_v, output_dim * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W3_m, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_W3_v, output_dim * input_dim * sizeof(float)));
    
    // Allocate device memory for layer outputs and working buffers
    CHECK_CUDA(cudaMalloc(&bmlp->d_layer1_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_layer2_output, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_error_hidden, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_error_output, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_outer_product, batch_size * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&bmlp->d_outer_grad, batch_size * hidden_dim * hidden_dim * sizeof(float)));
    
    // Initialize device memory
    CHECK_CUDA(cudaMemcpy(bmlp->d_W1, W1, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W2, W2, output_dim * hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W3, W3, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(bmlp->d_W1_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W1_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W2_m, 0, output_dim * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W2_v, 0, output_dim * hidden_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W3_m, 0, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W3_v, 0, output_dim * input_dim * sizeof(float)));
    
    // Free local host memory
    free(W1); free(W2); free(W3);
    
    return bmlp;
}

// Free network memory
void free_bmlp(BMLP* bmlp) {
    // Free device memory
    cudaFree(bmlp->d_W1); cudaFree(bmlp->d_W2); cudaFree(bmlp->d_W3);
    cudaFree(bmlp->d_W1_grad); cudaFree(bmlp->d_W2_grad); cudaFree(bmlp->d_W3_grad);
    cudaFree(bmlp->d_W1_m); cudaFree(bmlp->d_W1_v);
    cudaFree(bmlp->d_W2_m); cudaFree(bmlp->d_W2_v);
    cudaFree(bmlp->d_W3_m); cudaFree(bmlp->d_W3_v);
    cudaFree(bmlp->d_layer1_output); cudaFree(bmlp->d_layer2_output);
    cudaFree(bmlp->d_error_output); cudaFree(bmlp->d_error_hidden);
    cudaFree(bmlp->d_outer_product); cudaFree(bmlp->d_outer_grad);
    free(bmlp);
}

// Forward pass
void forward_pass_bmlp(BMLP* bmlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // H = XW₁
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            bmlp->hidden_dim, bmlp->batch_size, bmlp->input_dim,
                            &alpha, bmlp->d_W1, bmlp->input_dim,
                            d_X, bmlp->input_dim,
                            &beta, bmlp->d_layer1_output, bmlp->hidden_dim));

    // Compute outer products (H ⊗ H)
    CHECK_CUBLAS(cublasSgemmStridedBatched(bmlp->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_T,
                                          bmlp->hidden_dim, bmlp->hidden_dim, 1,
                                          &alpha,
                                          bmlp->d_layer1_output, bmlp->hidden_dim, bmlp->hidden_dim,
                                          bmlp->d_layer1_output, bmlp->hidden_dim, bmlp->hidden_dim,
                                          &beta,
                                          bmlp->d_outer_product, bmlp->hidden_dim, bmlp->hidden_dim * bmlp->hidden_dim,
                                          bmlp->batch_size));

    // Y = (H ⊗ H) @ W2^T
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            bmlp->output_dim, bmlp->batch_size, bmlp->hidden_dim * bmlp->hidden_dim,
                            &alpha, bmlp->d_W2, bmlp->hidden_dim * bmlp->hidden_dim,
                            bmlp->d_outer_product, bmlp->hidden_dim * bmlp->hidden_dim,
                            &beta, bmlp->d_layer2_output, bmlp->output_dim));

    // Y = Y + XW₃
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            bmlp->output_dim, bmlp->batch_size, bmlp->input_dim,
                            &alpha, bmlp->d_W3, bmlp->input_dim,
                            d_X, bmlp->input_dim,
                            &alpha, bmlp->d_layer2_output, bmlp->output_dim));
}

// Calculate loss
float calculate_loss_bmlp(BMLP* bmlp, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(bmlp->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            bmlp->output_dim, bmlp->batch_size,
                            &alpha, bmlp->d_layer2_output, bmlp->output_dim,
                            &beta, d_y, bmlp->output_dim,
                            bmlp->d_error_output, bmlp->output_dim));
    CHECK_CUBLAS(cublasSdot(bmlp->cublas_handle, (bmlp->batch_size * bmlp->output_dim), bmlp->d_error_output, 1, bmlp->d_error_output, 1, &loss));
    
    return loss / (bmlp->batch_size * bmlp->output_dim);
}

// Zero gradients
void zero_gradients_bmlp(BMLP* bmlp) {
    CHECK_CUDA(cudaMemset(bmlp->d_W1_grad, 0, bmlp->hidden_dim * bmlp->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W2_grad, 0, bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(bmlp->d_W3_grad, 0, bmlp->output_dim * bmlp->input_dim * sizeof(float)));
}

// Backward pass
void backward_pass_bmlp(BMLP* bmlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ∂L/∂W₃ = Xᵀ(∂L/∂Y)
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            bmlp->input_dim, bmlp->output_dim, bmlp->batch_size,
                            &alpha, d_X, bmlp->input_dim,
                            bmlp->d_error_output, bmlp->output_dim,
                            &alpha, bmlp->d_W3_grad, bmlp->input_dim));

    // ∂L/∂W₂ = (∂L/∂Y)ᵀ @ (H ⊗ H)
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            bmlp->output_dim, bmlp->hidden_dim * bmlp->hidden_dim, bmlp->batch_size,
                            &alpha, bmlp->d_error_output, bmlp->output_dim,
                            bmlp->d_outer_product, bmlp->hidden_dim * bmlp->hidden_dim,
                            &alpha, bmlp->d_W2_grad, bmlp->output_dim));

    // Compute gradient w.r.t (H ⊗ H)
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            bmlp->hidden_dim * bmlp->hidden_dim, bmlp->batch_size, bmlp->output_dim,
                            &alpha, bmlp->d_W2, bmlp->hidden_dim * bmlp->hidden_dim,
                            bmlp->d_error_output, bmlp->output_dim,
                            &beta, bmlp->d_outer_grad, bmlp->hidden_dim * bmlp->hidden_dim));

    // ∂L/∂H using chain rule for (H ⊗ H)
    // ∂L/∂(H⊗H) @ H
    CHECK_CUBLAS(cublasSgemmStridedBatched(bmlp->cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          bmlp->hidden_dim, 1, bmlp->hidden_dim,
                                          &alpha,
                                          bmlp->d_outer_grad, bmlp->hidden_dim, bmlp->hidden_dim * bmlp->hidden_dim,
                                          bmlp->d_layer1_output, bmlp->hidden_dim, bmlp->hidden_dim,
                                          &beta,
                                          bmlp->d_error_hidden, bmlp->hidden_dim, bmlp->hidden_dim,
                                          bmlp->batch_size));

    // ∂L/∂(H⊗H)^T @ H
    CHECK_CUBLAS(cublasSgemmStridedBatched(bmlp->cublas_handle,
                                          CUBLAS_OP_T, CUBLAS_OP_N,
                                          bmlp->hidden_dim, 1, bmlp->hidden_dim,
                                          &alpha,
                                          bmlp->d_outer_grad, bmlp->hidden_dim, bmlp->hidden_dim * bmlp->hidden_dim,
                                          bmlp->d_layer1_output, bmlp->hidden_dim, bmlp->hidden_dim,
                                          &alpha,
                                          bmlp->d_error_hidden, bmlp->hidden_dim, bmlp->hidden_dim,
                                          bmlp->batch_size));

    // ∂L/∂W₁ = Xᵀ(∂L/∂H)
    CHECK_CUBLAS(cublasSgemm(bmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            bmlp->input_dim, bmlp->hidden_dim, bmlp->batch_size,
                            &alpha, d_X, bmlp->input_dim,
                            bmlp->d_error_hidden, bmlp->hidden_dim,
                            &alpha, bmlp->d_W1_grad, bmlp->input_dim));
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_bmlp(float* weight, float* grad, float* m, float* v,
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
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_bmlp(BMLP* bmlp, float learning_rate) {
    bmlp->t++;
    
    float beta1_t = powf(bmlp->beta1, bmlp->t);
    float beta2_t = powf(bmlp->beta2, bmlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update W1 weights
    int W1_size = bmlp->hidden_dim * bmlp->input_dim;
    int W1_blocks = (W1_size + block_size - 1) / block_size;
    adamw_update_kernel_bmlp<<<W1_blocks, block_size>>>(
        bmlp->d_W1, bmlp->d_W1_grad, bmlp->d_W1_m, bmlp->d_W1_v,
        bmlp->beta1, bmlp->beta2, bmlp->epsilon, learning_rate, bmlp->weight_decay,
        alpha_t, W1_size, bmlp->batch_size
    );
    
    // Update W2 weights
    int W2_size = bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim;
    int W2_blocks = (W2_size + block_size - 1) / block_size;
    adamw_update_kernel_bmlp<<<W2_blocks, block_size>>>(
        bmlp->d_W2, bmlp->d_W2_grad, bmlp->d_W2_m, bmlp->d_W2_v,
        bmlp->beta1, bmlp->beta2, bmlp->epsilon, learning_rate, bmlp->weight_decay,
        alpha_t, W2_size, bmlp->batch_size
    );
    
    // Update W3 weights
    int W3_size = bmlp->output_dim * bmlp->input_dim;
    int W3_blocks = (W3_size + block_size - 1) / block_size;
    adamw_update_kernel_bmlp<<<W3_blocks, block_size>>>(
        bmlp->d_W3, bmlp->d_W3_grad, bmlp->d_W3_m, bmlp->d_W3_v,
        bmlp->beta1, bmlp->beta2, bmlp->epsilon, learning_rate, bmlp->weight_decay,
        alpha_t, W3_size, bmlp->batch_size
    );
}

// Save model weights to binary file
void save_bmlp(BMLP* bmlp, const char* filename) {
    // Allocate temporary host memory for weights
    float* W1 = (float*)malloc(bmlp->hidden_dim * bmlp->input_dim * sizeof(float));
    float* W2 = (float*)malloc(bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float));
    float* W3 = (float*)malloc(bmlp->output_dim * bmlp->input_dim * sizeof(float));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(W1, bmlp->d_W1, bmlp->hidden_dim * bmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W2, bmlp->d_W2, bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W3, bmlp->d_W3, bmlp->output_dim * bmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        free(W1); free(W2); free(W3);
        return;
    }
    
    fwrite(&bmlp->input_dim, sizeof(int), 1, file);
    fwrite(&bmlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&bmlp->output_dim, sizeof(int), 1, file);
    fwrite(&bmlp->batch_size, sizeof(int), 1, file);
    fwrite(W1, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(W2, sizeof(float), bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim, file);
    fwrite(W3, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);
    fwrite(&bmlp->t, sizeof(int), 1, file);
    
    // Also save Adam state variables
    float* W1_m = (float*)malloc(bmlp->hidden_dim * bmlp->input_dim * sizeof(float));
    float* W1_v = (float*)malloc(bmlp->hidden_dim * bmlp->input_dim * sizeof(float));
    float* W2_m = (float*)malloc(bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float));
    float* W2_v = (float*)malloc(bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float));
    float* W3_m = (float*)malloc(bmlp->output_dim * bmlp->input_dim * sizeof(float));
    float* W3_v = (float*)malloc(bmlp->output_dim * bmlp->input_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(W1_m, bmlp->d_W1_m, bmlp->hidden_dim * bmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W1_v, bmlp->d_W1_v, bmlp->hidden_dim * bmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W2_m, bmlp->d_W2_m, bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W2_v, bmlp->d_W2_v, bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W3_m, bmlp->d_W3_m, bmlp->output_dim * bmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W3_v, bmlp->d_W3_v, bmlp->output_dim * bmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(W1_m, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(W1_v, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(W2_m, sizeof(float), bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim, file);
    fwrite(W2_v, sizeof(float), bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim, file);
    fwrite(W3_m, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);
    fwrite(W3_v, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);
    
    // Free temporary host memory
    free(W1); free(W2); free(W3);
    free(W1_m); free(W1_v);
    free(W2_m); free(W2_v);
    free(W3_m); free(W3_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
BMLP* load_bmlp(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
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
    
    BMLP* bmlp = init_bmlp(input_dim, hidden_dim, output_dim, batch_size, cublas_handle);
    
    // Allocate temporary host memory for weights
    float* W1 = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* W2 = (float*)malloc(output_dim * hidden_dim * hidden_dim * sizeof(float));
    float* W3 = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    fread(W1, sizeof(float), hidden_dim * input_dim, file);
    fread(W2, sizeof(float), output_dim * hidden_dim * hidden_dim, file);
    fread(W3, sizeof(float), output_dim * input_dim, file);
    fread(&bmlp->t, sizeof(int), 1, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(bmlp->d_W1, W1, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W2, W2, output_dim * hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W3, W3, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state variables
    float* W1_m = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* W1_v = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* W2_m = (float*)malloc(output_dim * hidden_dim * hidden_dim * sizeof(float));
    float* W2_v = (float*)malloc(output_dim * hidden_dim * hidden_dim * sizeof(float));
    float* W3_m = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* W3_v = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    fread(W1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(W1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(W2_m, sizeof(float), output_dim * hidden_dim * hidden_dim, file);
    fread(W2_v, sizeof(float), output_dim * hidden_dim * hidden_dim, file);
    fread(W3_m, sizeof(float), output_dim * input_dim, file);
    fread(W3_v, sizeof(float), output_dim * input_dim, file);
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(bmlp->d_W1_m, W1_m, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W1_v, W1_v, hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W2_m, W2_m, output_dim * hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W2_v, W2_v, output_dim * hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W3_m, W3_m, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bmlp->d_W3_v, W3_v, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(W1); free(W2); free(W3);
    free(W1_m); free(W1_v);
    free(W2_m); free(W2_v);
    free(W3_m); free(W3_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return bmlp;
}