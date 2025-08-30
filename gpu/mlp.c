#include "mlp.h"

// Initialize the network with configurable dimensions
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int num_layers, int batch_size, cublasHandle_t cublas_handle) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    
    // Store dimensions
    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;
    mlp->num_layers = num_layers;
    mlp->batch_size = batch_size;
    
    // Initialize Adam parameters
    mlp->beta1 = 0.9f;
    mlp->beta2 = 0.999f;
    mlp->epsilon = 1e-8f;
    mlp->t = 0;
    mlp->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    mlp->cublas_handle = cublas_handle;
    
    // Allocate arrays of device pointers
    mlp->d_W1 = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W2 = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W3 = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W1_grad = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W2_grad = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W3_grad = (float**)malloc(num_layers * sizeof(float*));
    
    mlp->d_W1_m = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W1_v = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W2_m = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W2_v = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W3_m = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_W3_v = (float**)malloc(num_layers * sizeof(float*));
    
    mlp->d_layer_preact = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_layer_postact = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_layer_output = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_error_hidden = (float**)malloc(num_layers * sizeof(float*));
    mlp->d_error_output = (float**)malloc(num_layers * sizeof(float*));
    
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : hidden_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : hidden_dim;
        
        int w1_size = hidden_dim * input_size;
        int w2_size = output_size * hidden_dim;
        int w3_size = output_size * input_size;
        
        // Allocate host memory for weight initialization
        float* W1 = (float*)malloc(w1_size * sizeof(float));
        float* W2 = (float*)malloc(w2_size * sizeof(float));
        float* W3 = (float*)malloc(w3_size * sizeof(float));
        
        // Initialize weights on host
        float scale_W1 = 1.0f / sqrtf(input_size);
        float scale_W2 = 1.0f / sqrtf(hidden_dim);
        float scale_W3 = 1.0f / sqrtf(input_size);
        
        for (int i = 0; i < w1_size; i++) {
            W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
        }
        
        for (int i = 0; i < w2_size; i++) {
            W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
        }
        
        for (int i = 0; i < w3_size; i++) {
            W3[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W3;
        }
        
        // Allocate device memory for weights and gradients
        CHECK_CUDA(cudaMalloc(&mlp->d_W1[layer], w1_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W2[layer], w2_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W3[layer], w3_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W1_grad[layer], w1_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W2_grad[layer], w2_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W3_grad[layer], w3_size * sizeof(float)));
        
        // Allocate device memory for Adam parameters
        CHECK_CUDA(cudaMalloc(&mlp->d_W1_m[layer], w1_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W1_v[layer], w1_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W2_m[layer], w2_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W2_v[layer], w2_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W3_m[layer], w3_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_W3_v[layer], w3_size * sizeof(float)));
        
        // Allocate device memory for layer outputs and working buffers
        CHECK_CUDA(cudaMalloc(&mlp->d_layer_preact[layer], batch_size * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_layer_postact[layer], batch_size * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_layer_output[layer], batch_size * output_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_error_hidden[layer], batch_size * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mlp->d_error_output[layer], batch_size * output_size * sizeof(float)));
        
        // Initialize device memory
        CHECK_CUDA(cudaMemcpy(mlp->d_W1[layer], W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W2[layer], W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W3[layer], W3, w3_size * sizeof(float), cudaMemcpyHostToDevice));
        
        CHECK_CUDA(cudaMemset(mlp->d_W1_m[layer], 0, w1_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W1_v[layer], 0, w1_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W2_m[layer], 0, w2_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W2_v[layer], 0, w2_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W3_m[layer], 0, w3_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W3_v[layer], 0, w3_size * sizeof(float)));
        
        // Free local host memory
        free(W1); free(W2); free(W3);
    }
    
    return mlp;
}

// Free network memory
void free_mlp(MLP* mlp) {
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        // Free device memory
        cudaFree(mlp->d_W1[layer]); cudaFree(mlp->d_W2[layer]); cudaFree(mlp->d_W3[layer]);
        cudaFree(mlp->d_W1_grad[layer]); cudaFree(mlp->d_W2_grad[layer]); cudaFree(mlp->d_W3_grad[layer]);
        cudaFree(mlp->d_W1_m[layer]); cudaFree(mlp->d_W1_v[layer]);
        cudaFree(mlp->d_W2_m[layer]); cudaFree(mlp->d_W2_v[layer]);
        cudaFree(mlp->d_W3_m[layer]); cudaFree(mlp->d_W3_v[layer]);
        cudaFree(mlp->d_layer_preact[layer]); cudaFree(mlp->d_layer_postact[layer]); 
        cudaFree(mlp->d_layer_output[layer]);
        cudaFree(mlp->d_error_output[layer]); cudaFree(mlp->d_error_hidden[layer]);
    }
    
    free(mlp->d_W1); free(mlp->d_W2); free(mlp->d_W3);
    free(mlp->d_W1_grad); free(mlp->d_W2_grad); free(mlp->d_W3_grad);
    free(mlp->d_W1_m); free(mlp->d_W1_v);
    free(mlp->d_W2_m); free(mlp->d_W2_v);
    free(mlp->d_W3_m); free(mlp->d_W3_v);
    free(mlp->d_layer_preact); free(mlp->d_layer_postact); free(mlp->d_layer_output);
    free(mlp->d_error_output); free(mlp->d_error_hidden);
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
    
    float* input = d_X;
    
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        // H = XW₁
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                mlp->hidden_dim, mlp->batch_size, input_size,
                                &alpha, mlp->d_W1[layer], input_size,
                                input, input_size,
                                &beta, mlp->d_layer_preact[layer], mlp->hidden_dim));

        // S = Hσ(H)
        int block_size = 256;
        int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
        swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
            mlp->d_layer_postact[layer],
            mlp->d_layer_preact[layer],
            mlp->batch_size * mlp->hidden_dim
        );

        // Y = SW₂
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                output_size, mlp->batch_size, mlp->hidden_dim,
                                &alpha, mlp->d_W2[layer], mlp->hidden_dim,
                                mlp->d_layer_postact[layer], mlp->hidden_dim,
                                &beta, mlp->d_layer_output[layer], output_size));

        // Y = Y + XW₃
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                output_size, mlp->batch_size, input_size,
                                &alpha, mlp->d_W3[layer], input_size,
                                input, input_size,
                                &alpha, mlp->d_layer_output[layer], output_size));
        
        // Set input for next layer
        if (layer < mlp->num_layers - 1) {
            input = mlp->d_layer_output[layer];
        }
    }
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    int last_layer = mlp->num_layers - 1;
    float loss = 0.0f;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(mlp->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            mlp->output_dim, mlp->batch_size,
                            &alpha, mlp->d_layer_output[last_layer], mlp->output_dim,
                            &beta, d_y, mlp->output_dim,
                            mlp->d_error_output[last_layer], mlp->output_dim));
    CHECK_CUBLAS(cublasSdot(mlp->cublas_handle, (mlp->batch_size * mlp->output_dim), 
                           mlp->d_error_output[last_layer], 1, mlp->d_error_output[last_layer], 1, &loss));
    
    return loss / (mlp->batch_size * mlp->output_dim);
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        CHECK_CUDA(cudaMemset(mlp->d_W1_grad[layer], 0, w1_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W2_grad[layer], 0, w2_size * sizeof(float)));
        CHECK_CUDA(cudaMemset(mlp->d_W3_grad[layer], 0, w3_size * sizeof(float)));
    }
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    for (int layer = mlp->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? d_X : mlp->d_layer_output[layer - 1];
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;

        // ∂L/∂W₂ = S^T(∂L/∂Y)
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                mlp->hidden_dim, output_size, mlp->batch_size,
                                &alpha, mlp->d_layer_postact[layer], mlp->hidden_dim,
                                mlp->d_error_output[layer], output_size,
                                &alpha, mlp->d_W2_grad[layer], mlp->hidden_dim));

        // ∂L/∂W₃ = X^T(∂L/∂Y)
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                input_size, output_size, mlp->batch_size,
                                &alpha, input, input_size,
                                mlp->d_error_output[layer], output_size,
                                &alpha, mlp->d_W3_grad[layer], input_size));

        // ∂L/∂S = (∂L/∂Y)(W₂)^T
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                mlp->hidden_dim, mlp->batch_size, output_size,
                                &alpha, mlp->d_W2[layer], mlp->hidden_dim,
                                mlp->d_error_output[layer], output_size,
                                &beta, mlp->d_error_hidden[layer], mlp->hidden_dim));

        // ∂L/∂H = ∂L/∂S ⊙ [σ(H) + Hσ(H)(1-σ(H))]
        int block_size = 256;
        int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
        swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
            mlp->d_error_hidden[layer],
            mlp->d_layer_preact[layer],
            mlp->batch_size * mlp->hidden_dim
        );

        // ∂L/∂W₁ = X^T(∂L/∂H)
        CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                input_size, mlp->hidden_dim, mlp->batch_size,
                                &alpha, input, input_size,
                                mlp->d_error_hidden[layer], mlp->hidden_dim,
                                &alpha, mlp->d_W1_grad[layer], input_size));

        // Propagate error to previous layer
        if (layer > 0) {
            // ∂L/∂X = (∂L/∂H)(W₁)^T + (∂L/∂Y)(W₃)^T
            CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    input_size, mlp->batch_size, mlp->hidden_dim,
                                    &alpha, mlp->d_W1[layer], input_size,
                                    mlp->d_error_hidden[layer], mlp->hidden_dim,
                                    &beta, mlp->d_error_output[layer - 1], input_size));
            
            CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    input_size, mlp->batch_size, output_size,
                                    &alpha, mlp->d_W3[layer], input_size,
                                    mlp->d_error_output[layer], output_size,
                                    &alpha, mlp->d_error_output[layer - 1], input_size));
        }
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
    
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        // Update W1 weights
        int W1_blocks = (w1_size + block_size - 1) / block_size;
        adamw_update_kernel_mlp<<<W1_blocks, block_size>>>(
            mlp->d_W1[layer], mlp->d_W1_grad[layer], mlp->d_W1_m[layer], mlp->d_W1_v[layer],
            mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
            alpha_t, w1_size, mlp->batch_size
        );
        
        // Update W2 weights
        int W2_blocks = (w2_size + block_size - 1) / block_size;
        adamw_update_kernel_mlp<<<W2_blocks, block_size>>>(
            mlp->d_W2[layer], mlp->d_W2_grad[layer], mlp->d_W2_m[layer], mlp->d_W2_v[layer],
            mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
            alpha_t, w2_size, mlp->batch_size
        );
        
        // Update W3 weights
        int W3_blocks = (w3_size + block_size - 1) / block_size;
        adamw_update_kernel_mlp<<<W3_blocks, block_size>>>(
            mlp->d_W3[layer], mlp->d_W3_grad[layer], mlp->d_W3_m[layer], mlp->d_W3_v[layer],
            mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
            alpha_t, w3_size, mlp->batch_size
        );
    }
}

// Save model weights to binary file
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
    fwrite(&mlp->num_layers, sizeof(int), 1, file);
    fwrite(&mlp->batch_size, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        // Allocate temporary host memory for weights
        float* W1 = (float*)malloc(w1_size * sizeof(float));
        float* W2 = (float*)malloc(w2_size * sizeof(float));
        float* W3 = (float*)malloc(w3_size * sizeof(float));
        
        // Copy weights from device to host
        CHECK_CUDA(cudaMemcpy(W1, mlp->d_W1[layer], w1_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W2, mlp->d_W2[layer], w2_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W3, mlp->d_W3[layer], w3_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        fwrite(W1, sizeof(float), w1_size, file);
        fwrite(W2, sizeof(float), w2_size, file);
        fwrite(W3, sizeof(float), w3_size, file);
        
        // Free temporary host memory
        free(W1); free(W2); free(W3);
    }
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        // Allocate temporary host memory for Adam state
        float* W1_m = (float*)malloc(w1_size * sizeof(float));
        float* W1_v = (float*)malloc(w1_size * sizeof(float));
        float* W2_m = (float*)malloc(w2_size * sizeof(float));
        float* W2_v = (float*)malloc(w2_size * sizeof(float));
        float* W3_m = (float*)malloc(w3_size * sizeof(float));
        float* W3_v = (float*)malloc(w3_size * sizeof(float));
        
        CHECK_CUDA(cudaMemcpy(W1_m, mlp->d_W1_m[layer], w1_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W1_v, mlp->d_W1_v[layer], w1_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W2_m, mlp->d_W2_m[layer], w2_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W2_v, mlp->d_W2_v[layer], w2_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W3_m, mlp->d_W3_m[layer], w3_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(W3_v, mlp->d_W3_v[layer], w3_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        fwrite(W1_m, sizeof(float), w1_size, file);
        fwrite(W1_v, sizeof(float), w1_size, file);
        fwrite(W2_m, sizeof(float), w2_size, file);
        fwrite(W2_v, sizeof(float), w2_size, file);
        fwrite(W3_m, sizeof(float), w3_size, file);
        fwrite(W3_v, sizeof(float), w3_size, file);
        
        // Free temporary host memory
        free(W1_m); free(W1_v);
        free(W2_m); free(W2_v);
        free(W3_m); free(W3_v);
    }

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
MLP* load_mlp(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, num_layers, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, num_layers, batch_size, cublas_handle);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : hidden_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : hidden_dim;
        
        int w1_size = hidden_dim * input_size;
        int w2_size = output_size * hidden_dim;
        int w3_size = output_size * input_size;
        
        // Allocate temporary host memory for weights
        float* W1 = (float*)malloc(w1_size * sizeof(float));
        float* W2 = (float*)malloc(w2_size * sizeof(float));
        float* W3 = (float*)malloc(w3_size * sizeof(float));
        
        fread(W1, sizeof(float), w1_size, file);
        fread(W2, sizeof(float), w2_size, file);
        fread(W3, sizeof(float), w3_size, file);
        
        // Copy weights to device
        CHECK_CUDA(cudaMemcpy(mlp->d_W1[layer], W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W2[layer], W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W3[layer], W3, w3_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Free temporary host memory
        free(W1); free(W2); free(W3);
    }
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : hidden_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : hidden_dim;
        
        int w1_size = hidden_dim * input_size;
        int w2_size = output_size * hidden_dim;
        int w3_size = output_size * input_size;
        
        // Load Adam state variables
        float* W1_m = (float*)malloc(w1_size * sizeof(float));
        float* W1_v = (float*)malloc(w1_size * sizeof(float));
        float* W2_m = (float*)malloc(w2_size * sizeof(float));
        float* W2_v = (float*)malloc(w2_size * sizeof(float));
        float* W3_m = (float*)malloc(w3_size * sizeof(float));
        float* W3_v = (float*)malloc(w3_size * sizeof(float));
        
        fread(W1_m, sizeof(float), w1_size, file);
        fread(W1_v, sizeof(float), w1_size, file);
        fread(W2_m, sizeof(float), w2_size, file);
        fread(W2_v, sizeof(float), w2_size, file);
        fread(W3_m, sizeof(float), w3_size, file);
        fread(W3_v, sizeof(float), w3_size, file);
        
        // Copy Adam state to device
        CHECK_CUDA(cudaMemcpy(mlp->d_W1_m[layer], W1_m, w1_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W1_v[layer], W1_v, w1_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W2_m[layer], W2_m, w2_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W2_v[layer], W2_v, w2_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W3_m[layer], W3_m, w3_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mlp->d_W3_v[layer], W3_v, w3_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Free temporary host memory
        free(W1_m); free(W1_v);
        free(W2_m); free(W2_v);
        free(W3_m); free(W3_v);
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}