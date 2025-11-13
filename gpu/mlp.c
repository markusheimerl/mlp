#include "mlp.h"

// Initialize the MLP
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, cublasLtHandle_t cublaslt_handle) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    
    // Store dimensions
    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;
    mlp->batch_size = batch_size;
    
    // Initialize Adafactor parameters
    mlp->beta1 = 0.9f;
    mlp->beta2 = 0.999f;
    mlp->epsilon = 1e-8f;
    mlp->t = 0;
    mlp->weight_decay = 0.01f;
    
    // Initialize cuBLASLt
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
    
    // Allocate device memory for Adafactor parameters
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_m, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_r, input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_c, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_m, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_r, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_c, output_dim * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&mlp->d_preact, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_postact, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_output, output_buffer_size * sizeof(float)));
    
    // Alias device memory for backward pass buffers
    mlp->d_grad_postact = mlp->d_postact;
    mlp->d_grad_output = mlp->d_output;
    
    // Allocate single device float for loss and row mean computation
    CHECK_CUDA(cudaMalloc(&mlp->d_loss_result, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_row_mean, sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adafactor parameters to zero
    CHECK_CUDA(cudaMemset(mlp->d_W1_m, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_r, 0, input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_c, 0, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_m, 0, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_r, 0, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_c, 0, output_dim * sizeof(float)));
    
    // Create cuBLASLt matrix multiplication descriptor
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&mlp->matmul_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    
    // Row-major layout order
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    
    // Create matrix layout descriptors
    // W1 and W1_grad: [input_dim x hidden_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W1_layout, CUDA_R_32F, input_dim, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W1_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // W2 and W2_grad: [hidden_dim x output_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->W2_layout, CUDA_R_32F, hidden_dim, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->W2_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // X and grad_X: [batch_size x input_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->batch_input_layout, CUDA_R_32F, batch_size, input_dim, input_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->batch_input_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // H, S, grad_hidden, layer_preact, layer_postact: [batch_size x hidden_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->batch_hidden_layout, CUDA_R_32F, batch_size, hidden_dim, hidden_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->batch_hidden_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Y, grad_output, layer_output: [batch_size x output_dim]
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&mlp->batch_output_layout, CUDA_R_32F, batch_size, output_dim, output_dim));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(mlp->batch_output_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    
    // Free host memory
    free(h_W1); free(h_W2);
    
    return mlp;
}

// Free MLP memory
void free_mlp(MLP* mlp) {
    // Destroy cuBLASLt descriptor
    cublasLtMatmulDescDestroy(mlp->matmul_desc);
    
    // Destroy matrix layouts
    cublasLtMatrixLayoutDestroy(mlp->W1_layout);
    cublasLtMatrixLayoutDestroy(mlp->W2_layout);
    cublasLtMatrixLayoutDestroy(mlp->batch_input_layout);
    cublasLtMatrixLayoutDestroy(mlp->batch_hidden_layout);
    cublasLtMatrixLayoutDestroy(mlp->batch_output_layout);
    
    // Free device memory
    cudaFree(mlp->d_W1); cudaFree(mlp->d_W2);
    cudaFree(mlp->d_W1_grad); cudaFree(mlp->d_W2_grad);
    cudaFree(mlp->d_W1_m); cudaFree(mlp->d_W1_r); cudaFree(mlp->d_W1_c);
    cudaFree(mlp->d_W2_m); cudaFree(mlp->d_W2_r); cudaFree(mlp->d_W2_c);
    cudaFree(mlp->d_preact); cudaFree(mlp->d_postact);
    cudaFree(mlp->d_output);
    
    // Free loss computation buffer
    cudaFree(mlp->d_loss_result);
    cudaFree(mlp->d_row_mean);
    
    free(mlp);
}

// CUDA kernel for Swish activation
__global__ static void swish_forward_kernel_mlp(float* output, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = pre_activation[idx];
        output[idx] = h / (1.0f + expf(-h));
    }
}

// CUDA kernel for Swish derivative
__global__ static void swish_backward_kernel_mlp(float* grad_hidden, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = pre_activation[idx];
        float sigmoid = 1.0f / (1.0f + expf(-h));
        grad_hidden[idx] *= sigmoid + h * sigmoid * (1.0f - sigmoid);
    }
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // H = XW₁
    LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              d_X, mlp->batch_input_layout,
              mlp->d_W1, mlp->W1_layout,
              &beta, mlp->d_preact, mlp->batch_hidden_layout);

    // S = H⊙σ(H)
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_postact,
        mlp->d_preact,
        mlp->batch_size * mlp->hidden_dim
    );

    // Y = SW₂
    LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_N, &alpha,
              mlp->d_postact, mlp->batch_hidden_layout,
              mlp->d_W2, mlp->W2_layout,
              &beta, mlp->d_output, mlp->batch_output_layout);
}

// CUDA kernel for computing loss and gradient
__global__ static void compute_loss_and_gradient_kernel_mlp(float* grad_output, float* predictions, float* targets, float* loss_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_output[idx] = predictions[idx] - targets[idx];
        atomicAdd(loss_result, grad_output[idx] * grad_output[idx]);
    }
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    int total_elements = mlp->batch_size * mlp->output_dim;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Reset loss accumulator to zero
    CHECK_CUDA(cudaMemset(mlp->d_loss_result, 0, sizeof(float)));
    
    // Compute gradient and accumulate loss
    compute_loss_and_gradient_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_grad_output, mlp->d_output, d_y, mlp->d_loss_result, total_elements
    );
    
    // Copy result back to host
    float total_loss;
    CHECK_CUDA(cudaMemcpy(&total_loss, mlp->d_loss_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / total_elements;
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

    // ∂L/∂W₂ = Sᵀ(∂L/∂Y)
    LT_MATMUL(mlp, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              mlp->d_postact, mlp->batch_hidden_layout,
              mlp->d_grad_output, mlp->batch_output_layout,
              &alpha, mlp->d_W2_grad, mlp->W2_layout);

    // ∂L/∂S = (∂L/∂Y)W₂ᵀ
    LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
              mlp->d_grad_output, mlp->batch_output_layout,
              mlp->d_W2, mlp->W2_layout,
              &beta, mlp->d_grad_postact, mlp->batch_hidden_layout);

    // ∂L/∂H = ∂L/∂S⊙[σ(H)+H⊙σ(H)⊙(1-σ(H))]
    int block_size = 256;
    int num_blocks = (mlp->batch_size * mlp->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_grad_postact,
        mlp->d_preact,
        mlp->batch_size * mlp->hidden_dim
    );

    // ∂L/∂W₁ = Xᵀ(∂L/∂H)
    LT_MATMUL(mlp, CUBLAS_OP_T, CUBLAS_OP_N, &alpha,
              d_X, mlp->batch_input_layout,
              mlp->d_grad_postact, mlp->batch_hidden_layout,
              &alpha, mlp->d_W1_grad, mlp->W1_layout);
    
    if (d_grad_X != NULL) {
        // ∂L/∂X = (∂L/∂H)W₁ᵀ
        LT_MATMUL(mlp, CUBLAS_OP_N, CUBLAS_OP_T, &alpha,
                  mlp->d_grad_postact, mlp->batch_hidden_layout,
                  mlp->d_W1, mlp->W1_layout,
                  &beta, d_grad_X, mlp->batch_input_layout);
    }
}

// CUDA kernel to update row statistics
__global__ static void update_row_stats_adafactor(
    float* row_stats, const float* grad, 
    float beta2, int rows, int cols, float batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum_sq = 0.0f;
        for (int j = 0; j < cols; j++) {
            float g = grad[i * cols + j] / batch_size;
            sum_sq += g * g;
        }
        float mean_sq = sum_sq / cols;
        row_stats[i] = beta2 * row_stats[i] + (1.0f - beta2) * mean_sq;
    }
}

// CUDA kernel to update column statistics
__global__ static void update_col_stats_adafactor(
    float* col_stats, const float* grad,
    float beta2, int rows, int cols, float batch_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum_sq = 0.0f;
        for (int i = 0; i < rows; i++) {
            float g = grad[i * cols + j] / batch_size;
            sum_sq += g * g;
        }
        float mean_sq = sum_sq / rows;
        col_stats[j] = beta2 * col_stats[j] + (1.0f - beta2) * mean_sq;
    }
}

// CUDA kernel to compute mean of row statistics
__global__ static void compute_row_mean_adafactor(
    float* mean_result, const float* row_stats, int rows)
{
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < (unsigned int)rows) ? row_stats[i] : 0.0f;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(mean_result, sdata[0]);
    }
}

// CUDA kernel to update weights using Adafactor
__global__ static void update_weights_adafactor(
    float* weight, const float* grad, float* m, 
    const float* row_stats, const float* col_stats, float row_mean,
    float beta1, float epsilon, float learning_rate, float weight_decay,
    float alpha_t, int rows, int cols, float batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        
        float v = (row_stats[i] * col_stats[j]) / (row_mean + epsilon);
        
        float update = alpha_t * m[idx] / (sqrtf(v) + epsilon);
        
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using Adafactor
void update_weights_mlp(MLP* mlp, float learning_rate, int effective_batch_size) {
    mlp->t++;
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    
    // Update W₁ weights with Adafactor
    {
        int rows = mlp->input_dim;
        int cols = mlp->hidden_dim;
        
        int row_blocks = (rows + block_size - 1) / block_size;
        update_row_stats_adafactor<<<row_blocks, block_size>>>(
            mlp->d_W1_r, mlp->d_W1_grad, mlp->beta2, rows, cols, effective_batch_size);
        
        int col_blocks = (cols + block_size - 1) / block_size;
        update_col_stats_adafactor<<<col_blocks, block_size>>>(
            mlp->d_W1_c, mlp->d_W1_grad, mlp->beta2, rows, cols, effective_batch_size);
        
        CHECK_CUDA(cudaMemset(mlp->d_row_mean, 0, sizeof(float)));
        compute_row_mean_adafactor<<<row_blocks, block_size, block_size * sizeof(float)>>>(
            mlp->d_row_mean, mlp->d_W1_r, rows);
        
        float row_mean;
        CHECK_CUDA(cudaMemcpy(&row_mean, mlp->d_row_mean, sizeof(float), cudaMemcpyDeviceToHost));
        row_mean /= rows;
        
        int total_blocks = (rows * cols + block_size - 1) / block_size;
        update_weights_adafactor<<<total_blocks, block_size>>>(
            mlp->d_W1, mlp->d_W1_grad, mlp->d_W1_m,
            mlp->d_W1_r, mlp->d_W1_c, row_mean,
            mlp->beta1, mlp->epsilon, learning_rate, mlp->weight_decay,
            alpha_t, rows, cols, effective_batch_size);
    }
    
    // Update W₂ weights with Adafactor
    {
        int rows = mlp->hidden_dim;
        int cols = mlp->output_dim;
        
        int row_blocks = (rows + block_size - 1) / block_size;
        update_row_stats_adafactor<<<row_blocks, block_size>>>(
            mlp->d_W2_r, mlp->d_W2_grad, mlp->beta2, rows, cols, effective_batch_size);
        
        int col_blocks = (cols + block_size - 1) / block_size;
        update_col_stats_adafactor<<<col_blocks, block_size>>>(
            mlp->d_W2_c, mlp->d_W2_grad, mlp->beta2, rows, cols, effective_batch_size);
        
        CHECK_CUDA(cudaMemset(mlp->d_row_mean, 0, sizeof(float)));
        compute_row_mean_adafactor<<<row_blocks, block_size, block_size * sizeof(float)>>>(
            mlp->d_row_mean, mlp->d_W2_r, rows);
        
        float row_mean;
        CHECK_CUDA(cudaMemcpy(&row_mean, mlp->d_row_mean, sizeof(float), cudaMemcpyDeviceToHost));
        row_mean /= rows;
        
        int total_blocks = (rows * cols + block_size - 1) / block_size;
        update_weights_adafactor<<<total_blocks, block_size>>>(
            mlp->d_W2, mlp->d_W2_grad, mlp->d_W2_m,
            mlp->d_W2_r, mlp->d_W2_c, row_mean,
            mlp->beta1, mlp->epsilon, learning_rate, mlp->weight_decay,
            alpha_t, rows, cols, effective_batch_size);
    }
}

// Reset optimizer state
void reset_optimizer_mlp(MLP* mlp) {
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    CHECK_CUDA(cudaMemset(mlp->d_W1_m, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_r, 0, mlp->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_c, 0, mlp->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_m, 0, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_r, 0, mlp->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_c, 0, mlp->output_dim * sizeof(float)));
    
    mlp->t = 0;
}

// Serialize MLP to a file
void serialize_mlp(MLP* mlp, FILE* file) {
    fwrite(&mlp->input_dim, sizeof(int), 1, file);
    fwrite(&mlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&mlp->output_dim, sizeof(int), 1, file);
    
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_r = (float*)malloc(mlp->input_dim * sizeof(float));
    float* h_W1_c = (float*)malloc(mlp->hidden_dim * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_r = (float*)malloc(mlp->hidden_dim * sizeof(float));
    float* h_W2_c = (float*)malloc(mlp->output_dim * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_W1, mlp->d_W1, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, mlp->d_W2, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W1_m, mlp->d_W1_m, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W1_r, mlp->d_W1_r, mlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W1_c, mlp->d_W1_c, mlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_m, mlp->d_W2_m, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_r, mlp->d_W2_r, mlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_c, mlp->d_W2_c, mlp->output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W1, sizeof(float), w1_size, file);
    fwrite(h_W2, sizeof(float), w2_size, file);
    fwrite(&mlp->t, sizeof(int), 1, file);
    fwrite(h_W1_m, sizeof(float), w1_size, file);
    fwrite(h_W1_r, sizeof(float), mlp->input_dim, file);
    fwrite(h_W1_c, sizeof(float), mlp->hidden_dim, file);
    fwrite(h_W2_m, sizeof(float), w2_size, file);
    fwrite(h_W2_r, sizeof(float), mlp->hidden_dim, file);
    fwrite(h_W2_c, sizeof(float), mlp->output_dim, file);
    
    free(h_W1); free(h_W2);
    free(h_W1_m); free(h_W1_r); free(h_W1_c);
    free(h_W2_m); free(h_W2_r); free(h_W2_c);
}

// Deserialize MLP from a file
MLP* deserialize_mlp(FILE* file, int batch_size, cublasLtHandle_t cublaslt_handle) {
    int input_dim, hidden_dim, output_dim;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublaslt_handle);
    
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_r = (float*)malloc(input_dim * sizeof(float));
    float* h_W1_c = (float*)malloc(hidden_dim * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_r = (float*)malloc(hidden_dim * sizeof(float));
    float* h_W2_c = (float*)malloc(output_dim * sizeof(float));
    
    fread(h_W1, sizeof(float), w1_size, file);
    fread(h_W2, sizeof(float), w2_size, file);
    fread(&mlp->t, sizeof(int), 1, file);
    fread(h_W1_m, sizeof(float), w1_size, file);
    fread(h_W1_r, sizeof(float), input_dim, file);
    fread(h_W1_c, sizeof(float), hidden_dim, file);
    fread(h_W2_m, sizeof(float), w2_size, file);
    fread(h_W2_r, sizeof(float), hidden_dim, file);
    fread(h_W2_c, sizeof(float), output_dim, file);
    
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_m, h_W1_m, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_r, h_W1_r, input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_c, h_W1_c, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_m, h_W2_m, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_r, h_W2_r, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_c, h_W2_c, output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_W1); free(h_W2);
    free(h_W1_m); free(h_W1_r); free(h_W1_c);
    free(h_W2_m); free(h_W2_r); free(h_W2_c);
    
    return mlp;
}