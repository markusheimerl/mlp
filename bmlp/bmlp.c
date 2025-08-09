#include "bmlp.h"

// Initialize the bilinear network with configurable dimensions
BMLP* init_bmlp(int input_dim, int hidden_dim, int output_dim, int batch_size) {
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
    
    // Allocate and initialize base weights and gradients
    bmlp->W1 = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    bmlp->W2 = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    bmlp->W3 = (float*)malloc(output_dim * input_dim * sizeof(float));
    bmlp->W1_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    bmlp->W2_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    bmlp->W3_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate scaling parameters
    bmlp->u1 = (float*)malloc(input_dim * sizeof(float));
    bmlp->u2 = (float*)malloc(hidden_dim * sizeof(float));
    bmlp->u1_grad = (float*)malloc(input_dim * sizeof(float));
    bmlp->u2_grad = (float*)malloc(hidden_dim * sizeof(float));
    
    // Allocate Adam buffers for base weights
    bmlp->W1_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    bmlp->W1_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    bmlp->W2_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    bmlp->W2_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    bmlp->W3_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    bmlp->W3_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    
    // Allocate Adam buffers for scaling weights
    bmlp->u1_m = (float*)calloc(input_dim, sizeof(float));
    bmlp->u1_v = (float*)calloc(input_dim, sizeof(float));
    bmlp->u2_m = (float*)calloc(hidden_dim, sizeof(float));
    bmlp->u2_v = (float*)calloc(hidden_dim, sizeof(float));
    
    // Allocate layer outputs and working buffers
    bmlp->layer1_preact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    bmlp->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    bmlp->layer2_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    bmlp->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    bmlp->error_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Allocate temporary buffers for scaling operations
    bmlp->W1_scaled = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    bmlp->W2_scaled = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    bmlp->input_scale = (float*)malloc(input_dim * sizeof(float));
    bmlp->hidden_scale = (float*)malloc(hidden_dim * sizeof(float));
    
    // Initialize weights with appropriate scaling
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    float scale_W3 = 1.0f / sqrtf(input_dim);
    float scale_u = 0.1f;  // Small scale for scaling parameters
    
    // Initialize base weights
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        bmlp->W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        bmlp->W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        bmlp->W3[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W3;
    }
    
    // Initialize scaling parameters
    for (int i = 0; i < input_dim; i++) {
        bmlp->u1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_u;
    }
    
    for (int i = 0; i < hidden_dim; i++) {
        bmlp->u2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_u;
    }
    
    return bmlp;
}

// Free network memory
void free_bmlp(BMLP* bmlp) {
    // Free base weights
    free(bmlp->W1); free(bmlp->W2); free(bmlp->W3);
    free(bmlp->W1_grad); free(bmlp->W2_grad); free(bmlp->W3_grad);
    
    // Free scaling weights
    free(bmlp->u1); free(bmlp->u2);
    free(bmlp->u1_grad); free(bmlp->u2_grad);
    
    // Free Adam buffers
    free(bmlp->W1_m); free(bmlp->W1_v);
    free(bmlp->W2_m); free(bmlp->W2_v);
    free(bmlp->W3_m); free(bmlp->W3_v);
    free(bmlp->u1_m); free(bmlp->u1_v);
    free(bmlp->u2_m); free(bmlp->u2_v);
    
    // Free buffers
    free(bmlp->layer1_preact); free(bmlp->layer1_output); free(bmlp->layer2_output);
    free(bmlp->error_output); free(bmlp->error_hidden);
    free(bmlp->W1_scaled); free(bmlp->W2_scaled);
    free(bmlp->input_scale); free(bmlp->hidden_scale);
    
    free(bmlp);
}

// Forward pass with input-dependent weights
void forward_pass_bmlp(BMLP* bmlp, float* X) {
    // Compute mean of input batch for scaling
    for (int i = 0; i < bmlp->input_dim; i++) {
        bmlp->input_scale[i] = 0.0f;
        for (int j = 0; j < bmlp->batch_size; j++) {
            bmlp->input_scale[i] += X[j * bmlp->input_dim + i];
        }
        bmlp->input_scale[i] /= bmlp->batch_size;
        // Apply scaling function: scale = 1 + sigmoid(mean_input * u1)
        bmlp->input_scale[i] = 1.0f + 1.0f / (1.0f + expf(-bmlp->input_scale[i] * bmlp->u1[i]));
    }
    
    // Create scaled W1: W1_scaled[h][i] = W1[h][i] * input_scale[i]
    for (int h = 0; h < bmlp->hidden_dim; h++) {
        for (int i = 0; i < bmlp->input_dim; i++) {
            bmlp->W1_scaled[h * bmlp->input_dim + i] = bmlp->W1[h * bmlp->input_dim + i] * bmlp->input_scale[i];
        }
    }
    
    // H = X * W1_scaled^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                bmlp->batch_size, bmlp->hidden_dim, bmlp->input_dim,
                1.0f, X, bmlp->input_dim,
                bmlp->W1_scaled, bmlp->input_dim,
                0.0f, bmlp->layer1_output, bmlp->hidden_dim);
    
    // Copy to preact for compatibility
    memcpy(bmlp->layer1_preact, bmlp->layer1_output, bmlp->batch_size * bmlp->hidden_dim * sizeof(float));
    
    // Compute mean of hidden layer for scaling
    for (int i = 0; i < bmlp->hidden_dim; i++) {
        bmlp->hidden_scale[i] = 0.0f;
        for (int j = 0; j < bmlp->batch_size; j++) {
            bmlp->hidden_scale[i] += bmlp->layer1_output[j * bmlp->hidden_dim + i];
        }
        bmlp->hidden_scale[i] /= bmlp->batch_size;
        // Apply scaling function: scale = 1 + sigmoid(mean_hidden * u2)
        bmlp->hidden_scale[i] = 1.0f + 1.0f / (1.0f + expf(-bmlp->hidden_scale[i] * bmlp->u2[i]));
    }
    
    // Create scaled W2: W2_scaled[o][h] = W2[o][h] * hidden_scale[h]
    for (int o = 0; o < bmlp->output_dim; o++) {
        for (int h = 0; h < bmlp->hidden_dim; h++) {
            bmlp->W2_scaled[o * bmlp->hidden_dim + h] = bmlp->W2[o * bmlp->hidden_dim + h] * bmlp->hidden_scale[h];
        }
    }
    
    // Y = H * W2_scaled^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                bmlp->batch_size, bmlp->output_dim, bmlp->hidden_dim,
                1.0f, bmlp->layer1_output, bmlp->hidden_dim,
                bmlp->W2_scaled, bmlp->hidden_dim,
                0.0f, bmlp->layer2_output, bmlp->output_dim);
    
    // Y = Y + X * W3^T (residual connection)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                bmlp->batch_size, bmlp->output_dim, bmlp->input_dim,
                1.0f, X, bmlp->input_dim,
                bmlp->W3, bmlp->input_dim,
                1.0f, bmlp->layer2_output, bmlp->output_dim);
}

// Calculate loss (same as regular MLP)
float calculate_loss_bmlp(BMLP* bmlp, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < bmlp->batch_size * bmlp->output_dim; i++) {
        bmlp->error_output[i] = bmlp->layer2_output[i] - y[i];
        loss += bmlp->error_output[i] * bmlp->error_output[i];
    }
    return loss / (bmlp->batch_size * bmlp->output_dim);
}

// Zero gradients
void zero_gradients_bmlp(BMLP* bmlp) {
    memset(bmlp->W1_grad, 0, bmlp->hidden_dim * bmlp->input_dim * sizeof(float));
    memset(bmlp->W2_grad, 0, bmlp->output_dim * bmlp->hidden_dim * sizeof(float));
    memset(bmlp->W3_grad, 0, bmlp->output_dim * bmlp->input_dim * sizeof(float));
    memset(bmlp->u1_grad, 0, bmlp->input_dim * sizeof(float));
    memset(bmlp->u2_grad, 0, bmlp->hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass_bmlp(BMLP* bmlp, float* X) {
    // ∂L/∂W3 = X^T * error_output
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->input_dim, bmlp->output_dim, bmlp->batch_size,
                1.0f, X, bmlp->input_dim,
                bmlp->error_output, bmlp->output_dim,
                1.0f, bmlp->W3_grad, bmlp->output_dim);
    
    // ∂L/∂W2_scaled = layer1_output^T * error_output
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->hidden_dim, bmlp->output_dim, bmlp->batch_size,
                1.0f, bmlp->layer1_output, bmlp->hidden_dim,
                bmlp->error_output, bmlp->output_dim,
                1.0f, bmlp->W2_grad, bmlp->output_dim);
    
    // Scale W2_grad by hidden_scale since W2_scaled = W2 * hidden_scale
    for (int o = 0; o < bmlp->output_dim; o++) {
        for (int h = 0; h < bmlp->hidden_dim; h++) {
            bmlp->W2_grad[o * bmlp->hidden_dim + h] *= bmlp->hidden_scale[h];
        }
    }
    
    // ∂L/∂H = error_output * W2_scaled
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                bmlp->batch_size, bmlp->hidden_dim, bmlp->output_dim,
                1.0f, bmlp->error_output, bmlp->output_dim,
                bmlp->W2_scaled, bmlp->hidden_dim,
                0.0f, bmlp->error_hidden, bmlp->hidden_dim);
    
    // ∂L/∂W1_scaled = X^T * error_hidden
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->input_dim, bmlp->hidden_dim, bmlp->batch_size,
                1.0f, X, bmlp->input_dim,
                bmlp->error_hidden, bmlp->hidden_dim,
                1.0f, bmlp->W1_grad, bmlp->hidden_dim);
    
    // Scale W1_grad by input_scale since W1_scaled = W1 * input_scale
    for (int h = 0; h < bmlp->hidden_dim; h++) {
        for (int i = 0; i < bmlp->input_dim; i++) {
            bmlp->W1_grad[h * bmlp->input_dim + i] *= bmlp->input_scale[i];
        }
    }
    
    // Compute gradients for scaling parameters u1 and u2
    // This requires computing how the scaling affects the loss
    
    // For u1: ∂L/∂u1[i] = ∑_h (∂L/∂W1_scaled[h][i]) * W1[h][i] * ∂scale/∂u1[i]
    // where ∂scale/∂u1[i] = input_mean[i] * sigmoid(input_mean[i] * u1[i]) * (1 - sigmoid(input_mean[i] * u1[i]))
    
    // Compute input means again (could be optimized by storing)
    for (int i = 0; i < bmlp->input_dim; i++) {
        float input_mean = 0.0f;
        for (int j = 0; j < bmlp->batch_size; j++) {
            input_mean += X[j * bmlp->input_dim + i];
        }
        input_mean /= bmlp->batch_size;
        
        float sigmoid_val = 1.0f / (1.0f + expf(-input_mean * bmlp->u1[i]));
        float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
        float scale_deriv = input_mean * sigmoid_deriv;
        
        float grad_sum = 0.0f;
        for (int h = 0; h < bmlp->hidden_dim; h++) {
            grad_sum += bmlp->W1_grad[h * bmlp->input_dim + i] * bmlp->W1[h * bmlp->input_dim + i];
        }
        bmlp->u1_grad[i] += grad_sum * scale_deriv;
    }
    
    // For u2: similar computation
    for (int h = 0; h < bmlp->hidden_dim; h++) {
        float hidden_mean = 0.0f;
        for (int j = 0; j < bmlp->batch_size; j++) {
            hidden_mean += bmlp->layer1_output[j * bmlp->hidden_dim + h];
        }
        hidden_mean /= bmlp->batch_size;
        
        float sigmoid_val = 1.0f / (1.0f + expf(-hidden_mean * bmlp->u2[h]));
        float sigmoid_deriv = sigmoid_val * (1.0f - sigmoid_val);
        float scale_deriv = hidden_mean * sigmoid_deriv;
        
        float grad_sum = 0.0f;
        for (int o = 0; o < bmlp->output_dim; o++) {
            grad_sum += bmlp->W2_grad[o * bmlp->hidden_dim + h] * bmlp->W2[o * bmlp->hidden_dim + h];
        }
        bmlp->u2_grad[h] += grad_sum * scale_deriv;
    }
}

// Update weights using AdamW
void update_weights_bmlp(BMLP* bmlp, float learning_rate) {
    bmlp->t++;
    
    float beta1_t = powf(bmlp->beta1, bmlp->t);
    float beta2_t = powf(bmlp->beta2, bmlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update base weights W1, W2, W3 (same as regular MLP)
    for (int i = 0; i < bmlp->hidden_dim * bmlp->input_dim; i++) {
        float grad = bmlp->W1_grad[i] / bmlp->batch_size;
        bmlp->W1_m[i] = bmlp->beta1 * bmlp->W1_m[i] + (1.0f - bmlp->beta1) * grad;
        bmlp->W1_v[i] = bmlp->beta2 * bmlp->W1_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        float update = alpha_t * bmlp->W1_m[i] / (sqrtf(bmlp->W1_v[i]) + bmlp->epsilon);
        bmlp->W1[i] = bmlp->W1[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
    
    for (int i = 0; i < bmlp->output_dim * bmlp->hidden_dim; i++) {
        float grad = bmlp->W2_grad[i] / bmlp->batch_size;
        bmlp->W2_m[i] = bmlp->beta1 * bmlp->W2_m[i] + (1.0f - bmlp->beta1) * grad;
        bmlp->W2_v[i] = bmlp->beta2 * bmlp->W2_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        float update = alpha_t * bmlp->W2_m[i] / (sqrtf(bmlp->W2_v[i]) + bmlp->epsilon);
        bmlp->W2[i] = bmlp->W2[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
    
    for (int i = 0; i < bmlp->output_dim * bmlp->input_dim; i++) {
        float grad = bmlp->W3_grad[i] / bmlp->batch_size;
        bmlp->W3_m[i] = bmlp->beta1 * bmlp->W3_m[i] + (1.0f - bmlp->beta1) * grad;
        bmlp->W3_v[i] = bmlp->beta2 * bmlp->W3_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        float update = alpha_t * bmlp->W3_m[i] / (sqrtf(bmlp->W3_v[i]) + bmlp->epsilon);
        bmlp->W3[i] = bmlp->W3[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
    
    // Update scaling weights u1, u2
    for (int i = 0; i < bmlp->input_dim; i++) {
        float grad = bmlp->u1_grad[i] / bmlp->batch_size;
        bmlp->u1_m[i] = bmlp->beta1 * bmlp->u1_m[i] + (1.0f - bmlp->beta1) * grad;
        bmlp->u1_v[i] = bmlp->beta2 * bmlp->u1_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        float update = alpha_t * bmlp->u1_m[i] / (sqrtf(bmlp->u1_v[i]) + bmlp->epsilon);
        bmlp->u1[i] = bmlp->u1[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
    
    for (int i = 0; i < bmlp->hidden_dim; i++) {
        float grad = bmlp->u2_grad[i] / bmlp->batch_size;
        bmlp->u2_m[i] = bmlp->beta1 * bmlp->u2_m[i] + (1.0f - bmlp->beta1) * grad;
        bmlp->u2_v[i] = bmlp->beta2 * bmlp->u2_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        float update = alpha_t * bmlp->u2_m[i] / (sqrtf(bmlp->u2_v[i]) + bmlp->epsilon);
        bmlp->u2[i] = bmlp->u2[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
}

// Save and load functions
void save_bmlp(BMLP* bmlp, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&bmlp->input_dim, sizeof(int), 1, file);
    fwrite(&bmlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&bmlp->output_dim, sizeof(int), 1, file);
    fwrite(&bmlp->batch_size, sizeof(int), 1, file);
    
    // Save base weights
    fwrite(bmlp->W1, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(bmlp->W2, sizeof(float), bmlp->output_dim * bmlp->hidden_dim, file);
    fwrite(bmlp->W3, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);
    
    // Save scaling weights
    fwrite(bmlp->u1, sizeof(float), bmlp->input_dim, file);
    fwrite(bmlp->u2, sizeof(float), bmlp->hidden_dim, file);
    
    fclose(file);
    printf("BMLP model saved to %s\n", filename);
}

BMLP* load_bmlp(const char* filename, int custom_batch_size) {
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
    
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    BMLP* bmlp = init_bmlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(bmlp->W1, sizeof(float), hidden_dim * input_dim, file);
    fread(bmlp->W2, sizeof(float), output_dim * hidden_dim, file);
    fread(bmlp->W3, sizeof(float), output_dim * input_dim, file);
    fread(bmlp->u1, sizeof(float), input_dim, file);
    fread(bmlp->u2, sizeof(float), hidden_dim, file);
    
    fclose(file);
    printf("BMLP model loaded from %s\n", filename);
    
    return bmlp;
}