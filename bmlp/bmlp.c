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
    
    // Initialize scaling parameters with very small values
    for (int i = 0; i < input_dim; i++) {
        bmlp->u1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.01f;  // Much smaller scale
    }
    
    for (int i = 0; i < hidden_dim; i++) {
        bmlp->u2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * 0.01f;  // Much smaller scale
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
    // Standard H = X * W1^T (no scaling for now, just add input-dependent bias)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                bmlp->batch_size, bmlp->hidden_dim, bmlp->input_dim,
                1.0f, X, bmlp->input_dim,
                bmlp->W1, bmlp->input_dim,
                0.0f, bmlp->layer1_preact, bmlp->hidden_dim);
    
    // Add input-dependent bias: H += X_mean * u1 (broadcasted)
    for (int i = 0; i < bmlp->input_dim; i++) {
        bmlp->input_scale[i] = 0.0f;
        for (int j = 0; j < bmlp->batch_size; j++) {
            bmlp->input_scale[i] += X[j * bmlp->input_dim + i];
        }
        bmlp->input_scale[i] /= bmlp->batch_size;
    }
    
    // Add input-dependent modulation: H[b][h] += sum_i(X_mean[i] * u1[i])
    for (int b = 0; b < bmlp->batch_size; b++) {
        for (int h = 0; h < bmlp->hidden_dim; h++) {
            float input_bias = 0.0f;
            for (int i = 0; i < bmlp->input_dim; i++) {
                input_bias += bmlp->input_scale[i] * bmlp->u1[i];
            }
            bmlp->layer1_preact[b * bmlp->hidden_dim + h] += input_bias;
        }
    }
    
    // Copy preact to output (no activation, replacing swish)
    memcpy(bmlp->layer1_output, bmlp->layer1_preact, bmlp->batch_size * bmlp->hidden_dim * sizeof(float));
    
    // Standard Y = H * W2^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                bmlp->batch_size, bmlp->output_dim, bmlp->hidden_dim,
                1.0f, bmlp->layer1_output, bmlp->hidden_dim,
                bmlp->W2, bmlp->hidden_dim,
                0.0f, bmlp->layer2_output, bmlp->output_dim);
    
    // Add hidden-dependent bias: Y += H_mean * u2 (broadcasted)
    for (int h = 0; h < bmlp->hidden_dim; h++) {
        bmlp->hidden_scale[h] = 0.0f;
        for (int j = 0; j < bmlp->batch_size; j++) {
            bmlp->hidden_scale[h] += bmlp->layer1_output[j * bmlp->hidden_dim + h];
        }
        bmlp->hidden_scale[h] /= bmlp->batch_size;
    }
    
    for (int b = 0; b < bmlp->batch_size; b++) {
        for (int o = 0; o < bmlp->output_dim; o++) {
            float hidden_bias = 0.0f;
            for (int h = 0; h < bmlp->hidden_dim; h++) {
                hidden_bias += bmlp->hidden_scale[h] * bmlp->u2[h];
            }
            bmlp->layer2_output[b * bmlp->output_dim + o] += hidden_bias;
        }
    }
    
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
    
    // ∂L/∂W2 = layer1_output^T * error_output
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->hidden_dim, bmlp->output_dim, bmlp->batch_size,
                1.0f, bmlp->layer1_output, bmlp->hidden_dim,
                bmlp->error_output, bmlp->output_dim,
                1.0f, bmlp->W2_grad, bmlp->output_dim);
    
    // ∂L/∂H = error_output * W2
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                bmlp->batch_size, bmlp->hidden_dim, bmlp->output_dim,
                1.0f, bmlp->error_output, bmlp->output_dim,
                bmlp->W2, bmlp->hidden_dim,
                0.0f, bmlp->error_hidden, bmlp->hidden_dim);
    
    // ∂L/∂W1 = X^T * error_hidden
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->input_dim, bmlp->hidden_dim, bmlp->batch_size,
                1.0f, X, bmlp->input_dim,
                bmlp->error_hidden, bmlp->hidden_dim,
                1.0f, bmlp->W1_grad, bmlp->hidden_dim);
    
    // Compute gradients for bias parameters u1 and u2
    // For u1: ∂L/∂u1[i] = ∑_b,h (∂L/∂H[b][h]) * X_mean[i]
    for (int i = 0; i < bmlp->input_dim; i++) {
        float grad_sum = 0.0f;
        for (int b = 0; b < bmlp->batch_size; b++) {
            for (int h = 0; h < bmlp->hidden_dim; h++) {
                grad_sum += bmlp->error_hidden[b * bmlp->hidden_dim + h];
            }
        }
        bmlp->u1_grad[i] += grad_sum * bmlp->input_scale[i];
    }
    
    // For u2: ∂L/∂u2[h] = ∑_b,o (∂L/∂Y[b][o]) * H_mean[h]
    for (int h = 0; h < bmlp->hidden_dim; h++) {
        float grad_sum = 0.0f;
        for (int b = 0; b < bmlp->batch_size; b++) {
            for (int o = 0; o < bmlp->output_dim; o++) {
                grad_sum += bmlp->error_output[b * bmlp->output_dim + o];
            }
        }
        bmlp->u2_grad[h] += grad_sum * bmlp->hidden_scale[h];
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