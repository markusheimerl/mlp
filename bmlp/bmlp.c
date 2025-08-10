#include "bmlp.h"

// Initialize the network with configurable dimensions
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
    
    // Allocate and initialize weights and gradients
    bmlp->W1 = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    bmlp->W2 = (float*)malloc(output_dim * hidden_dim * hidden_dim * sizeof(float));
    bmlp->W3 = (float*)malloc(output_dim * input_dim * sizeof(float));
    bmlp->W1_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    bmlp->W2_grad = (float*)malloc(output_dim * hidden_dim * hidden_dim * sizeof(float));
    bmlp->W3_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate Adam buffers
    bmlp->W1_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    bmlp->W1_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    bmlp->W2_m = (float*)calloc(output_dim * hidden_dim * hidden_dim, sizeof(float));
    bmlp->W2_v = (float*)calloc(output_dim * hidden_dim * hidden_dim, sizeof(float));
    bmlp->W3_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    bmlp->W3_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    
    // Allocate layer outputs and working buffers
    bmlp->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    bmlp->layer2_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    bmlp->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    bmlp->error_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    bmlp->outer_product = (float*)malloc(batch_size * hidden_dim * hidden_dim * sizeof(float));
    
    // Initialize weights
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 0.1f / sqrtf(hidden_dim);
    float scale_W3 = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        bmlp->W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim * hidden_dim; i++) {
        bmlp->W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        bmlp->W3[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W3;
    }
    
    return bmlp;
}

// Free network memory
void free_bmlp(BMLP* bmlp) {
    free(bmlp->W1); free(bmlp->W2); free(bmlp->W3);
    free(bmlp->W1_grad); free(bmlp->W2_grad); free(bmlp->W3_grad);
    free(bmlp->W1_m); free(bmlp->W1_v);
    free(bmlp->W2_m); free(bmlp->W2_v);
    free(bmlp->W3_m); free(bmlp->W3_v);
    free(bmlp->layer1_output); free(bmlp->layer2_output);
    free(bmlp->error_output); free(bmlp->error_hidden);
    free(bmlp->outer_product);
    free(bmlp);
}

// Forward pass
void forward_pass_bmlp(BMLP* bmlp, float* X) {
    // H = XW₁
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                bmlp->batch_size, bmlp->hidden_dim, bmlp->input_dim,
                1.0f, X, bmlp->input_dim,
                bmlp->W1, bmlp->hidden_dim,
                0.0f, bmlp->layer1_output, bmlp->hidden_dim);
    
    // Compute outer products (H ⊗ H)
    for (int b = 0; b < bmlp->batch_size; b++) {
        cblas_sger(CblasRowMajor, bmlp->hidden_dim, bmlp->hidden_dim,
                   1.0f, &bmlp->layer1_output[b * bmlp->hidden_dim], 1,
                   &bmlp->layer1_output[b * bmlp->hidden_dim], 1,
                   &bmlp->outer_product[b * bmlp->hidden_dim * bmlp->hidden_dim], bmlp->hidden_dim);
    }
    
    // Y = (H ⊗ H) @ W2^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                bmlp->batch_size, bmlp->output_dim, bmlp->hidden_dim * bmlp->hidden_dim,
                1.0f, bmlp->outer_product, bmlp->hidden_dim * bmlp->hidden_dim,
                bmlp->W2, bmlp->hidden_dim * bmlp->hidden_dim,
                0.0f, bmlp->layer2_output, bmlp->output_dim);
    
    // Y = Y + XW₃
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                bmlp->batch_size, bmlp->output_dim, bmlp->input_dim,
                1.0f, X, bmlp->input_dim,
                bmlp->W3, bmlp->output_dim,
                1.0f, bmlp->layer2_output, bmlp->output_dim);
}

// Calculate loss
float calculate_loss_bmlp(BMLP* bmlp, float* y) {
    // ∂L/∂Y = Y - Y_true
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
    memset(bmlp->W2_grad, 0, bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim * sizeof(float));
    memset(bmlp->W3_grad, 0, bmlp->output_dim * bmlp->input_dim * sizeof(float));
}

// Backward pass
void backward_pass_bmlp(BMLP* bmlp, float* X) {
    // ∂L/∂W₃ = Xᵀ(∂L/∂Y)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->input_dim, bmlp->output_dim, bmlp->batch_size,
                1.0f, X, bmlp->input_dim,
                bmlp->error_output, bmlp->output_dim,
                1.0f, bmlp->W3_grad, bmlp->output_dim);
    
    // ∂L/∂W₂ = (∂L/∂Y)ᵀ @ (H ⊗ H)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->output_dim, bmlp->hidden_dim * bmlp->hidden_dim, bmlp->batch_size,
                1.0f, bmlp->error_output, bmlp->output_dim,
                bmlp->outer_product, bmlp->hidden_dim * bmlp->hidden_dim,
                1.0f, bmlp->W2_grad, bmlp->hidden_dim * bmlp->hidden_dim);
    
    // Compute gradient w.r.t (H ⊗ H)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                bmlp->batch_size, bmlp->hidden_dim * bmlp->hidden_dim, bmlp->output_dim,
                1.0f, bmlp->error_output, bmlp->output_dim,
                bmlp->W2, bmlp->hidden_dim * bmlp->hidden_dim,
                0.0f, bmlp->outer_product, bmlp->hidden_dim * bmlp->hidden_dim);
    
    // ∂L/∂H using chain rule with (H ⊗ H)
    for (int b = 0; b < bmlp->batch_size; b++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, bmlp->hidden_dim, bmlp->hidden_dim,
                    1.0f, &bmlp->outer_product[b * bmlp->hidden_dim * bmlp->hidden_dim], bmlp->hidden_dim,
                    &bmlp->layer1_output[b * bmlp->hidden_dim], 1,
                    0.0f, &bmlp->error_hidden[b * bmlp->hidden_dim], 1);
        
        cblas_sgemv(CblasRowMajor, CblasTrans, bmlp->hidden_dim, bmlp->hidden_dim,
                    1.0f, &bmlp->outer_product[b * bmlp->hidden_dim * bmlp->hidden_dim], bmlp->hidden_dim,
                    &bmlp->layer1_output[b * bmlp->hidden_dim], 1,
                    1.0f, &bmlp->error_hidden[b * bmlp->hidden_dim], 1);
    }
    
    // ∂L/∂W₁ = Xᵀ(∂L/∂H)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                bmlp->input_dim, bmlp->hidden_dim, bmlp->batch_size,
                1.0f, X, bmlp->input_dim,
                bmlp->error_hidden, bmlp->hidden_dim,
                1.0f, bmlp->W1_grad, bmlp->hidden_dim);
}

// Update weights using AdamW
void update_weights_bmlp(BMLP* bmlp, float learning_rate) {
    bmlp->t++;  // Increment time step
    
    float beta1_t = powf(bmlp->beta1, bmlp->t);
    float beta2_t = powf(bmlp->beta2, bmlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update W1 weights
    for (int i = 0; i < bmlp->hidden_dim * bmlp->input_dim; i++) {
        float grad = bmlp->W1_grad[i] / bmlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        bmlp->W1_m[i] = bmlp->beta1 * bmlp->W1_m[i] + (1.0f - bmlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        bmlp->W1_v[i] = bmlp->beta2 * bmlp->W1_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        
        float update = alpha_t * bmlp->W1_m[i] / (sqrtf(bmlp->W1_v[i]) + bmlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        bmlp->W1[i] = bmlp->W1[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
    
    // Update W2 weights
    for (int i = 0; i < bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim; i++) {
        float grad = bmlp->W2_grad[i] / bmlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        bmlp->W2_m[i] = bmlp->beta1 * bmlp->W2_m[i] + (1.0f - bmlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        bmlp->W2_v[i] = bmlp->beta2 * bmlp->W2_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        
        float update = alpha_t * bmlp->W2_m[i] / (sqrtf(bmlp->W2_v[i]) + bmlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        bmlp->W2[i] = bmlp->W2[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
    
    // Update W3 weights
    for (int i = 0; i < bmlp->output_dim * bmlp->input_dim; i++) {
        float grad = bmlp->W3_grad[i] / bmlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        bmlp->W3_m[i] = bmlp->beta1 * bmlp->W3_m[i] + (1.0f - bmlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        bmlp->W3_v[i] = bmlp->beta2 * bmlp->W3_v[i] + (1.0f - bmlp->beta2) * grad * grad;
        
        float update = alpha_t * bmlp->W3_m[i] / (sqrtf(bmlp->W3_v[i]) + bmlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        bmlp->W3[i] = bmlp->W3[i] * (1.0f - learning_rate * bmlp->weight_decay) - update;
    }
}

// Function to save model weights to binary file
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
    
    // Save weights
    fwrite(bmlp->W1, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(bmlp->W2, sizeof(float), bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim, file);
    fwrite(bmlp->W3, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);
    
    // Save Adam state
    fwrite(&bmlp->t, sizeof(int), 1, file);
    fwrite(bmlp->W1_m, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(bmlp->W1_v, sizeof(float), bmlp->hidden_dim * bmlp->input_dim, file);
    fwrite(bmlp->W2_m, sizeof(float), bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim, file);
    fwrite(bmlp->W2_v, sizeof(float), bmlp->output_dim * bmlp->hidden_dim * bmlp->hidden_dim, file);
    fwrite(bmlp->W3_m, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);
    fwrite(bmlp->W3_v, sizeof(float), bmlp->output_dim * bmlp->input_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to load model weights from binary file
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
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    BMLP* bmlp = init_bmlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(bmlp->W1, sizeof(float), hidden_dim * input_dim, file);
    fread(bmlp->W2, sizeof(float), output_dim * hidden_dim * hidden_dim, file);
    fread(bmlp->W3, sizeof(float), output_dim * input_dim, file);
    
    // Load Adam state
    fread(&bmlp->t, sizeof(int), 1, file);
    fread(bmlp->W1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(bmlp->W1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(bmlp->W2_m, sizeof(float), output_dim * hidden_dim * hidden_dim, file);
    fread(bmlp->W2_v, sizeof(float), output_dim * hidden_dim * hidden_dim, file);
    fread(bmlp->W3_m, sizeof(float), output_dim * input_dim, file);
    fread(bmlp->W3_v, sizeof(float), output_dim * input_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return bmlp;
}