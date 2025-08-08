#include "mlp.h"

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
    
    // Allocate and initialize weights and gradients
    mlp->W1 = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    mlp->W2 = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    mlp->W3 = (float*)malloc(output_dim * input_dim * sizeof(float));
    mlp->W1_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    mlp->W2_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    mlp->W3_grad = (float*)malloc(output_dim * input_dim * sizeof(float));
    
    // Allocate Adam buffers
    mlp->W1_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    mlp->W1_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    mlp->W2_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    mlp->W2_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    mlp->W3_m = (float*)calloc(output_dim * input_dim, sizeof(float));
    mlp->W3_v = (float*)calloc(output_dim * input_dim, sizeof(float));
    
    // Allocate layer outputs and working buffers
    mlp->layer1_preact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->layer2_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    mlp->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->error_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize weights
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    float scale_W3 = 1.0f / sqrtf(input_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        mlp->W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        mlp->W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        mlp->W3[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W3;
    }
    
    return mlp;
}

// Free network memory
void free_mlp(MLP* mlp) {
    free(mlp->W1); free(mlp->W2); free(mlp->W3);
    free(mlp->W1_grad); free(mlp->W2_grad); free(mlp->W3_grad);
    free(mlp->W1_m); free(mlp->W1_v);
    free(mlp->W2_m); free(mlp->W2_v);
    free(mlp->W3_m); free(mlp->W3_v);
    free(mlp->layer1_preact); free(mlp->layer1_output); free(mlp->layer2_output);
    free(mlp->error_output); free(mlp->error_hidden);
    free(mlp);
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* X) {
    // H = XW₁
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->hidden_dim, mlp->input_dim,
                1.0f, X, mlp->input_dim,
                mlp->W1, mlp->hidden_dim,
                0.0f, mlp->layer1_preact, mlp->hidden_dim);
    
    // S = Hσ(H)
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
        mlp->layer1_output[i] = mlp->layer1_preact[i] / (1.0f + expf(-mlp->layer1_preact[i]));
    }
    
    // Y = SW₂
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->output_dim, mlp->hidden_dim,
                1.0f, mlp->layer1_output, mlp->hidden_dim,
                mlp->W2, mlp->output_dim,
                0.0f, mlp->layer2_output, mlp->output_dim);
    
    // Y = Y + XW₃
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->output_dim, mlp->input_dim,
                1.0f, X, mlp->input_dim,
                mlp->W3, mlp->output_dim,
                1.0f, mlp->layer2_output, mlp->output_dim);
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;
    for (int i = 0; i < mlp->batch_size * mlp->output_dim; i++) {
        mlp->error_output[i] = mlp->layer2_output[i] - y[i];
        loss += mlp->error_output[i] * mlp->error_output[i];
    }
    return loss / (mlp->batch_size * mlp->output_dim);
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    memset(mlp->W1_grad, 0, mlp->hidden_dim * mlp->input_dim * sizeof(float));
    memset(mlp->W2_grad, 0, mlp->output_dim * mlp->hidden_dim * sizeof(float));
    memset(mlp->W3_grad, 0, mlp->output_dim * mlp->input_dim * sizeof(float));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* X) {
    // ∂L/∂W₂ = Sᵀ(∂L/∂Y)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                mlp->hidden_dim, mlp->output_dim, mlp->batch_size,
                1.0f, mlp->layer1_output, mlp->hidden_dim,
                mlp->error_output, mlp->output_dim,
                1.0f, mlp->W2_grad, mlp->output_dim);
    
    // ∂L/∂W₃ = Xᵀ(∂L/∂Y)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                mlp->input_dim, mlp->output_dim, mlp->batch_size,
                1.0f, X, mlp->input_dim,
                mlp->error_output, mlp->output_dim,
                1.0f, mlp->W3_grad, mlp->output_dim);
    
    // ∂L/∂S = (∂L/∂Y)(W₂)ᵀ
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                mlp->batch_size, mlp->hidden_dim, mlp->output_dim,
                1.0f, mlp->error_output, mlp->output_dim,
                mlp->W2, mlp->output_dim,
                0.0f, mlp->error_hidden, mlp->hidden_dim);
    
    // ∂L/∂H = ∂L/∂S ⊙ [σ(H) + Hσ(H)(1-σ(H))]
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
        float h = mlp->layer1_preact[i];
        float sigmoid = 1.0f / (1.0f + expf(-h));
        mlp->error_hidden[i] *= sigmoid + h * sigmoid * (1.0f - sigmoid);
    }
    
    // ∂L/∂W₁ = Xᵀ(∂L/∂H)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                mlp->input_dim, mlp->hidden_dim, mlp->batch_size,
                1.0f, X, mlp->input_dim,
                mlp->error_hidden, mlp->hidden_dim,
                1.0f, mlp->W1_grad, mlp->hidden_dim);
}

// Update weights using AdamW
void update_weights_mlp(MLP* mlp, float learning_rate) {
    mlp->t++;  // Increment time step
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update W1 weights
    for (int i = 0; i < mlp->hidden_dim * mlp->input_dim; i++) {
        float grad = mlp->W1_grad[i] / mlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->W1_m[i] = mlp->beta1 * mlp->W1_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->W1_v[i] = mlp->beta2 * mlp->W1_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->W1_m[i] / (sqrtf(mlp->W1_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->W1[i] = mlp->W1[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
    }
    
    // Update W2 weights
    for (int i = 0; i < mlp->output_dim * mlp->hidden_dim; i++) {
        float grad = mlp->W2_grad[i] / mlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->W2_m[i] = mlp->beta1 * mlp->W2_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->W2_v[i] = mlp->beta2 * mlp->W2_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->W2_m[i] / (sqrtf(mlp->W2_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->W2[i] = mlp->W2[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
    }
    
    // Update W3 weights
    for (int i = 0; i < mlp->output_dim * mlp->input_dim; i++) {
        float grad = mlp->W3_grad[i] / mlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->W3_m[i] = mlp->beta1 * mlp->W3_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->W3_v[i] = mlp->beta2 * mlp->W3_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->W3_m[i] / (sqrtf(mlp->W3_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->W3[i] = mlp->W3[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
    }
}

// Function to save model weights to binary file
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
    
    // Save weights
    fwrite(mlp->W1, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->W2, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(mlp->W3, sizeof(float), mlp->output_dim * mlp->input_dim, file);
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    fwrite(mlp->W1_m, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->W1_v, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->W2_m, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(mlp->W2_v, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(mlp->W3_m, sizeof(float), mlp->output_dim * mlp->input_dim, file);
    fwrite(mlp->W3_v, sizeof(float), mlp->output_dim * mlp->input_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to load model weights from binary file
MLP* load_mlp(const char* filename, int custom_batch_size) {
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
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(mlp->W1, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->W2, sizeof(float), output_dim * hidden_dim, file);
    fread(mlp->W3, sizeof(float), output_dim * input_dim, file);
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    fread(mlp->W1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->W1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->W2_m, sizeof(float), output_dim * hidden_dim, file);
    fread(mlp->W2_v, sizeof(float), output_dim * hidden_dim, file);
    fread(mlp->W3_m, sizeof(float), output_dim * input_dim, file);
    fread(mlp->W3_v, sizeof(float), output_dim * input_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}