#include "mlp.h"

// Initialize the network
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
    
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    
    // Allocate weights and gradients
    mlp->W1 = (float*)malloc(w1_size * sizeof(float));
    mlp->W2 = (float*)malloc(w2_size * sizeof(float));
    mlp->W1_grad = (float*)malloc(w1_size * sizeof(float));
    mlp->W2_grad = (float*)malloc(w2_size * sizeof(float));
    
    // Allocate Adam buffers
    mlp->W1_m = (float*)calloc(w1_size, sizeof(float));
    mlp->W1_v = (float*)calloc(w1_size, sizeof(float));
    mlp->W2_m = (float*)calloc(w2_size, sizeof(float));
    mlp->W2_v = (float*)calloc(w2_size, sizeof(float));
    
    // Allocate layer outputs and working buffers
    mlp->preact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->postact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->output = (float*)malloc(batch_size * output_dim * sizeof(float));
    mlp->grad_postact = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->grad_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize weights
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    
    for (int i = 0; i < w1_size; i++) {
        mlp->W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < w2_size; i++) {
        mlp->W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    return mlp;
}

// Free network memory
void free_mlp(MLP* mlp) {
    free(mlp->W1); free(mlp->W2);
    free(mlp->W1_grad); free(mlp->W2_grad);
    free(mlp->W1_m); free(mlp->W1_v);
    free(mlp->W2_m); free(mlp->W2_v);
    free(mlp->preact); free(mlp->postact); free(mlp->output);
    free(mlp->grad_output); free(mlp->grad_postact);
    free(mlp);
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* X) {
    // H = XW₁
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->hidden_dim, mlp->input_dim,
                1.0f, X, mlp->input_dim,
                mlp->W1, mlp->hidden_dim,
                0.0f, mlp->preact, mlp->hidden_dim);
    
    // S = H⊙σ(H)
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
        mlp->postact[i] = mlp->preact[i] / (1.0f + expf(-mlp->preact[i]));
    }
    
    // Y = SW₂
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                mlp->batch_size, mlp->output_dim, mlp->hidden_dim,
                1.0f, mlp->postact, mlp->hidden_dim,
                mlp->W2, mlp->output_dim,
                0.0f, mlp->output, mlp->output_dim);
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* y) {
    // ∂L/∂Y = Y - Y_true
    cblas_scopy(mlp->batch_size * mlp->output_dim, 
                mlp->output, 1, 
                mlp->grad_output, 1);
    cblas_saxpy(mlp->batch_size * mlp->output_dim, 
                -1.0f, y, 1, 
                mlp->grad_output, 1);
    
    float loss = cblas_sdot(mlp->batch_size * mlp->output_dim, 
                           mlp->grad_output, 1, 
                           mlp->grad_output, 1);
    return loss / (mlp->batch_size * mlp->output_dim);
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    memset(mlp->W1_grad, 0, w1_size * sizeof(float));
    memset(mlp->W2_grad, 0, w2_size * sizeof(float));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* X, float* grad_X) {
    // ∂L/∂W₂ = Sᵀ(∂L/∂Y)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                mlp->hidden_dim, mlp->output_dim, mlp->batch_size,
                1.0f, mlp->postact, mlp->hidden_dim,
                mlp->grad_output, mlp->output_dim,
                1.0f, mlp->W2_grad, mlp->output_dim);
    
    // ∂L/∂S = (∂L/∂Y)W₂ᵀ
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                mlp->batch_size, mlp->hidden_dim, mlp->output_dim,
                1.0f, mlp->grad_output, mlp->output_dim,
                mlp->W2, mlp->output_dim,
                0.0f, mlp->grad_postact, mlp->hidden_dim);
    
    // ∂L/∂H = ∂L/∂S⊙[σ(H)+H⊙σ(H)⊙(1-σ(H))]
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
        float h = mlp->preact[i];
        float sigmoid = 1.0f / (1.0f + expf(-h));
        mlp->grad_postact[i] *= sigmoid + h * sigmoid * (1.0f - sigmoid);
    }
    
    // ∂L/∂W₁ = Xᵀ(∂L/∂H)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                mlp->input_dim, mlp->hidden_dim, mlp->batch_size,
                1.0f, X, mlp->input_dim,
                mlp->grad_postact, mlp->hidden_dim,
                1.0f, mlp->W1_grad, mlp->hidden_dim);
    
    if (grad_X != NULL) {
        // ∂L/∂X = (∂L/∂H)W₁ᵀ
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    mlp->batch_size, mlp->input_dim, mlp->hidden_dim,
                    1.0f, mlp->grad_postact, mlp->hidden_dim,
                    mlp->W1, mlp->hidden_dim,
                    0.0f, grad_X, mlp->input_dim);
    }
}

// Update weights using AdamW
void update_weights_mlp(MLP* mlp, float learning_rate, int effective_batch_size) {
    mlp->t++;  // Increment time step
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Update W₁ weights
    for (int i = 0; i < w1_size; i++) {
        float grad = mlp->W1_grad[i] / effective_batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->W1_m[i] = mlp->beta1 * mlp->W1_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->W1_v[i] = mlp->beta2 * mlp->W1_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->W1_m[i] / (sqrtf(mlp->W1_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->W1[i] = mlp->W1[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
    }
    
    // Update W₂ weights
    for (int i = 0; i < w2_size; i++) {
        float grad = mlp->W2_grad[i] / effective_batch_size;

        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->W2_m[i] = mlp->beta1 * mlp->W2_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->W2_v[i] = mlp->beta2 * mlp->W2_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->W2_m[i] / (sqrtf(mlp->W2_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->W2[i] = mlp->W2[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
    }
}

// Reset optimizer state
void reset_optimizer_mlp(MLP* mlp) {
    int w1_size = mlp->input_dim * mlp->hidden_dim;
    int w2_size = mlp->hidden_dim * mlp->output_dim;
    
    // Reset Adam moment estimates to zero
    memset(mlp->W1_m, 0, w1_size * sizeof(float));
    memset(mlp->W1_v, 0, w1_size * sizeof(float));
    memset(mlp->W2_m, 0, w2_size * sizeof(float));
    memset(mlp->W2_v, 0, w2_size * sizeof(float));
    
    // Reset time step
    mlp->t = 0;
}

// Save model weights and Adam state to binary file
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
    
    // Save weights
    fwrite(mlp->W1, sizeof(float), w1_size, file);
    fwrite(mlp->W2, sizeof(float), w2_size, file);
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    fwrite(mlp->W1_m, sizeof(float), w1_size, file);
    fwrite(mlp->W1_v, sizeof(float), w1_size, file);
    fwrite(mlp->W2_m, sizeof(float), w2_size, file);
    fwrite(mlp->W2_v, sizeof(float), w2_size, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights and Adam state from binary file
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
    
    int w1_size = input_dim * hidden_dim;
    int w2_size = hidden_dim * output_dim;
    
    // Load weights
    fread(mlp->W1, sizeof(float), w1_size, file);
    fread(mlp->W2, sizeof(float), w2_size, file);
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    fread(mlp->W1_m, sizeof(float), w1_size, file);
    fread(mlp->W1_v, sizeof(float), w1_size, file);
    fread(mlp->W2_m, sizeof(float), w2_size, file);
    fread(mlp->W2_v, sizeof(float), w2_size, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}