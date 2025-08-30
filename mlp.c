#include "mlp.h"

// Initialize the network with configurable dimensions
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int num_layers, int batch_size) {
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
    
    // Allocate and initialize weights and gradients
    mlp->W1 = (float**)malloc(num_layers * sizeof(float*));
    mlp->W2 = (float**)malloc(num_layers * sizeof(float*));
    mlp->W3 = (float**)malloc(num_layers * sizeof(float*));
    mlp->W1_grad = (float**)malloc(num_layers * sizeof(float*));
    mlp->W2_grad = (float**)malloc(num_layers * sizeof(float*));
    mlp->W3_grad = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate Adam buffers
    mlp->W1_m = (float**)malloc(num_layers * sizeof(float*));
    mlp->W1_v = (float**)malloc(num_layers * sizeof(float*));
    mlp->W2_m = (float**)malloc(num_layers * sizeof(float*));
    mlp->W2_v = (float**)malloc(num_layers * sizeof(float*));
    mlp->W3_m = (float**)malloc(num_layers * sizeof(float*));
    mlp->W3_v = (float**)malloc(num_layers * sizeof(float*));
    
    // Allocate layer outputs and working buffers
    mlp->layer_preact = (float**)malloc(num_layers * sizeof(float*));
    mlp->layer_postact = (float**)malloc(num_layers * sizeof(float*));
    mlp->layer_output = (float**)malloc(num_layers * sizeof(float*));
    mlp->error_hidden = (float**)malloc(num_layers * sizeof(float*));
    mlp->error_output = (float**)malloc(num_layers * sizeof(float*));
    
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : hidden_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : hidden_dim;
        
        int w1_size = hidden_dim * input_size;
        int w2_size = output_size * hidden_dim;
        int w3_size = output_size * input_size;
        
        mlp->W1[layer] = (float*)malloc(w1_size * sizeof(float));
        mlp->W2[layer] = (float*)malloc(w2_size * sizeof(float));
        mlp->W3[layer] = (float*)malloc(w3_size * sizeof(float));
        mlp->W1_grad[layer] = (float*)malloc(w1_size * sizeof(float));
        mlp->W2_grad[layer] = (float*)malloc(w2_size * sizeof(float));
        mlp->W3_grad[layer] = (float*)malloc(w3_size * sizeof(float));
        
        mlp->W1_m[layer] = (float*)calloc(w1_size, sizeof(float));
        mlp->W1_v[layer] = (float*)calloc(w1_size, sizeof(float));
        mlp->W2_m[layer] = (float*)calloc(w2_size, sizeof(float));
        mlp->W2_v[layer] = (float*)calloc(w2_size, sizeof(float));
        mlp->W3_m[layer] = (float*)calloc(w3_size, sizeof(float));
        mlp->W3_v[layer] = (float*)calloc(w3_size, sizeof(float));
        
        mlp->layer_preact[layer] = (float*)malloc(batch_size * hidden_dim * sizeof(float));
        mlp->layer_postact[layer] = (float*)malloc(batch_size * hidden_dim * sizeof(float));
        mlp->layer_output[layer] = (float*)malloc(batch_size * output_size * sizeof(float));
        mlp->error_hidden[layer] = (float*)malloc(batch_size * hidden_dim * sizeof(float));
        mlp->error_output[layer] = (float*)malloc(batch_size * output_size * sizeof(float));
        
        // Initialize weights
        float scale_W1 = 1.0f / sqrtf(input_size);
        float scale_W2 = 1.0f / sqrtf(hidden_dim);
        float scale_W3 = 1.0f / sqrtf(input_size);
        
        for (int i = 0; i < w1_size; i++) {
            mlp->W1[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
        }
        
        for (int i = 0; i < w2_size; i++) {
            mlp->W2[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
        }
        
        for (int i = 0; i < w3_size; i++) {
            mlp->W3[layer][i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W3;
        }
    }
    
    return mlp;
}

// Free network memory
void free_mlp(MLP* mlp) {
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        free(mlp->W1[layer]); free(mlp->W2[layer]); free(mlp->W3[layer]);
        free(mlp->W1_grad[layer]); free(mlp->W2_grad[layer]); free(mlp->W3_grad[layer]);
        free(mlp->W1_m[layer]); free(mlp->W1_v[layer]);
        free(mlp->W2_m[layer]); free(mlp->W2_v[layer]);
        free(mlp->W3_m[layer]); free(mlp->W3_v[layer]);
        free(mlp->layer_preact[layer]); free(mlp->layer_postact[layer]); free(mlp->layer_output[layer]);
        free(mlp->error_output[layer]); free(mlp->error_hidden[layer]);
    }
    
    free(mlp->W1); free(mlp->W2); free(mlp->W3);
    free(mlp->W1_grad); free(mlp->W2_grad); free(mlp->W3_grad);
    free(mlp->W1_m); free(mlp->W1_v);
    free(mlp->W2_m); free(mlp->W2_v);
    free(mlp->W3_m); free(mlp->W3_v);
    free(mlp->layer_preact); free(mlp->layer_postact); free(mlp->layer_output);
    free(mlp->error_output); free(mlp->error_hidden);
    free(mlp);
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* X) {
    float* input = X;
    
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        // H = XW₁
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    mlp->batch_size, mlp->hidden_dim, input_size,
                    1.0f, input, input_size,
                    mlp->W1[layer], mlp->hidden_dim,
                    0.0f, mlp->layer_preact[layer], mlp->hidden_dim);
        
        // S = Hσ(H)
        for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
            mlp->layer_postact[layer][i] = mlp->layer_preact[layer][i] / (1.0f + expf(-mlp->layer_preact[layer][i]));
        }
        
        // Y = SW₂
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    mlp->batch_size, output_size, mlp->hidden_dim,
                    1.0f, mlp->layer_postact[layer], mlp->hidden_dim,
                    mlp->W2[layer], output_size,
                    0.0f, mlp->layer_output[layer], output_size);
        
        // Y = Y + XW₃
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    mlp->batch_size, output_size, input_size,
                    1.0f, input, input_size,
                    mlp->W3[layer], output_size,
                    1.0f, mlp->layer_output[layer], output_size);
        
        // Set input for next layer
        if (layer < mlp->num_layers - 1) {
            input = mlp->layer_output[layer];
        }
    }
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* y) {
    // ∂L/∂Y = Y - Y_true
    int last_layer = mlp->num_layers - 1;
    float loss = 0.0f;
    for (int i = 0; i < mlp->batch_size * mlp->output_dim; i++) {
        mlp->error_output[last_layer][i] = mlp->layer_output[last_layer][i] - y[i];
        loss += mlp->error_output[last_layer][i] * mlp->error_output[last_layer][i];
    }
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
        
        memset(mlp->W1_grad[layer], 0, w1_size * sizeof(float));
        memset(mlp->W2_grad[layer], 0, w2_size * sizeof(float));
        memset(mlp->W3_grad[layer], 0, w3_size * sizeof(float));
    }
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* X) {
    for (int layer = mlp->num_layers - 1; layer >= 0; layer--) {
        float* input = (layer == 0) ? X : mlp->layer_output[layer - 1];
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        // ∂L/∂W₂ = S^T(∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    mlp->hidden_dim, output_size, mlp->batch_size,
                    1.0f, mlp->layer_postact[layer], mlp->hidden_dim,
                    mlp->error_output[layer], output_size,
                    1.0f, mlp->W2_grad[layer], output_size);
        
        // ∂L/∂W₃ = X^T(∂L/∂Y)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, output_size, mlp->batch_size,
                    1.0f, input, input_size,
                    mlp->error_output[layer], output_size,
                    1.0f, mlp->W3_grad[layer], output_size);
        
        // ∂L/∂S = (∂L/∂Y)(W₂)^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    mlp->batch_size, mlp->hidden_dim, output_size,
                    1.0f, mlp->error_output[layer], output_size,
                    mlp->W2[layer], output_size,
                    0.0f, mlp->error_hidden[layer], mlp->hidden_dim);
        
        // ∂L/∂H = ∂L/∂S ⊙ [σ(H) + Hσ(H)(1-σ(H))]
        for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
            float h = mlp->layer_preact[layer][i];
            float sigmoid = 1.0f / (1.0f + expf(-h));
            mlp->error_hidden[layer][i] *= sigmoid + h * sigmoid * (1.0f - sigmoid);
        }
        
        // ∂L/∂W₁ = X^T(∂L/∂H)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    input_size, mlp->hidden_dim, mlp->batch_size,
                    1.0f, input, input_size,
                    mlp->error_hidden[layer], mlp->hidden_dim,
                    1.0f, mlp->W1_grad[layer], mlp->hidden_dim);
        
        // Propagate error to previous layer
        if (layer > 0) {
            // ∂L/∂X = (∂L/∂H)(W₁)^T + (∂L/∂Y)(W₃)^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        mlp->batch_size, input_size, mlp->hidden_dim,
                        1.0f, mlp->error_hidden[layer], mlp->hidden_dim,
                        mlp->W1[layer], mlp->hidden_dim,
                        0.0f, mlp->error_output[layer - 1], input_size);
            
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        mlp->batch_size, input_size, output_size,
                        1.0f, mlp->error_output[layer], output_size,
                        mlp->W3[layer], output_size,
                        1.0f, mlp->error_output[layer - 1], input_size);
        }
    }
}

// Update weights using AdamW
void update_weights_mlp(MLP* mlp, float learning_rate) {
    mlp->t++;  // Increment time step
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        // Update W1 weights
        for (int i = 0; i < w1_size; i++) {
            float grad = mlp->W1_grad[layer][i] / mlp->batch_size;
            
            // m = β₁m + (1-β₁)(∂L/∂W)
            mlp->W1_m[layer][i] = mlp->beta1 * mlp->W1_m[layer][i] + (1.0f - mlp->beta1) * grad;
            // v = β₂v + (1-β₂)(∂L/∂W)²
            mlp->W1_v[layer][i] = mlp->beta2 * mlp->W1_v[layer][i] + (1.0f - mlp->beta2) * grad * grad;
            
            float update = alpha_t * mlp->W1_m[layer][i] / (sqrtf(mlp->W1_v[layer][i]) + mlp->epsilon);
            // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
            mlp->W1[layer][i] = mlp->W1[layer][i] * (1.0f - learning_rate * mlp->weight_decay) - update;
        }
        
        // Update W2 weights
        for (int i = 0; i < w2_size; i++) {
            float grad = mlp->W2_grad[layer][i] / mlp->batch_size;

            // m = β₁m + (1-β₁)(∂L/∂W)
            mlp->W2_m[layer][i] = mlp->beta1 * mlp->W2_m[layer][i] + (1.0f - mlp->beta1) * grad;
            // v = β₂v + (1-β₂)(∂L/∂W)²
            mlp->W2_v[layer][i] = mlp->beta2 * mlp->W2_v[layer][i] + (1.0f - mlp->beta2) * grad * grad;
            
            float update = alpha_t * mlp->W2_m[layer][i] / (sqrtf(mlp->W2_v[layer][i]) + mlp->epsilon);
            // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
            mlp->W2[layer][i] = mlp->W2[layer][i] * (1.0f - learning_rate * mlp->weight_decay) - update;
        }
        
        // Update W3 weights
        for (int i = 0; i < w3_size; i++) {
            float grad = mlp->W3_grad[layer][i] / mlp->batch_size;
            
            // m = β₁m + (1-β₁)(∂L/∂W)
            mlp->W3_m[layer][i] = mlp->beta1 * mlp->W3_m[layer][i] + (1.0f - mlp->beta1) * grad;
            // v = β₂v + (1-β₂)(∂L/∂W)²
            mlp->W3_v[layer][i] = mlp->beta2 * mlp->W3_v[layer][i] + (1.0f - mlp->beta2) * grad * grad;
            
            float update = alpha_t * mlp->W3_m[layer][i] / (sqrtf(mlp->W3_v[layer][i]) + mlp->epsilon);
            // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
            mlp->W3[layer][i] = mlp->W3[layer][i] * (1.0f - learning_rate * mlp->weight_decay) - update;
        }
    }
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
    fwrite(&mlp->num_layers, sizeof(int), 1, file);
    fwrite(&mlp->batch_size, sizeof(int), 1, file);
    
    // Save weights for each layer
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        fwrite(mlp->W1[layer], sizeof(float), w1_size, file);
        fwrite(mlp->W2[layer], sizeof(float), w2_size, file);
        fwrite(mlp->W3[layer], sizeof(float), w3_size, file);
    }
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    for (int layer = 0; layer < mlp->num_layers; layer++) {
        int input_size = (layer == 0) ? mlp->input_dim : mlp->hidden_dim;
        int output_size = (layer == mlp->num_layers - 1) ? mlp->output_dim : mlp->hidden_dim;
        
        int w1_size = mlp->hidden_dim * input_size;
        int w2_size = output_size * mlp->hidden_dim;
        int w3_size = output_size * input_size;
        
        fwrite(mlp->W1_m[layer], sizeof(float), w1_size, file);
        fwrite(mlp->W1_v[layer], sizeof(float), w1_size, file);
        fwrite(mlp->W2_m[layer], sizeof(float), w2_size, file);
        fwrite(mlp->W2_v[layer], sizeof(float), w2_size, file);
        fwrite(mlp->W3_m[layer], sizeof(float), w3_size, file);
        fwrite(mlp->W3_v[layer], sizeof(float), w3_size, file);
    }

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
    int input_dim, hidden_dim, output_dim, num_layers, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, num_layers, batch_size);
    
    // Load weights for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : hidden_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : hidden_dim;
        
        int w1_size = hidden_dim * input_size;
        int w2_size = output_size * hidden_dim;
        int w3_size = output_size * input_size;
        
        fread(mlp->W1[layer], sizeof(float), w1_size, file);
        fread(mlp->W2[layer], sizeof(float), w2_size, file);
        fread(mlp->W3[layer], sizeof(float), w3_size, file);
    }
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    for (int layer = 0; layer < num_layers; layer++) {
        int input_size = (layer == 0) ? input_dim : hidden_dim;
        int output_size = (layer == num_layers - 1) ? output_dim : hidden_dim;
        
        int w1_size = hidden_dim * input_size;
        int w2_size = output_size * hidden_dim;
        int w3_size = output_size * input_size;
        
        fread(mlp->W1_m[layer], sizeof(float), w1_size, file);
        fread(mlp->W1_v[layer], sizeof(float), w1_size, file);
        fread(mlp->W2_m[layer], sizeof(float), w2_size, file);
        fread(mlp->W2_v[layer], sizeof(float), w2_size, file);
        fread(mlp->W3_m[layer], sizeof(float), w3_size, file);
        fread(mlp->W3_v[layer], sizeof(float), w3_size, file);
    }

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}