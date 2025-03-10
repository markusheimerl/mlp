#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients
    float* fc1_weight;     // hidden_dim x input_dim
    float* fc2_weight;     // output_dim x hidden_dim
    float* fc1_weight_grad; // hidden_dim x input_dim
    float* fc2_weight_grad; // output_dim x hidden_dim
    
    // Adam parameters
    float* fc1_m;  // First moment for fc1
    float* fc1_v;  // Second moment for fc1
    float* fc2_m;  // First moment for fc2
    float* fc2_v;  // Second moment for fc2
    float beta1;   // Exponential decay rate for first moment
    float beta2;   // Exponential decay rate for second moment
    float epsilon; // Small constant for numerical stability
    int t;         // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Helper arrays for forward/backward pass
    float* layer1_output;   // batch_size x hidden_dim
    float* predictions;     // batch_size x output_dim
    float* error;          // batch_size x output_dim
    float* pre_activation; // batch_size x hidden_dim
    float* error_hidden;   // batch_size x hidden_dim
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} MLP;

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
    mlp->fc1_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    mlp->fc2_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    mlp->fc1_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    mlp->fc2_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate Adam buffers
    mlp->fc1_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    mlp->fc1_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    mlp->fc2_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    mlp->fc2_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    
    // Allocate helper arrays
    mlp->layer1_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    mlp->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    mlp->pre_activation = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    mlp->error_hidden = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    // Initialize weights
    float scale1 = 1.0f / sqrt(input_dim);
    float scale2 = 1.0f / sqrt(hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        mlp->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale1;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        mlp->fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2 - 1) * scale2;
    }
    
    return mlp;
}

// Free network memory
void free_mlp(MLP* mlp) {
    free(mlp->fc1_weight);
    free(mlp->fc2_weight);
    free(mlp->fc1_weight_grad);
    free(mlp->fc2_weight_grad);
    free(mlp->fc1_m);
    free(mlp->fc1_v);
    free(mlp->fc2_m);
    free(mlp->fc2_v);
    free(mlp->layer1_output);
    free(mlp->predictions);
    free(mlp->error);
    free(mlp->pre_activation);
    free(mlp->error_hidden);
    free(mlp);
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* X) {
    // Z = XW₁
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                mlp->batch_size,
                mlp->hidden_dim,
                mlp->input_dim,
                1.0f,
                X,
                mlp->input_dim,
                mlp->fc1_weight,
                mlp->hidden_dim,
                0.0f,
                mlp->layer1_output,
                mlp->hidden_dim);
    
    // Store Z for backward pass
    memcpy(mlp->pre_activation, mlp->layer1_output, 
           mlp->batch_size * mlp->hidden_dim * sizeof(float));
    
    // A = Zσ(Z)
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
        mlp->layer1_output[i] = mlp->layer1_output[i] / (1.0f + expf(-mlp->layer1_output[i]));
    }
    
    // Y = AW₂
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                mlp->batch_size,
                mlp->output_dim,
                mlp->hidden_dim,
                1.0f,
                mlp->layer1_output,
                mlp->hidden_dim,
                mlp->fc2_weight,
                mlp->output_dim,
                0.0f,
                mlp->predictions,
                mlp->output_dim);
}

// Calculate loss
float calculate_loss_mlp(MLP* mlp, float* y) {
    // ∂L/∂Y = Y - Y_true
    float loss = 0.0f;
    for (int i = 0; i < mlp->batch_size * mlp->output_dim; i++) {
        mlp->error[i] = mlp->predictions[i] - y[i];
        loss += mlp->error[i] * mlp->error[i];
    }
    return loss / (mlp->batch_size * mlp->output_dim);
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    memset(mlp->fc1_weight_grad, 0, mlp->hidden_dim * mlp->input_dim * sizeof(float));
    memset(mlp->fc2_weight_grad, 0, mlp->output_dim * mlp->hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* X) {
    // ∂L/∂W₂ = Aᵀ(∂L/∂Y)
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                mlp->hidden_dim,
                mlp->output_dim,
                mlp->batch_size,
                1.0f,
                mlp->layer1_output,
                mlp->hidden_dim,
                mlp->error,
                mlp->output_dim,
                0.0f,
                mlp->fc2_weight_grad,
                mlp->output_dim);
    
    // ∂L/∂A = (∂L/∂Y)(W₂)ᵀ
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                mlp->batch_size,
                mlp->hidden_dim,
                mlp->output_dim,
                1.0f,
                mlp->error,
                mlp->output_dim,
                mlp->fc2_weight,
                mlp->output_dim,
                0.0f,
                mlp->error_hidden,
                mlp->hidden_dim);
    
    // ∂L/∂Z = ∂L/∂A ⊙ [σ(Z) + Zσ(Z)(1-σ(Z))]
    for (int i = 0; i < mlp->batch_size * mlp->hidden_dim; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-mlp->pre_activation[i]));
        mlp->error_hidden[i] *= sigmoid + mlp->pre_activation[i] * sigmoid * (1.0f - sigmoid);
    }
    
    // ∂L/∂W₁ = Xᵀ(∂L/∂Z)
    cblas_sgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                mlp->input_dim,
                mlp->hidden_dim,
                mlp->batch_size,
                1.0f,
                X,
                mlp->input_dim,
                mlp->error_hidden,
                mlp->hidden_dim,
                0.0f,
                mlp->fc1_weight_grad,
                mlp->hidden_dim);
}

// Update weights using AdamW
void update_weights_mlp(MLP* mlp, float learning_rate) {
    mlp->t++;  // Increment time step
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update fc1 weights
    for (int i = 0; i < mlp->hidden_dim * mlp->input_dim; i++) {
        float grad = mlp->fc1_weight_grad[i] / mlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->fc1_m[i] = mlp->beta1 * mlp->fc1_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->fc1_v[i] = mlp->beta2 * mlp->fc1_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->fc1_m[i] / (sqrtf(mlp->fc1_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->fc1_weight[i] = mlp->fc1_weight[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
    }
    
    // Update fc2 weights
    for (int i = 0; i < mlp->output_dim * mlp->hidden_dim; i++) {
        float grad = mlp->fc2_weight_grad[i] / mlp->batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        mlp->fc2_m[i] = mlp->beta1 * mlp->fc2_m[i] + (1.0f - mlp->beta1) * grad;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        mlp->fc2_v[i] = mlp->beta2 * mlp->fc2_v[i] + (1.0f - mlp->beta2) * grad * grad;
        
        float update = alpha_t * mlp->fc2_m[i] / (sqrtf(mlp->fc2_v[i]) + mlp->epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        mlp->fc2_weight[i] = mlp->fc2_weight[i] * (1.0f - learning_rate * mlp->weight_decay) - update;
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
    fwrite(mlp->fc1_weight, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->fc2_weight, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    fwrite(mlp->fc1_m, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->fc1_v, sizeof(float), mlp->hidden_dim * mlp->input_dim, file);
    fwrite(mlp->fc2_m, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);
    fwrite(mlp->fc2_v, sizeof(float), mlp->output_dim * mlp->hidden_dim, file);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Function to load model weights from binary file
MLP* load_mlp(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&batch_size, sizeof(int), 1, file);
    
    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Load weights
    fread(mlp->fc1_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->fc2_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    fread(mlp->fc1_m, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->fc1_v, sizeof(float), hidden_dim * input_dim, file);
    fread(mlp->fc2_m, sizeof(float), output_dim * hidden_dim, file);
    fread(mlp->fc2_v, sizeof(float), output_dim * hidden_dim, file);

    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}

#endif