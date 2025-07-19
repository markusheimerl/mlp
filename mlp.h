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

// Function prototypes
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size);
void free_mlp(MLP* mlp);
void forward_pass_mlp(MLP* mlp, float* X);
float calculate_loss_mlp(MLP* mlp, float* y);
void zero_gradients_mlp(MLP* mlp);
void backward_pass_mlp(MLP* mlp, float* X);
void update_weights_mlp(MLP* mlp, float learning_rate);
void save_mlp(MLP* mlp, const char* filename);
MLP* load_mlp(const char* filename, int custom_batch_size);

#endif