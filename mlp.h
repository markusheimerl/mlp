#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Weights and gradients
    float** W1;      // [num_layers][hidden_dim x input_dim]
    float** W2;      // [num_layers][output_dim x hidden_dim]
    float** W3;      // [num_layers][output_dim x input_dim]
    float** W1_grad; // [num_layers][hidden_dim x input_dim]
    float** W2_grad; // [num_layers][output_dim x hidden_dim]
    float** W3_grad; // [num_layers][output_dim x input_dim]
    
    // Adam parameters
    float** W1_m;    // First moment estimates for W1
    float** W1_v;    // Second moment estimates for W1
    float** W2_m;    // First moment estimates for W2
    float** W2_v;    // Second moment estimates for W2
    float** W3_m;    // First moment estimates for W3
    float** W3_v;    // Second moment estimates for W3
    float beta1;     // Exponential decay rate for first moment estimates
    float beta2;     // Exponential decay rate for second moment estimates
    float epsilon;   // Small constant for numerical stability
    int t;           // Time step
    float weight_decay; // Weight decay parameter for AdamW regularization
    
    // Layer outputs and working buffers
    float** layer_preact;  // [num_layers][batch_size x hidden_dim]
    float** layer_postact; // [num_layers][batch_size x hidden_dim]
    float** layer_output;  // [num_layers][batch_size x output_dim]
    float** error_hidden;  // [num_layers][batch_size x hidden_dim]
    float** error_output;  // [num_layers][batch_size x output_dim]
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int num_layers;
    int batch_size;
} MLP;

// Function prototypes
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int num_layers, int batch_size);
void free_mlp(MLP* mlp);
void forward_pass_mlp(MLP* mlp, float* X);
float calculate_loss_mlp(MLP* mlp, float* y);
void zero_gradients_mlp(MLP* mlp);
void backward_pass_mlp(MLP* mlp, float* X);
void update_weights_mlp(MLP* mlp, float learning_rate);
void save_mlp(MLP* mlp, const char* filename);
MLP* load_mlp(const char* filename, int custom_batch_size);

#endif