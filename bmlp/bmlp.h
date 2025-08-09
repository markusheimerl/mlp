#ifndef BMLP_H
#define BMLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

typedef struct {
    // Base weights and gradients
    float* W1;     // hidden_dim x input_dim
    float* W2;     // output_dim x hidden_dim
    float* W3;     // output_dim x input_dim
    float* W1_grad; // hidden_dim x input_dim
    float* W2_grad; // output_dim x hidden_dim
    float* W3_grad; // output_dim x input_dim
    
    // Input-dependent scaling parameters
    float* u1;     // input_dim (for W1 modulation)
    float* u2;     // hidden_dim (for W2 modulation)
    float* u1_grad; // input_dim
    float* u2_grad; // hidden_dim
    
    // Adam parameters for base weights
    float* W1_m;  // First moment for W1
    float* W1_v;  // Second moment for W1
    float* W2_m;  // First moment for W2
    float* W2_v;  // Second moment for W2
    float* W3_m;  // First moment for W3
    float* W3_v;  // Second moment for W3
    
    // Adam parameters for scaling weights
    float* u1_m;  // First moment for u1
    float* u1_v;  // Second moment for u1
    float* u2_m;  // First moment for u2
    float* u2_v;  // Second moment for u2
    
    float beta1;   // Exponential decay rate for first moment
    float beta2;   // Exponential decay rate for second moment
    float epsilon; // Small constant for numerical stability
    int t;         // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Layer outputs and working buffers
    float* layer1_preact;   // batch_size x hidden_dim
    float* layer1_output;   // batch_size x hidden_dim (same as preact for BMLP)
    float* layer2_output;   // batch_size x output_dim
    float* error_hidden;    // batch_size x hidden_dim
    float* error_output;    // batch_size x output_dim
    
    // Temporary buffers for scaling operations
    float* W1_scaled;          // hidden_dim x input_dim (input-scaled weights)
    float* W2_scaled;          // output_dim x hidden_dim (hidden-scaled weights)
    float* input_scale;        // input_dim (scaling factors from input)
    float* hidden_scale;       // hidden_dim (scaling factors from hidden)
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
} BMLP;

// Function prototypes
BMLP* init_bmlp(int input_dim, int hidden_dim, int output_dim, int batch_size);
void free_bmlp(BMLP* bmlp);
void forward_pass_bmlp(BMLP* bmlp, float* X);
float calculate_loss_bmlp(BMLP* bmlp, float* y);
void zero_gradients_bmlp(BMLP* bmlp);
void backward_pass_bmlp(BMLP* bmlp, float* X);
void update_weights_bmlp(BMLP* bmlp, float learning_rate);
void save_bmlp(BMLP* bmlp, const char* filename);
BMLP* load_bmlp(const char* filename, int custom_batch_size);

#endif