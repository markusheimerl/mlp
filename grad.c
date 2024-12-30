#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { ADD, MATMUL, RELU, SIGMOID, RESHAPE } OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape;

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float sigmoid_derivative(float x) { float s = sigmoid(x); return s * (1 - s); }
static float relu(float x) { return x > 0 ? x : 0; }
static float relu_derivative(float x) { return x > 0 ? 1 : 0; }

static int calc_size(const int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i < ndims; i++) size *= dims[i];
    return size;
}

Tensor* tensor_new(int ndims, int* dims, float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->size = calc_size(dims, ndims);
    t->dims = malloc(ndims * sizeof(int));
    t->data = malloc(t->size * sizeof(float));
    memcpy(t->dims, dims, ndims * sizeof(int));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    if ((t->requires_grad = requires_grad)) t->grad = calloc(t->size, sizeof(float));
    return t;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data); free(t->grad); free(t->dims); free(t);
}

Tensor* tensor_reshape(Tensor* t, int new_ndims, int* new_dims) {
    if (calc_size(new_dims, new_ndims) != t->size) return NULL;
    Tensor* result = tensor_new(new_ndims, new_dims, t->data, t->requires_grad);
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){RESHAPE, result, t, NULL};
    return result;
}

static Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (op == RELU || op == SIGMOID) {
        Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == RELU ? relu(a->data[i]) : sigmoid(a->data[i]);
        if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){op, result, a, NULL};
        return result;
    }

    int out_dims[MAX_DIMS], out_ndims = op == ADD ? a->ndims : MAX(a->ndims, b->ndims);
    if (op == ADD) {
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else {
        for (int i = 0; i < out_ndims - 2; i++) out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[out_ndims-2] = a->dims[a->ndims-2];
        out_dims[out_ndims-1] = b->dims[b->ndims-1];
    }

    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    
    if (op == ADD) {
        for (int i = 0; i < result->size; i++) result->data[i] = a->data[i] + b->data[i];
    } else if (op == MATMUL) {
        int batch_size = calc_size(out_dims, out_ndims - 2);
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++) {
            float *out = &result->data[batch * m * p];
            const float *a_data = &a->data[batch * m * n];
            const float *b_data = &b->data[batch * n * p];
            
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++) {
                    float aik = a_data[i * n + k];
                    for (int j = 0; j < p; j++)
                        out[i * p + j] += aik * b_data[k * p + j];
                }
        }
    }
    
    if (result->requires_grad) tape.entries[tape.len++] = (TapeEntry){op, result, a, b};
    return result;
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)
#define tensor_relu(a) tensor_op(a, NULL, RELU)
#define tensor_sigmoid(a) tensor_op(a, NULL, SIGMOID)

void backward() {
    if (tape.len == 0) return;
    
    Tensor* final = tape.entries[tape.len - 1].result;
    if (!final->grad) final->grad = calloc(final->size, sizeof(float));
    for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
    
    for (int i = tape.len - 1; i >= 0; i--) {
        TapeEntry* entry = &tape.entries[i];
        Tensor *t = entry->result, *a = entry->input1, *b = entry->input2;
        
        if (a->requires_grad && !a->grad) a->grad = calloc(a->size, sizeof(float));
        if (b && b->requires_grad && !b->grad) b->grad = calloc(b->size, sizeof(float));
        
        switch (entry->op) {
            case RESHAPE:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++) a->grad[j] += t->grad[j];
                break;
                
            case RELU:
            case SIGMOID:
                if (a->requires_grad)
                    for (int j = 0; j < a->size; j++)
                        a->grad[j] += t->grad[j] * (entry->op == RELU ? 
                            relu_derivative(a->data[j]) : sigmoid_derivative(a->data[j]));
                break;
                
            case ADD:
                if (a->requires_grad)
                    for (int j = 0; j < a->size; j++) a->grad[j] += t->grad[j];
                if (b && b->requires_grad)
                    for (int j = 0; j < b->size; j++) b->grad[j] += t->grad[j];
                break;
                
            case MATMUL: {
                int batch_size = calc_size(t->dims, t->ndims - 2);
                int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
                
                for (int batch = 0; batch < batch_size; batch++) {
                    int a_off = batch * m * n, b_off = batch * n * p, t_off = batch * m * p;
                    
                    if (a->requires_grad)
                        for (int i = 0; i < m; i++)
                            for (int k = 0; k < n; k++) {
                                float sum = 0;
                                for (int j = 0; j < p; j++)
                                    sum += t->grad[t_off + i * p + j] * b->data[b_off + k * p + j];
                                a->grad[a_off + i * n + k] += sum;
                            }
                    
                    if (b->requires_grad)
                        for (int k = 0; k < n; k++)
                            for (int j = 0; j < p; j++) {
                                float sum = 0;
                                for (int i = 0; i < m; i++)
                                    sum += t->grad[t_off + i * p + j] * a->data[a_off + i * n + k];
                                b->grad[b_off + k * p + j] += sum;
                            }
                }
                break;
            }
        }
    }
}

int main() {
    // Initialize input tensor
    int input_dims[] = {2, 3, 4, 4};
    float *input_data = malloc(96 * sizeof(float));
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < 4; h++)
                for (int w = 0; w < 4; w++)
                    input_data[b*48 + c*16 + h*4 + w] = (float)(h + w)/8.0f + (float)c/3.0f + (float)b*0.1f;
    
    Tensor *x = tensor_new(4, input_dims, input_data, 1);
    printf("Input tensor (first batch, first channel):\n");
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) printf("%.3f ", x->data[h*4 + w]);
        printf("\n");
    }
    printf("\n");

    // Create and initialize weights
    int w1_dims[] = {48, 64}, w2_dims[] = {64, 32};
    float *w1_data = malloc(48 * 64 * sizeof(float));
    float *w2_data = malloc(64 * 32 * sizeof(float));
    for (int i = 0; i < 48 * 64; i++) w1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < 64 * 32; i++) w2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    Tensor *w1 = tensor_new(2, w1_dims, w1_data, 1);
    Tensor *w2 = tensor_new(2, w2_dims, w2_data, 1);

    // Forward pass
    int reshaped_dims[] = {2, 48};
    Tensor *x_reshaped = tensor_reshape(x, 2, reshaped_dims);
    printf("Reshaped input dimensions: [%d, %d]\n", x_reshaped->dims[0], x_reshaped->dims[1]);

    Tensor *h1 = tensor_relu(tensor_matmul(x_reshaped, w1));
    printf("First hidden layer dimensions: [%d, %d]\n", h1->dims[0], h1->dims[1]);

    Tensor *output = tensor_sigmoid(tensor_matmul(h1, w2));
    printf("Output dimensions: [%d, %d]\n\n", output->dims[0], output->dims[1]);

    printf("Output sample (first batch):\n");
    for (int i = 0; i < 8; i++) printf("%.3f ", output->data[i]);
    printf("...\n\n");

    // Backward pass and gradient statistics
    backward();
    
    float w1_grad_mean = 0, w2_grad_mean = 0, w1_grad_max = 0, w2_grad_max = 0;
    for (int i = 0; i < w1->size; i++) {
        w1_grad_mean += fabsf(w1->grad[i]);
        w1_grad_max = fmaxf(w1_grad_max, fabsf(w1->grad[i]));
    }
    for (int i = 0; i < w2->size; i++) {
        w2_grad_mean += fabsf(w2->grad[i]);
        w2_grad_max = fmaxf(w2_grad_max, fabsf(w2->grad[i]));
    }
    printf("Gradient statistics:\n");
    printf("W1 gradients - Mean: %.6f, Max: %.6f\n", w1_grad_mean/w1->size, w1_grad_max);
    printf("W2 gradients - Mean: %.6f, Max: %.6f\n", w2_grad_mean/w2->size, w2_grad_max);

    // Cleanup
    tensor_free(output); tensor_free(w2); tensor_free(h1);
    tensor_free(x_reshaped); tensor_free(w1); tensor_free(x);
    free(input_data); free(w1_data); free(w2_data);
    tape.len = 0;

    return 0;
}