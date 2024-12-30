#ifndef __GRAD_H__
#define __GRAD_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DIMS 8
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef enum { 
    ADD, MATMUL, RELU, SIGMOID, RESHAPE, SLICE, 
    PERMUTE, GATHER, HADAMARD, POW, EXP, REDUCE_SUM 
} OpType;

typedef struct {
    float *data, *grad;
    int *dims, ndims, size;
    int requires_grad;
} Tensor;

typedef struct {
    OpType op;
    Tensor *result, *input1, *input2;
    int *aux_data1, *aux_data2, *aux_data3;
    int aux_int;
} TapeEntry;

static struct { TapeEntry entries[1000]; int len; } tape;

// Helper functions
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-fmaxf(fminf(x, 88.0f), -88.0f))); }
static float d_sigmoid(float x) { float s = sigmoid(x); return s * (1 - s); }
static float relu(float x) { return x > 0 ? x : 0; }
static float d_relu(float x) { return x > 0 ? 1 : 0; }
static float safe_exp(float x) { return expf(fmaxf(fminf(x, 88.0f), -88.0f)); }

static int coords_to_index(const int* coords, const int* dims, int ndims) {
    int index = 0, stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        index += coords[i] * stride;
        stride *= dims[i];
    }
    return index;
}

static void index_to_coords(int index, int* coords, const int* dims, int ndims) {
    for (int i = ndims - 1; i >= 0; i--) {
        coords[i] = index % dims[i];
        index /= dims[i];
    }
}

Tensor* tensor_new(int ndims, const int* dims, const float* data, int requires_grad) {
    Tensor* t = calloc(1, sizeof(Tensor));
    t->ndims = ndims;
    t->dims = malloc(ndims * sizeof(int));
    memcpy(t->dims, dims, ndims * sizeof(int));
    
    t->size = 1;
    for (int i = 0; i < ndims; i++) t->size *= dims[i];
    
    t->data = malloc(t->size * sizeof(float));
    if (data) memcpy(t->data, data, t->size * sizeof(float));
    
    if ((t->requires_grad = requires_grad)) t->grad = calloc(t->size, sizeof(float));
    return t;
}

static void record_operation(OpType op, Tensor* result, Tensor* input1, Tensor* input2, 
                           int* aux1, int* aux2, int* aux3, int aux_int) {
    if (result->requires_grad) {
        tape.entries[tape.len++] = (TapeEntry){
            .op = op, .result = result, .input1 = input1, .input2 = input2,
            .aux_data1 = aux1, .aux_data2 = aux2, .aux_data3 = aux3, .aux_int = aux_int
        };
    }
}

Tensor* tensor_reduce_sum(Tensor* t, const int* axes, int num_axes) {
    if (!t || !axes || num_axes <= 0 || num_axes > t->ndims) return NULL;
    
    int reduce_dims[MAX_DIMS] = {0}, new_dims[MAX_DIMS];
    int new_ndims = 0;
    
    for (int i = 0; i < num_axes; i++) {
        if (axes[i] < 0 || axes[i] >= t->ndims) return NULL;
        reduce_dims[axes[i]] = 1;
    }
    
    for (int i = 0; i < t->ndims; i++) {
        if (!reduce_dims[i]) new_dims[new_ndims++] = t->dims[i];
    }
    
    Tensor* result = tensor_new(new_ndims, new_dims, NULL, t->requires_grad);
    memset(result->data, 0, result->size * sizeof(float));
    
    int coords[MAX_DIMS], result_coords[MAX_DIMS];
    for (int i = 0; i < t->size; i++) {
        index_to_coords(i, coords, t->dims, t->ndims);
        
        int idx = 0;
        for (int j = 0; j < t->ndims; j++) {
            if (!reduce_dims[j]) result_coords[idx++] = coords[j];
        }
        
        result->data[coords_to_index(result_coords, result->dims, new_ndims)] += t->data[i];
    }
    
    if (result->requires_grad) {
        int* axes_copy = malloc(num_axes * sizeof(int));
        memcpy(axes_copy, axes, num_axes * sizeof(int));
        record_operation(REDUCE_SUM, result, t, NULL, axes_copy, NULL, NULL, num_axes);
    }
    
    return result;
}

Tensor* tensor_gather(Tensor* t, int axis, const int* indices, int num_indices) {
    if (axis < 0 || axis >= t->ndims) return NULL;
    
    int new_dims[MAX_DIMS];
    memcpy(new_dims, t->dims, t->ndims * sizeof(int));
    new_dims[axis] = num_indices;
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS];
    for (int i = 0; i < result->size; i++) {
        index_to_coords(i, coords, result->dims, result->ndims);
        int original_coord = coords[axis];
        coords[axis] = indices[original_coord];
        result->data[i] = t->data[coords_to_index(coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int* indices_copy = malloc(num_indices * sizeof(int));
        memcpy(indices_copy, indices, num_indices * sizeof(int));
        record_operation(GATHER, result, t, NULL, indices_copy, NULL, NULL, axis);
    }
    return result;
}

Tensor* tensor_pow(Tensor* t, float exponent) {
    Tensor* result = tensor_new(t->ndims, t->dims, NULL, t->requires_grad);
    
    for (int i = 0; i < t->size; i++)
        result->data[i] = powf(t->data[i], exponent);
    
    if (result->requires_grad) {
        float* exp_ptr = malloc(sizeof(float));
        *exp_ptr = exponent;
        record_operation(POW, result, t, NULL, (int*)exp_ptr, NULL, NULL, 0);
    }
    return result;
}

Tensor* tensor_slice(Tensor* t, const int* start, const int* end) {
    int new_dims[MAX_DIMS];
    for (int i = 0; i < t->ndims; i++) {
        new_dims[i] = end[i] - start[i];
        if (start[i] < 0 || end[i] > t->dims[i] || start[i] >= end[i]) return NULL;
    }
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], src_coords[MAX_DIMS];
    for (int i = 0; i < result->size; i++) {
        index_to_coords(i, coords, result->dims, result->ndims);
        for (int j = 0; j < t->ndims; j++) src_coords[j] = coords[j] + start[j];
        result->data[i] = t->data[coords_to_index(src_coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int *slice_start = malloc(t->ndims * sizeof(int));
        int *slice_end = malloc(t->ndims * sizeof(int));
        memcpy(slice_start, start, t->ndims * sizeof(int));
        memcpy(slice_end, end, t->ndims * sizeof(int));
        record_operation(SLICE, result, t, NULL, slice_start, slice_end, NULL, 0);
    }
    return result;
}

Tensor* tensor_reshape(Tensor* t, int new_ndims, const int* new_dims) {
    int new_size = 1;
    for (int i = 0; i < new_ndims; i++) new_size *= new_dims[i];
    if (new_size != t->size) return NULL;
    
    Tensor* result = tensor_new(new_ndims, new_dims, t->data, t->requires_grad);
    if (result->requires_grad)
        record_operation(RESHAPE, result, t, NULL, NULL, NULL, NULL, 0);
    return result;
}

Tensor* tensor_permute(Tensor* t, const int* permutation) {
    int new_dims[MAX_DIMS];
    for (int i = 0; i < t->ndims; i++) new_dims[i] = t->dims[permutation[i]];
    
    Tensor* result = tensor_new(t->ndims, new_dims, NULL, t->requires_grad);
    
    int coords[MAX_DIMS], new_coords[MAX_DIMS];
    for (int i = 0; i < t->size; i++) {
        index_to_coords(i, new_coords, result->dims, result->ndims);
        for (int j = 0; j < t->ndims; j++) coords[permutation[j]] = new_coords[j];
        result->data[i] = t->data[coords_to_index(coords, t->dims, t->ndims)];
    }
    
    if (result->requires_grad) {
        int* perm_copy = malloc(t->ndims * sizeof(int));
        memcpy(perm_copy, permutation, t->ndims * sizeof(int));
        record_operation(PERMUTE, result, t, NULL, perm_copy, NULL, NULL, 0);
    }
    return result;
}

Tensor* tensor_op(Tensor* a, Tensor* b, OpType op) {
    if (op == RELU || op == SIGMOID || op == EXP) {
        Tensor* result = tensor_new(a->ndims, a->dims, NULL, a->requires_grad);
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == RELU ? relu(a->data[i]) : 
                             op == SIGMOID ? sigmoid(a->data[i]) : safe_exp(a->data[i]);
        if (result->requires_grad)
            record_operation(op, result, a, NULL, NULL, NULL, NULL, 0);
        return result;
    }

    int out_dims[MAX_DIMS], out_ndims = op == ADD ? a->ndims : MAX(a->ndims, b->ndims);
    if (op == ADD || op == HADAMARD) {
        memcpy(out_dims, a->dims, a->ndims * sizeof(int));
    } else {
        for (int i = 0; i < out_ndims - 2; i++) out_dims[i] = MAX(a->dims[i], b->dims[i]);
        out_dims[out_ndims-2] = a->dims[a->ndims-2];
        out_dims[out_ndims-1] = b->dims[b->ndims-1];
    }

    Tensor* result = tensor_new(out_ndims, out_dims, NULL, a->requires_grad || b->requires_grad);
    
    if (op == ADD || op == HADAMARD) {
        for (int i = 0; i < result->size; i++)
            result->data[i] = op == ADD ? a->data[i] + b->data[i] : a->data[i] * b->data[i];
    } else {  // MATMUL
        int batch_size = result->size / (result->dims[out_ndims-2] * result->dims[out_ndims-1]);
        int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
        
        for (int batch = 0; batch < batch_size; batch++) {
            float *out = result->data + batch * m * p;
            const float *a_data = a->data + batch * m * n;
            const float *b_data = b->data + batch * n * p;
            
            for (int i = 0; i < m; i++)
                for (int k = 0; k < n; k++) {
                    float aik = a_data[i * n + k];
                    for (int j = 0; j < p; j++)
                        out[i * p + j] += aik * b_data[k * p + j];
                }
        }
    }
    
    if (result->requires_grad)
        record_operation(op, result, a, b, NULL, NULL, NULL, 0);
    return result;
}

void backward() {
    if (!tape.len) return;
    
    Tensor* final = tape.entries[tape.len - 1].result;
    if (!final->grad) final->grad = calloc(final->size, sizeof(float));
    for (int i = 0; i < final->size; i++) final->grad[i] = 1.0f;
    
    for (int i = tape.len - 1; i >= 0; i--) {
        TapeEntry* e = &tape.entries[i];
        Tensor *t = e->result, *a = e->input1, *b = e->input2;
        
        if (a->requires_grad && !a->grad) a->grad = calloc(a->size, sizeof(float));
        if (b && b->requires_grad && !b->grad) b->grad = calloc(b->size, sizeof(float));
        
        switch (e->op) {
            case REDUCE_SUM: {
                if (a->requires_grad) {
                    int* axes = e->aux_data1;
                    int num_axes = e->aux_int;
                    int coords[MAX_DIMS], result_coords[MAX_DIMS];
                    
                    for (int i = 0; i < t->size; i++) {
                        index_to_coords(i, result_coords, t->dims, t->ndims);
                        int idx = 0;
                        for (int j = 0; j < a->ndims; j++) {
                            int is_reduced_dim = 0;
                            for (int k = 0; k < num_axes; k++) {
                                if (j == axes[k]) {
                                    is_reduced_dim = 1;
                                    coords[j] = 0;
                                    break;
                                }
                            }
                            if (!is_reduced_dim) coords[j] = result_coords[idx++];
                        }
                        
                        int done = 0;
                        while (!done) {
                            int input_idx = coords_to_index(coords, a->dims, a->ndims);
                            a->grad[input_idx] += t->grad[i];
                            done = 1;
                            for (int k = 0; k < num_axes; k++) {
                                int dim = axes[k];
                                coords[dim]++;
                                if (coords[dim] < a->dims[dim]) {
                                    done = 0;
                                    break;
                                }
                                coords[dim] = 0;
                            }
                        }
                    }
                }
                break;
            }
            case RESHAPE:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++) a->grad[j] += t->grad[j];
                break;
            case SLICE:
                if (a->requires_grad) {
                    int coords[MAX_DIMS], src_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, coords, t->dims, t->ndims);
                        for (int k = 0; k < a->ndims; k++) 
                            src_coords[k] = coords[k] + e->aux_data1[k];
                        a->grad[coords_to_index(src_coords, a->dims, a->ndims)] += t->grad[j];
                    }
                }
                break;
            case PERMUTE: {
                if (a->requires_grad) {
                    int inverse_perm[MAX_DIMS];
                    for (int j = 0; j < t->ndims; j++) 
                        inverse_perm[e->aux_data1[j]] = j;
                    
                    int old_coords[MAX_DIMS], new_coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, old_coords, t->dims, t->ndims);
                        for (int k = 0; k < t->ndims; k++) 
                            new_coords[inverse_perm[k]] = old_coords[k];
                        a->grad[coords_to_index(new_coords, a->dims, a->ndims)] += t->grad[j];
                    }
                }
                break;
            }
            case EXP:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++)
                        a->grad[j] += t->grad[j] * t->data[j];
                break;
            case POW:
                if (a->requires_grad) {
                    float exponent = *(float*)e->aux_data1;
                    for (int j = 0; j < t->size; j++)
                        a->grad[j] += t->grad[j] * exponent * powf(a->data[j], exponent - 1);
                }
                break;
            case HADAMARD:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++) 
                        a->grad[j] += t->grad[j] * b->data[j];
                if (b->requires_grad)
                    for (int j = 0; j < t->size; j++) 
                        b->grad[j] += t->grad[j] * a->data[j];
                break;
            case GATHER:
                if (a->requires_grad) {
                    int* indices = e->aux_data1;
                    int coords[MAX_DIMS];
                    for (int j = 0; j < t->size; j++) {
                        index_to_coords(j, coords, t->dims, t->ndims);
                        int original_coord = coords[e->aux_int];
                        coords[e->aux_int] = indices[original_coord];
                        a->grad[coords_to_index(coords, a->dims, a->ndims)] += t->grad[j];
                    }
                }
                break;
            case ADD:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++) a->grad[j] += t->grad[j];
                if (b && b->requires_grad)
                    for (int j = 0; j < t->size; j++) b->grad[j] += t->grad[j];
                break;
            case MATMUL: {
                int m = a->dims[a->ndims-2], n = a->dims[a->ndims-1], p = b->dims[b->ndims-1];
                int batch_size = t->size / (m * p);
                
                for (int batch = 0; batch < batch_size; batch++) {
                    float *t_grad = t->grad + batch * m * p;
                    float *a_data = a->data + batch * m * n;
                    float *b_data = b->data + batch * n * p;
                    
                    if (a->requires_grad) {
                        float *a_grad = a->grad + batch * m * n;
                        for (int i = 0; i < m; i++)
                            for (int k = 0; k < n; k++)
                                for (int j = 0; j < p; j++)
                                    a_grad[i * n + k] += t_grad[i * p + j] * b_data[k * p + j];
                    }
                    if (b->requires_grad) {
                        float *b_grad = b->grad + batch * n * p;
                        for (int k = 0; k < n; k++)
                            for (int j = 0; j < p; j++)
                                for (int i = 0; i < m; i++)
                                    b_grad[k * p + j] += t_grad[i * p + j] * a_data[i * n + k];
                    }
                }
                break;
            }
            case RELU:
            case SIGMOID:
                if (a->requires_grad)
                    for (int j = 0; j < t->size; j++)
                        a->grad[j] += t->grad[j] * (e->op == RELU ? d_relu(a->data[j]) : d_sigmoid(a->data[j]));
                break;
        }
    }
}

void cleanup_tape() {
    for (int i = 0; i < tape.len; i++) {
        free(tape.entries[i].aux_data1);
        free(tape.entries[i].aux_data2);
        free(tape.entries[i].aux_data3);
    }
    tape.len = 0;
}

void tensor_free(Tensor* t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->dims);
    free(t);
}

#define tensor_add(a, b) tensor_op(a, b, ADD)
#define tensor_matmul(a, b) tensor_op(a, b, MATMUL)
#define tensor_relu(a) tensor_op(a, NULL, RELU)
#define tensor_sigmoid(a) tensor_op(a, NULL, SIGMOID)
#define tensor_hadamard(a, b) tensor_op(a, b, HADAMARD)
#define tensor_exp(a) tensor_op(a, NULL, EXP)

#endif // __GRAD_H__