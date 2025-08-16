#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../data.h"
#include "mlp.h"

int main() {
    srand(time(NULL));

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // Parameters
    const int input_dim = 16;
    const int hidden_dim = 256;
    const int output_dim = 4;
    const int num_layers = 2;
    const int num_samples = 1024;
    const int batch_size = num_samples; // Full batch training
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim, -3.0f, 3.0f);

    // Convert synthetic data to FP16
    __half *X_fp16, *y_fp16;
    X_fp16 = (__half*)malloc(num_samples * input_dim * sizeof(__half));
    y_fp16 = (__half*)malloc(num_samples * output_dim * sizeof(__half));
    
    for (int i = 0; i < num_samples * input_dim; i++) {
        X_fp16[i] = __float2half(X[i]);
    }
    for (int i = 0; i < num_samples * output_dim; i++) {
        y_fp16[i] = __float2half(y[i]);
    }

    // Allocate device memory for input and output and copy FP16 data
    __half *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * input_dim * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * output_dim * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_X, X_fp16, batch_size * input_dim * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y_fp16, batch_size * output_dim * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, num_layers, batch_size, cublas_handle);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0002f;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        // Forward pass
        forward_pass_mlp(mlp, d_X);
        
        // Calculate loss
        float loss = calculate_loss_mlp(mlp, d_y);

        // Print progress
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, loss);
        }

        // Don't update weights after final evaluation
        if (epoch == num_epochs) break;

        // Backward pass
        zero_gradients_mlp(mlp);
        backward_pass_mlp(mlp, d_X);
        
        // Update weights
        update_weights_mlp(mlp, learning_rate);
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_mlp(mlp, model_fname);
    save_data(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    MLP* loaded_mlp = load_mlp(model_fname, batch_size, cublas_handle);
    
    // Allocate host memory for predictions (FP32)
    float* predictions = (float*)malloc(num_samples * output_dim * sizeof(float));
    __half* predictions_fp16 = (__half*)malloc(num_samples * output_dim * sizeof(__half));

    // Forward pass with loaded model
    forward_pass_mlp(loaded_mlp, d_X);
    
    // Copy predictions from device to host (from last layer) and convert to FP32
    int last_layer = loaded_mlp->num_layers - 1;
    CHECK_CUDA(cudaMemcpy(predictions_fp16, loaded_mlp->d_layer_output[last_layer], 
                         num_samples * output_dim * sizeof(__half),
                         cudaMemcpyDeviceToHost));
    
    // Convert FP16 predictions to FP32
    for (int i = 0; i < num_samples * output_dim; i++) {
        predictions[i] = __half2float(predictions_fp16[i]);
    }
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_mlp(loaded_mlp, d_y);
    printf("Loss with loaded model: %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores
    printf("\nR² scores:\n");
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            y_mean += y[j * output_dim + i];
        }
        y_mean /= num_samples;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            float diff_res = y[j * output_dim + i] - predictions[j * output_dim + i];
            float diff_tot = y[j * output_dim + i] - y_mean;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        float r2 = 1.0f - (ss_res / ss_tot);
        printf("R² score for output y%d: %.8f\n", i, r2);
    }

    // Print sample predictions
    printf("\nSample Predictions (first 15 samples):\n");
    printf("Output\t\tPredicted\tActual\t\tDifference\n");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        printf("\ny%d:\n", i);
        for (int j = 0; j < 15; j++) {
            float pred = predictions[j * output_dim + i];
            float actual = y[j * output_dim + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        
        // Calculate MAE for this output
        float mae = 0.0f;
        for (int j = 0; j < num_samples; j++) {
            mae += fabs(predictions[j * output_dim + i] - y[j * output_dim + i]);
        }
        mae /= num_samples;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_fp16);
    free(y_fp16);
    free(predictions);
    free(predictions_fp16);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_mlp(mlp);
    free_mlp(loaded_mlp);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}