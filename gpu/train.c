#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "mlp.h"

int main() {
    srand(time(NULL));

    // Initialize cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));

    // Parameters
    const int input_dim = 16;
    const int hidden_dim = 256;
    const int output_dim = 16;
    const int num_samples = 524288;
    const int batch_size = 512;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, num_samples, input_dim, output_dim, -30.0f, 30.0f);

    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, cublaslt_handle);
    
    // Training parameters
    const int num_epochs = 10;
    const float learning_rate = 0.0003f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate device memory for batch data
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * output_dim * sizeof(float)));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {

            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_X, &X[batch * batch_size * input_dim], batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, &y[batch * batch_size * output_dim], batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice));
            
            // Forward pass
            forward_pass_mlp(mlp, d_X);
            
            // Calculate loss
            float loss = calculate_loss_mlp(mlp, d_y);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_mlp(mlp);
            backward_pass_mlp(mlp, d_X, NULL);
            
            // Update weights
            update_weights_mlp(mlp, learning_rate, batch_size);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch % 2 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    // Save model
    FILE* model_file = fopen(model_fname, "wb");
    serialize_mlp(mlp, model_file);
    fclose(model_file);
    printf("Model saved to %s\n", model_fname);
    
    // Save data
    save_data(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    model_file = fopen(model_fname, "rb");
    MLP* loaded_mlp = deserialize_mlp(model_file, batch_size, cublaslt_handle);
    fclose(model_file);
    printf("Model loaded from %s\n", model_fname);
    
    // Forward pass with loaded model on first batch
    CHECK_CUDA(cudaMemcpy(d_X, X, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    forward_pass_mlp(loaded_mlp, d_X);
    
    // Copy predictions back to host
    float* layer_output = (float*)malloc(batch_size * output_dim * sizeof(float));
    CHECK_CUDA(cudaMemcpy(layer_output, loaded_mlp->d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // Evaluate model performance on first batch
    printf("Output\tR²\t\tMAE\t\tSample Predictions\n");
    printf("------\t--------\t--------\t--------------------------------\n");

    for (int i = 0; i < output_dim; i++) {
        // Calculate mean for R²
        float y_mean = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            y_mean += y[j * output_dim + i];
        }
        y_mean /= batch_size;
        
        // Calculate R² and MAE
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float pred = layer_output[j * output_dim + i];
            float actual = y[j * output_dim + i];
            float diff = pred - actual;
            
            ss_res += diff * diff;
            ss_tot += (actual - y_mean) * (actual - y_mean);
            mae += fabs(diff);
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= batch_size;
        
        // Print summary
        printf("y%d\t%.6f\t%.3f\t\t", i, r2, mae);
        for (int j = 0; j < 3; j++) {
            float pred = layer_output[j * output_dim + i];
            float actual = y[j * output_dim + i];
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free(layer_output);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
    free_mlp(mlp);
    free_mlp(loaded_mlp);
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    
    return 0;
}