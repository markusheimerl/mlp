#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../data.h"
#include "mlp.h"

int main() {
    srand(time(NULL));

    // Parameters
    const int input_dim = 16;
    const int hidden_dim = 1024;
    const int output_dim = 4;
    const int num_samples = 1024;
    const int batch_size = num_samples; // Full batch training
    
    // Generate synthetic data
    float *X, *y;
    generate_synthetic_data(&X, &y, num_samples, input_dim, output_dim);

    // Allocate device memory for input and output and copy data
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, batch_size * output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.001f;
    
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
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", 
             localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", 
             localtime(&now));

    // Save model and data with timestamped filenames
    save_mlp(mlp, model_fname);
    save_data_to_csv(X, y, num_samples, input_dim, output_dim, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    MLP* loaded_mlp = load_mlp(model_fname, batch_size);
    
    // Allocate host memory for predictions
    float* predictions = (float*)malloc(num_samples * output_dim * sizeof(float));

    // Forward pass with loaded model
    forward_pass_mlp(loaded_mlp, d_X);
    
    // Copy predictions from device to host
    CHECK_CUDA(cudaMemcpy(predictions, loaded_mlp->d_predictions, 
                         num_samples * output_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
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
    free(predictions);
    free_mlp(mlp);
    free_mlp(loaded_mlp);
    
    return 0;
}