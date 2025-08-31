#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "mlp.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int input_dim = 64;
    const int hidden_dim = 256;
    const int output_dim = 4;
    const int num_samples = 4096;
    const int batch_size = 512;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, num_samples, input_dim, output_dim, -3.0f, 3.0f);
    
    // Initialize network
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size);
    
    // Training parameters
    const int num_epochs = 10000;
    const float learning_rate = 0.0003f;
    const int num_batches = num_samples / batch_size;
    
    // Allocate batch buffers
    float* X_batch = (float*)malloc(batch_size * input_dim * sizeof(float));
    float* y_batch = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            
            // Copy batch data
            memcpy(X_batch, &X[start_idx * input_dim], batch_size * input_dim * sizeof(float));
            memcpy(y_batch, &y[start_idx * output_dim], batch_size * output_dim * sizeof(float));
            
            // Forward pass
            forward_pass_mlp(mlp, X_batch);
            
            // Calculate loss
            float loss = calculate_loss_mlp(mlp, y_batch);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_mlp(mlp);
            backward_pass_mlp(mlp, X_batch, NULL);
            
            // Update weights
            update_weights_mlp(mlp, learning_rate);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch > 0 && epoch % 100 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
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
    MLP* loaded_mlp = load_mlp(model_fname, batch_size);
    
    // Evaluate on first batch
    memcpy(X_batch, X, batch_size * input_dim * sizeof(float));
    memcpy(y_batch, y, batch_size * output_dim * sizeof(float));
    
    // Forward pass with loaded model
    forward_pass_mlp(loaded_mlp, X_batch);
    
    // Calculate and print loss with loaded model
    float verification_loss = calculate_loss_mlp(loaded_mlp, y_batch);
    printf("Loss with loaded model (first batch): %.8f\n", verification_loss);

    printf("\nEvaluating model performance...\n");

    // Calculate R² scores on first batch
    printf("\nR² scores (first batch):\n");
    for (int i = 0; i < output_dim; i++) {
        float y_mean = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            y_mean += y_batch[j * output_dim + i];
        }
        y_mean /= batch_size;

        float ss_res = 0.0f;
        float ss_tot = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            float diff_res = y_batch[j * output_dim + i] - loaded_mlp->layer_output[j * output_dim + i];
            float diff_tot = y_batch[j * output_dim + i] - y_mean;
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
            float pred = loaded_mlp->layer_output[j * output_dim + i];
            float actual = y_batch[j * output_dim + i];
            float diff = pred - actual;
            printf("Sample %d:\t%8.3f\t%8.3f\t%8.3f\n", j, pred, actual, diff);
        }
        
        // Calculate MAE for this output
        float mae = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            mae += fabs(loaded_mlp->layer_output[j * output_dim + i] - y_batch[j * output_dim + i]);
        }
        mae /= batch_size;
        printf("Mean Absolute Error for y%d: %.3f\n", i, mae);
    }
    
    // Cleanup
    free(X);
    free(y);
    free(X_batch);
    free(y_batch);
    free_mlp(mlp);
    free_mlp(loaded_mlp);
    
    return 0;
}