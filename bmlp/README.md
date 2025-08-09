# Bilinear Multilayer Perceptron (BMLP)

This directory contains the implementation of a Bilinear Multilayer Perceptron (BMLP) that replaces the swish activation function with input-dependent weight modifications.

## Architecture

The original MLP architecture is:
```
H = X * W₁
S = H ⊙ σ(H)    # swish activation
Y = S * W₂ + X * W₃
```

The BMLP replaces the swish activation with input-dependent bias terms:
```
H = X * W₁ + f₁(X)    # input-dependent bias
Y = H * W₂ + f₂(H) + X * W₃    # hidden-dependent bias
```

Where:
- `f₁(X) = mean(X) · u₁` - input-dependent bias added to hidden layer
- `f₂(H) = mean(H) · u₂` - hidden-dependent bias added to output layer
- `u₁` and `u₂` are learned parameter vectors

## Key Differences from Regular MLP

1. **No Swish Activation**: The nonlinearity comes from input-dependent bias rather than the swish function H⊙σ(H)
2. **Additional Parameters**: Two learned vectors `u₁` (size: input_dim) and `u₂` (size: hidden_dim)
3. **Adaptive Behavior**: The network's behavior changes based on the input values through the bias terms

## Files

- `bmlp.h` - Header file with BMLP structure and function declarations
- `bmlp.c` - Core BMLP implementation (forward pass, backward pass, weight updates)
- `train.c` - Training program demonstrating BMLP usage
- `data.c/data.h` - Synthetic data generation (copied from parent directory)
- `Makefile` - Build configuration

## Building and Running

```bash
make clean
make
./train.out
```

## Architecture Details

### Forward Pass
1. Compute `H = X * W₁ᵀ`
2. Add input-dependent bias: `H += mean(X) · u₁` (broadcasted)
3. Compute `Y = H * W₂ᵀ` 
4. Add hidden-dependent bias: `Y += mean(H) · u₂` (broadcasted)
5. Add residual connection: `Y += X * W₃ᵀ`

### Backward Pass
- Standard backpropagation for W₁, W₂, W₃
- Additional gradients for bias parameters u₁, u₂

### Training Stability
- Uses conservative learning rate (0.0001) for stability
- Small initialization for bias parameters (0.01 scale)
- AdamW optimizer with weight decay

## Performance Notes

The BMLP trains stably but may converge slower than the regular MLP due to the additional parameters and different nonlinearity structure. The input-dependent bias provides a different form of adaptivity compared to the swish activation.

## Comparison with Regular MLP

- Regular MLP: Uses swish activation H⊙σ(H) for nonlinearity
- BMLP: Uses input/hidden-dependent bias terms for adaptivity
- Both architectures maintain the residual connection X*W₃ and similar parameter counts