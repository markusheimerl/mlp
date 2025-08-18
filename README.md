# mlp
A multilayer perceptron implementation

Consider a multilayer perceptron operating on batched inputs of shape (batch_size × input_dim). The architecture consists of multiple layers, each performing two linear transformations with an intermediate swish activation and a learned residual connection. For each layer, the forward propagation follows:

$$
\begin{align*}
H &= XW_1 \\
S &= H\sigma(H) \\
Y &= SW_2 + XW_3
\end{align*}
$$

The input transformation matrix $W_1$ maps input features to hidden representations, the output projection matrix $W_2$ transforms activated hidden states to outputs, and the residual matrix $W_3$ provides direct input-output connections. The swish activation $H\sigma(H)$ interpolates between linear and nonlinear regimes, yielding the following backward pass for each layer through the chain rule, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_2} &= S^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial W_3} &= X^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial S} &= (\frac{\partial L}{\partial Y})(W_2)^\top \\
\frac{\partial L}{\partial H} &= \frac{\partial L}{\partial S} \odot [\sigma(H) + H\sigma(H)(1-\sigma(H))] \\
\frac{\partial L}{\partial W_1} &= X^\top(\frac{\partial L}{\partial H})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```

## Benchmarks

### NVIDIA Jetson Orin Nano Super Developer Kit

#### CPU
```
R² score for output y0: 0.99999571
R² score for output y1: 0.99998200
R² score for output y2: 0.99999940
R² score for output y3: 0.99994481
...
3 minutes 39 seconds elapsed
```

#### GPU
```
R² score for output y0: 0.99999571
R² score for output y1: 0.99998200
R² score for output y2: 0.99999940
R² score for output y3: 0.99994481
...
10 seconds elapsed
```