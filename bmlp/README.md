# bmlp
A bilinear multilayer perceptron implementation

Consider a bilinear feed-forward network operating on batched inputs of shape (batch_size × input_dim). The architecture consists of a linear transformation followed by a bilinear layer and a learned residual connection, where the forward propagation follows:

$$
\begin{align*}
H &= XW_1 \\
B &= H \otimes H \\
Y &= BW_2 + XW_3
\end{align*}
$$

The input transformation matrix $W_1$ maps input features to hidden representations, the bilinear transformation $H \otimes H$ computes outer products between hidden states, the bilinear projection matrix $W_2$ transforms the flattened outer products to outputs, and the residual matrix $W_3$ provides direct input-output connections. The outer product operation $H \otimes H$ creates a matrix of size (hidden_dim × hidden_dim) for each sample, capturing second-order interactions between hidden features.

The backward pass through the chain rule yields the following gradients, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_3} &= X^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial W_2} &= (\frac{\partial L}{\partial Y})^\top \otimes B \\
\frac{\partial L}{\partial B} &= (\frac{\partial L}{\partial Y})W_2^\top \\
\frac{\partial L}{\partial H} &= \frac{\partial L}{\partial B} \cdot H + (\frac{\partial L}{\partial B})^\top \cdot H \\
\frac{\partial L}{\partial W_1} &= X^\top(\frac{\partial L}{\partial H})
\end{align*}
$$

The gradient with respect to the hidden layer $H$ accounts for both left and right multiplication in the outer product operation, as each hidden unit contributes to multiple elements in the bilinear term.

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The bilinear architecture enables the model to capture quadratic interactions between input features through the learned hidden representation, providing greater expressivity than standard linear layers while maintaining computational efficiency through matrix operations.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
cd bmlp
make run
```