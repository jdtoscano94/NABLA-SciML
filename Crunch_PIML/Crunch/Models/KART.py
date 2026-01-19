import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import partial
from tqdm import trange
import optax
from flax import linen as nn
from sklearn.model_selection import train_test_split
from jaxopt import LBFGS
import jax
from jax import config
import jax.numpy as jnp
# -----------------------------
def get_psi(d, N, mu=2):
    M = N
    delta_step = 1 / (M * N)
    # --- Prepare lists to accumulate values ---
    xx = []
    yy = []
    d1f = []
    d2f = []

    # --- Loop over blocks ---
    for j in range(-1, M):  
        a0 = N * j * delta_step          # Left-hand point of gap
        b0 = a0 + delta_step             # Right-hand point of gap (and LHS of next interval)
        c0 = a0 + N * delta_step         # Right-hand point of each interval

        z1 = a0
        z2 = a0 + delta_step**mu         # Here mu >= 1/2 (mu=2 in our case)
        z3 = c0

        # --- First segment: the gap ---
        xx_interp = jnp.linspace(a0, b0, 600)
        t0 = (xx_interp - a0) / (b0 - a0)
        yy_interp = z1 + (z2 - z1) * (t0 - (1/(2*jnp.pi)) * jnp.sin(2*jnp.pi*t0))
        df_part1 = (z2 - z1) / (b0 - a0) * (1 - jnp.cos(2*jnp.pi*t0))
        d2f_part1 = (z2 - z1) / ((b0 - a0)**2) * (2 * jnp.pi * jnp.sin(2*jnp.pi*t0))

        # --- Second segment: the interval ---
        xx_part2 = jnp.linspace(b0, c0, 600)
        t = (xx_part2 - b0) / (c0 - b0)
        poly = 6*t**5 - 15*t**4 + 10*t**3
        yy_part2 = z2 + (z3 - z2) * poly
        df_part2 = (z3 - z2) / (c0 - b0) * (30*t**4 - 60*t**3 + 30*t**2)
        d2f_part2 = (z3 - z2) / ((c0 - b0)**2) * (120*t**3 - 180*t**2 + 60*t)

        # Append the computed points (skipping the first points of each segment to avoid duplicates)
        xx.extend(np.array(xx_interp[1:]).tolist() + np.array(xx_part2[1:]).tolist())
        yy.extend(np.array(yy_interp[1:]).tolist() + np.array(yy_part2[1:]).tolist())
        d1f.extend(np.array(df_part1[1:]).tolist() + np.array(df_part2[1:]).tolist())
        d2f.extend(np.array(d2f_part1[1:]).tolist() + np.array(d2f_part2[1:]).tolist())

    xx = np.array(xx)
    yy = np.array(yy)
    d1f = np.array(d1f)
    d2f = np.array(d2f)

    # Define extract_psi as a nested function so it can access xx, yy, and delta_step
    def extract_psi(k):
        tol = 1e-6
        if k == 0:
            xi = np.where(np.isclose(xx, 0, atol=tol))[0]
            xe = np.where(np.isclose(xx, 1, atol=tol))[0]
        else:
            xi = np.where(np.isclose(xx, -k * delta_step, atol=1e-4))[0]
            xe = np.where(np.isclose(xx, 1 - k * delta_step, atol=1e-4))[0]
        if len(xi) == 0 or len(xe) == 0:
            raise ValueError(f"Could not find proper indices for ψ_{k+1}")
        start_idx = xi[0]
        end_idx = xe[0]
        # Normalize x and y for the segment
        x_norm = xx[start_idx:end_idx+1] - xx[start_idx]
        psi_norm = (yy[start_idx:end_idx+1] - yy[start_idx]) / (yy[end_idx] - yy[start_idx])
        return x_norm, psi_norm

    # Total number of ψ functions is N (ψ₁, ψ₂, ..., ψ_N).
    num_psi = N
    x_common = np.linspace(0, 1, 600)
    psi_array = np.zeros((len(x_common), num_psi + 1))
    psi_array[:, 0] = x_common

    for k in range(num_psi):
        try:
            x_norm, psi_norm = extract_psi(k)
        except ValueError as e:
            print(e)
            continue
        psi_interp = np.interp(x_common, x_norm, psi_norm)
        psi_array[:, k+1] = psi_interp

    # --- Choice for lambda_p ---
    key = random.PRNGKey(0)
    theta = random.uniform(key, shape=(d,), minval=0.0, maxval=1.0)
    theta = jnp.sort(theta)
    exp_theta = jnp.array([jnp.pi / N, jnp.pi]) * jnp.exp(theta)
    lambda_params = exp_theta / jnp.sum(exp_theta)
    print("lambda parameters:", lambda_params)
    return psi_array, lambda_params


# Provided data initialization (assumed to be available)
# -------------------------------------------------------------------
class MLP(nn.Module):
    features: list  # list or tuple of feature sizes
    
    @nn.compact
    def __call__(self, inputs):
        H = inputs
        for fs in self.features[:-1]:
            H = nn.tanh(nn.Dense(fs)(H))
        H = nn.Dense(self.features[-1])(H)
        return H


def get_psi_params(d=2,num_psi=24,mu=2,N_LAYERS = 4 ,HIDDEN = 32  ,verbose=False, Train_LBFGs=False,n_epochs=100000,maxiter_LBFGs=10000,lr0=1e-3):
    config.update("jax_default_matmul_precision", "float32")
    #Generate Data
    psi_array, lambda_params = get_psi(d, num_psi, mu)
    # Set up the data using psi_array (shapes as provided)
    x = psi_array[:, 0:1]
    u = psi_array[:, 1:]
    decay_step = 1000             # Decay step for exponential decay
    decay_rate = 0.99             # Decay rate for exponential decay
    key = jax.random.PRNGKey(42)
    # Perform a random split (50% train, 50% test) using scikit-learn's train_test_split.
    x_train, x_test, u_train, u_test = train_test_split(x, u, test_size=0.5, random_state=42)

    # Create training and testing datasets
    train_data = (x_train, u_train)
    test_data = (x_test, u_test)

    # -------------------------------------------------------------------
    # Hyperparameters and Environment Configuration
    NC = x_train.shape[0]   
    FEATURES = u_train.shape[1]   # Output feature size (e.g., 32)
    # Set environment variables to use one device and avoid preallocation
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Set up initial PRNG key
    subkey = jax.random.PRNGKey(0)
    # -------------------------------------------------------------------
    # Model Definition: MLP using Flax
    # -------------------------------------------------------------------
    # Define feature sizes: repeat HIDDEN for the first N_LAYERS layers, then output layer of size FEATURES
    feat_sizes = tuple([HIDDEN for _ in range(N_LAYERS)] + [FEATURES])
    print("Feature sizes:", feat_sizes)

    # Initialize the model and its parameters
    model = MLP(feat_sizes)
    params = model.init(subkey, jnp.ones((NC, 1)))  # using a dummy input for shape
    # -------------------------------------------------------------------
    # Optimizer and Update Function Setup
    # -------------------------------------------------------------------
    optimizer = optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate, staircase=False))
    state = optimizer.init(params)

    @partial(jax.jit, static_argnums=(1,))  # Only optimizer is static, NOT key!
    def update_model(key, optimizer, gradient, params, state):
        updates, new_state = optimizer.update(gradient, state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state

    # -------------------------------------------------------------------
    # Loss Function and Training Step
    # -------------------------------------------------------------------
    def loss_fn(params, batch):
        predictions = model.apply(params, batch["inputs"])
        #loss = jnp.mean((predictions - batch["targets"]) ** 2)
        loss = jnp.linalg.norm(predictions - batch["targets"]) / jnp.linalg.norm(batch["targets"])
        return loss

    @jax.jit
    def train_step(params, state, batch, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        new_params, new_state = update_model(key, optimizer, grads, params, state)
        return new_params, new_state, loss

    # ------------------------------------------------------------------
    xc, u_train = train_data
    xc = jnp.asarray(xc)
    u_train = jnp.asarray(u_train)

    # -------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------
    pbar = trange(n_epochs)
    print('Training  ADAM:')
    for epoch in pbar:
        # Using the full dataset as a single batch for training
        batch = {"inputs": xc, "targets": u_train}
        key, subkey = jax.random.split(key)
        params, state, loss = train_step(params, state, batch, subkey)
        
        if verbose and epoch % 100 == 0:
            test_inputs, test_targets = test_data
            test_pred = model.apply(params, test_inputs)
            rel_l2_error = jnp.linalg.norm(test_pred - test_targets) / jnp.linalg.norm(test_targets)
            pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4e}, Rel L2 Error: {rel_l2_error:.4e}")
    if Train_LBFGs:
        print('Training LBFGs:')
        # Ensure that parameters and data are in float64
        jax.config.update("jax_enable_x64", True)
        params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)

        # Convert training and test data to float64
        xc64 = xc.astype(jnp.float64)
        u_train64 = u_train.astype(jnp.float64)
        x64 = test_data[0].astype(jnp.float64)
        u_gt64 = test_data[1].astype(jnp.float64)

        # For convenience, define apply_fn as your model's apply function.
        # (Assuming the model is already defined as before.)
        apply_fn = lambda params, inputs: model.apply(params, inputs)

        # Define a helper to compute relative L2 error
        def relative_l2(a, b):
            return jnp.linalg.norm(a - b) / jnp.linalg.norm(b)

        # Define the loss function using the training data
        def lbfgs_loss_fn(params):
            predictions = apply_fn(params, xc64)
            # Using sum of squared errors for training loss
            loss = jnp.sum((predictions - u_train64) ** 2)
            #loss = jnp.linalg.norm(predictions - u_train64) / jnp.linalg.norm(u_train64)
            return loss

        # Initialize the LBFGS solver with appropriate hyperparameters.
        lbfgs_solver = LBFGS(
            fun=lbfgs_loss_fn,
            value_and_grad=False,    # Let the solver compute gradients via automatic differentiation
            maxiter=maxiter_LBFGs,
            tol=1e-9,
            stepsize=0.0,            # 0.0 means that LBFGS will use a line search to compute the step size
            linesearch="zoom",       # Alternatively, you can use "hager-zhang"
            maxls=50,                # Maximum number of line search steps
            condition="strong-wolfe",# Convergence condition for the line search
            history_size=50,
            use_gamma=True,
            verbose=False
        )

        # Define a JIT-compiled update function for LBFGS
        @jax.jit
        def lbfgs_update(params, state):
            return lbfgs_solver.update(params, state)

        # Initialize the solver state
        state = lbfgs_solver.init_state(params)

        # Prepare lists to record history
        lbfgs_losses = []
        lbfgs_grad_norms = []
        lbfgs_errors = []

        # Optimization loop
        # Create a progress bar for the maximum number of iterations
        pbar = trange(lbfgs_solver.maxiter, desc="LBFGS Optimization")

        for i in range(lbfgs_solver.maxiter):
            # Perform one LBFGS update step
            opt_step = lbfgs_update(params, state)
            params = opt_step.params
            state = opt_step.state

            # Compute current loss from the LBFGS state
            loss = state.value

            # Compute gradient norm manually by squaring, summing over all leaves, and taking square root
            grad = state.grad
            grad_norm = jnp.sqrt(sum([jnp.sum(jnp.abs(g) ** 2) for g in jax.tree_util.tree_leaves(grad)]))

            # Evaluate on the test data
            u_refined = apply_fn(params, x64)
            error_refined = relative_l2(u_refined, u_gt64)

            # Append history if desired
            lbfgs_losses.append(loss)
            lbfgs_grad_norms.append(grad_norm)
            lbfgs_errors.append(error_refined)

            # Optionally print every 100 iterations (or remove this if you rely solely on the progress bar)
            if verbose and i % 100 == 0:
                pbar.set_description(
                    f"Iter {i+1}, Loss: {loss:.6e}, Rel L2 Error: {error_refined:.6e}, Grad Norm: {grad_norm:.6e}"
                )
            # Check for convergence: if the gradient norm is below tolerance, update description and break
            if grad_norm < lbfgs_solver.tol:
                pbar.set_description(f"Convergence met at iter {i+1}")
                break

        # Close the progress bar when finished
        pbar.close()
    predictions = model.apply(params, x)
    config.update("jax_default_matmul_precision", "float32")
    return jax.device_get(x),jax.device_get(u),params,lambda_params,jax.device_get(predictions)