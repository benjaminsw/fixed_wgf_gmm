import jax
from jax import vmap, grad
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple, NamedTuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree
from jax.lax import map
import jax.scipy as jsp
from functools import partial


class GMMComponent(NamedTuple):
    """Represents a single Gaussian component with mean and covariance"""
    mean: jax.Array  # Shape: (d_z,)
    cov: jax.Array   # Shape: (d_z, d_z)
    weight: float    # Scalar weight


class GMMState(NamedTuple):
    """State for GMM-based particle representation"""
    components: list[GMMComponent]  # List of Gaussian components
    n_components: int
    prev_components: list[GMMComponent] = None  # For Wasserstein regularization
    dirichlet_alpha: jax.Array = None  # Dirichlet prior parameters


class WGFGMMHyperparams(NamedTuple):
    """Hyperparameters for WGF-GMM algorithm with Dirichlet prior and entropy regularization"""
    lambda_reg: float = 0.1           # Wasserstein regularization strength
    lambda_dirichlet: float = 0.1     # Dirichlet prior weight
    entropy_weight: float = 0.01      # Entropy regularization coefficient (Step 1)
    alpha_value: float = 0.1          # Dirichlet concentration parameter (< 1 for sparsity)
    lr_mean: float = 0.01             # Learning rate for means
    lr_cov: float = 0.001             # Learning rate for covariances
    lr_weight: float = 0.01           # Learning rate for weights
    prune_threshold: float = 1e-3     # Threshold for component pruning
    min_components: int = 1           # Minimum number of components to keep


class WGFGMMMetrics(NamedTuple):
    """Metrics for WGF-GMM training with Dirichlet and entropy regularization"""
    elbo: float
    elbo_with_wasserstein: float
    elbo_with_dirichlet: float
    elbo_with_entropy: float
    elbo_with_all_regularization: float
    wasserstein_distance: float
    dirichlet_log_prior: float
    entropy: float
    n_active_components: int
    pruned_components: int
    lambda_reg: float
    lambda_dirichlet: float
    entropy_weight: float
    lr_mean: float
    lr_cov: float
    lr_weight: float


def initialize_dirichlet_alpha(n_components: int, alpha_value: float = 0.1) -> jax.Array:
    """
    Step 1: Define the Dirichlet Prior Parameters
    
    Args:
        n_components: Number of GMM components
        alpha_value: Concentration parameter (< 1 for sparsity, > 1 for uniformity)
    
    Returns:
        Dirichlet alpha parameters
    """
    return np.ones(n_components) * alpha_value


def compute_entropy(weights: jax.Array, eps: float = 1e-8) -> float:
    """
    Step 2: Compute the Entropy of the GMM Weights
    
    Args:
        weights: Component weights
        eps: Small constant for numerical stability
    
    Returns:
        Entropy of the weights: -sum(w * log(w))
    """
    return -np.sum(weights * np.log(weights + eps))


def compute_dirichlet_log_prior(weights: jax.Array, 
                               dirichlet_alpha: jax.Array, 
                               eps: float = 1e-8) -> float:
    """
    Compute log Dirichlet prior (ignoring normalization constants).
    
    Args:
        weights: Component weights
        dirichlet_alpha: Dirichlet parameters
        eps: Small constant for numerical stability
    
    Returns:
        Log Dirichlet prior value
    """
    return np.sum((dirichlet_alpha - 1) * np.log(weights + eps))


def compute_elbo_with_all_regularization(key: jax.random.PRNGKey,
                                       pid: PID,
                                       target: Target,
                                       gmm_state: GMMState,
                                       y: jax.Array,
                                       hyperparams: PIDParameters,
                                       lambda_reg: float = 0.1,
                                       lambda_dirichlet: float = 0.1,
                                       entropy_weight: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Step 3: Modify the Loss Function
    
    Compute ELBO with Wasserstein, Dirichlet, and entropy regularization:
    F(r) = ELBO(r) - λ₁ × W₂²(r, r_prev) + λ₂ × Dir(w | α) + λ₃ × H(w)
    
    Returns:
        Tuple of (elbo_with_all_regularization, wasserstein_distance, dirichlet_log_prior, entropy)
    """
    # Compute standard ELBO
    elbo = compute_elbo(key, pid, target, gmm_state, y, hyperparams)
    
    # Compute Wasserstein regularization
    wasserstein_reg = 0.0
    if gmm_state.prev_components is not None:
        prev_gmm = GMMState(
            components=gmm_state.prev_components,
            n_components=len(gmm_state.prev_components)
        )
        wasserstein_reg = wasserstein_distance_gmm(gmm_state, prev_gmm)
    
    # Compute Dirichlet prior
    weights = np.array([comp.weight for comp in gmm_state.components])
    dirichlet_log_prior = 0.0
    if gmm_state.dirichlet_alpha is not None:
        dirichlet_log_prior = compute_dirichlet_log_prior(weights, gmm_state.dirichlet_alpha)
    
    # Step 2: Compute entropy
    entropy = compute_entropy(weights)
    
    # Step 3: Combine all terms (add entropy as negative term to loss)
    elbo_with_regularization = (elbo 
                               - lambda_reg * wasserstein_reg 
                               + lambda_dirichlet * dirichlet_log_prior 
                               + entropy_weight * entropy)
    
    return elbo_with_regularization, wasserstein_reg, dirichlet_log_prior, entropy


def compute_gmm_gradients_with_entropy_and_dirichlet(key: jax.random.PRNGKey,
                                                   pid: PID,
                                                   target: Target,
                                                   gmm_state: GMMState,
                                                   y: jax.Array,
                                                   hyperparams: PIDParameters,
                                                   lambda_reg: float = 0.1,
                                                   lambda_dirichlet: float = 0.1,
                                                   entropy_weight: float = 0.01) -> Tuple[list, list, jax.Array]:
    """
    Step 4: Add Entropy Gradient to Weight Update
    Step 5: Combine with Dirichlet Prior Gradient
    
    Compute gradients for GMM parameters including both Dirichlet and entropy regularization.
    
    Returns:
        Tuple of (mean_grads, cov_grads, weight_grads_with_all_regularization)
    """
    def objective_fn(means, covs, weights):
        # Create temporary GMM state
        components = []
        for i in range(len(means)):
            components.append(GMMComponent(
                mean=means[i], cov=covs[i], weight=weights[i]
            ))
        temp_gmm = GMMState(
            components=components,
            n_components=len(components),
            prev_components=gmm_state.prev_components,
            dirichlet_alpha=gmm_state.dirichlet_alpha
        )
        
        elbo_reg, _, _, _ = compute_elbo_with_all_regularization(
            key, pid, target, temp_gmm, y, hyperparams, 
            lambda_reg, lambda_dirichlet, entropy_weight
        )
        return elbo_reg
    
    # Extract current parameters
    means = np.stack([comp.mean for comp in gmm_state.components])
    covs = np.stack([comp.cov for comp in gmm_state.components])
    weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Compute gradients from automatic differentiation
    grad_fn = jax.grad(objective_fn, argnums=(0, 1, 2))
    mean_grads, cov_grads, weight_grads = grad_fn(means, covs, weights)
    
    # Step 4: Add explicit entropy gradient to weights
    eps = 1e-8
    entropy_grad = -np.log(weights + eps)
    
    # Step 5: Combine with Dirichlet prior gradient
    total_weight_grad = weight_grads
    
    if gmm_state.dirichlet_alpha is not None:
        dirichlet_weight_grad = lambda_dirichlet * (gmm_state.dirichlet_alpha - 1) / (weights + eps)
        total_weight_grad = total_weight_grad + dirichlet_weight_grad
    
    # Add entropy gradient
    total_weight_grad = total_weight_grad + entropy_weight * entropy_grad
    
    return list(mean_grads), list(cov_grads), total_weight_grad


def normalize_weights(weights: jax.Array, eps: float = 1e-8) -> jax.Array:
    """
    Step 6: Update and Normalize Weights
    
    Ensure weights stay on the simplex after updates.
    """
    # Clip weights to valid range (Step 6)
    weights = np.clip(weights, eps, 1.0)
    # Normalize to simplex (Step 6)
    weights = weights / np.sum(weights)
    return weights


def prune_components(gmm_state: GMMState, 
                    threshold: float = 1e-3, 
                    min_components: int = 1) -> Tuple[GMMState, int]:
    """
    Component Pruning (enhanced by entropy regularization for better stability)
    
    Remove components with weights below threshold.
    
    Args:
        gmm_state: Current GMM state
        threshold: Weight threshold for pruning
        min_components: Minimum number of components to keep
    
    Returns:
        Tuple of (pruned_gmm_state, num_pruned_components)
    """
    weights = np.array([comp.weight for comp in gmm_state.components])
    active_mask = weights > threshold
    
    # Ensure we keep at least min_components
    n_active = np.sum(active_mask)
    if n_active < min_components:
        # Keep the top min_components by weight
        sorted_indices = np.argsort(weights)[::-1]  # Descending order
        active_mask = np.zeros_like(weights, dtype=bool)
        active_mask = active_mask.at[sorted_indices[:min_components]].set(True)
        n_active = min_components
    
    n_pruned = len(gmm_state.components) - n_active
    
    if n_pruned == 0:
        return gmm_state, 0
    
    # Prune components
    active_components = []
    active_alpha = None
    
    for i, comp in enumerate(gmm_state.components):
        if active_mask[i]:
            active_components.append(comp)
    
    # Renormalize weights
    active_weights = weights[active_mask]
    active_weights = active_weights / np.sum(active_weights)
    
    # Update components with renormalized weights
    final_components = []
    for i, comp in enumerate(active_components):
        final_components.append(GMMComponent(
            mean=comp.mean,
            cov=comp.cov,
            weight=active_weights[i]
        ))
    
    # Update Dirichlet alpha if it exists
    if gmm_state.dirichlet_alpha is not None:
        active_alpha = gmm_state.dirichlet_alpha[active_mask]
    
    pruned_gmm_state = GMMState(
        components=final_components,
        n_components=len(final_components),
        prev_components=gmm_state.prev_components,
        dirichlet_alpha=active_alpha
    )
    
    return pruned_gmm_state, n_pruned


def update_gmm_parameters_with_entropy_and_dirichlet(gmm_state: GMMState,
                                                   mean_grads: list,
                                                   cov_grads: list,
                                                   weight_grads: jax.Array,
                                                   lr_mean: float = 0.01,
                                                   lr_cov: float = 0.001,
                                                   lr_weight: float = 0.01,
                                                   prune_threshold: float = 1e-3,
                                                   min_components: int = 1) -> Tuple[GMMState, int]:
    """
    Update GMM parameters with entropy and Dirichlet regularization.
    
    Args:
        gmm_state: Current GMM state
        mean_grads: Gradients for means
        cov_grads: Gradients for covariances
        weight_grads: Gradients for weights (includes Dirichlet + entropy terms)
        lr_mean: Learning rate for means
        lr_cov: Learning rate for covariances
        lr_weight: Learning rate for weights
        prune_threshold: Threshold for component pruning
        min_components: Minimum components to keep after pruning
        
    Returns:
        Tuple of (updated_gmm_state, num_pruned_components)
    """
    new_components = []
    
    # Extract current weights
    current_weights = np.array([comp.weight for comp in gmm_state.components])
    
    # Step 6: Update weights using gradient descent with all regularization terms
    new_weights = current_weights - lr_weight * weight_grads
    
    # Step 6: Normalize weights (clip and normalize to simplex)
    new_weights = normalize_weights(new_weights)
    
    # Update each component
    for i, comp in enumerate(gmm_state.components):
        # Update mean using Euclidean gradient
        new_mean = comp.mean - lr_mean * riemannian_grad_mean(comp.mean, mean_grads[i])
        
        # Update covariance using Riemannian gradient
        riem_grad_cov = riemannian_grad_cov(cov_grads[i], comp.cov)
        new_cov = retraction_cov(comp.cov, -lr_cov * riem_grad_cov)
        
        # Create new component with updated weight
        new_components.append(GMMComponent(
            mean=new_mean,
            cov=new_cov,
            weight=new_weights[i]
        ))
    
    # Create updated GMM state
    updated_gmm_state = GMMState(
        components=new_components,
        n_components=gmm_state.n_components,
        prev_components=gmm_state.components,
        dirichlet_alpha=gmm_state.dirichlet_alpha
    )
    
    # Component Pruning (enhanced stability due to entropy regularization)
    if prune_threshold > 0:
        updated_gmm_state, n_pruned = prune_components(
            updated_gmm_state, prune_threshold, min_components
        )
    else:
        n_pruned = 0
    
    return updated_gmm_state, n_pruned


def particles_to_gmm_with_dirichlet(particles: jax.Array, 
                                   weights: jax.Array = None,
                                   use_em: bool = True,
                                   n_components: int = None,
                                   alpha_value: float = 0.1) -> GMMState:
    """
    Convert particle representation to GMM representation with Dirichlet prior.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Optional weights, defaults to uniform
        use_em: Whether to use EM algorithm to fit proper GMM
        n_components: Number of GMM components (if less than n_particles)
        alpha_value: Dirichlet concentration parameter
    
    Returns:
        GMMState with Gaussian components and Dirichlet prior
    """
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    if not use_em or n_components is None:
        n_components = n_particles
    
    # Step 1: Initialize Dirichlet alpha
    dirichlet_alpha = initialize_dirichlet_alpha(n_components, alpha_value)
    
    if use_em and n_components < n_particles:
        # Fit proper GMM using EM algorithm
        gmm_state = _fit_gmm_em(particles, weights, n_components)
        # Add Dirichlet alpha
        gmm_state = gmm_state._replace(dirichlet_alpha=dirichlet_alpha)
        return gmm_state
    else:
        # Initialize each particle as a Gaussian component
        components = []
        for i in range(n_particles):
            mean = particles[i]
            # Start with small identity covariance to avoid degeneracy
            cov = np.eye(d_z) * 0.1
            weight = weights[i]
            components.append(GMMComponent(mean=mean, cov=cov, weight=weight))
        
        return GMMState(
            components=components, 
            n_components=n_particles,
            dirichlet_alpha=dirichlet_alpha
        )


def wgf_gmm_pvi_step_with_entropy_and_dirichlet(key: jax.random.PRNGKey,
                                               carry: PIDCarry,
                                               target: Target,
                                               y: jax.Array,
                                               optim: PIDOpt,
                                               hyperparams: PIDParameters,
                                               wgf_hyperparams: WGFGMMHyperparams) -> Tuple[float, PIDCarry]:
    """
    Full WGF-GMM with PVI step including Dirichlet, entropy regularization, and component pruning.
    
    This implements all PDF steps:
    1. Define entropy regularization coefficient
    2. Compute entropy of GMM weights
    3. Modify loss function with entropy term
    4. Add entropy gradient to weight update
    5. Combine with Dirichlet prior gradient
    6. Update and normalize weights
    
    Args:
        key: PRNG key
        carry: PID carry state
        target: Target distribution
        y: Observations
        optim: PID optimizer
        hyperparams: PID parameters
        wgf_hyperparams: WGF-GMM specific hyperparameters
        
    Returns:
        Tuple of (loss_value, updated_carry)
    """
    theta_key, r_key, grad_key = jax.random.split(key, 3)
    
    # Step 1: Standard conditional parameter (theta) update
    def loss(key, params, static):
        pid = eqx.combine(params, static)
        _samples = pid.sample(key, hyperparams.mc_n_samples, None)
        logq = vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
        logp = vmap(target.log_prob, (0, None))(_samples, y)
        return np.mean(logq - logp, axis=0)
    
    # Update conditional parameters (theta)
    lval, pid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
    )
    
    # Step 2: Initialize or get GMM representation
    if not hasattr(carry, 'gmm_state') or carry.gmm_state is None:
        # Initialize GMM from particles with Dirichlet prior
        gmm_state = particles_to_gmm_with_dirichlet(
            pid.particles, 
            use_em=True, 
            n_components=None,
            alpha_value=wgf_hyperparams.alpha_value
        )
    else:
        gmm_state = carry.gmm_state
        # Ensure Dirichlet alpha is set
        if gmm_state.dirichlet_alpha is None:
            dirichlet_alpha = initialize_dirichlet_alpha(
                gmm_state.n_components, 
                wgf_hyperparams.alpha_value
            )
            gmm_state = gmm_state._replace(dirichlet_alpha=dirichlet_alpha)
    
    # Step 3-5: Compute gradients including entropy and Dirichlet regularization
    mean_grads, cov_grads, weight_grads = compute_gmm_gradients_with_entropy_and_dirichlet(
        grad_key, pid, target, gmm_state, y, hyperparams, 
        wgf_hyperparams.lambda_reg, wgf_hyperparams.lambda_dirichlet, wgf_hyperparams.entropy_weight
    )
    
    # Step 6: Update GMM parameters with normalization and pruning
    updated_gmm_state, n_pruned = update_gmm_parameters_with_entropy_and_dirichlet(
        gmm_state, mean_grads, cov_grads, weight_grads,
        wgf_hyperparams.lr_mean, wgf_hyperparams.lr_cov, wgf_hyperparams.lr_weight,
        wgf_hyperparams.prune_threshold, wgf_hyperparams.min_components
    )
    
    # Convert back to particle representation for compatibility
    updated_particles = gmm_to_particles(updated_gmm_state)
    pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
    
    # Create updated carry with GMM state
    updated_carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state
    )
    
    # Store GMM state in carry
    updated_carry.gmm_state = updated_gmm_state
    
    return lval, updated_carry


def wgf_gmm_pvi_step_with_full_monitoring(key: jax.random.PRNGKey,
                                        carry: PIDCarry,
                                        target: Target,
                                        y: jax.Array,
                                        optim: PIDOpt,
                                        hyperparams: PIDParameters,
                                        wgf_hyperparams: WGFGMMHyperparams) -> Tuple[float, PIDCarry, WGFGMMMetrics]:
    """
    WGF-GMM step with detailed monitoring including entropy metrics.
    
    Returns:
        Tuple of (loss_value, updated_carry, metrics)
    """
    # Call the main WGF-GMM step
    lval, updated_carry = wgf_gmm_pvi_step_with_entropy_and_dirichlet(
        key, carry, target, y, optim, hyperparams, wgf_hyperparams
    )
    
    # Compute detailed metrics
    metrics_key = jax.random.split(key)[0]
    
    if hasattr(updated_carry, 'gmm_state') and updated_carry.gmm_state is not None:
        gmm_state = updated_carry.gmm_state
        
        try:
            # Compute all ELBO variants
            elbo = compute_elbo(metrics_key, updated_carry.id, target, gmm_state, y, hyperparams)
            
            # Compute individual regularization terms
            weights = np.array([comp.weight for comp in gmm_state.components])
            
            # Wasserstein distance
            wasserstein_dist = 0.0
            if gmm_state.prev_components is not None:
                prev_gmm = GMMState(
                    components=gmm_state.prev_components,
                    n_components=len(gmm_state.prev_components)
                )
                wasserstein_dist = wasserstein_distance_gmm(gmm_state, prev_gmm)
            
            # Dirichlet prior
            dirichlet_prior = 0.0
            if gmm_state.dirichlet_alpha is not None:
                dirichlet_prior = compute_dirichlet_log_prior(weights, gmm_state.dirichlet_alpha)
            
            # Entropy
            entropy = compute_entropy(weights)
            
            # Combined ELBO variants
            elbo_with_wasserstein = elbo - wgf_hyperparams.lambda_reg * wasserstein_dist
            elbo_with_dirichlet = elbo + wgf_hyperparams.lambda_dirichlet * dirichlet_prior
            elbo_with_entropy = elbo + wgf_hyperparams.entropy_weight * entropy
            elbo_with_all = (elbo 
                           - wgf_hyperparams.lambda_reg * wasserstein_dist 
                           + wgf_hyperparams.lambda_dirichlet * dirichlet_prior 
                           + wgf_hyperparams.entropy_weight * entropy)
            
            # Count active components
            n_active = np.sum(weights > wgf_hyperparams.prune_threshold)
            n_pruned = len(gmm_state.prev_components or []) - len(gmm_state.components)
            
        except Exception as e:
            # Fallback metrics
            elbo = -float(lval)
            elbo_with_wasserstein = -float(lval)
            elbo_with_dirichlet = -float(lval)
            elbo_with_entropy = -float(lval)
            elbo_with_all = -float(lval)
            wasserstein_dist = 0.0
            dirichlet_prior = 0.0
            entropy = 0.0
            n_active = gmm_state.n_components if gmm_state else 0
            n_pruned = 0
    else:
        # Fallback if no GMM state
        elbo = -float(lval)
        elbo_with_wasserstein = -float(lval)
        elbo_with_dirichlet = -float(lval)
        elbo_with_entropy = -float(lval)
        elbo_with_all = -float(lval)
        wasserstein_dist = 0.0
        dirichlet_prior = 0.0
        entropy = 0.0
        n_active = 0
        n_pruned = 0
    
    metrics = WGFGMMMetrics(
        elbo=float(elbo),
        elbo_with_wasserstein=float(elbo_with_wasserstein),
        elbo_with_dirichlet=float(elbo_with_dirichlet),
        elbo_with_entropy=float(elbo_with_entropy),
        elbo_with_all_regularization=float(elbo_with_all),
        wasserstein_distance=float(wasserstein_dist),
        dirichlet_log_prior=float(dirichlet_prior),
        entropy=float(entropy),
        n_active_components=int(n_active),
        pruned_components=int(max(0, n_pruned)),
        lambda_reg=wgf_hyperparams.lambda_reg,
        lambda_dirichlet=wgf_hyperparams.lambda_dirichlet,
        entropy_weight=wgf_hyperparams.entropy_weight,
        lr_mean=wgf_hyperparams.lr_mean,
        lr_cov=wgf_hyperparams.lr_cov,
        lr_weight=wgf_hyperparams.lr_weight
    )
    
    return lval, updated_carry, metrics


# Include all the existing helper functions from the original code
# (compute_elbo, bures_wasserstein_distance_squared, wasserstein_distance_gmm, 
#  riemannian_grad_mean, riemannian_grad_cov, retraction_cov, sample_from_gmm,
#  gmm_to_particles, _fit_gmm_em, etc.)