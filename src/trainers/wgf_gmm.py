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
    """State for GMM-based particle representation - JAX-compatible"""
    means: jax.Array      # Shape: (n_components, d_z)
    covs: jax.Array       # Shape: (n_components, d_z, d_z)
    weights: jax.Array    # Shape: (n_components,)
    n_components: int
    prev_means: jax.Array = None     # For Wasserstein regularization
    prev_covs: jax.Array = None
    prev_weights: jax.Array = None


def particles_to_gmm(particles: jax.Array, 
                     weights: jax.Array = None,
                     use_em: bool = False,
                     n_components: int = None) -> GMMState:
    """
    Convert particle representation to GMM representation.
    
    Args:
        particles: Array of shape (n_particles, d_z)
        weights: Optional weights, defaults to uniform
        use_em: Whether to use EM algorithm to fit proper GMM
        n_components: Number of GMM components (if less than n_particles)
    
    Returns:
        GMMState with Gaussian components
    """
    n_particles, d_z = particles.shape
    
    if weights is None:
        weights = np.ones(n_particles) / n_particles
    
    if n_components is None:
        n_components = n_particles
    
    if use_em and n_components < n_particles:
        # Fit proper GMM using EM algorithm
        return _fit_gmm_em(particles, weights, n_components)
    else:
        # Initialize each particle as a Gaussian component
        means = particles
        # Start with small identity covariance to avoid degeneracy
        # Change from 0.1 to larger value:
        covs = np.tile(np.eye(d_z) * 0.5, (n_particles, 1, 1))  # Use 0.5 instead of 0.1
        
        return GMMState(
            means=means,
            covs=covs,
            weights=weights,
            n_components=n_particles
        )


def _fit_gmm_em(particles: jax.Array, weights: jax.Array, n_components: int, 
                max_iter: int = 10, tol: float = 1e-4) -> GMMState:
    """
    Fit GMM to particles using EM algorithm.
    """
    n_particles, d_z = particles.shape
    key = jax.random.PRNGKey(42)
    
    # Initialize GMM parameters using k-means++ style
    means = _kmeans_plus_plus_init(key, particles, n_components)
    covs = np.tile(np.eye(d_z) * 0.5, (n_components, 1, 1))
    gmm_weights = np.ones(n_components) / n_components
    
    def em_step(state):
        means, covs, gmm_weights = state
        
        # E-step
        responsibilities = _e_step(particles, weights, means, covs, gmm_weights)
        
        # M-step
        new_means, new_covs, new_weights = _m_step(particles, weights, responsibilities)
        
        return (new_means, new_covs, new_weights)
    
    # Run EM iterations
    state = (means, covs, gmm_weights)
    for _ in range(max_iter):
        state = em_step(state)
    
    final_means, final_covs, final_weights = state
    
    return GMMState(
        means=final_means,
        covs=final_covs,
        weights=final_weights,
        n_components=n_components
    )


def _kmeans_plus_plus_init(key: jax.random.PRNGKey, particles: jax.Array, 
                          n_components: int) -> jax.Array:
    """K-means++ initialization for GMM means."""
    n_particles, d_z = particles.shape
    means = np.zeros((n_components, d_z))
    
    # Choose first center randomly
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, n_particles)
    means = means.at[0].set(particles[first_idx])
    
    for i in range(1, n_components):
        # Compute distances to nearest centers
        distances = np.full(n_particles, np.inf)
        for j in range(i):
            dist_to_j = np.sum((particles - means[j]) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_j)
        
        # Choose next center with probability proportional to squared distance
        key, subkey = jax.random.split(key)
        probs = distances / np.sum(distances)
        next_idx = jax.random.categorical(subkey, np.log(probs))
        means = means.at[i].set(particles[next_idx])
    
    return means


def _e_step(particles: jax.Array, weights: jax.Array, means: jax.Array, 
           covs: jax.Array, gmm_weights: jax.Array) -> jax.Array:
    """E-step of EM algorithm."""
    n_particles, d_z = particles.shape
    n_components = means.shape[0]
    
    def compute_log_prob(particle, mean, cov):
        diff = particle - mean
        cov_reg = cov + 1e-6 * np.eye(d_z)
        cov_inv = np.linalg.inv(cov_reg)
        mahal_dist = np.sum(diff @ cov_inv * diff)
        log_det = np.linalg.slogdet(cov_reg)[1]
        return -0.5 * (d_z * np.log(2 * np.pi) + log_det + mahal_dist)
    
    # Vectorized computation of log probabilities
    log_probs = vmap(lambda p: vmap(lambda m, c, w: np.log(w) + compute_log_prob(p, m, c))(
        means, covs, gmm_weights))(particles)
    
    # Compute responsibilities using log-sum-exp trick
    log_sum = jsp.special.logsumexp(log_probs, axis=1, keepdims=True)
    responsibilities = np.exp(log_probs - log_sum)
    
    return responsibilities


def _m_step(particles: jax.Array, weights: jax.Array, 
           responsibilities: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """M-step of EM algorithm."""
    n_particles, d_z = particles.shape
    n_components = responsibilities.shape[1]
    
    # Effective counts
    weighted_resp = responsibilities * weights[:, None]
    nk = np.sum(weighted_resp, axis=0)
    
    # Update means
    def update_mean(k):
        return np.where(nk[k] > 1e-8, 
                       np.sum(weighted_resp[:, k:k+1] * particles, axis=0) / nk[k],
                       np.zeros(d_z))
    
    means = vmap(update_mean)(np.arange(n_components))
    
    # Update covariances
    def update_cov(k):
        def safe_cov():
            diff = particles - means[k]
            weighted_diff = weighted_resp[:, k:k+1] * diff
            cov = (weighted_diff.T @ diff) / nk[k]
            return cov + 1e-6 * np.eye(d_z)
        
        def fallback_cov():
            return np.eye(d_z) * 0.1
        
        return np.where(nk[k] > 1e-8, safe_cov(), fallback_cov())
    
    covs = vmap(update_cov)(np.arange(n_components))
    
    # Update weights
    gmm_weights = nk / np.sum(nk)
    
    return means, covs, gmm_weights


def gmm_to_particles(gmm_state: GMMState) -> jax.Array:
    """
    Extract particle locations from GMM (using means).
    """
    return gmm_state.means


def sample_from_gmm(key: jax.random.PRNGKey, gmm_state: GMMState, 
                   n_samples: int) -> jax.Array:
    """
    Sample from a GMM - JAX-compatible version with differentiable sampling.
    """
    # Sample component indices
    key, subkey = jax.random.split(key)
    component_indices = jax.random.categorical(
        subkey, np.log(gmm_state.weights), shape=(n_samples,)
    )
    
    # Sample from components using vectorized operations
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (n_samples, gmm_state.means.shape[1]))
    
    def sample_from_component(idx, noise_sample):
        mean = gmm_state.means[idx]
        cov = gmm_state.covs[idx]
        
        # FIXED: Use JAX's built-in Cholesky which is more autodiff-friendly
        try:
            # Add regularization for numerical stability
            reg_cov = cov + 1e-5 * np.eye(cov.shape[0])
            L = jax.scipy.linalg.cholesky(reg_cov, lower=True)
            return mean + L @ noise_sample
        except:
            # Fallback: use matrix square root via eigendecomposition but more carefully
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Clamp eigenvalues to be positive with larger minimum
            eigvals = np.maximum(eigvals, 1e-4)  
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            return mean + sqrt_cov @ noise_sample
    
    # Vectorized sampling
    samples = vmap(sample_from_component)(component_indices, noise)
    
    return samples


def bures_wasserstein_distance_squared(mu1: jax.Array, cov1: jax.Array,
                                     mu2: jax.Array, cov2: jax.Array) -> float:
    """
    Compute squared Bures-Wasserstein distance between two Gaussian distributions.
    """
    # Mean difference term
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Covariance term with numerical stability
    d = cov1.shape[0]
    reg = 1e-6 * np.eye(d)
    
    try:
        cov1_sqrt = jsp.linalg.sqrtm(cov1 + reg)
        temp = cov1_sqrt @ cov2 @ cov1_sqrt
        temp_sqrt = jsp.linalg.sqrtm(temp + reg)
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.trace(temp_sqrt)
    except:
        # Fallback approximation
        cov_term = np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(np.trace(cov1) * np.trace(cov2))
    
    return mean_diff + np.maximum(cov_term, 0.0)


def wasserstein_distance_gmm(gmm1: GMMState, gmm2: GMMState) -> float:
    """
    Compute approximate Wasserstein distance between two GMMs.
    """
    if gmm1.n_components != gmm2.n_components:
        # If different sizes, just compare first few components
        min_comp = min(gmm1.n_components, gmm2.n_components)
        means1, covs1, weights1 = gmm1.means[:min_comp], gmm1.covs[:min_comp], gmm1.weights[:min_comp]
        means2, covs2, weights2 = gmm2.means[:min_comp], gmm2.covs[:min_comp], gmm2.weights[:min_comp]
    else:
        means1, covs1, weights1 = gmm1.means, gmm1.covs, gmm1.weights
        means2, covs2, weights2 = gmm2.means, gmm2.covs, gmm2.weights
    
    # Compute pairwise distances and use simple matching
    def compute_distance(i, j):
        return bures_wasserstein_distance_squared(means1[i], covs1[i], means2[j], covs2[j])
    
    # Simple diagonal matching (could be improved with Hungarian algorithm)
    distances = vmap(lambda i: compute_distance(i, i))(np.arange(len(means1)))
    avg_weights = (weights1 + weights2) / 2
    
    return np.sum(avg_weights * distances)


def compute_elbo_with_wasserstein_regularization(key: jax.random.PRNGKey,
                                               pid: PID,
                                               target: Target,
                                               gmm_state: GMMState,
                                               y: jax.Array,
                                               hyperparams: PIDParameters,
                                               lambda_reg: float = 0.1) -> float:
    """
    Compute regularized ELBO: F(r) = ELBO(r) - λ * W₂²(r, r_prev)
    """
    # Sample from GMM
    key, subkey = jax.random.split(key)
    samples = sample_from_gmm(subkey, gmm_state, hyperparams.mc_n_samples)
    
    # Compute standard ELBO terms
    logq = vmap(pid.log_prob, (0, None))(samples, y)
    logp = vmap(target.log_prob, (0, None))(samples, y)
    elbo = np.mean(logp - logq)
    
    # Add Wasserstein regularization if previous state exists
    wasserstein_reg = 0.0
    if gmm_state.prev_means is not None:
        prev_gmm = GMMState(
            means=gmm_state.prev_means,
            covs=gmm_state.prev_covs,
            weights=gmm_state.prev_weights,
            n_components=gmm_state.prev_means.shape[0]
        )
        wasserstein_reg = wasserstein_distance_gmm(gmm_state, prev_gmm)
    
    return elbo - lambda_reg * wasserstein_reg


def riemannian_grad_mean(mean: jax.Array, euclidean_grad_mean: jax.Array) -> jax.Array:
    """Riemannian gradient for mean parameters (just Euclidean for means)."""
    return euclidean_grad_mean


def riemannian_grad_cov(euclidean_grad_cov: jax.Array, cov: jax.Array) -> jax.Array:
    """Riemannian gradient for covariance matrix."""
    product = euclidean_grad_cov @ cov
    symmetric_product = (product + product.T) / 2
    return symmetric_product #return 4 * symmetric_product


def retraction_cov(cov: jax.Array, tangent_vector: jax.Array) -> jax.Array:
    """Retraction operator for covariance matrices."""
    new_cov = cov + tangent_vector
    new_cov = (new_cov + new_cov.T) / 2
    d = new_cov.shape[0]
    regularization = 1e-6 * np.eye(d)
    return new_cov + regularization


def sinkhorn_weights_update(weights: jax.Array, grad_weights: jax.Array, 
                           lr: float = 0.01, n_iter: int = 5) -> jax.Array:
    """Update GMM weights using simplex projection."""
    # Project gradient onto tangent space of simplex
    grad_projected = grad_weights - np.mean(grad_weights)
    
    # Update in log space for numerical stability
    log_weights = np.log(weights + 1e-8)
    log_weights_new = log_weights - lr * grad_projected
    
    # Project back to simplex
    weights_new = np.exp(log_weights_new - np.max(log_weights_new))
    weights_new = weights_new / np.sum(weights_new)
    
    return weights_new


def debug_objective_components(means, covs, weights, grad_key, pid, target, gmm_state, y, hyperparams, lambda_reg):
    """Debug each component of the objective function"""
    
    # Test if basic GMM construction works
    try:
        temp_gmm = GMMState(
            means=means,
            covs=covs, 
            weights=weights,
            n_components=means.shape[0],
            prev_means=gmm_state.prev_means,
            prev_covs=gmm_state.prev_covs,
            prev_weights=gmm_state.prev_weights
        )
        print("✓ GMM construction OK")
    except Exception as e:
        print("✗ GMM construction failed:", e)
        return
    
    # Test sampling from GMM
    try:
        key, subkey = jax.random.split(grad_key)
        samples = sample_from_gmm(subkey, temp_gmm, hyperparams.mc_n_samples)
        print("✓ GMM sampling OK, sample shape:", samples.shape)
        print("  Sample stats: min=", float(np.min(samples)), "max=", float(np.max(samples)))
        if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
            print("✗ NaN/Inf in samples!")
            return
    except Exception as e:
        print("✗ GMM sampling failed:", e)
        return
    
    # Test log probabilities
    try:
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        print("✓ Log q OK, shape:", logq.shape)
        print("  Log q stats: min=", float(np.min(logq)), "max=", float(np.max(logq)))
        if np.any(np.isnan(logq)) or np.any(np.isinf(logq)):
            print("✗ NaN/Inf in logq!")
            return
    except Exception as e:
        print("✗ Log q computation failed:", e)
        return
        
    try:
        logp = vmap(target.log_prob, (0, None))(samples, y)
        print("✓ Log p OK, shape:", logp.shape) 
        print("  Log p stats: min=", float(np.min(logp)), "max=", float(np.max(logp)))
        if np.any(np.isnan(logp)) or np.any(np.isinf(logp)):
            print("✗ NaN/Inf in logp!")
            return
    except Exception as e:
        print("✗ Log p computation failed:", e)
        return
    
    # Test ELBO computation
    try:
        elbo = np.mean(logp - logq)
        print("✓ ELBO OK:", float(elbo))
        if np.isnan(elbo) or np.isinf(elbo):
            print("✗ NaN/Inf in ELBO!")
            return
    except Exception as e:
        print("✗ ELBO computation failed:", e)
        return
    
    # Test full objective
    try:
        obj_val = compute_elbo_with_wasserstein_regularization(
            grad_key, pid, target, temp_gmm, y, hyperparams, lambda_reg
        )
        print("✓ Full objective OK:", float(obj_val))
        if np.isnan(obj_val) or np.isinf(obj_val):
            print("✗ NaN/Inf in objective!")
            return
    except Exception as e:
        print("✗ Full objective failed:", e)
        return


def wgf_gmm_pvi_step(key: jax.random.PRNGKey,
                    carry: PIDCarry,
                    target: Target,
                    y: jax.Array,
                    optim: PIDOpt,
                    hyperparams: PIDParameters,
                    lambda_reg: float = 0.1,
                    lr_mean: float = 0.01,
                    lr_cov: float = 0.001,
                    lr_weight: float = 0.01) -> Tuple[float, PIDCarry]:
    """
    Fixed WGF-GMM with PVI step that avoids tracing issues.
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
    
    # Step 2: Convert particles to GMM representation
    if carry.gmm_state is None:
        # Initialize GMM from particles
        gmm_state = particles_to_gmm(pid.particles, use_em=False, n_components=None)

        # Right after particles_to_gmm initialization:
        print("Initial cov determinants:", [float(np.linalg.det(cov)) for cov in gmm_state.covs[:3]])
        print("Initial cov condition numbers:", [float(np.linalg.cond(cov)) for cov in gmm_state.covs[:3]])
    else:
        gmm_state = carry.gmm_state

    
    # Step 3: Define objective function for gradients
    def objective_fn(means, covs, weights):
        temp_gmm = GMMState(
            means=means,
            covs=covs,
            weights=weights,
            n_components=means.shape[0],
            prev_means=gmm_state.prev_means,
            prev_covs=gmm_state.prev_covs,
            prev_weights=gmm_state.prev_weights
        )
        return compute_elbo_with_wasserstein_regularization(
            grad_key, pid, target, temp_gmm, y, hyperparams, lambda_reg
        )
    
    # ADD DEBUGGING HERE
    print("=== DEBUGGING OBJECTIVE FUNCTION ===")
    debug_objective_components(
        gmm_state.means, gmm_state.covs, gmm_state.weights,
        grad_key, pid, target, gmm_state, y, hyperparams, lambda_reg
    )

    print("=== TESTING GRADIENT COMPUTATION ===")
    # Test gradient computation with simpler function first
    def simple_test_fn(means, covs, weights):
        return np.sum(means**2) + np.sum(covs**2) + np.sum(weights**2)

    try:
        simple_grad_fn = jax.grad(simple_test_fn, argnums=(0, 1, 2))
        simple_grads = simple_grad_fn(gmm_state.means, gmm_state.covs, gmm_state.weights)
        print("✓ Simple gradient computation works")
    except Exception as e:
        print("✗ Simple gradient computation failed:", e)

    # Now test your actual objective
    try:
        grad_fn = jax.grad(objective_fn, argnums=(0, 1, 2))
        mean_grads, cov_grads, weight_grads = grad_fn(
            gmm_state.means, gmm_state.covs, gmm_state.weights
        )
        print("✓ Actual gradient computation works")
    except Exception as e:
        print("✗ Actual gradient computation failed:", e)
        # Return early if gradient computation fails
        return lval, carry
    
    ###############################
    # LOG GRADIENT NORMS BEFORE CLIPPING
    mean_grad_norms = np.linalg.norm(mean_grads, axis=1)
    cov_grad_norms = np.linalg.norm(cov_grads.reshape(cov_grads.shape[0], -1), axis=1)
    weight_grad_norm = np.linalg.norm(weight_grads)
    print("BEFORE clipping - Mean grad norms: max=", float(np.max(mean_grad_norms)), "mean=", float(np.mean(mean_grad_norms)))
    print("BEFORE clipping - Cov grad norms: max=", float(np.max(cov_grad_norms)), "mean=", float(np.mean(cov_grad_norms)))
    print("BEFORE clipping - Weight grad norm:", float(weight_grad_norm))
    ###############################
    
    # Step 4: Update parameters using Riemannian gradients
    # Update means
    new_means = gmm_state.means - lr_mean * mean_grads
    
    # Update covariances using Riemannian gradients
    riem_cov_grads = vmap(riemannian_grad_cov)(cov_grads, gmm_state.covs)
    new_covs = vmap(retraction_cov)(gmm_state.covs, -lr_cov * riem_cov_grads)
    
    ###############################
    # LOG RIEMANNIAN GRADIENT NORMS
    riem_cov_grad_norms = np.linalg.norm(riem_cov_grads.reshape(riem_cov_grads.shape[0], -1), axis=1)
    print(f"AFTER Riemannian - Cov grad norms: max={np.max(riem_cov_grad_norms):.3e}, mean={np.mean(riem_cov_grad_norms):.3e}")
    ###############################


    # Update weights using Sinkhorn
    new_weights = sinkhorn_weights_update(gmm_state.weights, weight_grads, lr_weight)
    
    ###############################
    # LOG PARAMETER CHANGES
    mean_changes = np.linalg.norm(new_means - gmm_state.means, axis=1)
    weight_changes = np.linalg.norm(new_weights - gmm_state.weights)
    print(f"Parameter changes - Means: max={np.max(mean_changes):.3e}")
    print(f"Parameter changes - Weights: {weight_changes:.3e}")

    # CHECK FOR NAN/INF IN NEW PARAMETERS
    if np.any(np.isnan(new_means)) or np.any(np.isinf(new_means)):
        print("WARNING: NaN/Inf in new_means!")
    if np.any(np.isnan(new_covs)) or np.any(np.isinf(new_covs)):
        print("WARNING: NaN/Inf in new_covs!")
    ###############################

    # Create updated GMM state
    updated_gmm_state = GMMState(
        means=new_means,
        covs=new_covs,
        weights=new_weights,
        n_components=gmm_state.n_components,
        prev_means=gmm_state.means,  # Store current as previous
        prev_covs=gmm_state.covs,
        prev_weights=gmm_state.weights
    )
    
    # Step 5: Convert back to particle representation
    updated_particles = gmm_to_particles(updated_gmm_state)
    pid = eqx.tree_at(lambda tree: tree.particles, pid, updated_particles)
    
    # Create updated carry with GMM state included in constructor
    updated_carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state,
        gmm_state=updated_gmm_state  # Pass gmm_state in constructor
    )
    
    return lval, updated_carry