######
from tqdm import tqdm
import typer
from src.problems.toy import *
from src.id import *
from src.trainers.trainer import trainer
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from pathlib import Path
from src.base import *
from src.utils import (make_step_and_carry,
                       config_to_parameters,
                       parse_config)
import pickle
from ot.sliced import sliced_wasserstein_distance
import numpy
from mmdfuse import mmdfuse
import pandas as pd


app = typer.Typer()

PROBLEMS = {
    'banana': Banana,
    'multimodal': Multimodal,
    'xshape': XShape,
}

ALGORITHMS = ['wgf_gmm_entropy', 'wgf_gmm_dirichlet', 'pvi'] #, 'sm', 'svi', 'uvi']

def visualize(key, 
              ids,
              target,
              path,
              prefix=""):
    _max = 4.5
    _min = -4.5
    x_lin = np.linspace(_min, _max, 1000)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)
    log_p = lambda x : target.log_prob(x, None)
    log_true_ZZ = vmap(vmap(log_p))(XY)
    plt.clf()
    if 'pvi' in ids:
        model_log_p = lambda x: ids['pvi'].log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)

        diff = np.abs(np.exp(log_true_ZZ) - np.exp(log_model_ZZ))
        plt.imshow(diff, cmap=mpl.colormaps['Reds'])
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.locator_params(nbins=5)

        c_true = plt.contour(
            np.exp(log_true_ZZ),
            levels=5,
            colors='black',
            linewidths=6)
            #linestyles='dashed',
            #label='True')
        c_model = plt.contour(
            np.exp(log_model_ZZ),
            levels=c_true._levels,
            colors='deepskyblue',
            linewidths=2)
            #label='Model')
        labels = ['True', 'Model']
        # Fix for matplotlib compatibility
        try:
            c_true_color = c_true.collections[-1].get_edgecolor() if hasattr(c_true, 'collections') else 'black'
            c_model_color = c_model.collections[-1].get_edgecolor() if hasattr(c_model, 'collections') else 'deepskyblue'
        except (AttributeError, IndexError):
            c_true_color = 'black'
            c_model_color = 'deepskyblue'
            
        lines = [plt.Line2D([0], [0], color=c_true_color, lw=2),
                plt.Line2D([0], [0], color=c_model_color, lw=2)]
        #plt.legend(lines, labels)
        plt.xticks([])
        plt.yticks([])

        (path / 'pvi').mkdir(exist_ok=True, parents=True)
        plt.savefig(path / 'pvi' / f"{prefix}_pdf.pdf")

    for alg, id in ids.items():
        print(f"\n=== VISUALIZING {alg} ===")
        plt.clf()
        m_key, t_key, key = jax.random.split(key, 3)
        
        # Sample with error handling
        try:
            model_samples = id.sample(m_key, 100, None)
            target_samples = target.sample(t_key, 100, None)
            
            # Check for NaN/Inf values
            if np.any(np.isnan(model_samples)) or np.any(np.isinf(model_samples)):
                print(f"WARNING: {alg} model samples contain NaN/Inf values. Stats:")
                print(f"  Shape: {model_samples.shape}")
                print(f"  NaN count: {np.sum(np.isnan(model_samples))}")
                print(f"  Inf count: {np.sum(np.isinf(model_samples))}")
                print(f"  Min: {np.nanmin(model_samples)}")
                print(f"  Max: {np.nanmax(model_samples)}")
                # Skip this algorithm's visualization
                continue
                
            if np.any(np.isnan(target_samples)) or np.any(np.isinf(target_samples)):
                print(f"WARNING: Target samples contain NaN/Inf values. Skipping {alg} visualization.")
                continue
                
            print(f"✓ {alg} sampling test passed (100 samples)")
            print(f"  Model samples range: [{np.min(model_samples):.3f}, {np.max(model_samples):.3f}]")
            print(f"  Target samples range: [{np.min(target_samples):.3f}, {np.max(target_samples):.3f}]")
                
        except Exception as e:
            print(f"ERROR: Failed to sample from {alg}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        c = plt.contour(XX,
                        YY,
                        np.exp(log_true_ZZ),
                        levels=5,
                        cmap='Reds',)
        plt.scatter(model_samples[..., 0],
                    model_samples[..., 1],
                    alpha=0.5,
                    label='Model Samples')
        plt.scatter(target_samples[..., 0],
                    target_samples[..., 1],
                    alpha=0.5,
                    label='Target Samples')
        
        # Create algorithm-specific directory
        (path / f'{alg}').mkdir(exist_ok=True, parents=True)
        plt.savefig(path / f'{alg}' / f"{prefix}_samples.pdf")

        # ECDF plot with additional error handling
        plt.clf()
        try:
            print(f"Attempting large sample (10000) from {alg}...")
            model_samples_large = id.sample(key, 10000, None)
            
            # Check for NaN/Inf values in large sample
            if np.any(np.isnan(model_samples_large)) or np.any(np.isinf(model_samples_large)):
                print(f"WARNING: {alg} large model samples contain NaN/Inf values.")
                print(f"  NaN count: {np.sum(np.isnan(model_samples_large))}")
                print(f"  Inf count: {np.sum(np.isinf(model_samples_large))}")
                # Create a simple scatter plot instead using the valid small samples
                plt.scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.5)
                plt.title(f'{alg}: Using small sample due to NaN/Inf in large sample')
            else:
                # Check if samples have finite range
                x_range = [np.min(model_samples_large[:, 0]), np.max(model_samples_large[:, 0])]
                y_range = [np.min(model_samples_large[:, 1]), np.max(model_samples_large[:, 1])]
                
                print(f"  Large sample range: x=[{x_range[0]:.3f}, {x_range[1]:.3f}], y=[{y_range[0]:.3f}, {y_range[1]:.3f}]")
                
                if np.isfinite(x_range).all() and np.isfinite(y_range).all() and x_range[0] != x_range[1] and y_range[0] != y_range[1]:
                    plt.hist2d(model_samples_large[:, 0],
                              model_samples_large[:, 1],
                              bins=100,
                              cmap='Blues',
                              label='Samples')
                    print(f"✓ {alg} hist2d plot created successfully")
                else:
                    print(f"WARNING: {alg} samples have degenerate range. Using scatter plot.")
                    plt.scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.5)
                    plt.title(f'{alg}: Degenerate range, using scatter plot')
                    
        except Exception as e:
            print(f"ERROR: Failed to create ECDF plot for {alg}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to text
            plt.text(0.5, 0.5, f'{alg}: Visualization failed\n{str(e)[:50]}...', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        
        plt.savefig(path / f'{alg}' / f"{prefix}_ecdf.pdf")
        
        # Clean up large sample array to save memory
        if 'model_samples_large' in locals():
            del model_samples_large


def test(key, x, y):
    output = mmdfuse(x, y, key)
    return output


def compute_power(
        key,
        target,
        id,
        n_samples=500,
        n_retries=100):
    avg_rej = 0
    for _ in range(n_retries):
        m_key, t_key, test_key, key = jax.random.split(key, 4)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        avg_rej = avg_rej + test(
            test_key, model_samples, target_samples,
        )
    return avg_rej / n_retries


def compute_w1(key,
               target,
               id,
               n_samples=10000,
               n_retries=1):
    distance = 0
    for _ in range(n_retries):
        m_key, t_key, key = jax.random.split(key, 3)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        distance = distance + sliced_wasserstein_distance(
            numpy.array(model_samples), numpy.array(target_samples),
            n_projections=100,
        )
    return distance / n_retries


def metrics_fn(key,
            target,
            id):
    # Add safety check for metrics computation
    try:
        # Test sampling first
        test_samples = id.sample(key, 100, None)
        if np.any(np.isnan(test_samples)) or np.any(np.isinf(test_samples)):
            print("WARNING: Model produces NaN/Inf samples, skipping metrics")
            return {'power': float('nan'), 'sliced_w': float('nan')}
        
        power = compute_power(
            key, target, id, n_samples=1000, n_retries=100
        ) 
        sliced_w = compute_w1(key,
                              target,
                              id,
                              n_samples=10000,
                              n_retries=10)
        return {'power': power,
                'sliced_w': sliced_w}
    except Exception as e:
        print(f"ERROR in metrics computation: {e}")
        return {'power': float('nan'), 'sliced_w': float('nan')}


def extract_components_from_config(config_name):
    """Extract number of components from config name (e.g., 'toy-paper-run-100' -> '100')"""
    if '-' in config_name:
        parts = config_name.split('-')
        if parts[-1].isdigit():
            return parts[-1]
    return 'default'


# Add wrapper functions for the entropy version
def wgf_gmm_entropy_de_step(key, carry, target, y, optim, hyperparams):
    """
    Wrapper for WGF-GMM with entropy regularization step.
    """
    try:
        from src.trainers.wgf_gmm_entropy import (
            wgf_gmm_pvi_step_with_entropy_and_dirichlet,
            WGFGMMHyperparams
        )
    except ImportError as e:
        print(f"Error importing entropy WGF-GMM: {e}")
        raise ImportError("WGF-GMM entropy implementation is not available")
    
    # Handle the gmm_state attribute that WGF-GMM expects
    if not hasattr(carry, 'gmm_state'):
        # Create a temporary extended carry with gmm_state
        class ExtendedCarry:
            def __init__(self, original_carry):
                self.id = original_carry.id
                self.theta_opt_state = original_carry.theta_opt_state
                self.r_opt_state = original_carry.r_opt_state
                self.r_precon_state = original_carry.r_precon_state
                self.gmm_state = None  # Initialize as None
        
        extended_carry = ExtendedCarry(carry)
    else:
        extended_carry = carry
    
    # Set up WGF-GMM hyperparameters with entropy regularization
    wgf_hyperparams = WGFGMMHyperparams(
        lambda_reg=0.1,           # Wasserstein regularization
        lambda_dirichlet=0.1,     # Dirichlet prior
        entropy_weight=0.01,      # Entropy regularization (key addition)
        alpha_value=0.1,          # Dirichlet concentration
        lr_mean=0.01,            # Learning rates
        lr_cov=0.001,
        lr_weight=0.01,
        prune_threshold=1e-3,     # Component pruning
        min_components=1
    )
    
    # Call the WGF-GMM implementation with entropy
    lval, updated_extended_carry = wgf_gmm_pvi_step_with_entropy_and_dirichlet(
        key=key,
        carry=extended_carry,
        target=target,
        y=y,
        optim=optim,
        hyperparams=hyperparams,
        wgf_hyperparams=wgf_hyperparams
    )
    
    # Convert back to standard PIDCarry format
    updated_carry = type(carry)(
        id=updated_extended_carry.id,
        theta_opt_state=updated_extended_carry.theta_opt_state,
        r_opt_state=updated_extended_carry.r_opt_state,
        r_precon_state=updated_extended_carry.r_precon_state
    )
    
    return lval, updated_carry


def wgf_gmm_dirichlet_de_step(key, carry, target, y, optim, hyperparams):
    """
    Wrapper for WGF-GMM with Dirichlet prior step.
    """
    try:
        from src.trainers.wgf_gmm_dirichlet import (
            wgf_gmm_pvi_step_with_dirichlet,
            WGFGMMHyperparams
        )
    except ImportError as e:
        print(f"Error importing Dirichlet WGF-GMM: {e}")
        raise ImportError("WGF-GMM Dirichlet implementation is not available")
    
    # Handle the gmm_state attribute
    if not hasattr(carry, 'gmm_state'):
        class ExtendedCarry:
            def __init__(self, original_carry):
                self.id = original_carry.id
                self.theta_opt_state = original_carry.theta_opt_state
                self.r_opt_state = original_carry.r_opt_state
                self.r_precon_state = original_carry.r_precon_state
                self.gmm_state = None
        
        extended_carry = ExtendedCarry(carry)
    else:
        extended_carry = carry
    
    # Set up hyperparameters for Dirichlet version
    wgf_hyperparams = WGFGMMHyperparams(
        lambda_reg=0.1,
        lambda_dirichlet=0.1,
        alpha_value=0.1,
        lr_mean=0.01,
        lr_cov=0.001,
        lr_weight=0.01,
        prune_threshold=1e-3,
        min_components=1
    )
    
    # Call the Dirichlet implementation
    lval, updated_extended_carry = wgf_gmm_pvi_step_with_dirichlet(
        key=key,
        carry=extended_carry,
        target=target,
        y=y,
        optim=optim,
        hyperparams=hyperparams,
        wgf_hyperparams=wgf_hyperparams
    )
    
    # Convert back to standard format
    updated_carry = type(carry)(
        id=updated_extended_carry.id,
        theta_opt_state=updated_extended_carry.theta_opt_state,
        r_opt_state=updated_extended_carry.r_opt_state,
        r_precon_state=updated_extended_carry.r_precon_state
    )
    
    return lval, updated_carry


# Import PVI step
from src.trainers.pvi import de_step as pvi_de_step

# Define the step functions mapping
STEP_FUNCTIONS = {
    'wgf_gmm_entropy': wgf_gmm_entropy_de_step,
    'wgf_gmm_dirichlet': wgf_gmm_dirichlet_de_step,
    'pvi': pvi_de_step
}


@app.command()
def run(config_name: str,
        seed: int=2):
    config_path = Path(f"scripts/sec_5.2/config/{config_name}.yaml")
    assert config_path.exists()
    config = parse_config(config_path)

    n_rerun = config['experiment']['n_reruns']
    n_updates = config['experiment']['n_updates']
    name = config['experiment']['name']
    name = 'default' if len(name) == 0 else name
    compute_metrics = config['experiment']['compute_metrics']
    use_jit = config['experiment']['use_jit']

    # Extract components number from config name
    components = extract_components_from_config(config_name)
    
    parent_path = Path(f"output/sec_5.2/{name}")
    key = jax.random.PRNGKey(seed)
    histories = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    results = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    print(f"\n=== STARTING EXPERIMENT ===")
    print(f"Config: {config_name}")
    print(f"Components: {components}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Problems: {list(PROBLEMS.keys())}")
    print(f"Reruns: {n_rerun}, Updates: {n_updates}")
    print(f"Use JIT: {use_jit}")

    for prob_name, problem in PROBLEMS.items():
        print(f"\n{'='*50}")
        print(f"PROBLEM: {prob_name}")
        print(f"{'='*50}")
        
        for i in tqdm(range(n_rerun), desc=f"Rerun for {prob_name}"):
            print(f"\n--- Rerun {i+1}/{n_rerun} for {prob_name} ---")
            trainer_key, init_key, key = jax.random.split(key, 3)
            ids = {}
            target = problem()
            path = parent_path / f"{prob_name}"
            path.mkdir(parents=True, exist_ok=True)

            for algo in ALGORITHMS:
                print(f"\n=== RUNNING {algo} ===")
                m_key, key = jax.random.split(key, 2)
                
                try:
                    # Use the appropriate algorithm name for config parsing
                    config_algo = 'wgf_gmm_dirichlet' if algo in ['wgf_gmm_entropy', 'wgf_gmm_dirichlet'] else algo
                    parameters = config_to_parameters(config, config_algo)
                    print(f"✓ Parameters loaded for {algo}")
                    
                    # Debug the algorithm parameters
                    if hasattr(parameters, 'theta_opt_parameters'):
                        theta_opt = parameters.theta_opt_parameters
                        print(f"  Theta optimizer - lr: {theta_opt.lr}, optimizer: {theta_opt.optimizer}")
                        if hasattr(theta_opt, 'clip') and theta_opt.clip:
                            print(f"  Gradient clipping enabled: max_clip = {theta_opt.max_clip}")

                    if hasattr(parameters, 'r_opt_parameters'):
                        r_opt = parameters.r_opt_parameters
                        print(f"  R optimizer - lr: {r_opt.lr}, regularization: {r_opt.regularization}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to load parameters for {algo}: {e}")
                    continue
                
                try:
                    step, carry = make_step_and_carry(
                        init_key,
                        parameters,
                        target)
                    
                    # Replace the step function with our custom implementations
                    if algo in STEP_FUNCTIONS:
                        def custom_step(key, carry, target, y):
                            return STEP_FUNCTIONS[algo](
                                key, carry, target, y, 
                                step.__defaults__[0],  # optim 
                                step.__defaults__[1]   # hyperparams
                            )
                        step = custom_step
                    
                    print(f"✓ Step and carry initialized for {algo}")
                    
                    # Check initial model state
                    print(f"  Initial model type: {type(carry.id)}")
                    if hasattr(carry.id, 'particles'):
                        particles = carry.id.particles
                        print(f"  Initial particles shape: {particles.shape}")
                        print(f"  Initial particles stats: min={np.min(particles):.3f}, max={np.max(particles):.3f}, mean={np.mean(particles):.3f}, std={np.std(particles):.3f}")
                        if np.any(np.isnan(particles)) or np.any(np.isinf(particles)):
                            print("  WARNING: Initial particles contain NaN/Inf values!")
                    
                except Exception as e:
                    print(f"ERROR: Failed to initialize step/carry for {algo}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Training with detailed error handling
                try:
                    print(f"  Starting training for {algo}...")
                    metrics = compute_w1 if compute_metrics else None
                    
                    history, carry = trainer(
                        trainer_key,
                        carry,
                        target,
                        None,
                        step,
                        n_updates,
                        metrics=metrics,
                        use_jit=use_jit,
                    )
                    print(f"✓ {algo} training completed successfully")
                    
                    # Check final model state
                    if hasattr(carry.id, 'particles'):
                        final_particles = carry.id.particles
                        print(f"  Final particles stats: min={np.min(final_particles):.3f}, max={np.max(final_particles):.3f}, mean={np.mean(final_particles):.3f}, std={np.std(final_particles):.3f}")
                        if np.any(np.isnan(final_particles)) or np.any(np.isinf(final_particles)):
                            print("  WARNING: Final particles contain NaN/Inf values!")
                    
                    # Test sampling before storing
                    test_key, key = jax.random.split(key, 2)
                    try:
                        test_samples = carry.id.sample(test_key, 10, None)
                        if np.any(np.isnan(test_samples)) or np.any(np.isinf(test_samples)):
                            print(f"  WARNING: {algo} produces NaN/Inf samples!")
                        else:
                            print(f"  ✓ {algo} sampling test passed")
                    except Exception as sample_error:
                        print(f"  ERROR: {algo} sampling failed: {sample_error}")
                        
                except Exception as training_error:
                    print(f"ERROR: {algo} training failed: {training_error}")
                    import traceback
                    traceback.print_exc()
                    continue  # Skip this algorithm and continue with others

                # Store results only if training succeeded
                ids[algo] = carry.id
                
                # Save training history plots
                try:
                    for k, v in history.items():
                        plt.clf()
                        plt.plot(v, label=k)
                        (path / f"{algo}").mkdir(exist_ok=True, parents=True)
                        plt.savefig(path / f"{algo}" / f"iter{i}_{k}.pdf")
                        histories[prob_name][algo][k].append(np.stack(v, axis=0))
                except Exception as e:
                    print(f"WARNING: Failed to save history for {algo}: {e}")
                
                # Compute metrics
                try:
                    metrics = metrics_fn(
                        m_key,
                        target,
                        ids[algo])
                    for met_key, met_value in metrics.items():
                        results[prob_name][algo][met_key].append(met_value)
                    print(f"  ✓ Metrics computed for {algo}: {metrics}")
                except Exception as e:
                    print(f"WARNING: Failed to compute metrics for {algo}: {e}")

            # Visualization (only if we have valid models)
            if ids:  # Only visualize if we have at least one successful algorithm
                try:
                    print(f"\n=== VISUALIZATION for {prob_name} rerun {i} ===")
                    visualize_key, key = jax.random.split(key, 2)
                    visualize(visualize_key,
                              ids,
                              target,
                              path,
                              prefix=f"iter{i}")
                    print(f"✓ Visualization completed for {prob_name} rerun {i}")
                except Exception as e:
                    print(f"ERROR: Visualization failed for {prob_name} rerun {i}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"WARNING: No successful algorithms for {prob_name} rerun {i}, skipping visualization")
    
    #dump results
    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d

    results = default_to_regular(results)
    histories = default_to_regular(histories)
    
    # Create component-specific filenames
    results_filename = f'{name}_results_{components}.pkl'
    histories_filename = f'{name}_histories_{components}.pkl'
    csv_filename = f'{name}_comparison_{components}.csv'
    
    try:
        with open(parent_path / histories_filename, 'wb') as f:
            pickle.dump(histories, f)

        with open(parent_path / results_filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Results saved to {parent_path / results_filename}")
    except Exception as e:
        print(f"ERROR: Failed to dump results: {e}")

    # Create CSV comparison file
    csv_data = []
    for prob_name, problem in PROBLEMS.items():
        for algo in ALGORITHMS:
            if algo in results[prob_name].keys():
                for met_name, run in results[prob_name][algo].items():
                    if len(run) > 1:
                        run = np.stack(run, axis=-1)
                        mean = np.mean(run, axis=-1)
                        std = np.std(run, axis=-1)
                    else:
                        mean = run[0]
                        std = 0
                    csv_data.append({
                        'problem': prob_name,
                        'algorithm': algo,
                        'metric': met_name,
                        'components': components,
                        'mean': mean,
                        'std': std
                    })

    # Save CSV file
    if csv_data:
        try:
            df = pd.DataFrame(csv_data)
            df.to_csv(parent_path / csv_filename, index=False)
            print(f"✓ CSV comparison saved to: {parent_path / csv_filename}")
        except Exception as e:
            print(f"ERROR: Failed to save CSV: {e}")

    if compute_metrics:
        for prob_name, problem in PROBLEMS.items():
            for algo in ALGORITHMS:
                if algo in histories[prob_name]:
                    for metric_name, run in histories[prob_name][algo].items():
                        run = np.stack(run, axis=0)
                        assert run.shape == (n_rerun, n_updates)
                        last = run[:, -1]
                        if len(histories) > 1:
                            mean = np.mean(last, axis=-1)
                            std = np.std(last, axis=-1) 
                        else:
                            mean = last[0]
                            std = 0
                        print(f"{algo} on {prob_name} with {metric_name} has mean {mean:.3f} and std {std:.3f}")
    
    print(f"\n=== FINAL RESULTS SUMMARY ===")
    for prob_name, problem in PROBLEMS.items():
        for algo in ALGORITHMS:
            if algo in results[prob_name].keys():
                for met_name, run in  results[prob_name][algo].items():
                    if len(run) > 1:
                        run = np.stack(run, axis=-1)
                        mean = np.mean(run, axis=-1)
                        std = np.std(run, axis=-1)
                    else:
                        mean = run[0]
                        std = 0
                    print(f"{algo} on {prob_name} {met_name} has mean {mean:.3f} and std {std:.3f}")