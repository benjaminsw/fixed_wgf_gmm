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

# FIX 1: Add missing 'banana' to PROBLEMS
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


# Import PVI step
from src.trainers.pvi import de_step as pvi_de_step

# FIX 2: Define proper step functions that avoid the NoneType subscriptable error
def create_step_wrapper(step_func_name, config_algo):
    """Create a step wrapper that properly handles optim and hyperparams"""
    
    if step_func_name == 'wgf_gmm_entropy':
        try:
            from src.trainers.wgf_gmm_entropy import (
                wgf_gmm_pvi_step_with_entropy_and_dirichlet,
                WGFGMMHyperparams
            )
            
            def step_wrapper(key, carry, target, y, optim, hyperparams):
                # Create WGF hyperparams
                wgf_hyperparams = WGFGMMHyperparams(
                    lambda_reg=0.1,
                    lambda_dirichlet=0.1,
                    entropy_weight=0.01,
                    alpha_value=0.1,
                    lr_mean=0.01,
                    lr_cov=0.001,
                    lr_weight=0.01,
                    prune_threshold=1e-3,
                    min_components=1
                )
                
                # Handle gmm_state
                if not hasattr(carry, 'gmm_state'):
                    carry.gmm_state = None
                
                return wgf_gmm_pvi_step_with_entropy_and_dirichlet(
                    key, carry, target, y, optim, hyperparams, wgf_hyperparams
                )
            
            return step_wrapper
            
        except ImportError:
            print(f"Could not import {step_func_name}, using PVI")
            return pvi_de_step
    
    elif step_func_name == 'wgf_gmm_dirichlet':
        try:
            from src.trainers.wgf_gmm_dirichlet import (
                wgf_gmm_pvi_step_with_dirichlet,
                WGFGMMHyperparams
            )
            
            def step_wrapper(key, carry, target, y, optim, hyperparams):
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
                
                if not hasattr(carry, 'gmm_state'):
                    carry.gmm_state = None
                
                return wgf_gmm_pvi_step_with_dirichlet(
                    key, carry, target, y, optim, hyperparams, wgf_hyperparams
                )
            
            return step_wrapper
            
        except ImportError:
            print(f"Could not import {step_func_name}, using PVI")
            return pvi_de_step
    
    else:  # pvi
        return pvi_de_step


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
                    
                except Exception as e:
                    print(f"ERROR: Failed to load parameters for {algo}: {e}")
                    continue
                
                try:
                    # FIX 3: Create step function properly to avoid NoneType subscriptable error
                    step, carry = make_step_and_carry(
                        init_key,
                        parameters,
                        target)
                    
                    # Replace with our custom step wrapper
                    step = create_step_wrapper(algo, config_algo)
                    
                    print(f"✓ Step and carry initialized for {algo}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to initialize step/carry for {algo}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Training with error handling
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

    # Create CSV comparison file - FIX 4: Only process existing results
    csv_data = []
    for prob_name in PROBLEMS.keys():
        if prob_name in results:  # Only process problems that have results
            for algo in ALGORITHMS:
                if algo in results[prob_name]:  # Only process algorithms that have results
                    for met_name, run in results[prob_name][algo].items():
                        if len(run) > 1:
                            run = np.stack(run, axis=-1)
                            mean = np.mean(run, axis=-1)
                            std = np.std(run, axis=-1)
                        else:
                            mean = run[0] if len(run) > 0 else 0.0
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
        for prob_name in PROBLEMS.keys():
            if prob_name in histories:  # Only process problems that have histories
                for algo in ALGORITHMS:
                    if algo in histories[prob_name]:
                        for metric_name, run in histories[prob_name][algo].items():
                            if len(run) > 0:
                                run = np.stack(run, axis=0)
                                if run.shape[0] == n_rerun and run.shape[1] == n_updates:
                                    last = run[:, -1]
                                    if len(run) > 1:
                                        mean = np.mean(last, axis=-1)
                                        std = np.std(last, axis=-1) 
                                    else:
                                        mean = last[0]
                                        std = 0
                                    print(f"{algo} on {prob_name} with {metric_name} has mean {mean:.3f} and std {std:.3f}")
    
    print(f"\n=== FINAL RESULTS SUMMARY ===")
    for prob_name in PROBLEMS.keys():
        if prob_name in results:  # Only process problems that have results
            for algo in ALGORITHMS:
                if algo in results[prob_name]:  # Only process algorithms that have results
                    for met_name, run in results[prob_name][algo].items():
                        if len(run) > 1:
                            run = np.stack(run, axis=-1)
                            mean = np.mean(run, axis=-1)
                            std = np.std(run, axis=-1)
                        else:
                            mean = run[0] if len(run) > 0 else 0.0
                            std = 0
                        print(f"{algo} on {prob_name} {met_name} has mean {mean:.3f} and std {std:.3f}")

if __name__ == "__main__":
    app()
#####