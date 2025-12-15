#!/usr/bin/env python3
"""
Replication of GMM (Gaussian Mixture Model) experiments from 
"The Curse of Recursion: Training on Generated Data Makes Models Forget"

This demonstrates model collapse in the simplest case - fitting GMMs recursively.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import scipy.stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_l2_distance_gmms(gmm1, gmm2):
    """Calculate L2 distance between two GMM distributions."""
    try:
        l2_distance = 0
        
        # First term: integral of gmm1^2
        for i in range(gmm1.n_components):
            for j in range(gmm1.n_components):
                mean_diff = gmm1.means_[i] - gmm1.means_[j]
                cov_sum = gmm1.covariances_[i] + gmm1.covariances_[j]
                l2_distance += gmm1.weights_[i] * gmm1.weights_[j] * \
                    scipy.stats.multivariate_normal.pdf(mean_diff.flatten(), cov=cov_sum.flatten()[0])
        
        # Second term: integral of gmm2^2
        for i in range(gmm2.n_components):
            for j in range(gmm2.n_components):
                mean_diff = gmm2.means_[i] - gmm2.means_[j]
                cov_sum = gmm2.covariances_[i] + gmm2.covariances_[j]
                l2_distance += gmm2.weights_[i] * gmm2.weights_[j] * \
                    scipy.stats.multivariate_normal.pdf(mean_diff.flatten(), cov=cov_sum.flatten()[0])
        
        # Cross term: -2 * integral of gmm1 * gmm2
        for i in range(gmm1.n_components):
            for j in range(gmm2.n_components):
                mean_diff = gmm1.means_[i] - gmm2.means_[j]
                cov_sum = gmm1.covariances_[i] + gmm2.covariances_[j]
                l2_distance -= 2 * gmm1.weights_[i] * gmm2.weights_[j] * \
                    scipy.stats.multivariate_normal.pdf(mean_diff.flatten(), cov=cov_sum.flatten()[0])
        
        return l2_distance
    except:
        return np.nan

def run_gmm_collapse_experiment(n_samples=1000, n_components=2, n_generations=100):
    """
    Run the GMM model collapse experiment.
    
    Args:
        n_samples: Number of samples to generate at each generation
        n_components: Number of components in the GMM
        n_generations: Number of recursive generations
    
    Returns:
        Dictionary with results including L2 distances, means, and variances
    """
    
    print(f"Running GMM collapse experiment:")
    print(f"  Samples per generation: {n_samples}")
    print(f"  GMM components: {n_components}")
    print(f"  Number of generations: {n_generations}")
    
    # Initialize results storage
    means_evolution = []
    stds_evolution = []
    l2_distances = []
    
    # Generate original data from a true GMM
    print("\nGenerating original data...")
    true_means = np.array([[-2.0], [2.0]])[:n_components]
    true_variances = np.array([[1.0], [1.0]])[:n_components]
    
    # Create original data from mixture of Gaussians
    original_data = []
    for i in range(n_samples):
        component = np.random.choice(n_components)
        sample = np.random.normal(true_means[component], np.sqrt(true_variances[component]))
        original_data.append(sample)
    original_data = np.array(original_data)
    
    # Fit original GMM
    gmm_params = {
        "n_components": n_components,
        "covariance_type": "full",
        "max_iter": 2000,
        "tol": 1e-5
    }
    
    original_gmm = GMM(**gmm_params)
    original_gmm.fit(original_data)
    
    # Store original statistics
    means_evolution.append(original_gmm.means_.flatten())
    stds = [np.sqrt(np.trace(original_gmm.covariances_[i])/n_components) 
            for i in range(n_components)]
    stds_evolution.append(stds)
    
    # Recursive training loop
    print("\nRunning recursive generations...")
    current_data = original_data
    current_gmm = original_gmm
    
    for generation in range(n_generations):
        if generation % 10 == 0:
            print(f"  Generation {generation}/{n_generations}")
        
        # Generate new data from current GMM
        new_data, _ = current_gmm.sample(n_samples)
        
        # Fit new GMM to generated data
        new_gmm = GMM(**gmm_params)
        new_gmm.fit(new_data)
        
        # Calculate L2 distance from original
        l2_dist = calculate_l2_distance_gmms(original_gmm, new_gmm)
        l2_distances.append(l2_dist)
        
        # Store statistics
        means_evolution.append(new_gmm.means_.flatten())
        stds = [np.sqrt(np.trace(new_gmm.covariances_[i])/n_components) 
                for i in range(n_components)]
        stds_evolution.append(stds)
        
        # Update for next generation
        current_data = new_data
        current_gmm = new_gmm
    
    print("\nExperiment complete!")
    
    return {
        "means": np.array(means_evolution),
        "stds": np.array(stds_evolution),
        "l2_distances": np.array(l2_distances),
        "original_data": original_data,
        "final_data": current_data,
        "original_gmm": original_gmm,
        "final_gmm": current_gmm
    }

def plot_results(results):
    """Plot the results showing model collapse."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot 1: Original vs Final data distributions
    ax = axes[0, 0]
    ax.hist(results["original_data"], bins=30, alpha=0.5, label="Original", density=True)
    ax.hist(results["final_data"], bins=30, alpha=0.5, label="Final", density=True)
    ax.set_title("Data Distribution: Original vs Final Generation")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    
    # Plot 2: Means evolution
    ax = axes[0, 1]
    means = results["means"]
    for i in range(means.shape[1]):
        ax.plot(means[:, i], label=f"Component {i+1}")
    ax.set_title("Evolution of GMM Means")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Value")
    ax.legend()
    
    # Plot 3: Standard deviations evolution
    ax = axes[0, 2]
    stds = results["stds"]
    for i in range(stds.shape[1]):
        ax.plot(stds[:, i], label=f"Component {i+1}")
    ax.set_title("Evolution of GMM Standard Deviations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Standard Deviation")
    ax.legend()
    
    # Plot 4: L2 distance from original
    ax = axes[1, 0]
    ax.plot(results["l2_distances"])
    ax.set_title("L2 Distance from Original GMM")
    ax.set_xlabel("Generation")
    ax.set_ylabel("L2 Distance")
    ax.set_yscale('log')
    
    # Plot 5: Variance collapse ratio
    ax = axes[1, 1]
    original_var = results["stds"][0].mean()
    variance_ratio = [s.mean() / original_var for s in results["stds"]]
    ax.plot(variance_ratio)
    ax.set_title("Variance Collapse (Relative to Original)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Variance Ratio")
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
    Model Collapse Summary:
    
    Original variance: {results['stds'][0].mean():.4f}
    Final variance: {results['stds'][-1].mean():.4f}
    Variance reduction: {(1 - results['stds'][-1].mean()/results['stds'][0].mean())*100:.1f}%
    
    Final L2 distance: {results['l2_distances'][-1]:.4f}
    
    Key Finding: 
    Recursive training causes
    progressive loss of variance
    and distribution diversity.
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig("gmm_model_collapse_results.png", dpi=150)
    
    print("\nResults saved to 'gmm_model_collapse_results.png'")

def main():
    """Main function to run the experiment."""
    
    print("=" * 60)
    print("Replicating Model Collapse with Gaussian Mixture Models")
    print("From: 'The Curse of Recursion' Paper")
    print("=" * 60)
    
    # Run experiment with different sample sizes to show effect
    sample_sizes = [500, 1000]  # Reduced for faster execution
    all_results = {}
    
    for n_samples in sample_sizes:
        results = run_gmm_collapse_experiment(
            n_samples=n_samples,
            n_components=2,
            n_generations=20  # Reduced for faster execution
        )
        all_results[n_samples] = results
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Plot variance collapse for different sample sizes
    plt.subplot(1, 2, 1)
    for n_samples, results in all_results.items():
        original_var = results["stds"][0].mean()
        variance_ratio = [s.mean() / original_var for s in results["stds"]]
        plt.plot(variance_ratio, label=f"N={n_samples}")
    plt.title("Variance Collapse for Different Sample Sizes")
    plt.xlabel("Generation")
    plt.ylabel("Variance Ratio")
    plt.legend()
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    
    # Plot L2 distances
    plt.subplot(1, 2, 2)
    for n_samples, results in all_results.items():
        plt.plot(results["l2_distances"], label=f"N={n_samples}")
    plt.title("L2 Distance from Original Distribution")
    plt.xlabel("Generation")
    plt.ylabel("L2 Distance")
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("gmm_sample_size_comparison.png", dpi=150)
    
    print("\nComparison saved to 'gmm_sample_size_comparison.png'")
    
    # Show detailed results for n=1000
    print("\nDetailed results for N=1000:")
    plot_results(all_results[1000])
    
    # Save results to JSON file
    print("\nSaving results to JSON...")
    results_to_save = {}
    for n_samples, results in all_results.items():
        results_to_save[f"n_{n_samples}"] = {
            "means": results["means"].tolist(),
            "stds": results["stds"].tolist(),
            "l2_distances": [float(x) if not np.isnan(x) else None for x in results["l2_distances"]],
            "original_variance": float(results["stds"][0].mean()),
            "final_variance": float(results["stds"][-1].mean()),
            "variance_reduction_percent": float((1 - results["stds"][-1].mean()/results["stds"][0].mean())*100)
        }
    
    with open("gmm_collapse_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    print("Results saved to: gmm_collapse_results.json")

if __name__ == "__main__":
    main()