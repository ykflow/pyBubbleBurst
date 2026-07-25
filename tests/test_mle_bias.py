import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from simulation.e4_bubble_simulation import E4BubbleModelSimulator
from models.explosions_model import DynamicLocalExplosionsModel  # Using your explicit model class
from plotting_tools.set_plotting_theme import set_theme, colors


def run_monte_carlo(n_simulations: int = 50, T: int = 1000, burn_in: int = 1000):
    # Core reference parameter state
    true_params = {
        "delta": 0.1,
        "beta": 0.95,
        "gamma": 0.7,
        "omega": 0.2,
        "alpha": 1.03,
        "kappa": 7.0,
        "c": -0.1,
        "sigma2": 1.0
    }

    # Initialize simulator outside the loop
    simulator = E4BubbleModelSimulator()

    # Allocation for metrics tracking
    mc_results = {name: [] for name in true_params.keys()}
    failed_simulations = 0

    print(f"Starting Monte Carlo execution utilizing DynamicLocalExplosionsModel across {n_simulations} runs...")

    for i in tqdm(range(n_simulations)):
        try:
            # 1. Generate unique independent path trajectories
            simulator.generate(
                T, burn_in,
                true_params["delta"], true_params["beta"], true_params["gamma"], true_params["omega"],
                true_params["alpha"], true_params["kappa"], true_params["c"], true_params["sigma2"]
            )
            y_sim = simulator.y

            # 2. Bind the simulated data vector to your custom model class
            model = DynamicLocalExplosionsModel(y_sim)

            # 3. Trigger your nested optimization pipeline execution
            model.fit()

            # 4. Extract the final estimated parameters from your model instance.
            # (Note: If your model stores them in an attribute other than .estimated_params, adjust this reference line)
            estimates = model.estimated_params

            # Commit successfully tracked metrics to records
            for param_name in true_params.keys():
                mc_results[param_name].append(estimates[param_name])

        except Exception as e:
            failed_simulations += 1
            continue

    # Process metrics into analysis dataframe
    df_results = pd.DataFrame(mc_results)

    print(f"\nMonte Carlo Execution Complete. (Successful Runs: {len(df_results)} | Failed Runs: {failed_simulations})")

    # Print out summary analytical calculations
    print("\n--- MONTE CARLO ESTIMATION SUMMARY STATISTICS ---")
    summary_data = []
    for p_name, true_val in true_params.items():
        estimates_p = df_results[p_name]
        mean_est = estimates_p.mean()
        bias = mean_est - true_val
        rmse = np.sqrt(((estimates_p - true_val) ** 2).mean())

        summary_data.append({
            "Parameter": p_name,
            "True Value": true_val,
            "MC Mean": mean_est,
            "Bias": bias,
            "RMSE": rmse
        })
    print(pd.DataFrame(summary_data).to_string(index=False))

    # --- Plotting the Estimator Distributions ---
    set_theme()
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    for idx, (p_name, true_val) in enumerate(true_params.items()):
        ax = axes[idx]

        # Plot structural histogram of model estimates
        ax.hist(df_results[p_name], bins=15, alpha=0.6, edgecolor='white', label='Estimates')

        # Add visual reference points for evaluation
        ax.axvline(true_val, linestyle='--', linewidth=2.5, label=f'True ({true_val})')
        ax.axvline(df_results[p_name].mean(), color='black', linestyle=':', linewidth=2, label='MC Mean')

        ax.set_title(f"Distribution of Estimated Parameter: {p_name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_monte_carlo(n_simulations=50, T=1000, burn_in=1000)
