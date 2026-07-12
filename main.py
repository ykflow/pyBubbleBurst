import os
import numpy as np
import matplotlib.pyplot as plt
from models.explosions_model import DynamicLocalExplosionsModel
from utilities.utils import load_btc_data
from plotting_tools.set_plotting_theme import set_theme, colors




def main():
    set_theme()

    csv_path = os.path.join("datasets", "CBBTCUSD.csv")
    print(f"[*] Reading and preprocessing market data from: {csv_path}")

    data = load_btc_data(csv_path)
    data = data[data.index > '2016']
    raw_prices = data['BTC/USD'].values
    log_prices = np.log(raw_prices)


    print(f"[*] Successfully processed {len(log_prices)} trading intervals.")
    print(f"[*] Start Date: {data.index.min().strftime('%Y-%m-%d')}")
    print(f"[*] End Date:   {data.index.max().strftime('%Y-%m-%d')}")

    print("\n[*] Initializing Dynamic Local Explosions Model (Specification: E4)...")
    model = DynamicLocalExplosionsModel(y=log_prices, bubble_type="E4", detrend=True)

    print("[*] Launching maximum likelihood optimization sequence (SLSQP solver)...")
    model.fit(burn_in=15, max_iter_mle=300)

    print("\n" + "=" * 40)
    print("      ESTIMATED PARAMETERS SUMMARY      ")
    print("=" * 40)
    if model.estimated_params:
        for param_name, optimal_val in model.estimated_params.items():
            print(f"  {param_name:<12} : {optimal_val:.6f}")
    print("-" * 40)
    print(f"  Optimization Success : {model.mle_summary.success}")
    print(f"  Optimizer Message    : {model.mle_summary.message}")
    print("=" * 40)

    results = model.states
    results.index = data.index  # Re-bind the date indices back onto states

    # # Save the output arrays locally to disk for later storage access
    # output_csv = os.path.join("datasets", "CBBTCUSD_filtered_results.csv")
    # results.to_csv(output_csv)
    # print(f"\n[*] Structural states written successfully to: {output_csv}")

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(results.index, results['y'], label='Detrended Log Price ($y_t$)', color='black', alpha=0.8)
    plt.plot(results.index, results['mu'], label='Fundamental Value ($\mu_t$)', color=colors[0], linestyle='--')
    plt.title('Filtered Local Explosions for BTC/USD (E4 Model Specification)', size=25)
    plt.ylabel('Log Levels')
    plt.legend(fontsize='20')

    plt.subplot(2, 1, 2)
    plt.plot(results.index, results['b'], label='Bubble Value ($b_t$)', color=colors[2])
    plt.fill_between(results.index, 0, results['b'], where=(results['survival'] > 0), color=colors[2], alpha=0.15,
                     label='Active Bubble Window')
    plt.ylabel('Bubble State Value')
    plt.legend(fontsize='20')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
