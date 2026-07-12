# pyBubbleBurst
<img width="2000" height="1000" alt="filtered_explosions_btc" src="https://github.com/user-attachments/assets/2d681a07-10a9-4bd6-af47-e2221e89ce2d" />

A high-performance econometric time-series pipeline for identifying and tracking explosive speculative regimes (bubbles) and sudden collapses in asset prices. This project implements the **Dynamic Local Explosions Filter (E4 Specification)** of Blasques et al.(2022), whilst utilizing unconstrained Maximum Likelihood Estimation (MLE) routine driven by a decoupled abstract architecture.

## 🚀 Key Features

* **Blazing Fast Computation**: Core recursive filtering operations are fully compiled to native machine code via **Numba JIT (`nopython=True`, `fastmath=True`)**.
* **Unconstrained MLE Optimization**: Leverages parameter tracking layers to map constrained variable domains (like variances and coefficients) into \(\mathbb{R}\) space for robust `scipy.optimize` convergence.
* **Extensible Architecture**: Utilizes a strict Registry-driven Factory design pattern (`FiltersFactory`), letting you plug in new filter modules seamlessly.
* **Built-in Preprocessing**: Includes a clean, native linear detrender to isolate structural cycles from long-term asset drift.

## 🛠️ Project Structure

```text
pyBubbleBurst/
├── datasets/                 # Local data storage (.csv inputs/outputs)
├── estimation/               # Optimization engines & MLE routines
├── filters/                  # Abstract models & compiled Numba JIT filters
├── models/                   # High-level wrapper models (User API)
├── parameters/               # Bounds transformers & link functions
├── plotting_tools/           # A custom theme for Julia-inspired plots
├── utilities/                # Signal/detrending math processing tools
├── requirements.txt          # Third-party package dependencies
└── main.py                   # Master orchestration entry-point
```

## 📦 Installation & Setup

1. **Clone the repository** and navigate to your project directory:
   ```bash
   cd pyBubbleBurst
   ```

2. **Install the dependencies** using the verified `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project targets **Python 3.11** or **Python 3.12** due to Numba compilation specifications.*

## 📈 Quick Start

Place your asset price history (e.g., `CBBTCUSD.csv` formatted from FRED) inside the `datasets/` folder, then run the execution script:

```bash
python main.py
```

### Basic API Usage Example

```python
import numpy as np
from models.dynamic_local_explosions import DynamicLocalExplosionsModel

# 1. Load your raw time-series data vector (e.g., log prices)
log_prices = np.log([150.0, 155.0, 210.0, 450.0, 1100.0, 320.0])

# 2. Initialize the wrapper (automatically activates detrending and loads E4 filter)
model = DynamicLocalExplosionsModel(y=log_prices, bubble_type="E4", detrend=True)

# 3. Fit via maximum likelihood
model.fit(burn_in=2)

# 4. Access final tracking states matrix as a Pandas DataFrame
df_results = model.states
print(df_results[['y', 'mu', 'b', 'survival']])
```

## 📊 Outputs

Upon execution, the pipeline outputs:
1. An optimization summary specifying optimal parameter states ($\delta, \beta, \gamma, \omega, \alpha, \kappa, c, \sigma^2$).
2. An automated analytical chart plotting the **Model Mean ($\mu_t$)** and the **Explosive Bubble Component ($b_t$)** highlighting active bubble regimes.

## References
Francisco Blasques, Siem Jan Koopman*, Marc Nientker (2022). A time-varying parameter model for local explosions. Journal of Econometrics, (227) 65-84.
