# Algorithmic Trading Fraud Detection Using Mahalanobis Distance
### Pattern Recognition and Anomaly Detection — Semester 5 Course Project

> **Application Name:** TRADEGUARD  
> **Subtitle:** Mahalanobis Trading Anomaly Detection  
> **Design Theme:** Modern Clean Light Financial Analytics Platform  
> **Course:** Pattern Recognition and Anomaly Detection (Semester 5)  
> **Official Problem Statement:** *“Build a Mahalanobis distance pipeline to flag irregular high-frequency trades and volume spikes in simulated financial markets.”*

---

## ⚠️ Important Academic Clarification & Disclaimer

> [!IMPORTANT]
> **Simulated Financial Market Data:**  
> This project is an **academic anomaly-detection demonstration** developed using synthetic, simulated high-frequency trading (HFT) market data. It is **not** a real banking, regulatory, or live broker production system.
>
> In financial market surveillance and statistical pattern recognition, an **anomaly** is an observation that statistically diverges from the learned multivariate covariance structure or univariate volume threshold. It does **not** automatically establish fraudulent or illegal activity.
>
> In accordance with academic best practices, this application strictly adopts the following terminology:
> - **“Potential anomaly”**
> - **“Unusual trade”**
> - **“Potential irregular trading behavior”**
> - **“Candidate fraud / anomaly”**
>
> *We deliberately avoid asserting that any trade is "definitely fraudulent", as statistical outliers require human investigator verification.*

---

## 1. Project Objective

High-frequency algorithmic trading generates massive volumes of millisecond-level order and execution data. Traditional univariate thresholding (such as checking if price or quantity individually exceeds a limit) frequently fails to detect sophisticated market irregularities such as:
- Illiquid price impact gaps (moving a price substantially with microscopic order quantity).
- Excessive volume without commensurate liquidity support.
- Severe fat-finger block sweeps.
- Correlation breakdowns across trading attributes.

This project delivers a complete, full-stack analytical web platform (**TRADEGUARD**) designed with a modern light-mode data analytics aesthetic that:
1. Loads and preprocesses simulated high-frequency trading data across multiple equities (`TECH`, `BANK`, `AUTO`, `ENERGY`, `PHARMA`).
2. Extracts numerical features: `price`, `quantity`, `volume`, and `price_change`.
3. Learns the baseline multivariate normal pattern (mean vector and covariance matrix).
4. Calculates the **Mahalanobis Distance** for each trade using a numerically stable Moore-Penrose pseudo-inverse (`np.linalg.pinv`).
5. Identifies observations exceeding a configurable distance threshold ($\tau_M$) as potential anomalies.
6. Analyzes trade-volume spikes in parallel using statistical standard score ($Z$-score, $\tau_V$).
7. Synthesizes a transparent combined classification explaining *why* an observation was flagged (`Mahalanobis Distance`, `Volume Spike`, or `Both`).
8. Provides rich, interactive 2D scatter projections, distribution histograms, and time-series monitoring.
9. Enables individual trade inspection with contextual market comparisons and an interactive **Mahalanobis Distance Deviation Scale**.
10. Supports user-uploaded CSV datasets with automated schema validation and CSV report downloads.

---

## 2. Technology Stack

- **Frontend:**
  - Modern HTML5 semantic layout
  - Vanilla CSS3 (Modern Light Financial Analytics design system: crisp off-white background `#F8FAFC`, elevated white cards `#FFFFFF`, royal blue/indigo primary `#2563EB`, emerald normal state `#16A34A`, crimson anomaly alert `#DC2626`, and subtle slate borders `#E2E8F0`)
  - Vanilla JavaScript (Fetch API, DOM manipulation, state management, modal dialogues, toast notifications, light-mode Chart.js engine)
  - [Chart.js 4.4+](https://www.chartjs.org/) (bundled locally in `static/vendor/` for 100% offline portability)
- **Backend:**
  - Python 3.10+ / 3.14+
  - Flask (REST API microframework)
- **Data Science & ML Pipeline:**
  - `pandas` (tabular data manipulation and ingestion)
  - `numpy` (multivariate matrix operations, mean vectors, covariance, and pseudo-inverse)
  - `scikit-learn` & `scipy` (statistical distributions and classification metrics)
- **Dataset:**
  - CSV format (7,500+ simulated HFT records)

---

## 3. Machine Learning / Analytical Pipeline

```
Simulated Trading Dataset (CSV)
        ↓
Data Validation & Schema Verification (Column check, non-emptiness, numeric type verification)
        ↓
Missing Value Handling & Preprocessing (Median imputation & normalization)
        ↓
Feature Selection ([Price, Quantity, Volume, Price Change])
        ↓
Covariance Matrix Estimation & Regularization (Σ + ε·I, ε = 10⁻⁶)
        ↓
Moore-Penrose Pseudo-Inverse (Σ⁺ = np.linalg.pinv(Σ))
        ↓
Mahalanobis Distance Calculation (D_M = √((x - μ)ᵀ Σ⁺ (x - μ)))
        ↓
Distance Threshold Comparison (D_M > τ_M → Anomaly Flag)
        ↓
Volume Spike Z-Score Analysis (Z_v = (Volume - μ_v) / σ_v > τ_V → Volume Spike Flag)
        ↓
Combined Classification & Reason Formulation (Both, Mahalanobis Distance, Volume Spike, None)
        ↓
Interactive Web Dashboard & 2D Market Space Visualizations
        ↓
Exportable Diagnostic CSV Reports
```

---

## 4. Mathematical Foundation

### A. Why Mahalanobis Distance over Euclidean Distance?

In standard Euclidean space, distance is measured spherically assuming all dimensions are independent and have equal variance:
$$D_{\text{Euclidean}}(x, \mu) = \sqrt{\sum_{i=1}^p (x_i - \mu_i)^2}$$

In financial trading, this assumption is **fundamentally invalid**:
1. **Unequal variances:** Trade volume operates in tens or hundreds of thousands of dollars, whereas price change operates in cents.
2. **High correlation:** Order volume is intrinsically tied to price and quantity ($\text{Volume} = \text{Price} \times \text{Quantity}$), and price change often reflects trade size.

**Mahalanobis Distance** solves this by standardizing the variance of each feature and accounting for pairwise covariances:
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Where:
- $x = [x_{\text{price}}, x_{\text{quantity}}, x_{\text{volume}}, x_{\text{price\_change}}]^T$ is the observation vector.
- $\mu = \frac{1}{N}\sum_{i=1}^N x_i$ is the empirical feature mean vector.
- $\Sigma = \frac{1}{N-1}\sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T$ is the sample covariance matrix.
- $\Sigma^{-1}$ is the inverse covariance matrix.

### B. Numerical Stability with Ridge Regularization and Pseudo-Inverse
If features exhibit collinearity or variance drops to zero, the sample covariance matrix $\Sigma$ becomes singular or ill-conditioned. Calling `numpy.linalg.inv(cov)` will crash with a `LinAlgError`.

TRADEGUARD resolves this through:
1. **Ridge Regularization:** Adding a microscopic identity diagonal perturbation $\epsilon = 10^{-6}$:
   $$\Sigma_{\text{reg}} = \Sigma + 10^{-6} \cdot I_p$$
2. **Moore-Penrose Pseudo-Inverse:** Computing $\Sigma^+$ via singular value decomposition using:
   ```python
   self.inv_covariance_matrix = np.linalg.pinv(cov_reg)
   ```

### C. Statistically Justified Anomaly Thresholds
For a $p$-dimensional multivariate Gaussian distribution, the squared Mahalanobis distance follows a Chi-Square distribution with $p$ degrees of freedom:
$$D_M^2 \sim \chi^2(p)$$

For $p = 4$ features at a 99% confidence level ($\alpha = 0.01$):
$$\chi^2_{4, 0.99} \approx 13.277 \implies \tau_M = \sqrt{13.277} \approx 3.64$$

In practice, market data exhibits slightly heavier tails than pure Gaussian distributions. TRADEGUARD defaults to $\tau_M = 3.00$ while allowing users to dynamically adjust the threshold from $1.5$ to $6.0$ via the interactive control panel.

### D. Volume Spike Analysis
Unilateral order surges (such as fat-finger institutional sweeps) are detected using the volume $Z$-score:
$$Z_v = \frac{v - \mu_v}{\sigma_v}$$

Trades with $Z_v > \tau_V$ (default $\tau_V = 3.00$, corresponding to the upper $0.13\%$ tail) are flagged as potential volume spikes.

---

## 5. Dataset Description

The generated simulated dataset (`data/trading_data.csv`) contains **7,500 trade executions** spanning a simulated trading session across 5 diverse equity sectors:

| Symbol | Sector | Typical Base Price | Typical Quantity | Market Role |
| :--- | :--- | :--- | :--- | :--- |
| **`TECH`** | Technology | $182.50 | 180 shares | High notional value, moderate volatility |
| **`BANK`** | Financials | $46.20 | 350 shares | High share volume, tight spread |
| **`AUTO`** | Automotive | $68.75 | 220 shares | Cyclical consumer stock |
| **`ENERGY`** | Energy | $93.10 | 200 shares | Commodity-sensitive stock |
| **`PHARMA`** | Healthcare | $138.40 | 160 shares | High-margin healthcare stock |

### Dataset Columns
1. `trade_id`: Unique transaction identifier (`TRD-100000` to `TRD-107499`).
2. `timestamp`: High-frequency execution time (`YYYY-MM-DD HH:MM:SS.mmm`).
3. `symbol`: Ticker code (`TECH`, `BANK`, `AUTO`, `ENERGY`, `PHARMA`).
4. `price`: Executed transaction price ($).
5. `quantity`: Number of contracts/shares traded.
6. `volume`: Notional trade dollar value ($\text{Price} \times \text{Quantity}$).
7. `price_change`: Price difference ($\Delta$) relative to baseline.
8. `synthetic_anomaly`: Ground-truth label ($0$ = Normal, $1$ = Injected synthetic irregularity) used strictly for controlled academic validation.
9. `anomaly_type`: Ground-truth subtype (`normal`, `volume_spike`, `price_shock`, `multivariate_break`, `liquidity_vacuum`).

---

## 6. Directory Structure

```
algorithmic-trading-anomaly/
│
├── app.py                      # Flask backend application and REST endpoints
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive academic documentation and viva guide
├── test_ml.py                  # Verification unit test script
│
├── data/
│   ├── generate_data.py        # Realistic HFT synthetic dataset generator
│   └── trading_data.csv        # 7,500 records simulated trading dataset
│
├── ml/
│   ├── __init__.py             # ML package exports
│   └── mahalanobis_detector.py # Core Mahalanobis & Volume Spike pipeline
│
├── templates/
│   └── index.html              # Modern dark single-page dashboard template
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark financial analytics CSS styling
│   ├── js/
│   │   └── app.js              # Vanilla JS frontend controller & Chart.js logic
│   └── vendor/
│       └── chart.umd.min.js    # Self-contained offline Chart.js bundle
│
├── results/
│   └── detected_anomalies.csv  # Output CSV containing flagged potential anomalies
│
└── venv/                       # Local virtual environment (portable)
```

---

## 7. Installation & Quick Start

### Step 1: Clone or Navigate to Project Directory
```bash
cd "/Users/kenvilkalotra/Downloads/IBM Project"
# or navigate to your extracted folder
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Re-generate Synthetic Data
```bash
python data/generate_data.py
```

### Step 5: Start the Flask Application
```bash
python app.py
```

The application will start immediately at:
**`http://127.0.0.1:5001`**

Open your web browser and navigate to the address above.

---

## 8. Dashboard Walkthrough & User Guide

1. **Dashboard:**
   - Review live dynamic summary cards: Total Trades, Normal Trades, Potential Anomalies, Anomaly %, Average Volume, Max Volume, and Average Mahalanobis Distance.
   - Inspect the interactive charts: Normal vs Anomaly Doughnut, Mahalanobis Distance Histogram (with threshold line), Volume Distribution, Price Movement line chart, and Trading Activity Timeline.
   - View the Synthetic Evaluation Box (Precision, Recall, F1-Score, Confusion Matrix).
2. **Trade Analysis:**
   - Search for any specific Trade ID (e.g., `TRD-100063`, `TRD-100109`, `TRD-100138`) or pick from the Quick Select dropdown of flagged anomalies.
   - Inspect the diagnostic interpretation card explaining whether the anomaly arose from multivariate correlation deviation, a volume spike, or both.
3. **Market Visualization:**
   - Explore 2D scatter projections: Price vs Trade Volume, Price vs Quantity, and Trade Volume vs Price Change.
   - Toggle **“Show Anomalies Only”** to isolate outliers or click **“Show All”** to view the full market depth.
4. **Run Detection:**
   - Adjust the **Mahalanobis Distance Threshold** ($\tau_M$) and **Volume Z-Score Threshold** ($\tau_V$) using responsive sliders.
   - Click **“Run Anomaly Detection Pipeline”** to re-score the dataset in real-time.
   - Drag and drop or browse to upload a custom CSV dataset (with automatic schema and type validation).
   - Click **“Restore Default Synthetic Dataset”** anytime to reset back to the 7,500-record benchmark.
5. **Results & Trades:**
   - Browse paginated trade records (20 trades per page).
   - Filter by Asset/Symbol (`TECH`, `BANK`, etc.), Anomaly Status (`POTENTIAL ANOMALY`, `NORMAL`), or Flag Reason (`Mahalanobis Distance`, `Volume Spike`, `Both`).
   - Sort by Mahalanobis distance or Volume.
   - Click **“Inspect”** on any trade row to jump directly to its individual diagnostic card.
   - Click **“Download Anomaly Results (CSV)”** to export the flagged records.
6. **About & Methodology:**
   - Read the complete theoretical and viva cheat-sheet explaining the mathematical concepts in clear, intuitive language.

---

## 9. REST API Reference

| Endpoint | Method | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | None | Serves the main single-page web dashboard |
| `/api/dashboard` | `GET` | None | Returns dynamic summary metrics and evaluation statistics |
| `/api/market-data` | `GET` | None | Returns structured chart payloads for Chart.js rendering |
| `/api/trades` | `GET` | `page`, `page_size`, `search`, `symbol`, `status`, `reason`, `sort_by`, `sort_order` | Returns filtered and paginated trade records |
| `/api/trade/<id>` | `GET` | `trade_id` in URL path | Returns detailed diagnostic and market comparative profile for a trade |
| `/api/detect` | `POST` | `mahalanobis_threshold`, `volume_z_threshold`, optional `file` | Re-executes detection pipeline with updated parameters |
| `/api/results` | `GET` | None | Returns top flagged anomalies and summary statistics |
| `/api/upload` | `POST` | Multipart file (`file`) | Validates and ingests a custom CSV dataset |
| `/api/download-results` | `GET` | None | Downloads `results/detected_anomalies.csv` |
| `/api/reset-default-dataset` | `POST` | None | Restores the default 7,500-record synthetic market dataset |

---

## 10. College Viva Presentation Cheat-Sheet

### Q1: What is the core problem statement?
**Answer:** To build a Mahalanobis distance pipeline that flags irregular high-frequency trades and volume spikes in simulated financial markets.

### Q2: Why did you choose Mahalanobis Distance instead of Euclidean Distance?
**Answer:** Euclidean distance assumes all features are independent and have equal variance. In financial trading, features like price, quantity, and volume are strongly correlated and have vastly different units (cents vs thousands of dollars). Mahalanobis distance normalizes variance and incorporates the covariance matrix, creating an elliptical boundary that recognizes true multivariate relationships.

### Q3: How do you handle singular covariance matrices?
**Answer:** If features are collinear, the covariance matrix cannot be inverted with `np.linalg.inv()`. We apply ridge regularization by adding $\epsilon = 10^{-6}$ along the diagonal and compute the Moore-Penrose pseudo-inverse using `numpy.linalg.pinv()`, ensuring mathematical stability without crashes.

### Q4: How is a trade flagged as a potential anomaly?
**Answer:** A trade is flagged as a *Potential Anomaly* if:
1. Its Mahalanobis distance exceeds the configured threshold ($\tau_M > 3.0$), OR
2. Its Volume $Z$-Score exceeds the configured threshold ($\tau_V > 3.0$).  
The system records the exact reason: `Mahalanobis Distance`, `Volume Spike`, or `Both`.

### Q5: Is a flagged trade definitely fraud?
**Answer:** No. An anomaly is merely a statistical deviation from the learned pattern. In real markets, anomalies can occur due to legitimate institutional block rebalancing, sudden economic news, fat-finger errors, or market manipulation. The system flags *candidates* for human surveillance review, not proven fraud.

---

## 11. Limitations & Future Scope

### Current Limitations:
- The model is trained on simulated market data where price changes follow stylized stochastic random walks.
- Stationary covariance is assumed across the trading session; real markets experience intraday volatility clustering (GARCH effects).

### Future Scope:
- Implement rolling-window dynamic covariance estimation (EWMA / Exponentially Weighted Moving Average) to adapt to intraday volatility shifts.
- Integrate Level-2 Limit Order Book (LOB) depth features (bid-ask spread, order book imbalance, cancellation rates).
- Add support for real-time WebSocket trade streaming from live testnet exchanges (e.g., Binance or Interactive Brokers paper trading).

---

## 12. License & Academic Attribution
Developed as an academic semester project for **Pattern Recognition and Anomaly Detection (Semester 5)**. Designed for demonstration and educational purposes.
