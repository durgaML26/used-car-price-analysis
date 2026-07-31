# What Drives the Price of a Used Car?

**UC Berkeley Professional Certificate in Machine Learning & AI — Practical Application 11.1**

📓 [View the full analysis notebook](./used_car_price_analysis.ipynb)

---

## The Business Question

A used car dealership wants to know **what consumers actually value in a used car**, so they can fine-tune which vehicles they stock and how they price them.

I analyzed **426,880 used vehicle listings** to identify which characteristics drive price, and by how much.

> **Note on the dataset file:** The dataset (`vehicles.csv`, ~50 MB) is **not included in this
> repository** because it exceeds GitHub's 25 MB web-upload limit. It is the used-car dataset
> provided with the assignment (originally from Kaggle). To reproduce the notebook, place
> `vehicles.csv` in a local `data/` folder. All notebook outputs are already rendered, so the
> results are fully visible without re-running it.

---

## Headline Findings

### 1. Age and mileage are the universal price drivers

| Factor | Effect on price |
|---|---|
| Each additional **year of age** | **− $797** |
| Each additional **10,000 miles** | **− $840** |

These are the two most reliable levers. A 5-year-old car with 50,000 miles carries roughly **$8,200** less value than an otherwise identical new one.

### 2. Trucks and utility vehicles hold value best

Compared to a standard SUV:

| Body type | Price difference |
|---|---|
| Off-road | **+ $4,492** |
| Pickup | **+ $4,225** |
| Truck | **+ $4,007** |
| Convertible | **+ $3,962** |

**Diesel vehicles command the single largest fuel premium** — gas vehicles sell for roughly **$14,750 less** than comparable diesels. Since diesel is concentrated in heavy-duty trucks, this reinforces the same story: **work and utility vehicles retain value.**

### 3. Engine size tracks price directly

Compared to a 10-cylinder baseline, a 4-cylinder sells for **$9,087 less** and a 3-cylinder for **$11,453 less**. Larger engines consistently command higher prices.

### 4. Title status is a major, controllable factor

A **parts-only title reduces value by $7,170** versus a clean title. Title condition is one of the clearest quality signals in the market.

### 5. Brand premiums are real but concentrated

Compared to Acura, exotic and premium brands carry large premiums (Ferrari **+$53,185**, Porsche **+$9,814**, Aston Martin **+$9,495**), while budget brands sell below (Fiat **−$10,251**, Mitsubishi **−$8,115**, Kia **−$6,477**).

---

## Recommendations for the Dealership

1. **Prioritize trucks, pickups, and off-road vehicles.** They command a $4,000+ premium over SUVs and hold value better.
2. **Favor diesel inventory where practical** — the diesel premium is the largest single fuel effect in the data.
3. **Treat clean titles as non-negotiable.** The gap between clean and parts-only titles ($7,170) usually exceeds any acquisition discount.
4. **Price age and mileage explicitly.** Use ≈ $800/year and ≈ $840 per 10,000 miles as baseline depreciation when valuing trade-ins.
5. **Be cautious with budget brands.** Fiat, Mitsubishi, and Kia carry measurable price discounts.

---

## How I Got There

**Process:** CRISP-DM (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment)

**Data preparation** (426,880 → 364,758 records, 86.8% retained):

| Decision | Rationale |
|---|---|
| Dropped `id`, `VIN` | Identifiers — no predictive information |
| Dropped `model` (29,649 categories) | Encoding would create 29,649 columns; `manufacturer` captures brand |
| Dropped `size` (72% missing) | Too sparse to impute responsibly |
| Dropped `region` (404 categories) | Geographically redundant with `state` |
| Price limited to $500–$100,000 | Removed 32,895 $0 listings and 53 records above $1M |
| Odometer capped at 300,000 miles | Matches the 99th percentile (280,000) and IQR outlier fence (277,300) |
| Restricted to model year ≥ 1980 | Pre-1975 vehicles show *rising* prices with age — a collector-car market with different pricing dynamics |
| Engineered `age` from `year` | Makes the model intercept meaningful (price of a new car) |

**Models compared:** Linear Regression, Ridge, and Lasso, each evaluated with 5-fold cross-validation and hyperparameters tuned via grid search.

---

## Model Performance

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear Regression | 0.675 | $8,092 |
| Ridge (α = 10, grid-searched) | 0.675 | $8,092 |
| Lasso | 0.675 | $8,091 |

The model explains **67.5% of price variation**, with typical prediction error of about **$8,100**.

**Regularization made no difference** — expected, since ~292,000 training records against 150 features leaves no room for overfitting. Training R² (0.681) and test R² (0.675) are nearly identical, confirming the model generalizes well but is **underfitting**: too simple to capture the full relationship rather than over-tuned to the data.

---

## Limitations

- **Prediction error of ~$8,100 is too coarse for pricing individual vehicles.** Use these findings for inventory strategy, not per-car quotes.
- **`model` was excluded** for dimensionality reasons, so trim-level differences (a Camry LE vs. XSE) are not captured — likely a significant share of unexplained variance.
- **Correlation is not causation.** White vehicles are associated with higher prices, but this reflects white being standard for high-value work trucks — repainting a car will not increase its value.
- **Some coefficients are confounded.** Every Tesla is electric, so `manufacturer_tesla` (+$15,189) and `fuel_electric` (−$17,956) cannot be separated; neither should be read on its own.
- **Rare-brand estimates are unstable.** Ferrari and Datsun coefficients rest on very small samples.
- **Missing factors:** accident history, service records, trim, options, and local demand are absent from the data.

---

## Next Steps

1. **Model nonlinearity.** Depreciation is a curve, not a line — polynomial features or tree-based models (Random Forest, Gradient Boosting) should capture it better.
2. **Capture interactions.** Mileage likely affects a Honda differently than a Ferrari; linear models cannot represent this.
3. **Reintroduce `model` selectively** — top-N models plus an "other" bucket would recover trim-level signal without a dimensionality explosion.
4. **Enrich the data** with accident history and service records, the most likely sources of the missing 32.5% of variance.

*A log-transformed target was also tested to address nonlinear depreciation but performed slightly worse (R² 0.669 vs 0.675), suggesting residual error stems more from unmodeled interactions than from multiplicative price effects.*

---

## Repository Contents

```
├── used_car_price_analysis.ipynb   Full analysis notebook (CRISP-DM structure)
└── README.md      This summary
```
*(The `vehicles.csv` dataset is not committed — see the note above. The notebook expects it in
a local `data/` folder.)*

**Tools:** Python · pandas · scikit-learn · seaborn · matplotlib
