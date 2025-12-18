import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# 1. Download data
# ---------------------------------------------------

indices = [
    "^AXJO", "^N225", "^HSI", "^GDAXI", "^FCHI",
    "^GSPC", "^DJI", "^IXIC", "SPY"
]

start_date = "2008-01-02"
end_date = "2018-08-21"

data = yf.download(indices, start=start_date, end=end_date)

# Extract each index into its own DataFrame
aord   = data.xs("^AXJO", level=1, axis=1)
nikkei = data.xs("^N225", level=1, axis=1)
hsi    = data.xs("^HSI", level=1, axis=1)
daxi   = data.xs("^GDAXI", level=1, axis=1)
cac40  = data.xs("^FCHI", level=1, axis=1)
sp500  = data.xs("^GSPC", level=1, axis=1)
dji    = data.xs("^DJI", level=1, axis=1)
nasdaq = data.xs("^IXIC", level=1, axis=1)
spy    = data.xs("SPY", level=1, axis=1)

# ---------------------------------------------------
# 2. Build modeling DataFrame
# ---------------------------------------------------

indicepanel = pd.DataFrame(index=spy.index)

# Log returns (no look-ahead bias)
indicepanel["spy"]     = np.log(spy["Open"]).diff()
indicepanel["spy_lag1"]= indicepanel["spy"].shift(1)

indicepanel["sp500"]  = np.log(sp500["Open"]).diff()
indicepanel["nasdaq"] = np.log(nasdaq["Open"]).diff()
indicepanel["dji"]    = np.log(dji["Open"]).diff()
indicepanel["cac40"]  = np.log(cac40["Open"]).diff()
indicepanel["daxi"]   = np.log(daxi["Open"]).diff()

# Markets that close earlier (same-day close-to-open)
indicepanel["aord"]   = np.log(aord["Close"] / aord["Open"])
indicepanel["nikkei"] = np.log(nikkei["Close"] / nikkei["Open"])
indicepanel["hsi"]    = np.log(hsi["Close"] / hsi["Open"])

# Price (not used in model, informational only)
indicepanel["Price"] = spy["Open"]

# Clean NaNs
indicepanel = indicepanel.dropna()

# ---------------------------------------------------
# 3. Train / Test split
# ---------------------------------------------------

split = int(len(indicepanel) * 0.7)

Train = indicepanel.iloc[:split].copy()
Test  = indicepanel.iloc[split:].copy()

print("Train shape:", Train.shape)
print("Test shape:", Test.shape)

# ---------------------------------------------------
# 4. Correlation check
# ---------------------------------------------------

corr = Train.drop(columns=["Price"]).corr()["spy"].sort_values(ascending=False)
print("\nCorrelation with SPY returns:\n")
print(corr)

# ---------------------------------------------------
# 5. Regression Model
# ---------------------------------------------------

formula = """
spy ~ spy_lag1 + sp500 + nasdaq + dji + cac40
     + aord + daxi + nikkei + hsi
"""

lm = smf.ols(formula=formula, data=Train).fit()
print(lm.summary())

# ---------------------------------------------------
# 6. Predictions
# ---------------------------------------------------

Train["PredictedY"] = lm.predict(Train)
Test["PredictedY"]  = lm.predict(Test)

# ---------------------------------------------------
# 7. Model Evaluation
# ---------------------------------------------------

rmse_train = np.sqrt(mean_squared_error(Train["spy"], Train["PredictedY"]))
rmse_test  = np.sqrt(mean_squared_error(Test["spy"], Test["PredictedY"]))

r2_train = r2_score(Train["spy"], Train["PredictedY"])
r2_test  = r2_score(Test["spy"], Test["PredictedY"])

print("\nModel Performance:")
print(f"Train RMSE: {rmse_train:.6f}")
print(f"Test  RMSE: {rmse_test:.6f}")
print(f"Train R²  : {r2_train:.4f}")
print(f"Test  R²  : {r2_test:.4f}")

# ---------------------------------------------------
# 8. Visualization
# ---------------------------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(Test["spy"], Test["PredictedY"], alpha=0.5)
plt.xlabel("Actual SPY Returns")
plt.ylabel("Predicted SPY Returns")
plt.title("Out-of-Sample Predictions")
plt.show()
