import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate some example data
np.random.seed(123)
data = np.random.randn(1000)

# Fit an ARMA(p, q) model to the data
p = 1 # number of autoregressive (AR) lags
q = 1 # number of moving average (MA) lags
model = sm.tsa.AutoReg(data, lags=p, trend='c', old_names=False).fit()

# Print the model summary
print(model.summary())


# Plot the residuals
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(model.resid)
ax.set(title="Residuals of ARMA(p={}, q={}) model".format(p, q), xlabel="Time", ylabel="Residuals")
plt.show()

# Fit an ARMA(p, q) model to the data
p = 2 # number of autoregressive (AR) lags
q = 2 # number of moving average (MA) lags
model = sm.tsa.AutoReg(data, lags=[1, 2], trend='c', old_names=False).fit()

# Print the model summary
print(model.summary())

# Plot the residuals
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(model.resid)
ax.set(title="Residuals of ARMA(p={}, q={}) model".format(p, q), xlabel="Time", ylabel="Residuals")
plt.show()

# Plot the ACF and PACF
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
sm.graphics.tsa.plot_acf(model.resid, lags=20, ax=ax[0])
sm.graphics.tsa.plot_pacf(model.resid, lags=20, ax=ax[1])
plt.show()
