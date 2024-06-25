import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_predict

# Plot 1: Moving Average parameter is -0.7
plt.subplot(2, 1, 1)
AR1 = np.array([1])
MA1 = np.array([1, -0.7)
MA_model1 = ArmaProcess(AR1, MA1)
simulated_data_1 = MA_model1.generate_sample(nsample=5000)
plt.plot(simulated_data_1);

# Plot 2: Moving Average parameter is +0.7
plt.subplot(2, 1, 2)
AR2 = np.array([1])
MA2 = np.array([1, 0.7)
MA_model2 = ArmaProcess(AR2, MA2)
simulated_data_2 = MA_model2.generate_sample(nsample=5000)
plt.plot(simulated_data_2);

AR3 = np.array([1])
MA3 = np.array([1, -0.4)
MA_model3 = ArmaProcess(AR3, MA3)
simulated_data_3 = MA_model3.generate_sample(nsample=5000)
plt.plot(simulated_data_3)

from statsmodels.tsa.arima.model import ARIMA

# Fitting the ARMA(1,1) model to the first simulated data
Arma_model = ARIMA(simulated_data_1, order=(1, 1, 0))
Graph = Arma_model.fit()

# Forecasting ARMA(1,1) model
Graph.predict(900, 920);

# Reading the Dataset using pandas
df = pd.read_csv('dataset_ARMA.csv', header=None)
df = df.loc[:, :1]
print(df)

# Changing the first date to zero
df.iloc[0, 0] = 0

# Changing the column headers to 'DATE' and 'CLOSE'
tup=['DATE', 'CLOSE']
df.columns = ['DATE']
print(df)
# Converting DATE column to numeric
df[[tup[0]]] = pd.to_datetime(df[['DATE']])

# convert datetime to numeric
df[[tup[0]]] = df[[tup[0]]].apply(lambda x: x.timestamp())
               
# Making 'DATE' column the new index
df = df.set_index(df[[tup[0]]])

#Some rows are missing in the dataset 
print("Number of minutes data, without any missing values is 391")
print("Actual length of the DataFrame is:", len(df))

# For Everything
setting_everything = set(range(391))

# The df index as a set
setting_df = set(df.index)

# Calculating the difference
missing_values = setting_everything - setting_df

# Printing the difference
print("Missing rows: ", missing_values)

# Filling in the missing rows
df = df.reindex(range(391), method='ffill')

# Change the index to the df times
df.index = pd.date_range(start='2018-12-26 11:30', end='2018-12-26 18:00', freq='1min')

# Plot the df time series
df.plot(grid=True)
plt.savefig('ARMA.png')

"""**Fitting the model and computing the ACF**"""

from statsmodels.graphics.tsaplots import plot_acf
# Compute returns from prices and drop the NaN
returns = df.pct_change()
returns = returns.dropna()

# Plot ACF of returns with lags up to 60 minutes
plot_acf(returns, lags=60);

# fit the data to an ARMA(1,1) model
arma_model = ARMA(returns, order=(1, 1))
res = arma_model.fit()
print(res.params)
plt.savefig('Autocorrelation.png')
