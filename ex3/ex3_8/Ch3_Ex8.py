"""ISLR- Python: Ch3 -- Applied Question 8
   excerpted from Mr Matt Caudill
   @ user: Yan
"""


# perform standard imports
import statsmodels.api as sm
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import OLSInfluence

# %matplotlib inline
plt.style.use('ggplot') # emulate pretty r-style plots


# load data
auto_df = pd.read_csv('Auto.csv',na_values='?')
auto_df = auto_df.dropna() # drop rows with na values
print(len(auto_df.mpg),'rows')
auto_df.head()


# Construct Linear Estimate
# construct desgin matrix, model and fit
X = sm.add_constant(auto_df.horsepower)
y = auto_df.mpg
model = sm.OLS(y,X)
estimate = model.fit()
print(estimate.summary())


# Plot Data and Estimate
# plot the data and the estimate
fig,ax = plt.subplots(figsize=(8,6))
# scatter data
ax.scatter(X.horsepower.values,y.values, facecolors='none', edgecolors='b', label="data")
# plot estimate
ax.plot(X.horsepower.values, estimate.fittedvalues, 'g', label="OLS")
ax.legend(loc='best');


# Perform Estimate Diagnostics
#Plot the residuals, studentized residuals and the leverages
# Obtain the residuals, studentized residuals and the leverages
fitted_values = estimate.fittedvalues
residuals = estimate.resid.values
studentized_residuals = OLSInfluence(estimate).resid_studentized_internal
leverages = OLSInfluence(estimate).influence

# Plot
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,4))

# Residuals
ax1.scatter(fitted_values, residuals, facecolors='none', edgecolors='b');
ax1.set_xlabel('fitted values');
ax1.set_ylabel('residuals');
# Studentized Residuals
ax2.scatter(fitted_values, studentized_residuals, facecolors='none', edgecolors='b');
ax2.set_xlabel('fitted values');
ax2.set_ylabel('studentized residuals');
# Leverages
ax3.scatter(leverages, studentized_residuals, facecolors='none', edgecolors='b');
ax3.set_xlabel('Leverage');
ax3.set_ylabel('studentized residual');