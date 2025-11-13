# Reproduction with DoubleML of 'Hit or Miss? The Effect of Assassinations on Institutions and War' by Benjamin F. Jones and Benjamin A. Olken"

#The data and code for replication can be accessed at: https://www.openicpsr.org/openicpsr/project/114047/version/V1/view.
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import math
import matplotlib.pyplot as plt
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM, DoubleMLClusterData
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install doubleml

"""# Reading the uncleaned data for consistency"""

df = pd.read_stata('mergeddata.dta')
# for speed, drop later
df = df.drop(df.index[100:])

df.head()

print(list(df.columns))

"""# Reading the cleaned data obtained from Stata

We run the Stata code to clean and retrieve the data according to the authors' routine as outlined in their code. We then load the data in a Pandas dataframe to both reproduce the authors' OLS regressions and DoubleML specifications.
"""

df = pd.read_stata('country_year_data.dta')

df.head(2)

print(list(df.columns)[:30])



# Reproducing the OLS results in Jones and Olken
"""

# All regressions are ran on the subsample with only serious attempts
df = df.drop(df[df.seriousattempt != 1].index)

"""### With clustered standard errors (using StatsModels)"""

# We rename the variables for simplicity
y1 = df['absnpolity2dummy11']
y2 = df['npolity2dummy11']
y3 = df['perexitregularNC201']
y4 = df['npolity2dummy11']
y5 = df['perexitregularNC201']

Xnames = ['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X = df[Xnames]

X1names = ['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X1 = df[X1names]

"""- We first reproduce the results Table 5 using clustered standard errors at the ```cowcode``` level. Note that with StatsModels, the intercept needs to be added manually. We find exactly the same result as in the paper."""

# Regression (1) for Panel A in Table 5 (with clustering)

df1 = df[['absnpolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df1 = df1.dropna()

model = sm.OLS(y1, sm.add_constant(X), missing='drop')
results = model.fit(cov_type='cluster',cov_kwds={'groups':df1['cowcode']})
results.summary()

# Regression (2) for Panel A in Table 5 (with clustering)

df2 = df[['npolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df2 = df2.dropna()

model = sm.OLS(y2, sm.add_constant(X), missing='drop')
results = model.fit(cov_type='cluster',cov_kwds={'groups':df2['cowcode']})
results.summary()

# Regression (3) for Panel A in Table 5 (with clustering)

df3 = df[['perexitregularNC201']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df3 = df3.dropna()

model = sm.OLS(y3, sm.add_constant(X), missing='drop')
results = model.fit(cov_type='cluster',cov_kwds={'groups':df3['cowcode']})
results.summary()

# Regression (2) for Panel B in Table 5 (with clustering)

df4 = df[['npolity2dummy11']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df4 = df4.dropna()

model = sm.OLS(y4, sm.add_constant(X1), missing='drop')
results = model.fit(cov_type='cluster',cov_kwds={'groups':df4['cowcode']})
results.summary()

# Regression (3) for Panel B in Table 5 (with clustering)

df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df5 = df5.dropna()

model = sm.OLS(y5, sm.add_constant(X1), missing='drop')
results = model.fit(cov_type='cluster',cov_kwds={'groups':df5['cowcode']})
results.summary()

"""### Wihtout clustering (using StatsModels)

- In what follows, we recompute all the regressions in Table 5 but do not cluster standard errors at the ```cowcode``` level. In particular, the estimates for $\beta$ remain the same as in the paper but the standard errors naturally change. We do this because we will compute the parameter estimates and standard errors with DoubleML with and without clustering so that everything can be compared properly.
"""

# Regression (1) for Panel A in Table 5 (without clustering)

model = sm.OLS(y1, sm.add_constant(X), missing='drop')
results = model.fit()
results.summary()

# Regression (2) for Panel A in Table 5 (without clustering)

model = sm.OLS(y2, sm.add_constant(X), missing='drop')
results = model.fit()
results.summary()

# Regression (3) for Panel A in Table 5 (without clustering)

model = sm.OLS(y3, sm.add_constant(X), missing='drop')
results = model.fit()
results.summary()

# Regression (2) for Panel B in Table 5 (without clustering)

model = sm.OLS(y4, sm.add_constant(X1), missing='drop')
results = model.fit()
results.summary()

# Regression (3) for Panel B in Table 5 (without clustering)

model = sm.OLS(y5, sm.add_constant(X1), missing='drop')
results = model.fit()
results.summary()

"""### Without clustering (using SKLearn)

- For consistency, we re-run the regressions using SKLearn. Note that there is no direct method to compute standard errors with SKLearn, and so we do not report them. For this reason, we generally recommand using StatsModels for statistical analysis compared to SKLearn.

- Regression (1) for Panel A in Table 5:
"""

df1 = df[['absnpolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df1 = df1.dropna()

y = df1['absnpolity2dummy11']
Xnames = ['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X = df1[Xnames]

reg = LinearRegression().fit(X, y)
reg.coef_[0]

"""- Regression (2) for Panel A in Table 5:"""

df2 = df[['npolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df2 = df2.dropna()

y = df2['npolity2dummy11']
Xnames = ['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X = df2[Xnames]

reg = LinearRegression().fit(X, y)
reg.coef_[0]

"""- Regression (3) for Panel A in Table 5:"""

df3 = df[['perexitregularNC201']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df3 = df3.dropna()

y = df3['perexitregularNC201']
Xnames = ['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X = df3[Xnames]

reg = LinearRegression().fit(X, y)
reg.coef_[0]

"""- Regression (2) for Panel B in Table 5:"""

df4 = df[['npolity2dummy11']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df4 = df4.dropna()

y = df4['npolity2dummy11']
Xnames = ['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X = df4[Xnames]

reg = LinearRegression().fit(X, y)
reg.coef_[0:2]

"""- Regression (3) for Panel B in Table 5:"""

df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df5 = df5.dropna()

y = df5['perexitregularNC201']
Xnames = ['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]
X = df5[Xnames]

reg = LinearRegression().fit(X, y)
reg.coef_[0:2]

# Reproducing the results in Jones and Olken with DoubleML


ml_l_bonus = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth= 5)
ml_m_bonus = RandomForestClassifier(max_depth= 5, n_estimators=150)

# Regression (1) for Panel A in Table 5 (DoubleML)
#df1.drop('cowcode')
dml_data = DoubleMLData(df1,'absnpolity2dummy11','success')

# let us use median estimation with 100 repetitions, following Chernozhukov et al
theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Regression (2) for Panel A in Table 5 (DoubleML)

dml_data = DoubleMLData(df2,'npolity2dummy11','success')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Regression (3) for Panel A in Table 5 (DoubleML)

dml_data = DoubleMLData(df3,'perexitregularNC201','success')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Part 1 of Regression (2) for Panel B in Table 5 (DoubleML)

dml_data = DoubleMLData(df4,'npolity2dummy11','successlautoc')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Part 2 of Regression (2) for Panel B in Table 5 (DoubleML)

dml_data = DoubleMLData(df4,'npolity2dummy11','successldemoc')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Part 1 of Regression (3) for Panel B in Table 5 (DoubleML)

dml_data = DoubleMLData(df5,'perexitregularNC201','successlautoc')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Part 2 of Regression (3) for Panel B in Table 5 (DoubleML)

dml_data = DoubleMLData(df5,'perexitregularNC201','successldemoc')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_data, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

"""- We observe that the results (parameter estimates and standard errors) obtained with DoubleML in a PLR model are quite similar to the one obtained with a simple OLS regression in a linear model. As already mentionned, this was expected given that: 1. the controls are low-dimensional; 2. the OLS results are almost invariant by the inclusion or exclusion of them; 3. the ML methods should work with constant and linear functions (as supposed from 2.).

# Clustering standard errors with DoubleML
"""

df1 = df[['absnpolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df1 = df1.dropna()
dml_d = DoubleMLClusterData(df1,'absnpolity2dummy11','success', cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Panel A regression (2)
df2 = df[['npolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df2 = df2.dropna()
dml_d = DoubleMLClusterData(df2,'npolity2dummy11','success', cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Panel A regression (3)

df3 = df[['perexitregularNC201']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df3 = df3.dropna()

dml_d = DoubleMLClusterData(df3,'perexitregularNC201','success', cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Panel B regression (1) part 1
df4 = df[['npolity2dummy11']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df4 = df4.dropna()

dml_d = DoubleMLClusterData(df4,'npolity2dummy11','successlautoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# panel B regression (1) part 2

dml_d = DoubleMLClusterData(df4,'npolity2dummy11','successldemoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Panel B regression (2) part 1
df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df5 = df5.dropna()

dml_d = DoubleMLClusterData(df5,'perexitregularNC201','successlautoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

# Panel B regression (2) part 2
df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df5 = df5.dropna()

dml_d = DoubleMLClusterData(df5,'perexitregularNC201','successldemoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLPLR(dml_d, ml_l_bonus, ml_m_bonus)
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

"""# Robustness Check"""

# let us run probit regression following the authors
d_pr = df.drop(df[df.seriousattempt != 1].index)
d_pr = d_pr[['success','age', 'year', 'country',
             'lpol2dum', 'pol2duml1l3', 'lzGledAnywar', 'anywarl1l3', 'llnenergy_pc', 'llnpop', 'lage', 'lclock',
          'regdumAfrica', 'regdumAsia', 'regdumMENA', 'regdumLatAm', 'regdumEEur', 'cowcode', 'weapondum2',
          'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6']]
d_pr = d_pr.dropna()

y = d_pr['success']
X = d_pr[['lpol2dum', 'pol2duml1l3', 'lzGledAnywar', 'anywarl1l3', 'llnenergy_pc', 'llnpop', 'lage', 'lclock',
          'regdumAfrica', 'regdumAsia', 'regdumMENA', 'regdumLatAm', 'regdumEEur', 'weapondum2',
          'weapondum3', 'weapondum5', 'weapondum6']]

probit_model=sm.Probit(y,X)
result=probit_model.fit(cov_type='cluster', cov_kwds={'groups': d_pr['cowcode']})
print(result.summary2())

# finding the classifier that gives higher ROC AUC
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=seed)

light = LGBMClassifier(learning_rate = 0.01255, num_leaves=3, n_estimators=80,
                      class_weight = {0: 1.0, 1: 3.32}).fit(X_train,y_train)
y_l_pred = light.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_l_pred))
print("ROC AUC", roc_auc_score(y_test, light.predict_proba(X_test)[:, 1]))

# let us use the classifier with higher roc auc as a function to estimate nuisance parameters
ml_m_1 = LGBMClassifier(learning_rate = 0.01255, num_leaves=3, n_estimators=80,
                      class_weight = {0: 1.0, 1: 3.32}).fit(X_train,y_train)
ml_l_1 = RandomForestRegressor(n_estimators = 150, max_features = 'sqrt', max_depth= 5)

# here we run the same DML procedure for Table 5 from paper but with a new classifier
df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
df1 = df[['absnpolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df1 = df1.dropna()

dml_d = DoubleMLClusterData(df1,'absnpolity2dummy11','success', cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df2 = df[['npolity2dummy11']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df2 = df2.dropna()
dml_d = DoubleMLClusterData(df2,'npolity2dummy11','success', cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df3 = df[['perexitregularNC201']+['success']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df3 = df3.dropna()

dml_d = DoubleMLClusterData(df3,'perexitregularNC201','success', cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df4 = df[['npolity2dummy11']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df4 = df4.dropna()

dml_d = DoubleMLClusterData(df4,'npolity2dummy11','successlautoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

dml_d = DoubleMLClusterData(df4,'npolity2dummy11','successldemoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df5 = df5.dropna()

dml_d = DoubleMLClusterData(df5,'perexitregularNC201','successlautoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]+['cowcode']]
df5 = df5.dropna()

dml_d = DoubleMLClusterData(df5,'perexitregularNC201','successldemoc',cluster_cols = 'cowcode')

theta_est = []
se_est = []
for i in range(1,100):
  obj_dml_plr = DoubleMLIRM(dml_d, ml_l_1, ml_m_1, score = 'ATE', dml_procedure='dml1')
  obj_dml_plr.fit()
  theta_est.append(obj_dml_plr.coef)
  se_est.append(obj_dml_plr.se)

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

"""## Causal forest

"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install econml

from econml.dml import CausalForestDML
from sklearn.linear_model import MultiTaskLassoCV

# for robustness check we run Casual Forest model to estimate coefficients from Table 5 from the paper
treatment = 'success'
outcome = 'absnpolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df1, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.4, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'success'
outcome = 'npolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df2, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'success'
outcome = 'perexitregularNC201'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df3, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=120, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'successlautoc'
outcome = 'npolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df4, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=120, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'successldemoc'
outcome = 'npolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df4, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=120, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'successlautoc'
outcome = 'perexitregularNC201'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df5, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'successldemoc'
outcome = 'perexitregularNC201'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df5, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=5,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

"""**With twofolds**"""

# sence we have a few observations, we rerun casual forest model with twofolds cross-validation to compare the results
treatment = 'success'
outcome = 'absnpolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df1, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'success'
outcome = 'npolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df2, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.4, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'success'
outcome = 'perexitregularNC201'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df3, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=120, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df4 = df[['npolity2dummy11']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df4 = df4.dropna()
df4 = df4.astype(np.float64)

treatment = 'successlautoc'
outcome = 'npolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df4, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=120, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.42, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'successldemoc'
outcome = 'npolity2dummy11'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df4, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=120, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

df5 = df[['perexitregularNC201']+['successlautoc']+['successldemoc']+['lautoc']+[x for x in df.columns if x.startswith("weapondum")]+[x for x in df.columns if x.startswith("numserdum")]]
df5 = df5.dropna()
df5 = df5.astype(np.float64)

treatment = 'successlautoc'
outcome = 'perexitregularNC201'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df5, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)

treatment = 'successldemoc'
outcome = 'perexitregularNC201'
covariates = ['weapondum2', 'weapondum3', 'weapondum4', 'weapondum5', 'weapondum6', 'numserdum2', 'numserdum3', 'numserdum4']
train, test = train_test_split(df5, test_size = 0.25)
Y_cf, T_cf, X_cf, W_cf, X_cf_test = train[outcome], train[treatment], train[covariates], None, test[covariates]

theta_est = []
se_est = []
for i in range(1,100):
  causal_forest = CausalForestDML(criterion='het', n_estimators=100, min_samples_leaf = 5, max_depth=5,
                                max_samples=0.5, discrete_treatment=True, honest=True, inference=True,
                                cv=2,)
  causal_forest.fit(Y_cf, T_cf, X=X_cf, W=W_cf)
  theta_est.append(causal_forest.ate(X_cf_test))
  se_est.append(causal_forest.ate_stderr_[0])

theta_median = np.median(theta_est)
se_median = np.sqrt(np.median(np.array(se_est)*np.array(se_est) + (np.median(theta_est) - np.array(theta_est))*(np.median(theta_est) - np.array(theta_est))))

print("Estimated median theta: ", theta_median)
print("Estimated median standard error: ", se_median)
