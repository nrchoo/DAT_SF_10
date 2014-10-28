import pandas as pd
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as sm

df = pd.read_csv('student_logins.csv')

df['session_end_time'] = pd.to_datetime(df['session_end_time'])
df['session_start_time'] = pd.to_datetime(df['session_start_time'])
df['student_account_created'] = pd.to_datetime(df['student_account_created'])
df['Duration'] = df['session_end_time'] - df['session_start_time']
df['Duration'] = df['Duration'].map(lambda x: x.astype('float64')/(1e9*60))

MaxCreated = pd.to_datetime(max(df['session_end_time']))
df['AccountAge'] = MaxCreated - df['student_account_created']
df['AccountAge'] = df['AccountAge'].map(lambda x: x.astype('float64')/(1e9*60*60*24))

class_id = df['class_id']
class_dummies = pd.get_dummies(class_id)
df1 = pd.merge(df,class_dummies,left_index=True,right_index=True,how='left')
df1 = df1[(df1['Duration'] < 225) & (df1['Duration'] > 0)]

dum = df1[['Duration','a','c','e','g','m']]
scat2 = scatter_matrix(dum, figsize = (10,10))

scat = scatter_matrix(df, figsize = (25,25))
df2 = df1[['problems_completed','AccountAge','student_previous_logins_total','student_previous_class_logins','Duration']]
scat2 = scatter_matrix(df2, figsize = (10,10))

x = df1[['problems_completed','AccountAge','student_previous_logins_total','student_previous_class_logins','a','c','e','g','m']].values
y = df1['Duration'].values
X = sm.add_constant(x, prepend=True)
results = sm.OLS(y, X).fit()
results.summary()


X = sm.add_constant(x, prepend=True)
results = sm.OLS(y, X).fit()
intercept, slope = results.params
r2 = results.rsquared
plt.plot(x, y, 'bo')
plt.title("Duration VS problems_completed")
xl = np.array([min(x), max(x)])
yl = intercept + slope * xl
plt.plot(xl, yl, 'r-')
plt.show
