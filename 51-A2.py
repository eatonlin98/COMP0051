#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

start_sp = datetime.datetime(2014, 1, 1)
end_sp = datetime.datetime(2020, 1, 1)


SPY = yf.download('SPY',start_sp,end_sp)
SPY


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from full_fred.fred import Fred
import os

os.environ['FRED_API_KEY'] = 'c84279a32142d5e22091c05e294bc665'

fred = Fred()
fred.env_api_key_found()

effr = fred.get_series_df('EFFR')
effr.tail()


# In[3]:


effr['date'] = pd.to_datetime(effr['date'])


# In[4]:


effr['date']=effr["date"].values.astype('datetime64')  #如果已为日期格式则此步骤可省略
import datetime
s_date = datetime.datetime.strptime('2014-01-01', '%Y-%m-%d').date()  #起始日期
e_date = datetime.datetime.strptime('2019-12-31', '%Y-%m-%d').date()  #结束日期


# In[5]:


effr_filter=effr[(effr.date>=pd.Timestamp(s_date))&(effr.date<=pd.Timestamp(e_date))]
len(effr_filter)


# In[6]:


df=pd.merge(SPY,effr_filter, left_on = 'Date', right_on = 'date', how = 'inner')
type(df['value'])
df['value_1'] = pd.to_numeric(df['value'],errors='coerce')


# In[7]:


df['daily_return'] = df['value_1'].multiply((1/252)/100)


# In[8]:


pd.set_option('display.max_rows', None)


# In[9]:


df


# In[10]:


#len(df_clear)


# In[11]:


df=df.drop(df[df['value']=='.'].index)


# In[12]:


df['Close_change_rate'] =((df['Close']  - df['Close'] .shift(1))/df['Close'] .shift(1)).dropna()


# In[13]:


df['excess return']=df['Close_change_rate']-df['daily_return'].shift(1)


# In[14]:


df


# In[15]:


plt.plot(df['date'],df['excess return'],color='red',label = "Excess return")
#df['excess return'].plot(color='red',label = "Excess return")
plt.tight_layout()
plt.grid()

plt.plot(df['date'],df['Close_change_rate'],color='green',label = "SPY return")
#df['Close_change_rate'].plot(color='green',label = "SPY return")
plt.tight_layout()
plt.title('Plot of three properties')
plt.grid()

plt.plot(df['date'],df['daily_return'],color='orange',label = "EFFR")
plt.legend()
plt.show()


# In[ ]:





# In[16]:


df['Close_change_rate'].plot()
plt.tight_layout()
plt.title('SPY return')
plt.grid()
plt.show()


# In[138]:


df['value_1'].plot()
plt.tight_layout()
plt.title('EFFR')
plt.grid()
plt.show()


# In[140]:


df['daily_return'].plot()
plt.tight_layout()
plt.title('EFFR_daily return')
plt.grid()
plt.show()


# In[144]:


plt.plot(df['date'],df['excess return'],color='red',label = "Excess return")
#df['excess return'].plot(color='red',label = "Excess return")
plt.tight_layout()
plt.title('Excess return')
plt.grid()
plt.legend()
plt.show()


# In[145]:



plt.plot(df['date'],df['Close_change_rate'],color='green',label = "SPY return")
#df['Close_change_rate'].plot(color='green',label = "SPY return")
plt.tight_layout()
plt.title('SPY return')
plt.grid()
plt.legend()
plt.show()


# In[18]:


train, test = df['Close'][0:len(df['Close'])-449], df['Close'][len(df['Close'])-449:]


# In[19]:


len(train)


# # trending 

# In[20]:


import pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

import statsmodels


close = train


T = 10
#mu = 0.01
#sigma = 0.02
#S0 = 20
dt = 0.006
N = len(close)
t = np.linspace(0, T, N)

F=df['daily_return']
S = close.values


plt.plot(t, S)
plt.show()


#TRADING: TREND FOLLOWING 
    
#time_window = 3*int(1/dt)
time_window = 30
cumsum = [0]

ma = np.zeros(np.shape(S))

w = np.zeros(np.shape(S))
cash = np.zeros(np.shape(S))

cash[0] = 200000*5

v= np.zeros(np.shape(S))
v[0]=200000
r=df['excess return'].values
rf=df['daily_return'].values
interst=np.zeros(np.shape(S))

upperlimit = np.zeros(np.shape(train))
lowerlimit = np.zeros(np.shape(train))

for i, x in enumerate(S[:-1], 0):
    cumsum.append(cumsum[i] + x)
    ma[i] = x
    if i>=time_window:
        moving_ave = (cumsum[i] - cumsum[i-time_window])/(time_window)
        ma[i] = moving_ave
        
    upperlimit[i]=5*v[i]
    lowerlimit[i]=-5*v[i]
    
    if x*w[i] <= upperlimit[i] and x*w[i] >= lowerlimit[i]:
        
        if ma[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
    
        if ma[i] < x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
        
        if ma[i] > x:
            cash[i+1] = 0.5*w[i]*x + cash[i]
            w[i+1] = 0.5*w[i]
            
    
    else:
        w[i+1]=w[i]-(x*w[i] - 5*v[i])/x
        
        
    v[i+1]=v[i]+w[i+1]*S[i]*r[i+1]+ cash[i]*(rf[i+1])
    
ma[i+1] = S[len(S)-1]
 

tf_strategy = [a*b for a,b in zip(w,S)]+ cash


# In[21]:


print(v[100:])


# In[22]:


plt.plot(t, tf_strategy)


# In[23]:


plt.plot(cash)


# In[24]:


plt.plot(t, tf_strategy)
plt.plot(t,upperlimit,color='red')
plt.plot(t,lowerlimit,color='red')


# In[25]:


w


# In[26]:


S


# ### turn over dollar

# In[27]:


tf_strategy


# In[28]:


delta_theta_tf=np.diff(tf_strategy)
delta_theta_tf


# In[29]:


sum(abs(delta_theta_tf))


# ### turn over units

# In[30]:


len(tf_strategy)


# In[31]:


turnover_units=[]
for i in range(1048):
    turnover_units.append(tf_strategy[i+1]/S[i+1] -tf_strategy[i]/S[i])
sum(np.abs(turnover_units))


# ### TF: ∆Vt, ∆Vt cap, ∆Vt total

# In[32]:


delta_v_tf_total=np.diff(v)
plt.plot(delta_v_tf_total)


# In[33]:


delta_v_tf=[]
for i in range(1049):
    delta_v_tf.append(w[i+1]*S[i]*r[i+1])


# In[34]:


delta_v_tf_cap=[]
for i in range(1049):
    delta_v_tf_cap.append(cash[i]*(rf[i+1])/5)
plt.plot(delta_v_tf_cap)


# In[ ]:





# # mean reverting

# In[35]:


import pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

import statsmodels


# close = train
# T = 10
# dt = 0.006
# N = len(close)
# t = np.linspace(0, T, N)

# F=df['daily_return']
# #S = close.values


plt.plot(t, S)
plt.show()


#TRADING: TREND FOLLOWING 
    
#time_window = 3*int(1/dt)
time_window = 30
cumsum = [0]

ma = np.zeros(np.shape(S))

w = np.zeros(np.shape(S))
cash = np.zeros(np.shape(S))

cash[0] = 200000*5

v_mr= np.zeros(np.shape(S))
v_mr[0]=200000
r=df['excess return'].values
rf=df['daily_return'].values
interst=np.zeros(np.shape(S))

upperlimit = np.zeros(np.shape(train))
lowerlimit = np.zeros(np.shape(train))

for i, x in enumerate(S[:-1], 0):
    cumsum.append(cumsum[i] + x)
    ma[i] = x
    if i>=time_window:
        moving_ave = (cumsum[i] - cumsum[i-time_window])/(time_window)
        ma[i] = moving_ave
    upperlimit[i]=5*v_mr[i]
    lowerlimit[i]=-5*v_mr[i]
    if x*w[i] <= upperlimit[i] and x*w[i] >= lowerlimit[i]:
        
        if ma[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
    
        if ma[i] > x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
        
        if ma[i] < x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0
    else:
        w[i+1]=w[i]-(x*w[i] - 5*v_mr[i])/x
        
        
    v_mr[i+1]=v_mr[i]+w[i+1]*S[i]*r[i+1]+ cash[i]*(rf[i+1])
    
ma[i+1] = S[len(S)-1]
 

mr_strategy = [a*b for a,b in zip(w,S)]+ cash


# In[36]:


plt.plot(t, mr_strategy)
plt.plot(t,upperlimit,color='red')
plt.plot(t,lowerlimit,color='red')


# In[37]:


print(v_mr)


# In[38]:


delta_theta_mr=np.diff(mr_strategy)
delta_theta_mr


# In[39]:


mr_strategy


# In[40]:


sum(abs(delta_theta_mr))


# In[41]:


turnover_units_mr=[]
for i in range(1049):
    turnover_units_mr.append(mr_strategy[i+1]/S[i+1] -mr_strategy[i]/S[i])
sum(np.abs(turnover_units_mr))


# ### MR: ∆Vt, ∆Vt cap, ∆Vt total

# In[42]:


delta_v_mr_total=np.diff(v_mr)
plt.plot(delta_v_mr_total)


# In[43]:


delta_v_mr=[0]
for i in range(1049):
    delta_v_mr.append(w[i+1]*S[i]*r[i+1])
plt.plot(delta_v_mr)


# In[ ]:





# In[44]:


delta_v_mr_cap=[0]
for i in range(1049):
    delta_v_mr_cap.append(cash[i]*(rf[i+1])/5)
#delta_v_ar
plt.plot(delta_v_mr_cap)


# # AR

# In[45]:


from statsmodels.tsa.ar_model import AutoReg


# In[46]:


import pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

import statsmodels

t=np.linspace(0,len(S),len(S))
plt.plot(t, S)
plt.show()


#TRADING: TREND FOLLOWING 

time_window = 30
cumsum = [0]

w = np.zeros(np.shape(S))
cash = np.zeros(np.shape(S))

cash[0] = 200000*5

v_ar= np.zeros(np.shape(S))
v_ar[0]=200000
r=df['excess return'].values
rf=df['daily_return'].values
interst=np.zeros(np.shape(S))

upperlimit = np.zeros(np.shape(S))
lowerlimit = np.zeros(np.shape(S))


ar_prediction = np.zeros(np.shape(S))


for i, x in enumerate(S[:-1], 0):
    cumsum.append(cumsum[i] + x)
    ar_prediction[i] = x
    if i>=time_window:
        X = S[0:i]
        train = X
        # train autoregression
        model = AutoReg(train,lags=1)
        model_fit = model.fit()
        predictions = model_fit.predict(start=len(train), end=len(train), dynamic=False)
        ar_prediction[i] = predictions
        
    upperlimit[i]=5*v_ar[i]
    lowerlimit[i]=-5*v_ar[i]
    if x*w[i] <= upperlimit[i] and x*w[i] >= lowerlimit[i]:
    
        if ar_prediction[i] == x:
            w[i+1] = w[i]
            cash[i+1] = cash[i]
    
        if ar_prediction[i] > x: 
            w[i+1] = cash[i]/x  + w[i]
            cash[i+1] = 0
        
        if ar_prediction[i] < x:
            cash[i+1] = w[i]*x + cash[i]
            w[i+1] = 0
    else:
        w[i+1]=w[i]-(x*w[i] - 5*v_ar[i])/x
        
    v_ar[i+1]=v_ar[i]+w[i+1]*S[i]*r[i+1]+ cash[i]*(rf[i+1])/5
ma[i+1] = S[len(S)-1]

ar_strategy = [a*b for a,b in zip(w,S)]+ cash


#plt.plot(t, tf_strategy)
plt.plot(t, cash[0]*S/S[0])
#plt.plot(t, mr_strategy)
plt.plot(t, ar_strategy)

plt.plot(t,upperlimit,color='red')
plt.plot(t,lowerlimit,color='red')

plt.show()


# In[47]:


print(v_ar)


# In[48]:


ar_strategy


# ### turn over dollar & turn over unit

# In[49]:


delta_theta_ar=np.diff(ar_strategy)
delta_theta_ar


# In[50]:


sum(abs(delta_theta_ar))


# In[51]:


turnover_units_ar=[]
for i in range(104):
    turnover_units_ar.append(ar_strategy[i+1]/S[i+1] -ar_strategy[i]/S[i])
sum(np.abs(turnover_units_ar))


# ### AR: ∆Vt, ∆Vt cap, ∆Vt total

# In[52]:


delta_v_ar_total=np.diff(v_ar)


# In[53]:


plt.plot(delta_v_ar_total)


# In[54]:


delta_v_ar=[0]
for i in range(1049):
    delta_v_ar.append(w[i+1]*S[i]*r[i+1])


# In[55]:


#delta_v_ar
plt.plot(delta_v_ar)


# In[56]:


delta_v_ar_cap=[0]
for i in range(1049):
    delta_v_ar_cap.append(cash[i]*(rf[i+1])/5)
#delta_v_ar
plt.plot(delta_v_ar_cap)


# In[ ]:





# # Q3

# ## TF

# In[57]:


#Sharpe ratio


# In[58]:


pnl_tf=delta_v_tf


# In[59]:


var_tf=np.var(pnl_tf)
sd_tf=var_tf**(1/2)


# In[60]:


mean_tf=np.mean(pnl_tf)


# In[61]:


print(sd_tf)
print(mean_tf)


# In[62]:


mean_tf/sd_tf


# In[63]:


#sortino


# In[64]:


positive_numbers_tf = [num for num in pnl_tf if num < 0]


# In[65]:


mean_tf_1=np.mean(positive_numbers_tf)
var_tf_1=np.var(positive_numbers_tf)
sd_tf_1=var_tf_1**(1/2)


# In[66]:


mean_tf/sd_tf_1


# In[67]:


#maximum drawdown


# In[68]:


len(v)


# In[69]:


#print(delta_v_tf)


# In[70]:


delta_v_tf_neg = [min(num,0) for num in pnl_tf ]


# In[71]:


dd_tf_pos=[abs(num) for num in delta_v_tf_neg]


# In[72]:


dd_tf=[]
for i in np.arange(1,1049):
    dd_tf.append(dd_tf_pos[i]/v[i-1])


# In[73]:


max(dd_tf)


# In[74]:


min(pnl_tf)


# In[75]:


min(delta_v_tf_neg)


# In[76]:


max(dd_tf_pos)


# In[77]:


# window = 30

# # Calculate the max drawdown in the past window days for each day in the series.
# # Use min_periods=1 if you want to let the first 252 days data have an expanding window
# Roll_Max = fsl.rolling(window, min_periods=1).max()
# Daily_Drawdown = fsl/Roll_Max - 1.0

# # Next we calculate the minimum (negative) daily drawdown in that window.
# # Again, use min_periods=1 if you want to allow the expanding window
# Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

# # Plot the results
# Daily_Drawdown.plot()
# Max_Daily_Drawdown.plot()
# plt.show()


# In[78]:


fsl = pd.DataFrame(pnl_tf)
fsl=(1+fsl).cumprod()-1


# In[79]:


fsl_v = pd.DataFrame(v)


# In[80]:


plt.plot(pnl_tf)


# In[81]:


plt.plot(v[300:500])


# In[ ]:





# In[82]:


previous_peaks = fsl.cummax()


# In[83]:


previous_peaks.plot.line()


# In[84]:


drawdown = (fsl - previous_peaks)/previous_peaks


# In[85]:


drawdown.plot.line()


# In[86]:


drawdown.min()


# In[87]:


drawdown.idxmin()


# In[88]:


plt.plot(v)


# In[89]:


def MDD(returns):
    cum_rets = (1 + returns).cumprod() - 1
    nav = ((1 + cum_rets) * 100).fillna(100)
    nav = pd.Series([100]).append(nav) # start at 100
    hwm = nav.cummax()
    dd = nav / hwm - 1
    return min(dd)


# In[90]:


fsl = pd.DataFrame(pnl_tf)
fsl=pd.to_numeric(pnl_tf,errors='coerce')


# In[91]:


fsl=pd.DataFrame(fsl)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## MR

# In[92]:


#sharpe ratio


# In[93]:


pnl_mr=delta_v_mr
var_mr=np.var(pnl_mr)
sd_mr=var_mr**(1/2)
mean_mr=np.mean(pnl_mr)


# In[94]:


mean_mr/sd_mr


# In[95]:


#sortino


# In[96]:


positive_numbers_mr = [num for num in pnl_mr if num < 0]


# In[97]:


mean_mr_1=np.mean(positive_numbers_mr)
var_mr_1=np.var(positive_numbers_mr)
sd_mr_1=var_mr_1**(1/2)


# In[98]:


mean_mr/sd_mr_1


# In[ ]:





# In[ ]:





# ## AR

# In[99]:


#sharpe ratio


# In[100]:


pnl_ar=delta_v_ar
var_ar=np.var(pnl_ar)
sd_ar=var_ar**(1/2)
mean_ar=np.mean(pnl_ar)


# In[101]:


mean_ar/sd_ar


# In[102]:


#sortino


# In[103]:


positive_numbers_ar = [num for num in pnl_ar if num < 0]


# In[104]:


var_ar_1=np.var(positive_numbers_ar)
sd_ar_1=var_ar_1**(1/2)


# In[105]:


mean_ar/sd_ar_1


# In[ ]:





# In[106]:


len(lxy)


# ## 3b

# In[108]:


import pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

import statsmodels


close_test = test


T = 10
# #mu = 0.01
# #sigma = 0.02
# #S0 = 20
# dt = 0.006
N = len(close_test)
t = np.linspace(0, T, N)

F=df['daily_return']
S_test = close_test.values


plt.plot(t, S_test)
plt.show()


#TRADING: TREND FOLLOWING 
    
#time_window = 3*int(1/dt)
time_window = 30
cumsum = [0]

ma_test = np.zeros(np.shape(S_test))

w_test = np.zeros(np.shape(S_test))
cash_test = np.zeros(np.shape(S_test))

cash_test[0] = 200000*5

v_test= np.zeros(np.shape(S_test))
v_test[0]=200000
r_test=df['excess return'].values
rf_test=df['daily_return'].values
rf_test=rf_test[-449:]
r_test=r_test[-449:]

interst=np.zeros(np.shape(S_test))

upperlimit = np.zeros(np.shape(test))
lowerlimit = np.zeros(np.shape(test))

for i, x in enumerate(S_test[:-1], 0):
    cumsum.append(cumsum[i] + x)
    ma_test[i] = x
    if i>=time_window:
        moving_ave_test = (cumsum[i] - cumsum[i-time_window])/(time_window)
        ma_test[i] = moving_ave_test
        
    upperlimit[i]=5*v_test[i]
    lowerlimit[i]=-5*v_test[i]
    
    if x*w_test[i] <= upperlimit[i] and x*w_test[i] >= lowerlimit[i]:
        
        if ma_test[i] == x:
            w_test[i+1] = w_test[i]
            cash_test[i+1] = cash_test[i]
    
        if ma_test[i] < x: 
            w_test[i+1] = cash_test[i]/x  + w_test[i]
            cash_test[i+1] = 0
        
        if ma_test[i] > x:
            cash_test[i+1] = 0.5*w_test[i]*x + cash_test[i]
            w_test[i+1] = 0.5*w_test[i]
            
    
    else:
        w_test[i+1]=w_test[i]-(x*w_test[i] - 5*v_test[i])/x
        
        
    v_test[i+1]=v_test[i]+w_test[i+1]*S_test[i]*r_test[i+1]+ cash_test[i]*(rf_test[i+1])
    
ma_test[i+1] = S_test[len(S_test)-1]
 

tf_strategy_test = [a*b for a,b in zip(w_test,S_test)]+ cash_test


# In[109]:


plt.plot(t,tf_strategy_test)


# In[110]:


len(test)


# In[ ]:





# In[111]:


plt.plot(t,tf_strategy_test)

plt.plot(t,upperlimit,color='red')
plt.plot(t,lowerlimit,color='red')
plt.show()


# In[112]:


delta_v_tf_test=[]
for i in range(448):
    delta_v_tf_test.append(w_test[i+1]*S_test[i]*r_test[i+1])


# In[113]:


np.mean(delta_v_tf_test)
np.std(delta_v_tf_test)
sharpe_test_tf=np.mean(delta_v_tf_test)/np.std(delta_v_tf_test)
sharpe_test_tf


# In[ ]:





# ## MR

# In[114]:


import pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

import statsmodels

plt.plot(t, S_test)
plt.show()


#TRADING: TREND FOLLOWING 
    
#time_window = 3*int(1/dt)
time_window = 30
cumsum = [0]

ma_test = np.zeros(np.shape(S_test))

w_test = np.zeros(np.shape(S_test))
cash_test = np.zeros(np.shape(S_test))

cash_test[0] = 200000*5

v_mr_test= np.zeros(np.shape(S_test))
v_mr_test[0]=200000
r_test=df['excess return'].values
rf_test=df['daily_return'].values
interst=np.zeros(np.shape(S_test))
rf_test=rf_test[-449:]
r_test=r_test[-449:]

upperlimit = np.zeros(np.shape(test))
lowerlimit = np.zeros(np.shape(test))

for i, x in enumerate(S_test[:-1], 0):
    cumsum.append(cumsum[i] + x)
    ma_test[i] = x
    if i>=time_window:
        moving_ave = (cumsum[i] - cumsum[i-time_window])/(time_window)
        ma_test[i] = moving_ave
    upperlimit[i]=5*v_mr_test[i]
    lowerlimit[i]=-5*v_mr_test[i]
    if x*w_test[i] <= upperlimit[i] and x*w_test[i] >= lowerlimit[i]:
        
        if ma_test[i] == x:
            w_test[i+1] = w_test[i]
            cash_test[i+1] = cash_test[i]
    
        if ma_test[i] > x: 
            w_test[i+1] = cash_test[i]/x  + w_test[i]
            cash_test[i+1] = 0
        
        if ma_test[i] < x:
            cash_test[i+1] = w_test[i]*x + cash_test[i]
            w_test[i+1] = 0
    else:
        w_test[i+1]=w_test[i]-(x*w_test[i] - 5*v_mr_test[i])/x
        
        
    v_mr_test[i+1]=v_mr_test[i]+w_test[i+1]*S_test[i]*r_test[i+1]+ cash_test[i]*(rf_test[i+1])
    
ma_test[i+1] = S_test[len(S_test)-1]
 

mr_strategy_test = [a*b for a,b in zip(w_test,S_test)]+ cash_test


# In[115]:


plt.plot(t,mr_strategy_test)


# In[116]:


plt.figure(figsize=(30,30))
plt.plot(t,mr_strategy_test)

plt.plot(t,upperlimit,color='red')
plt.plot(t,lowerlimit,color='red')

plt.show()


# In[ ]:





# In[117]:


delta_v_mr_test=[]
for i in range(448):
    delta_v_mr_test.append(w_test[i+1]*S_test[i]*r_test[i+1])


# In[118]:


np.mean(delta_v_mr_test)
np.std(delta_v_mr_test)
sharpe_test_mr=np.mean(delta_v_mr_test)/np.std(delta_v_mr_test)
sharpe_test_mr


# ## AR

# In[119]:


import pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
from pandas.plotting import lag_plot

from pandas.plotting import autocorrelation_plot

import statsmodels

t=np.linspace(0,len(S_test),len(S_test))
plt.plot(t, S_test)
plt.show()


#TRADING: TREND FOLLOWING 

time_window = 30
cumsum = [0]

w_test = np.zeros(np.shape(S_test))
cash_test = np.zeros(np.shape(S_test))

cash_test[0] = 200000*5

v_ar_test= np.zeros(np.shape(S_test))
v_ar_test[0]=200000
r_test=df['excess return'].values
rf_test=df['daily_return'].values
interst=np.zeros(np.shape(S_test))

upperlimit = np.zeros(np.shape(S_test))
lowerlimit = np.zeros(np.shape(S_test))


ar_prediction_test = np.zeros(np.shape(S_test))


for i, x in enumerate(S_test[:-1], 0):
    cumsum.append(cumsum[i] + x)
    ar_prediction_test[i] = x
    if i>=time_window:
        X = S_test[0:i]
        test = X
        # train autoregression
        model_test = AutoReg(test,lags=1)
        model_fit_test = model_test.fit()
        predictions_test = model_fit_test.predict(start=len(test), end=len(test), dynamic=False)
        ar_prediction_test[i] = predictions_test
        
    upperlimit[i]=5*v_ar_test[i]
    lowerlimit[i]=-5*v_ar_test[i]
    if x*w_test[i] <= upperlimit[i] and x*w_test[i] >= lowerlimit[i]:
    
        if ar_prediction_test[i] == x:
            w_test[i+1] = w_test[i]
            cash_test[i+1] = cash_test[i]
    
        if ar_prediction_test[i] > x: 
            w_test[i+1] = cash_test[i]/x  + w_test[i]
            cash_test[i+1] = 0
        
        if ar_prediction_test[i] < x:
            cash_test[i+1] = w_test[i]*x + cash_test[i]
            w_test[i+1] = 0
    else:
        w_test[i+1]=w_test[i]-(x*w_test[i] - 5*v_ar_test[i])/x
        
    v_ar_test[i+1]=v_ar_test[i]+w_test[i+1]*S_test[i]*r_test[i+1]+ cash_test[i]*(rf_test[i+1])/5
ma_test[i+1] = S_test[len(S_test)-1]

ar_strategy_test = [a*b for a,b in zip(w_test,S_test)]+ cash_test

plt.figure(figsize=(30,30))
#plt.plot(t, tf_strategy)
#plt.plot(t, cash_test[0]*S_test/S_test[0])
#plt.plot(t, mr_strategy)
plt.plot(t, ar_strategy_test)

plt.plot(t,upperlimit,color='red')
plt.plot(t,lowerlimit,color='red')

plt.show()


# In[120]:


delta_v_ar_test=[]
for i in range(448):
    delta_v_ar_test.append(w_test[i+1]*S_test[i]*r_test[i+1])


# In[121]:


np.mean(delta_v_ar_test)
np.std(delta_v_ar_test)
sharpe_test_ar=np.mean(delta_v_ar_test)/np.std(delta_v_ar_test)
sharpe_test_ar


# In[ ]:


## original sharpe ratio


# In[126]:


ex_mean=np.mean(df['excess return'][-449:])


# In[127]:


ex_var=np.var(df['excess return'][-449:])


# In[129]:


sharpe_original=ex_mean/(ex_var**(0.5))


# In[130]:


mean_list=[np.mean(delta_v_tf_test),np.mean(delta_v_mr_test),np.mean(delta_v_ar_test),ex_mean]
sd_list=[np.std(delta_v_tf_test),np.std(delta_v_mr_test),np.std(delta_v_ar_test),ex_var]


# In[133]:


plt.plot(sd_list, mean_list, 'o')  # 'o' 表示点状
plt.show()


# In[136]:


ex_var**(0.5)


# In[137]:


sd_list


# In[ ]:




