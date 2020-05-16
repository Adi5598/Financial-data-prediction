import numpy as np
from scipy.fftpack import dct,fft,idct
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score

# READING THE DATA

df = pd.read_csv('data.csv')
df = df.dropna(axis=0, how='all')
print(df)

# VISUALIZE THE DATA

import matplotlib.pyplot as pt
pt.plot(df.iloc[:,[2]])
pt.show()

# FEATURE GENERATION

data = df.to_numpy()
stock_data = (data[:,1]).astype(float)
stock_data = stock_data[~np.isnan(stock_data)]

#FILTERING THE SIGNAL

stock_data_dct = dct(stock_data,norm='ortho')
stock_data_dct[20:] = 0
pt.plot(stock_data,'y')
pt.plot(idct(stock_data_dct,norm='ortho'),'r')
pt.show()

# PREDICTING 
n = 5000
train = stock_data[:n]
test = stock_data[n:]

reg = RandomForestRegressor(n_estimators=1,max_depth = 20)
reg.fit(np.arange(1,n+1)[:,None],train[:,None])
y_pred = reg.predict(np.arange(1,len(stock_data))[:,None])
#explained_variance_score(test,y_pred)
#print(mean_absolute_error(test,y_pred))
#print(r2_score(np.ravel(test),np.ravel(y_pred)))
#r2_score(np.ravel(test),np.ravel(y_pred))

#pt.plot(np.linspace(n,n+len(test),len(test)),y_pred,color = 'r')
#pt.plot(np.linspace(n,n+len(test),len(test)),test,color = 'b')
pt.plot(y_pred)
pt.show()
