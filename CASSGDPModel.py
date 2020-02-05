import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import statsmodels.formula.api as fm    # This imports the Formula method of using the OLS module.
#
#   I use the Pandas DataReader to import the CASS Index and US GDP from the St Louis
#   Federal Reserve FRED database. I use join statements to create a single DataFrame with
#   all of therelevant data.It builds a simple linear regression model with the GDP as the
#   dependent variable.  
#
startdate='1990-01-01'
CFRED=pdr.DataReader(name='FRGEXPUSM649NCIS',data_source='fred',start=startdate)
CFRED.rename({'FRGEXPUSM649NCIS':'CASS_Expenditure'},axis=1,inplace='True')
CSFRED=pdr.DataReader(name='FRGSHPUSM649NCIS',data_source='fred',start=startdate)
CSFRED.rename({'FRGSHPUSM649NCIS':'CASS_Shipment'},axis=1,inplace='True')
CF=CFRED
CF=CFRED.join(CSFRED,how='inner')
CF.plot()
plt.show()
#
#   Read in the GDP and change to Quarterly period index (instead of months)
#
GDP=pdr.DataReader(name='A191RP1Q027SBEA',data_source='fred',start=startdate)
GDP.rename({'A191RP1Q027SBEA':'USGDP'},axis=1,inplace='True')
GDP=GDP.to_period('Q')
#
#   Construct quarterly dataframe. Resample the monthly CASS data to create a quarterly series.
#
QData=pd.DataFrame()
QData['CIndex']=CFRED.CASS_Expenditure.resample('Q').mean()
QData=QData.to_period('Q')
QData=QData.join(GDP,how='inner')
#
#   Shift the GDP data forward to create the values to be predicted. Difference the CASS data
#   to compute the predictor.
#
QData['FGDP']=QData['USGDP'].shift(-1)
QData['DCASS']=QData['CIndex']-QData['CIndex'].shift(1)
#
print(QData.tail())
result=fm.ols(formula="FGDP~DCASS",data=QData).fit()
print(result.summary())
r=result.fittedvalues-QData['FGDP']
r.plot()
plt.title('Model residuals')
plt.show()



