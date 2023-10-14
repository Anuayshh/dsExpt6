# Feature_Transformation
# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

# PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
# OUTPUT:
![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/6980d4ae-eb90-4ec9-94fb-39cb109e2b78)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/5c71fef7-ba79-4425-8505-c449bd9ffd0b)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/36e9260d-5f6c-4ff6-ae9d-9d0330263b78)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/e11a48bc-bee8-4a8f-9f76-59e933eb261f)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/563413bf-bfda-4098-879c-e84e05030858)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/b165f726-6ef0-447c-9966-94357d1bc0b3)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/32d903b6-999d-4d3f-ae85-577e43cb7457)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/e41d5a21-ee5f-4637-a9f3-6d901668b2c4)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/209ac860-9523-41e4-acb9-ad11d61127ec)

![image](https://github.com/Anuayshh/dsExpt6/assets/127651217/1ffb9e27-6493-4553-b4e2-ac4f5e86ea1b)



# RESULT:
Thus feature transformation is done for the given dataset.
