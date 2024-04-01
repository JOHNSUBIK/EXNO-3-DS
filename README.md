## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/80df6014-4c1e-4bf6-ae83-4379fe6bd129)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/e871a52e-6a49-493e-9484-c8d69383021a)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/f39a8fed-a1de-4616-b2f0-e6272fd38993)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/c1a9c612-5feb-478b-bd04-cd30e1067775)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/54af5429-d16a-4078-b651-e10f00ae76b5)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/ba39eda4-2655-4635-9637-e3b07381d825)
```
pip install --upgrade category_encoders
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/94b291aa-d192-4628-9a90-d13dcef171cd)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/59cd22b6-afda-4853-be49-7e22e235b660)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/a14316b4-27f7-4e3d-85f6-13e9b6be6afb)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/380d4e48-87ea-48aa-abd5-ca8acb95b7ee)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/e14bd39b-e165-49f6-b18e-802290c77794)
```
df.skew()
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/af8c716d-c097-4e92-8031-dbb262f5476d)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/66654b26-ddbc-417d-82df-08306583480d)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/3a787bf2-c9f7-43bc-b10c-12babbc8ad8a)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/1187b64b-d58c-44fd-8b7d-1f5f69df2346)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/ad20e73b-1e27-4a12-a594-711bd25e707a)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/cffbaf5b-4b97-4c59-af86-7da01774f287)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/3504109e-bce3-4290-8a14-7aed2cd41733)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/JOHNSUBIK/EXNO-3-DS/assets/150279319/fd77aac9-7397-4754-a1ee-055a2a906095)





# RESULT:
       Finally,perform Feature Encoding and Transformation process is executed successfully

       
