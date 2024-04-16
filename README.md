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
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv('/content/Encoding Data.csv')
df
```
<img width="294" alt="Screenshot 2024-04-16 at 10 35 15 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/f8aaa91f-6a59-4783-ad41-4743cc8edf34">


```
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="140" alt="Screenshot 2024-04-16 at 10 36 14 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/7480e46e-88d0-447c-b8e2-06b75c298a0f">


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="349" alt="Screenshot 2024-04-16 at 10 36 23 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/7ce3a351-6c7d-46c7-b0ed-54496716e107">


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="343" alt="Screenshot 2024-04-16 at 10 36 39 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/22035b94-2514-469d-a9f0-dd796426f37d">


```
on=OneHotEncoder(sparse=False)
df2=df.copy()
en=pd.DataFrame(on.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,en],axis=1)
pd.get_dummies(df2,columns=["nom_0"])

```
<img width="1354" alt="Screenshot 2024-04-16 at 10 36 57 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/ce361de7-af80-4c4e-b629-773eb919f683">


```
pip install --upgrade category_encoders
```
<img width="1271" alt="Screenshot 2024-04-16 at 10 37 18 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/8a3782be-c9f6-4c89-80d0-f18137878f74">


```
from category_encoders import BinaryEncoder
fd=pd.read_csv('/content/data.csv')
fd
```
<img width="485" alt="Screenshot 2024-04-16 at 10 37 32 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/775ba1d2-9b6e-4f56-8406-d7c35fd24257">


```
be=BinaryEncoder()
nd=be.fit_transform(fd['Ord_2'])
fd
```
<img width="489" alt="Screenshot 2024-04-16 at 10 37 44 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/59d7881d-3b89-4007-b829-637c912d48a3">



```
dfb=pd.concat([fd,nd],axis=1)
dfb=fd.copy()
dfb
```
<img width="493" alt="Screenshot 2024-04-16 at 10 37 50 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/ebe45bfa-7ad3-4ddd-a40d-89368e5132a8">


```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=fd.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
<img width="563" alt="Screenshot 2024-04-16 at 10 37 56 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/3d1c2ad9-8560-4e75-9a5f-de816340b803">


```
from scipy import stats
import numpy as np
ab=pd.read_csv('/content/Data_to_Transform.csv')
ab
```
<img width="829" alt="Screenshot 2024-04-16 at 10 38 04 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/f0671f47-9243-4569-b306-3075d59cea6f">



```
ab.skew()
```
<img width="332" alt="Screenshot 2024-04-16 at 10 38 10 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/2dff201c-ff30-4e52-ae85-0702438ee592">



```
np.log(ab['Highly Positive Skew'])
```
<img width="507" alt="Screenshot 2024-04-16 at 10 38 17 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/71df6c74-4a65-47c7-acfe-2feb4edcda5d">



```
np.reciprocal(ab["Moderate Negative Skew"])
```
<img width="531" alt="Screenshot 2024-04-16 at 10 38 25 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/e214c109-155f-4d2e-8fc5-9e4da4699cd8">



 
```
np.sqrt(ab["Highly Negative Skew"])
```
<img width="1040" alt="Screenshot 2024-04-16 at 10 38 34 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/716d908f-0682-468e-99a0-53e9b613169c">




```
np.square(ab["Highly Positive Skew"])
```
<img width="541" alt="Screenshot 2024-04-16 at 11 07 48 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/0828b8b7-1de2-45e9-b54a-51f6d1be60e9">





```
ab['Highly Positive Skew_boxcox'], parameters=stats.boxcox(ab['Highly Positive Skew'])
ab
```
<img width="1084" alt="Screenshot 2024-04-16 at 11 07 59 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/230b4fc5-579b-4308-8d43-31e5e962aa46">



```
ab.skew()
```
<img width="358" alt="Screenshot 2024-04-16 at 11 08 06 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/09ab17e9-2306-4808-b368-0ffbd6e2f5b3">





```
ab['Moderate Negative Skew_yeojohnson'], parameters=stats.yeojohnson(ab['Moderate Negative Skew'])
ab
```

<img width="1317" alt="Screenshot 2024-04-16 at 11 08 18 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/099582e7-d7c9-4bd6-abc4-c5fea02feeb1">






```
ab.skew()
```

<img width="393" alt="Screenshot 2024-04-16 at 11 08 26 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/5332ac84-a7bf-49f5-83f4-2237bfae2edd">




```
ab['Highly Negative Skew_yeojohnson'], parameters=stats.yeojohnson(ab['Highly Negative Skew'])
ab
```

<img width="1319" alt="Screenshot 2024-04-16 at 11 08 39 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/c074a7b4-c892-47b8-93e0-18c7dcb1bc56">




```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
ab["Moderate Negative Skew_1"]=qt.fit_transform(ab[["Moderate Negative Skew"]])
ab
```
<img width="1320" alt="Screenshot 2024-04-16 at 11 08 50 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/0f29b611-a6e7-4c98-8586-617e756c6723">





```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(ab["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="579" alt="Screenshot 2024-04-16 at 11 09 08 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/1da360bf-5fbf-4b92-9d49-e3eb497c2d01">



```
sm.qqplot(np.reciprocal(ab["Moderate Negative Skew"]),line='45')
```
<img width="576" alt="Screenshot 2024-04-16 at 11 09 21 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/fd527974-48f4-45f0-8013-8956619c6bd4">
<img width="575" alt="Screenshot 2024-04-16 at 11 09 35 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/b67b4b3d-abe8-427e-89b7-7a87e098a726">




```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
ab["Moderate Negative Skew"]=qt.fit_transform(ab[["Moderate Negative Skew"]])
sm.qqplot(ab["Moderate Negative Skew"],line="45")
plt.show()
```
<img width="563" alt="Screenshot 2024-04-16 at 11 09 46 AM" src="https://github.com/Richard01072002/EXNO-3-DS/assets/141472248/488f6af3-ddfd-4dfa-97b7-bf0b6d86976f">



       
# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
