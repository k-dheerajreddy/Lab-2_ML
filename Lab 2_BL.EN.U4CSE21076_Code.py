import pandas as pd
import numpy as np
df = pd.read_excel("19CSE305_LabData_Set3.1.xlsx", sheet_name = "thyroid0387_UCI")
print (df)
nom=['sex', 'on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'T3 measured', 'TT4 measured', 'T4U measured',
       'FTI measured', 'TBG measured', 'referral source',
       'Condition']
df.replace("?", np.nan, inplace=True)
ratio=['age']
interval=['TSH','T3','TT4',  'T4U','FTI',  'TBG']
nom_encoded = pd.get_dummies(df,columns=nom)
df=nom_encoded
df

"""datatypes after one hot code encoding"""

df.info()
df.describe()

"""Study the data range for numeric variables."""

numeric_attributes = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
numeric_range = df[numeric_attributes].max() - df[numeric_attributes].min()
print(numeric_range)

"""Study the presence of missing values in each attribute."""

df.isnull().sum()

"""Study presence of outliers in data."""

#outliers for age
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_age =df[(df['age'] < lower_bound) | (df['age'] > upper_bound)]
outliers_age

#outliers for TSH
Q1 = df['TSH'].quantile(0.25)
Q3 = df['TSH'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TSH =df[(df['TSH'] < lower_bound) | (df['TSH'] > upper_bound)]
outliers_TSH

#outliers for T3
Q1 = df['T3'].quantile(0.25)
Q3 = df['T3'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_T3 =df[(df['T3'] < lower_bound) | (df['T3'] > upper_bound)]
outliers_T3

#outliers for TT4
Q1 = df['TT4'].quantile(0.25)
Q3 = df['TT4'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['TT4'] < lower_bound) | (df['TT4'] > upper_bound)]
outliers_TT4

#outliers for T4U
Q1 = df['T4U'].quantile(0.25)
Q3 = df['T4U'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_T4U =df[(df['T4U'] < lower_bound) | (df['T4U'] > upper_bound)]
outliers_T4U

#outliers for FTI
Q1 = df['FTI'].quantile(0.25)
Q3 = df['FTI'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_FTI =df[(df['FTI'] < lower_bound) | (df['FTI'] > upper_bound)]
outliers_FTI

#outliers for TBG
Q1 = df['TBG'].quantile(0.25)
Q3 = df['TBG'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_TT4 =df[(df['TBG'] < lower_bound) | (df['TBG'] > upper_bound)]
outliers_TT4

"""For numeric variables, calculate the mean and variance (or standard deviation)."""

mean_age= df['age'].mean()
variance_age = df['age'].var()
print("Mean of age is:", mean_age)
print("Variance of age is:", variance_age)
mean_TSH= df['TSH'].mean()
variance_TSH = df['TSH'].var()
print("Mean of TSH is:", mean_TSH)
print("Variance of TSH is:", variance_TSH)
mean_T3 = df['T3'].mean()
variance_T3 = df['T3'].var()
print("Mean of T3 is:", mean_T3)
print("Variance of T3 is:", variance_T3)
mean_TT4 = df['TT4'].mean()
variance_TT4 = df['TT4'].var()
print("Mean of TT4 is:", mean_TT4)
print("Variance of TT4 is:", variance_TT4)
mean_T4U = df['T4U'].mean()
variance_T4U = df['T4U'].var()
print("Mean of T4U is:", mean_T4U)
print("Variance of T4U is:", variance_T4U)
mean_FTI = df['FTI'].mean()
variance_FTI = df['FTI'].var()
print("Mean of FTI is:", mean_FTI)
print("Variance of FTI is:", variance_FTI)
mean_TBG = df['TBG'].mean()
variance_TBG = df['TBG'].var()
print("Mean of TBG is:", mean_TBG)
print("Variance of TBG is:", variance_TBG)

"""A2. Data Imputation:employ appropriate central tendencies to fill the missing values in the data variables. Employ following guidance.
•Mean may be used when the attributeis numeric with no outliers
•Median may be employed for attributes which are numeric and contain outliers
•Mode may be employed for categorical attributes
"""

numeric_no_outliers = ['age']
numeric_with_outliers = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']


# Imputing missing values for numeric attributes without outliers using mean
for attribute in numeric_no_outliers:
    df[attribute].fillna(df[attribute].mean(), inplace=True)

# Imputing missing values for numeric attributes with outliers using median
for attribute in numeric_with_outliers:
    df[attribute].fillna(df[attribute].median(), inplace=True)
df
#no categorical attributes as the categories are converted into binary by one hot code encoding

"""A3. Data Normalization / Scaling:from the data study, identify the attributes which may need normalization. Employ appropriate normalization techniques to create normalized set of data."""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale = ['age','TSH','T3','TT4','T4U','FTI','TBG']
df[scale] = scaler.fit_transform(df[scale])
df

"""A4. Similarity Measure:Take the first 2 observation vectors from the dataset. Consider only the attributes (direct or derived) with binary values for these vectors (ignore other attributes). Calculate the Jaccard Coefficient(JC) and Simple Matching Coefficient (SMC)between the document vectors. Use first vector for each document for this.Compare the values for JC and SMC and judge the appropriateness of each of them.JC = (f11) / (f01+ f10+ f11)SMC= (f11+ f00) / (f00 + f01+ f10+ f11)f11= number of attributes where the attribute carries value of 1 in both the vectors."""

v1 = df['sex_M']
v2 = df['Condition_M']
f11 = sum([1 for a, b in zip(v1, v2) if a == b == 1])
f01 = sum([1 for a, b in zip(v1, v2) if a == 0 and b == 1])
f10 = sum([1 for a, b in zip(v1, v2) if a == 1 and b == 0])
f00 = sum([1 for a, b in zip(v1, v2) if a == b == 0])
print("f00",f00)
print("f01",f01)
print("f10",f10)
print("f11",f11)
smc = (f11 + f00) / (f00 + f01 + f10 + f11)
print("smc:",smc)
jc = f11 / (f01 + f10 + f11)
print("jc:",jc)

"""optional section by trying another attributes"""

v1 = df['sex_F']
v2 = df['Condition_MK']
f11 = sum([1 for a, b in zip(v1, v2) if a == b == 1])
f01 = sum([1 for a, b in zip(v1, v2) if a == 0 and b == 1])
f10 = sum([1 for a, b in zip(v1, v2) if a == 1 and b == 0])
f00 = sum([1 for a, b in zip(v1, v2) if a == b == 0])
print("f00",f00)
print("f01",f01)
print("f10",f10)
print("f11",f11)
smc = (f11 + f00) / (f00 + f01 + f10 + f11)
print("smc:",smc)
jc = f11 / (f01 + f10 + f11)
print("jc:",jc)

""" Cosine Similarity Measure:Now take the complete vectors for these two observations (including all the attributes). Calculate the Cosine similarity between the documents by using the second feature vector for each document."""

from sklearn.metrics.pairwise import cosine_similarity
v1 = np.array(v1).reshape(1, -2)
v2 = np.array(v2).reshape(1, -2)
cosine_sim = cosine_similarity(v1, v2)
print("Cosine Similarity:", cosine_sim[0][0])

"""Optional section by checking another vector"""

from sklearn.metrics.pairwise import cosine_similarity
v1 = np.array(v1).reshape(1, -3)
v2 = np.array(v2).reshape(1, -2)
cosine_sim = cosine_similarity(v1, v2)
print("Cosine Similarity:", cosine_sim[0][0])

"""A6. Heatmap Plot:Consider the first 20 observation vectors. Calculate the JC, SMC and COS between the pairs of vectors for these 20 vectors. Employ similar strategies for coefficient calculation as in A4& A5. Employ a heatmap plot to visualize the similarities."""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, pairwise_distances
data = np.random.randint(0, 2, size=(20, 5))
smc_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        smc_matrix[i, j] = np.mean(data[i] == data[j])
jc_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        jc_matrix[i, j] = jaccard_score(data[i], data[j])
cos_matrix = pairwise_distances(data, metric='cosine')
smc_df = pd.DataFrame(smc_matrix, index=range(8, 28), columns=range(8, 28))
jc_df = pd.DataFrame(jc_matrix, index=range(8, 28), columns=range(8, 28))
cos_df = pd.DataFrame(cos_matrix, index=range(8, 28), columns=range(8, 28))
fig, axes = plt.subplots(1, 3, figsize=(20, 9))
sns.heatmap(smc_df, annot=True, cmap='coolwarm', ax=axes[1])
axes[1].set_title('Simple Matching Coefficient')
sns.heatmap(jc_df, annot=True, cmap='coolwarm', ax=axes[0])
axes[0].set_title('Jaccard Coefficient')
sns.heatmap(cos_df, annot=True, cmap='coolwarm', ax=axes[2])
axes[2].set_title('Cosine Similarity')
plt.tight_layout()
plt.show()