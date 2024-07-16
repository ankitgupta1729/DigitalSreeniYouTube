# Random Forest takes the subset of original data  and create as bootstrap data which
# has same size as original data by making repeated sampling. This makes one part of randomness
# Other part comes while building decision trees (random subset of features, say select 3
# features from the original 9 features)

import  pandas as pd
from matplotlib import pyplot as plt
import  numpy as np

df=pd.read_csv('images_analyzed_productivity1.csv')
#print(df.head(5))

sizes=df['Productivity'].value_counts(sort=True)
#print(sizes)

# drop irrelevant columns
df.drop(['Images_Analyzed','User'],axis=1,inplace=True)

# Handle Missing values

df=df.dropna()

# convert non-numeric data to numeric

df.Productivity[df.Productivity=='Good']=1
df.Productivity[df.Productivity=='Bad']=2

#print(df.head())

Y=df['Productivity'].values
Y=Y.astype(int)

# Define Independent variables

X= df.drop(labels=['Productivity'],axis=1)

# Split data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, random_state=30)

model.fit(X_train, y_train)

prediction_test=model.predict(X_test)

print(prediction_test)

from sklearn import metrics

print("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance)