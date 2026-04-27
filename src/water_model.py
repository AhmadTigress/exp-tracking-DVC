import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dvclive import Live


data = pd.read_csv("/home/ahmadtigress/Documents/exp-tracking-dvc/data/water_potability.csv")

train_data, test_data = train_test_split(data, test_size=0.2, random_state=40)

def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values : {e}")


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

X_train = train_processed_data.drop(['Potability'], axis=1)
y_train = train_processed_data['Potability']

n_estimators=500

clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train)

# Save
pickle.dump(clf, open("model.pkl", "wb"))


X_test = test_processed_data.drop(['Potability'], axis=1)
y_test = test_processed_data['Potability']


model = pickle.load(open("model.pkl", "rb"))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with Live(save_dvc_exp=True) as live:
    live.log_metric("accuracy:", accuracy)
    live.log_metric("precision:", precision)
    live.log_metric("recall_score:", recall)
    live.log_metric("f1_score:", f1)

    live.log_param("n_estimators:", n_estimators)