## Libraries
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from collections import Counter
from sklearn.utils import resample
import time
import warnings
warnings.filterwarnings("ignore")

## Read the CICIDS2017 dataset
data_dir = 'D:/Capstone/intial_attempt'
files = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

dfs = [pd.read_csv(os.path.join(data_dir, file)) for file in files]

df = pd.concat(dfs)
del dfs

df.columns = df.columns.str.strip()
nRow, nCol = df.shape
print(f'The table has {nRow} rows and {nCol} columns')
print(df.head())

df['Label'] = df['Label'].map({
    'BENIGN': 'Normal',
    'Bot': 'Botnet',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'DoS Hulk': 'DoS/DDoS',
    'DoS GoldenEye': 'DoS/DDoS',
    'DoS slowloris': 'DoS/DDoS',
    'DoS Slowhttptest': 'DoS/DDoS',
    'DDoS': 'DoS/DDoS',
    'Heartbleed': 'DoS/DDoS',
    'Infiltration': 'Infiltration',
    'PortScan': 'Port Scan',
    'Web Attack – Brute Force': 'Web Attack',
    'Web Attack – XSS': 'Web Attack',
    'Web Attack – Sql Injection': 'Web Attack'
})

print(df['Label'].value_counts())

# Remove class 'Infiltration'
df = df[df['Label'] != 'Infiltration']

# Define the sample size range for each class
min_samples = 2000
max_samples = 2200

# Function to randomly sample within the range [min_samples, max_samples]
def random_sample_class(df, label, min_samples, max_samples):
    n_samples = np.random.randint(min_samples, max_samples + 1)
    return resample(df[df['Label'] == label], n_samples=n_samples, random_state=42)

# Define the labels to be resampled
class_labels = ['Normal', 'DoS/DDoS', 'Port Scan', 'Brute Force']

# Apply the random_sample_class function to each class
resampled_dfs = [random_sample_class(df, label, min_samples, max_samples) for label in class_labels]
df_resampled = pd.concat(resampled_dfs + [df[df['Label'] == label] for label in df['Label'].unique() if label not in class_labels])
df = df_resampled

print(df['Label'].value_counts())

# Preprocessing (normalization and padding values)
# Min-max normalization
numeric_features = df.dtypes[df.dtypes != 'object'].index
df[numeric_features] = df[numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))

# Fill empty values by 0
df = df.fillna(0)

# Split train set and test set
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
X = df.drop(['Label'],axis=1).values
y = df.iloc[:, -1].values.reshape(-1,1)
y = np.ravel(y)
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
print(pd.Series(y_train).value_counts())

# Train models
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_feature = dt.feature_importances_

rf = RandomForestClassifier(max_depth=5)
rf.fit(X_train, y_train)
rf_feature = rf.feature_importances_

et = ExtraTreesClassifier(max_depth=5)
et.fit(X_train, y_train)
et_feature = et.feature_importances_

xg = xgb.XGBClassifier(n_estimators=3)
xg.fit(X_train, y_train)
xgb_feature = xg.feature_importances_

# Calculate and print feature importance
avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature) / 4
feature = df.drop(['Label'], axis=1).columns.values

f_list = sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)
print("Features sorted by their score:")
for importance, fname in f_list:
    print(f"{fname}: {importance}")

Sum = 0
fs = []
for importance, fname in f_list:
    Sum += importance
    fs.append(fname)
    if Sum >= 0.9:
        break

print(f"Selected features ({len(fs)}): {fs}")


X_fs = df[fs].values
X_train, X_test, y_train, y_test = train_test_split(X_fs, y, train_size=0.7, test_size=0.3, random_state=0, stratify=y)
X_train.shape
print(pd.Series(y_train).value_counts())

# Continue with training models after feature selection
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
y_predict = dt.predict(X_test)
y_true = y_test
print('Accuracy of DT: ' + str(dt_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of DT: ' + (str(precision)))
print('Recall of DT: ' + (str(recall)))
print('F1-score of DT: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

rf = RandomForestClassifier(max_depth=5)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
y_predict = rf.predict(X_test)
y_true = y_test
print('Accuracy of RF: ' + str(rf_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of RF: ' + (str(precision)))
print('Recall of RF: ' + (str(recall)))
print('F1-score of RF: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

et = ExtraTreesClassifier(max_depth=5)
et.fit(X_train, y_train)
et_score = et.score(X_test, y_test)
y_predict = et.predict(X_test)
y_true = y_test
print('Accuracy of ET: ' + str(et_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of ET: ' + (str(precision)))
print('Recall of ET: ' + (str(recall)))
print('F1-score of ET: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

xg = xgb.XGBClassifier(n_estimators=3)
xg.fit(X_train, y_train)
xg_score = xg.score(X_test, y_test)
y_predict = xg.predict(X_test)
y_true = y_test
print('Accuracy of XGBoost: ' + str(xg_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of XGBoost: ' + (str(precision)))
print('Recall of XGBoost: ' + (str(recall)))
print('F1-score of XGBoost: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# Stacking model construction
base_predictions_train = pd.DataFrame({
    'DecisionTree': dt.predict(X_train).ravel(),
    'RandomForest': rf.predict(X_train).ravel(),
    'ExtraTrees': et.predict(X_train).ravel(),
    'XGBoost': xg.predict(X_train).ravel(),
})

base_predictions_test = pd.DataFrame({
    'DecisionTree': dt.predict(X_test).ravel(),
    'RandomForest': rf.predict(X_test).ravel(),
    'ExtraTrees': et.predict(X_test).ravel(),
    'XGBoost': xg.predict(X_test).ravel(),
})

stk = xgb.XGBClassifier().fit(base_predictions_train, y_train)
y_predict = stk.predict(base_predictions_test)
y_true = y_test
stk_score = accuracy_score(y_true, y_predict)
print('Accuracy of Stacking: ' + str(stk_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of Stacking: ' + (str(precision)))
print('Recall of Stacking: ' + (str(recall)))
print('F1-score of Stacking: ' + (str(fscore)))
print(classification_report(y_true, y_predict))
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title('Confusion Matrix for Stacking Model')
plt.show()
