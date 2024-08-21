import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import requests
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# List of file paths
file_paths = [
    '/mnt/SharedCapstone/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Monday-WorkingHours.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Tuesday-WorkingHours.pcap_ISCX.csv',
    '/mnt/SharedCapstone/Wednesday-workingHours.pcap_ISCX.csv'
]

# Define a comprehensive set of features
features = [
    'Init_Win_bytes_forward', 'Fwd Packet Length Max', 'Fwd Packet Length Mean',
    'Subflow Fwd Bytes', 'Avg Fwd Segment Size', 'Subflow Fwd Packets',
    'Total Length of Fwd Packets', 'Bwd Packet Length Min', 'act_data_pkt_fwd',
    'Fwd IAT Std', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'Average Packet Size', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s'
]

# Initialize an empty DataFrame to hold combined data
combined_df = pd.DataFrame()

# Load and preprocess each file, then concatenate
print("Loading and preprocessing data...")
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path, engine='python')
        print(f"Processing {file_path}...")

        df.columns = df.columns.str.strip()
        df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].median(), inplace=True)
        df.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)
        max_value = np.finfo(np.float32).max
        df['Flow Bytes/s'] = df['Flow Bytes/s'].clip(lower=-max_value, upper=max_value)
        df['Flow Packets/s'] = df['Flow Packets/s'].clip(lower=-max_value, upper=max_value)
        combined_df = pd.concat([combined_df, df])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Data loading and preprocessing complete.")

# Grouping labels into broader categories
label_mapping = {
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
}

combined_df['Label'] = combined_df['Label'].map(label_mapping)

print("Handling missing values...")
combined_df.dropna(subset=['Label'], inplace=True)
print("Missing values handled.")
print(f"Final label distribution:\n{combined_df['Label'].value_counts()}\n")

X = combined_df[features]
y = combined_df['Label']

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set label distribution:\n{y_train.value_counts()}")
print("Data split complete.")

print("Applying controlled RandomUnderSampler...")
undersample_strategy = {k: min(v, 50000) for k, v in y_train.value_counts().items()}
rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"Training set label distribution after controlled undersampling:\n{pd.Series(y_train_rus).value_counts()}")

print("Applying SMOTE to balance classes in the training set...")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_rus, y_train_rus)
print(f"Training set label distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

print("Setting up ensemble models...")
classifiers = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('et', ExtraTreesClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
]

ensemble = VotingClassifier(estimators=classifiers, voting='soft')

print("Starting hyperparameter tuning...")
param_grid = {
    'rf__n_estimators': [50, 100],
    'et__n_estimators': [50, 100],
    'xgb__n_estimators': [50, 100],
}

grid_search = GridSearchCV(ensemble, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train_smote)
print("Hyperparameter tuning complete.")

model = grid_search.best_estimator_

print("Evaluating the model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

print("Saving the model and scaler...")
joblib.dump(model, '/mnt/SharedCapstone/ensemble_multi_attack_model.pkl')
joblib.dump(scaler, '/mnt/SharedCapstone/feature_scaler_all_features.pkl')
print("Model and scaler saved.")

# Federated learning part: Update local model with global weights
try:
    response = requests.get('http://10.0.2.15:5000/get_model')
    global_weights = np.array(response.json()['weights'])

    for i, estimator in enumerate(model.estimators_):
        if hasattr(estimator, 'coef_'):
            estimator.coef_ = global_weights[i]
        elif hasattr(estimator, 'feature_importances_'):
            # Custom setting of the attribute if it's allowed by the model
            estimator._final_estimator.set_params(feature_importances_=global_weights[i])
        else:
            raise AttributeError(f"Estimator {i} does not have 'coef_' or 'feature_importances_' attributes")
except Exception as e:
    print(f"Failed to update the local model with global weights: {e}")

# Save the updated global model
joblib.dump(model, '/mnt/SharedCapstone/global_model.pkl')

# Log the successful completion
print("Model updated with global weights and saved successfully.")
