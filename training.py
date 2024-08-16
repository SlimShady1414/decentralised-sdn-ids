import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# List of file paths
file_paths = [
    '/mnt/shared/combine.csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Monday-WorkingHours.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Tuesday-WorkingHours.pcap_ISCX.csv',
    '/mnt/shared/combine.csv/Wednesday-workingHours.pcap_ISCX.csv'
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
for file_path in file_paths:
    df = pd.read_csv(file_path)

    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Fill missing values in 'Flow Bytes/s' with the median value
    df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].median(), inplace=True)

    # Convert 'Label' to numerical values
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Replace infinity values with a large finite number
    df.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)

    # Cap extremely large values
    max_value = np.finfo(np.float32).max
    df['Flow Bytes/s'] = df['Flow Bytes/s'].clip(lower=-max_value, upper=max_value)
    df['Flow Packets/s'] = df['Flow Packets/s'].clip(lower=-max_value, upper=max_value)

    # Append to the combined DataFrame
    combined_df = pd.concat([combined_df, df])

# Split the data into features and target
X = combined_df[features]
y = combined_df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for SGDClassifier
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'loss': ['hinge', 'modified_huber'],
    'penalty': ['l2', 'l1', 'elasticnet']
}

grid_search = GridSearchCV(SGDClassifier(random_state=42, max_iter=1000, tol=1e-3), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
model = grid_search.best_estimator_

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Save the model and scaler
joblib.dump(model, '/mnt/shared/ddos_model_all_features.pkl')
joblib.dump(scaler, '/mnt/shared/feature_scaler_all_features.pkl')