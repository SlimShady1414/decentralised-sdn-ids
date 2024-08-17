import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
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

        # Remove leading and trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Fill missing values in 'Flow Bytes/s' with the median value
        df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].median(), inplace=True)

        # Replace infinity values with a large finite number
        df.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)

        # Cap extremely large values
        max_value = np.finfo(np.float32).max
        df['Flow Bytes/s'] = df['Flow Bytes/s'].clip(lower=-max_value, upper=max_value)
        df['Flow Packets/s'] = df['Flow Packets/s'].clip(lower=-max_value, upper=max_value)

        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, df])

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("Data loading and preprocessing complete.")

# Handle missing values in the target variable
print("Handling missing values...")
combined_df.dropna(subset=['Label'], inplace=True)
print("Missing values handled.")
print(f"Final label distribution:\n{combined_df['Label'].value_counts()}\n")

# Convert 'Label' to numerical values for multi-class classification
label_mapping = {label: idx for idx, label in enumerate(combined_df['Label'].unique())}
combined_df['Label'] = combined_df['Label'].map(label_mapping)

# Split the data into features and target
X = combined_df[features]
y = combined_df['Label']

# Check if there are multiple classes in the target variable
print(f"Unique classes in the target variable: {y.unique()}")

if len(y.unique()) == 1:
    raise ValueError("The dataset only contains one class. Cannot train a model.")

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set label distribution:\n{y_train.value_counts()}")
print("Data split complete.")

# Controlled undersampling
print("Applying controlled RandomUnderSampler...")
# Here, we allow the majority class to be reduced but not as drastically
undersample_strategy = {k: min(v, 50000) for k, v in y_train.value_counts().items()} # Adjust number based on your hardware capability
rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"Training set label distribution after controlled undersampling:\n{pd.Series(y_train_rus).value_counts()}")

# Apply SMOTE to balance the classes in the training set
print("Applying SMOTE to balance classes in the training set...")
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_rus, y_train_rus)
print(f"Training set label distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

# Hyperparameter tuning for SGDClassifier
print("Starting hyperparameter tuning...")
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'loss': ['hinge', 'modified_huber'],
    'penalty': ['l2', 'l1', 'elasticnet']
}

grid_search = GridSearchCV(SGDClassifier(random_state=42, max_iter=1000, tol=1e-3), param_grid, cv=5, scoring='accuracy', error_score='raise')
grid_search.fit(X_train_scaled, y_train_smote)
print("Hyperparameter tuning complete.")

# Get the best model from the grid search
model = grid_search.best_estimator_

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Save the model and scaler
print("Saving the model and scaler...")
joblib.dump(model, '/mnt/SharedCapstone/multi_attack_model.pkl')
joblib.dump(scaler, '/mnt/SharedCapstone/feature_scaler_all_features.pkl')
print("Model and scaler saved.")
