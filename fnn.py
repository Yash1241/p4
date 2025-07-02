# Part 1 - Data Pre-Processing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# --- IMPORTANT: Change this variable for each run (e.g., 'SA', 'SB', 'SC') ---
current_scenario = 'SA' # <<< CHANGE THIS FOR EACH SCENARIO RUN!

# Define file names based on your Task 2 output.
# Assumes your .csv files are in the same directory as fnn_sample.py
train_file = f"{current_scenario}_Train.csv"
test_file = f"{current_scenario}_Test.csv"

print(f"--- Starting Anomaly Detection for Scenario: {current_scenario} ---")
print(f"Loading training data from: {train_file}")
print(f"Loading testing data from: {test_file}")


# Standardized Batch Size and Number of Epochs as per project instructions
BatchSize = 32
NumEpoch = 50


# Load separate training and testing datasets created in Task 2
# Ensuring header=None as your extracted CSVs likely don't have headers
# encoding="ISO-8859-1" is often needed for NSL-KDD due to character issues
df_train = pd.read_csv(train_file, header=None, encoding="ISO-8859-1")
df_test = pd.read_csv(test_file, header=None, encoding="ISO-8859-1")


# Separate Features (X) and Labels (y) for both train and test data
# NSL-KDD typically has 41 features, then the attack_type label, then the normal/attack label.
# We are taking all columns except the last two as features (X)
# And the second-to-last column as the binary classification label (y)
X_train_raw = df_train.iloc[:, :-2]
y_train_raw = df_train.iloc[:, -2]

X_test_raw = df_test.iloc[:, :-2]
y_test_raw = df_test.iloc[:, -2]


# Convert raw labels ('normal'/'attack_type') to binary (0 for normal, 1 for attack)
y_train = np.array([0 if label == 'normal' else 1 for label in y_train_raw])
y_test = np.array([0 if label == 'normal' else 1 for label in y_test_raw])


# Encoding Categorical Data (Features)
# Refer to NSL-KDD dataset description (CS-ML-00101) for column definitions.
# Columns 1, 2, 3 (0-indexed) are typically categorical: protocol_type, service, flag
categorical_features_indices = [1, 2, 3] # Verify these indices match your dataset's categorical columns

# Using ColumnTransformer for OneHotEncoding of categorical features
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_indices)],
    remainder='passthrough' # Keep all other columns (numerical features) as they are
)

# Fit the ColumnTransformer on the training data and transform both train and test sets
X_train_processed = ct.fit_transform(X_train_raw)
X_test_processed = ct.transform(X_test_raw) # IMPORTANT: Use transform, not fit_transform on test data

# Convert the processed data to float32 numpy arrays for Keras
X_train = np.array(X_train_processed, dtype=np.float32)
X_test = np.array(X_test_processed, dtype=np.float32)


# Perform Feature Scaling (StandardScaler)
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # Fit scaler on training data, then transform
X_test = sc.transform(X_test)      # IMPORTANT: Use only transform on testing data


#################################################################
# Part 2: Building FNN
#################################################################

# Initialising the ANN (Feed-forward Neural Network)
classifier = Sequential()

# Standardized FNN Architecture and Parameters for consistency across scenarios
input_dim = X_train.shape[1] # Number of features after preprocessing (dynamic)

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))

# Adding the second hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
print(f"\n--- Training FNN for Scenario: {current_scenario} ({NumEpoch} epochs, batch size {BatchSize}) ---")
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch, verbose=1)

# Evaluate the keras model for the provided model and dataset
print(f"\n--- Evaluating Model for Scenario: {current_scenario} ---")
loss, accuracy = classifier.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


#################################################################
# Part 3: Making predictions and evaluating the model
#################################################################

# Predicting the Test set results (get probabilities from sigmoid output)
y_pred_prob = classifier.predict(X_test)

# Applying a standard threshold (0.5) to convert probabilities to binary predictions (0 or 1)
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)

# Summarize the first 5 cases (for debugging/quick check)
print("\n--- Sample Predictions (First 5 Test Cases) ---")
for i in range(5):
    # Access y_pred[i][0] because classifier.predict outputs a 2D array for a single output neuron
    print(f"Predicted: {y_pred[i][0]} (Expected: {y_test[i]})")


# Making the Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Interpretation of Confusion Matrix (Rows: Actual, Columns: Predicted):
# [[True Negatives (Normal classified as Normal)  False Positives (Normal classified as Attack)]
#  [False Negatives (Attack classified as Normal) True Positives (Attack classified as Attack)]]

# Classification Report - provides Precision, Recall, F1-Score for each class
print("\n--- Classification Report ---")
# Define target names for clarity (0: Normal, 1: Attack).
target_names = ['Normal', 'Attack']
print(classification_report(y_test, y_pred, target_names=target_names))


#################################################################
# Part 4 - Visualizing
#################################################################

# Plotting the accuracy history
plt.figure(figsize=(10, 5))
plt.plot(classifierHistory.history['accuracy'], label='train_accuracy')
plt.title(f'Model Accuracy for Scenario {current_scenario}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(f'accuracy_scenario_{current_scenario}.png') # Saves plot with scenario name
plt.show()

# Plotting the loss history
plt.figure(figsize=(10, 5))
plt.plot(classifierHistory.history['loss'], label='train_loss')
plt.title(f'Model Loss for Scenario {current_scenario}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(f'loss_scenario_{current_scenario}.png') # Saves plot with scenario name
plt.show()

print(f"\nCompleted Scenario {current_scenario}. Results saved and plots generated.")
