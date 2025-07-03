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


current_scenario = 'SA' 

train_file = f"{current_scenario}_Train.csv"
test_file = f"{current_scenario}_Test.csv"

print(f"--- Starting Anomaly Detection for Scenario: {current_scenario} ---")
print(f"Loading training data from: {train_file}")
print(f"Loading testing data from: {test_file}")

BatchSize = 32 # Yash Patel: Standardized batch size as per project instructions
NumEpoch = 50 # Yash Patel: Standardized number of epochs as per project instructions

df_train = pd.read_csv(train_file, header=None, encoding="ISO-8859-1")
df_test = pd.read_csv(test_file, header=None, encoding="ISO-8859-1")

X_train_raw = df_train.iloc[:, :-2]
y_train_raw = df_train.iloc[:, -2]

X_test_raw = df_test.iloc[:, :-2]
y_test_raw = df_test.iloc[:, -2]

y_train = np.array([0 if label == 'normal' else 1 for label in y_train_raw]) # Yash Patel: Converted raw labels to binary (0 for normal, 1 for attack)
y_test = np.array([0 if label == 'normal' else 1 for label in y_test_raw]) # Yash Patel: Converted raw labels to binary (0 for normal, 1 for attack)

categorical_features_indices = [1, 2, 3] # Yash Patel: Defined indices for categorical features

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_indices)],
    remainder='passthrough' 
)

X_train_processed = ct.fit_transform(X_train_raw)
X_test_processed = ct.transform(X_test_raw) 

X_train = np.array(X_train_processed, dtype=np.float32) # Yash Patel: Converted processed data to float32 numpy array for Keras
X_test = np.array(X_test_processed, dtype=np.float32) # Yash Patel: Converted processed data to float32 numpy array for Keras

sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)      

#################################################################
# Part 2: Building FNN
#################################################################

classifier = Sequential()

input_dim = X_train.shape[1] 

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim)) # Yash Patel: Added input layer and first hidden layer

classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) # Yash Patel: Added second hidden layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Yash Patel: Added output layer

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Yash Patel: Compiled the ANN

print(f"\n--- Training FNN for Scenario: {current_scenario} ({NumEpoch} epochs, batch size {BatchSize}) ---")
classifierHistory = classifier.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch, verbose=1) # Yash Patel: Fitted the ANN to the Training set

print(f"\n--- Evaluating Model for Scenario: {current_scenario} ---")
loss, accuracy = classifier.evaluate(X_test, y_test, verbose=0) # Yash Patel: Evaluated the Keras model
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


#################################################################
# Part 3: Making predictions and evaluating the model
#################################################################

y_pred_prob = classifier.predict(X_test) # Yash Patel: Predicted the Test set results

threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int) # Yash Patel: Applied threshold to convert probabilities to binary predictions

print("\n--- Sample Predictions (First 5 Test Cases) ---")
for i in range(5):
    print(f"Predicted: {y_pred[i][0]} (Expected: {y_test[i]})")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred) # Yash Patel: Generated the Confusion Matrix
print(cm)

print("\n--- Classification Report ---")
target_names = ['Normal', 'Attack'] # Yash Patel: Defined target names for clarity
print(classification_report(y_test, y_pred, target_names=target_names)) # Yash Patel: Generated the Classification Report


#################################################################
# Part 4 - Visualizing
#################################################################

plt.figure(figsize=(10, 5))
plt.plot(classifierHistory.history['accuracy'], label='train_accuracy')
plt.title(f'Model Accuracy for Scenario {current_scenario}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(f'accuracy_scenario_{current_scenario}.png') # Yash Patel: Saved accuracy plot
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(classifierHistory.history['loss'], label='train_loss')
plt.title(f'Model Loss for Scenario {current_scenario}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig(f'loss_scenario_{current_scenario}.png') # Yash Patel: Saved loss plot
plt.show()

print(f"\nCompleted Scenario {current_scenario}. Results saved and plots generated.")
