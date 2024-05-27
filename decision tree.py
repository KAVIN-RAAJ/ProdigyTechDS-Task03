import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data
np.random.seed(0)
n_samples = 1000
age = np.random.randint(18, 80, n_samples)
income = np.random.randint(20000, 150000, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
behavior = np.random.choice(['High', 'Medium', 'Low'], n_samples)
purchased = np.random.choice([0, 1], n_samples)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Gender': gender,
    'Behavior': behavior,
    'Purchased': purchased
})

# Step 2: Data preprocessing
# Encoding categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Splitting into features and target variable
X = data.drop(columns=["Purchased"])
y = data["Purchased"]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the decision tree classifier with limited depth
max_depth = 3  # Set the maximum depth here
model = DecisionTreeClassifier(max_depth=max_depth)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Visualize the decision tree
plt.figure(figsize=(10,6))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Purchased', 'Purchased'])
plt.show()
