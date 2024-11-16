# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('Churning_project.csv')

# Step 2: Initial data exploration
print("Initial Dataset Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values Per Column:\n", df.isnull().sum())

# Step 3: Data Cleaning
# Removing duplicates
df = df.drop_duplicates()

# Filling missing values for categorical columns
df['Email'] = df['Email'].fillna('Unknown')
df['Address'] = df['Address'].fillna('Unknown')
df['Avatar'] = df['Avatar'].fillna('Unknown')

# Filling missing values for numerical columns with the mean
df['Avg. Session Length'] = df['Avg. Session Length'].fillna(df['Avg. Session Length'].mean())
df['Time on App'] = df['Time on App'].fillna(df['Time on App'].mean())
df['Time on Website'] = df['Time on Website'].fillna(df['Time on Website'].mean())
df['Length of Membership'] = df['Length of Membership'].fillna(df['Length of Membership'].mean())
df['Yearly Amount Spent'] = df['Yearly Amount Spent'].fillna(df['Yearly Amount Spent'].mean())

# Confirming that all missing values have been handled
print("\nMissing Values Per Column After Filling:\n", df.isnull().sum())

# Step 4: Visualizing the data
# Visualizing missing data before and after cleaning
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=False, ax=ax[0])
ax[0].set_title("Missing Data Before Cleaning")
sns.heatmap(df.isnull(), cbar=False, ax=ax[1])
ax[1].set_title("Missing Data After Cleaning")
plt.show()

# Adding a 'Churn' column based on 'Yearly Amount Spent'
df['Churn'] = df['Yearly Amount Spent'].apply(lambda x: 1 if x < 500 else 0)

# Visualize the distribution of churned vs. non-churned customers
sns.countplot(x='Churn', data=df)
plt.title('Count of Churned vs. Non-Churned Customers')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Preparing the features and target variable
y = df['Churn']
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]

# Train-test split with the updated feature set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=500)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()