# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import streamlit as st

# Streamlit app title
st.title("Customer Churn Prediction Model")

# Step 1: Load the dataset
df = pd.read_csv('Churning_project.csv')
st.write("### Initial Dataset")
st.write(df.head())

# Step 2: Initial data exploration
st.write("### Dataset Information")
st.write("Shape of the dataset:", df.shape)
st.write("Data Types and Missing Values:")
st.write(df.info())
st.write("Missing Values Per Column:", df.isnull().sum())

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
st.write("### Missing Values After Data Cleaning")
st.write(df.isnull().sum())

# Step 4: Feature Engineering
# Adding a 'Churn' column based on 'Yearly Amount Spent'
df['Churn'] = df['Yearly Amount Spent'].apply(lambda x: 1 if x < 500 else 0)
st.write("### Updated Dataset with 'Churn' Column")
st.write(df.head())

# Step 5: Data Visualization
# Visualize the distribution of churned vs. non-churned customers
st.write("### Churn Distribution")
sns.countplot(x='Churn', data=df)
plt.title('Count of Churned vs. Non-Churned Customers')
plt.xlabel('Churn')
plt.ylabel('Count')
st.pyplot(plt.gcf())  # Displaying the plot in Streamlit

# Step 6: Model Training
# Splitting the dataset into features (X) and target (y)
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
# Displaying evaluation metrics
st.write("### Model Performance Metrics")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")

# Step 8: Confusion Matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix with a specified figure
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)  # Display the confusion matrix plot

# Display classification report
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))