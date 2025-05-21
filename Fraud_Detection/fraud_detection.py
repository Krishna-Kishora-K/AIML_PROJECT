import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle

# Load data
data = pd.read_csv("creditcard.csv")

# Preprocessing
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data['normalizedTime'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)

X = data.drop('Class', axis=1)
y = data['Class']

# Balance dataset
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0].sample(n=len(fraud), random_state=1)
balanced_data = pd.concat([fraud, normal])

X_balanced = balanced_data.drop('Class', axis=1)
y_balanced = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=1)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
y_scores = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
