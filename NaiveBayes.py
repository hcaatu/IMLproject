from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("train.csv")
df = df.drop(["id", "date", "partlybad"], axis=1)

y = df["class4"]
X = df.drop("class4", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds)
nb_cm = confusion_matrix(y_test, nb_preds)

print("Naive Bayes Accuracy:", nb_acc)

plt.figure(figsize=(4,3))
sns.heatmap(nb_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()