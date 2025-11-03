from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("train.csv")
df = df.drop(["id", "date", "partlybad"], axis=1)

y = df["class4"]
X = df.drop("class4", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_preds)
knn_cm = confusion_matrix(y_test, knn_preds)

print("kNN Accuracy:", knn_acc)

plt.figure(figsize=(4,3))
sns.heatmap(knn_cm, annot=True, cmap="Blues", fmt="d")
plt.title("kNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()