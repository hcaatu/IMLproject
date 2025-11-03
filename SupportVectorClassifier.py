from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.svm import SVC

df = pd.read_csv("train.csv")
df = df.drop(["id", "date", "partlybad"], axis=1)

df["class2"] = df["class4"].apply(lambda x: "nonevent" if x == "nonevent" else "event")
y = df["class2"]
print(y.head()) 

X = df.drop(["class4", "class2"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
svm_cm = confusion_matrix(y_test, svm_preds)

print("SVM Accuracy:", svm_acc)

plt.figure(figsize=(4, 3))
sns.heatmap(svm_cm, annot=True, cmap="Blues", fmt="d")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()