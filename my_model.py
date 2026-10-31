import pandas as pd
data = pd.read_csv("crop_data.csv")
print(data.shape)

print(data.isnull().sum())
print("data duplicates: ", data.duplicated().sum())
print(data["label"].value_counts().head(3))

x = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = data["label"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(x_train.shape, x_test.shape)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(x_train, y_train)
print("training accuracy:", forest.score(x_train, y_train))

print("test accuracy:", forest.score(x_test, y_test))

from sklearn.metrics import classification_report

predictions = forest.predict(x_test)
print(classification_report(y_test, predictions))

importances = pd.Series(forest.feature_importances_, index=x.columns)
print(importances.sort_values(ascending=False).round(3))

new_record = pd.DataFrame([[90, 42, 43, 20.8, 82, 6.5, 203]], columns=x.columns)
print("predicted crop:", forest.predict(new_record)[0])


import pickle

with open("best_model.pkl", "wb") as file:
    pickle.dump(forest, file)
    print("model succcessfully saved")