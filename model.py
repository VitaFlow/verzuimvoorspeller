import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib

# Dataset inladen
df = pd.read_excel("verzuimdata_simulatie.xlsx")

# Voorbereiden
feature_cols = [
    "Leeftijd", "Geslacht", "Functie", "ContractType",
    "Dienstjaren", "Werkuren"
]
X = pd.get_dummies(df[feature_cols], drop_first=True)
joblib.dump(X.columns.tolist(), "model_features.pkl")

# Doelvariabelen
y_class = df["VerzuimVolgendJaar"]
y_reg = df["VerwachteVerzuimdagen"]

# Train/test split
X_train, _, y_class_train, _ = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, X_test, _, y_reg_train = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Modellen trainen
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_class_train)
reg = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_reg_train)

# Opslaan
joblib.dump(clf, "model_classification.pkl")
joblib.dump(reg, "model_regression.pkl")
print("✅ Modellen succesvol getraind en opgeslagen.")
