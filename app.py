import streamlit as st
import pandas as pd
import joblib

# Data & modellen laden
df = pd.read_excel("verzuimdata_simulatie.xlsx")
clf = joblib.load("model_classification.pkl")
reg = joblib.load("model_regression.pkl")
features = joblib.load("model_features.pkl")

st.title("🧠 AI VerzuimVoorspeller - Volledig Dashboard")

# Functie voor model-input
def prepare(df):
    X = pd.get_dummies(df, drop_first=True)
    for col in features:
        if col not in X: X[col] = 0
    return X[features]

df["Verzuimkans"] = reg.predict(prepare(df))
df["VerwachteVerzuimdagen"] = clf.predict_proba(prepare(df))[:,1]
df["Risicoscore"] = df["Verzuimkans"] * df["VerwachteVerzuimdagen"]

# Sidebar filters
functie = st.sidebar.selectbox("Functie", ["Alle"] + sorted(df["Functie"].unique()))
contract = st.sidebar.selectbox("ContractType", ["Alle"] + sorted(df["ContractType"].unique()))

mask = [(df["Functie"] == functie) if functie!="Alle" else True,
        (df["ContractType"] == contract) if contract!="Alle" else True]
df_filt = df[mask[0] & mask[1]]

# Kritieke medewerkers
st.header("🚨 Kritieke Medewerkers")
st.dataframe(df_filt.sort_values("Risicoscore", ascending=False).
             loc[:, ["Naam","Functie","ContractType","Verzuimkans","VerwachteVerzuimdagen","Risicoscore"]])

# Detail & uitleg factor
sel = st.selectbox("Medewerker", df_filt["Naam"])
rec = df_filt[df_filt["Naam"]==sel]
st.subheader("📋 Detail gegevens")
st.write(rec.T)

st.subheader("📈 Belangrijkste factoren")
import matplotlib.pyplot as plt
fimp = pd.Series(clf.feature_importances_, index=features)
st.bar_chart(fimp.sort_values(ascending=False).head(5))
