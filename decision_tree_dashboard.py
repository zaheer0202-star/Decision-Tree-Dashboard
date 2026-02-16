import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------
# Page Title
# ----------------------------------------
st.title("Decision Tree Hyperparameter Dashboard")
st.write("Water Potability Dataset")

# ----------------------------------------
# Load Data
# ----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("water_potability_preprocessed_labeled.csv")

data = load_data()

X = data.drop(columns=["Potability", "Potability_Label"])
y = data["Potability"]

# ----------------------------------------
# Sidebar Hyperparameter Controls
# ----------------------------------------
st.sidebar.header("Hyperparameter Controls")

criterion = st.sidebar.selectbox(
    "Criterion",
    ("gini", "entropy")
)

max_depth = st.sidebar.slider(
    "Max Depth",
    1, 20, 5
)

min_samples_split = st.sidebar.slider(
    "Min Samples Split",
    2, 50, 2
)

min_samples_leaf = st.sidebar.slider(
    "Min Samples Leaf",
    1, 20, 1
)

test_size = st.sidebar.slider(
    "Test Size (%)",
    10, 50, 30
) / 100

# ----------------------------------------
# Train/Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# ----------------------------------------
# Train Model
# ----------------------------------------
model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------
# Performance Metrics
# ----------------------------------------
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.subheader("Model Performance")
st.write(f"Training Accuracy: {train_acc:.4f}")
st.write(f"Test Accuracy: {test_acc:.4f}")
st.write(f"Tree Depth: {model.get_depth()}")
st.write(f"Number of Leaves: {model.get_n_leaves()}")

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, model.predict(X_test))

fig_cm, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-potable", "Potable"],
    yticklabels=["Non-potable", "Potable"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig_cm)

# ----------------------------------------
# Decision Tree Visualization
# ----------------------------------------
st.subheader("Decision Tree Structure")

fig_tree = plt.figure(figsize=(18, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Non-potable", "Potable"],
    filled=True,
    rounded=True,
    fontsize=8
)

st.pyplot(fig_tree)
