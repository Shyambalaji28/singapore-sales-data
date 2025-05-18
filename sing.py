# streamlit_app.py
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------- Configuration ----------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Analysis",
    page_icon="🏨",
    layout="wide"
)

# ---------------------- Data Loading ----------------------
DATA_PATH = Path('C:/shyam balaji/combined_resale_prices.csv')
if not DATA_PATH.exists():
    st.error(f"Resale data file not found at {DATA_PATH}. Please upload the CSV to /mnt/data/.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Preprocess: derive storey_median and remaining_lease if not present
if 'storey_median' not in df.columns and 'storey_range' in df.columns:
    df['storey_median'] = (
        df['storey_range']
          .str.split(' TO ')
          .apply(lambda x: np.median([float(i) for i in x]))
    )
if 'remaining_lease' not in df.columns and 'lease_commence_date' in df.columns:
    df['remaining_lease'] = 99 - (2023 - df['lease_commence_date'])

# Identify numeric columns for EDA/modeling
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# --------------------- Sidebar ------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["About", "EDA", "Preprocessing", "Modeling"],
        icons=["info-circle", "bar-chart", "sliders", "cpu"],
        default_index=0
    )

# ---------------------- About -------------------------
if selected == "About":
    st.title("Singapore Resale Flat Prices Analysis")
    st.markdown(
        "This app provides Exploratory Data Analysis, preprocessing transformations, "
        "and a simple modeling pipeline on the Singapore HDB resale dataset."
    )
    st.write("**Dataset shape:**", df.shape)
    st.write(df.head())

# ----------------------- EDA --------------------------
elif selected == "EDA":
    st.header("Exploratory Data Analysis")
    col       = st.selectbox("Select numeric column", numeric_cols)
    transform = st.checkbox("Log Transform", value=False)
    data = df[col].dropna()
    if transform:
        data = data[data > 0]
        data = np.log(data)
    fig, ax = plt.subplots()
    sns.boxplot(x=data, ax=ax)
    ax.set_title(f"Boxplot of {'log ' if transform else ''}{col}")
    st.pyplot(fig)
    st.write(data.describe())

# -------------------- Preprocessing -------------------
elif selected == "Preprocessing":
    st.header("Data Preprocessing")
    st.markdown("Apply log transformation to skewed features and scale them.")
    to_log   = st.multiselect("Columns to log-transform (positive only)", numeric_cols)
    to_scale = st.multiselect("Columns to scale", numeric_cols)
    if st.button("Run transformations"):
        df_proc = df.copy()
        # Log-transform
        for c in to_log:
            df_proc = df_proc[df_proc[c] > 0]
            df_proc[c] = np.log(df_proc[c])
        # Scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_proc[to_scale] = scaler.fit_transform(df_proc[to_scale])
        st.write("Transformed sample:")
        st.write(df_proc.head())
        st.write("Scaled columns summary:")
        st.write(df_proc[to_scale].describe().round(2))

# -------------------- Modeling -----------------------
elif selected == "Modeling":
    st.header("Modeling")
    st.markdown("Train a Decision Tree Regressor on your processed data.")
    default_target = 'resale_price' if 'resale_price' in numeric_cols else numeric_cols[0]
    target   = st.selectbox("Select target variable", numeric_cols, index=numeric_cols.index(default_target))
    features = st.multiselect("Select feature columns", [c for c in numeric_cols if c != target])
    if st.button("Train Model"):
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        X = df[features].dropna()
        y = df.loc[X.index, target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.2f}  |  **R²:** {r2:.2f}")

        st.markdown("**Feature importances:**")
        imp = dict(zip(features, model.feature_importances_))
        st.write(imp)
