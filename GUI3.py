import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np  # Added numpy import
import xgboost as xgb  # Importing XGBoost
from sklearn.metrics import r2_score

# --- Model Selection ---
st.title("Wall Characteristics Prediction")

# Model selection dropdown with only RF and XGBoost (ANN removed)
model_choice = st.selectbox(
    "Select Prediction Model",
    ("Random Forest", "XGBoost")
)

# --- Load Selected Model ---
@st.cache_resource
def load_model(model_choice):
    if model_choice == "Random Forest":
        model_file = "rf_model.pkl"
    elif model_choice == "XGBoost":
        model_file = "xgb_model.pkl"
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(model_choice)

# --- Wall Parameter Inputs ---
st.header("Wall Parameters")
cols = st.columns(3)
lw = cols[0].number_input("Length (lw)", value=1400.0, format="%.5f")
hw = cols[1].number_input("Height (hw)", value=2000.0, format="%.5f")
tw = cols[2].number_input("Thickness (tw)", value=150.0, format="%.5f")

st.header("Material Properties")
cols = st.columns(3)
fc = cols[0].number_input("Concrete compressive strength (f'c)", value=46.8, format="%.5f")
fyt = cols[1].number_input("Transverse web reinforcement yield strength (fyt)", value=490.0, format="%.5f")
fysh = cols[2].number_input("Transverse boundary reinforcement yield strength (fysh)", value=490.0, format="%.5f")

cols = st.columns(2)
fyl = cols[0].number_input("Vertical web reinforcement yield strength (fyl)", value=490.0, format="%.5f")
fybl = cols[1].number_input("Vertical boundary reinforcement yield strength (fybl)", value=598.0, format="%.5f")

st.header("Reinforcement Ratios")
cols = st.columns(2)
pt = cols[0].number_input("Transverse web reinforcement ratio (ρt)", value=0.005236, format="%.5f")
psh = cols[1].number_input("Transverse boundary reinforcement ratio (ρsh)", value=0.004725, format="%.5f")

cols = st.columns(2)
pl = cols[0].number_input("Vertical web reinforcement ratio (ρl)", value=0.005236, format="%.5f")
pbl = cols[1].number_input("Vertical boundary reinforcement ratio (ρbl)", value=0.004675, format="%.5f")

st.header("Global Behavior Parameters")
cols = st.columns(3)
p_agfc = cols[0].number_input("Axial Load Ratio (P/Agf'c)", value=0.01, format="%.5f")
b0 = cols[1].number_input("Boundary element width (b0)", value=150.0, format="%.5f")
db = cols[2].number_input("Boundary element length (db)", value=80.0, format="%.5f")

cols = st.columns(3)
s_db = cols[0].number_input("Hoop spacing / Boundary element length (s/db)", value=2.5, format="%.5f")
ar = cols[1].number_input("Aspect Ratio (AR)", value=1.428571429, format="%.5f")
m_vlw = cols[2].number_input("Shear span ratio (M/Vlw)", value=1.428571429, format="%.5f")

# --- Drift Ratios ---
st.header("Drift Ratios")
drift_default_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
edited_drifts = []
cols = st.columns(len(drift_default_values))
for i, default_val in enumerate(drift_default_values):
    edited_val = cols[i].number_input(f"Drift {i+1}", value=default_val, step=0.01, format="%.5f", key=f"drift_{i}")
    edited_drifts.append(edited_val)

# --- Predict Button ---
if st.button("Predict"):
    try:
        # Wall parameter inputs (no 'DampingRatio' in here)
        wall_inputs = {
            'lw': lw, 'hw': hw, 'tw': tw, "f′c": fc,
            'fyt': fyt, 'fysh': fysh, 'fyl': fyl, 'fybl': fybl,
            'ρt': pt, 'ρsh': psh, 'ρl': pl, 'ρbl': pbl,
            "P/(Agf′c)": p_agfc, 'b0': b0, 'db': db, 's/db': s_db,
            'AR': ar, 'M/Vlw': m_vlw
        }

        # Prepare input data for prediction (no 'DampingRatio' in here)
        input_df = pd.DataFrame(wall_inputs, index=[0])  # Start with a single row
        input_df = input_df.loc[np.repeat(input_df.index.values, len(edited_drifts))]  # Repeat rows for each drift
        input_df['DriftRatio'] = edited_drifts  # Set the 'DriftRatio' values as a new column

        # Check model type and predict accordingly
        if model_choice == "XGBoost":
            damping_results = model.predict(input_df)  # Use DataFrame directly for XGBoost
        else:
            damping_results = model.predict(input_df)  # RandomForest or other sklearn model prediction

        # Show Prediction Results
        st.success("✅ Prediction Successful!")

        # Show Table with Results
        st.markdown("### Damping Ratio Table")
        result_df = pd.DataFrame([damping_results], columns=edited_drifts, index=["Damping Ratio"])
        st.dataframe(result_df.T)

        # Show Plot
        plot_df = pd.DataFrame({
            "Drift Ratio": edited_drifts,
            "Damping Ratio": damping_results
        })
        fig = px.line(plot_df, x="Drift Ratio", y="Damping Ratio", markers=True, title="Damping Ratio vs Drift Ratio")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
