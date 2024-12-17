import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Hong Kong Rent Trends with Regression", layout="wide")
st.title("Hong Kong Rent Price Trends by Class and Region with Regression Models")

# Load and clean data
file_path = './1.1M.csv'
data = pd.read_csv(file_path)

cleaned_data = data.iloc[1:, :].dropna(how='all', axis=1)
cleaned_data.columns = [
    "Month", "Class_A_Hong_Kong", "Remarks_A_HK", "Class_A_Kowloon",
    "Remarks_A_Kowloon", "Class_A_New_Territories", "Remarks_A_NT",
    "Class_B_Hong_Kong", "Remarks_B_HK", "Class_B_Kowloon", "Remarks_B_Kowloon",
    "Class_B_New_Territories", "Remarks_B_NT", "Class_C_Hong_Kong",
    "Remarks_C_HK", "Class_C_Kowloon", "Remarks_C_Kowloon", "Class_C_New_Territories",
    "Remarks_C_NT", "Class_D_Hong_Kong", "Remarks_D_HK", "Class_D_Kowloon",
    "Remarks_D_Kowloon", "Class_D_New_Territories", "Remarks_D_NT", "Class_E_Hong_Kong",
    "Remarks_E_HK", "Class_E_Kowloon", "Remarks_E_Kowloon", "Class_E_New_Territories",
    "Remarks_E_NT"
]

columns_to_drop = [col for col in cleaned_data.columns if "Remarks" in col]
cleaned_data = cleaned_data.drop(columns=columns_to_drop)

for col in cleaned_data.columns[1:]:
    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')

melted_data = cleaned_data.melt(id_vars=["Month"], 
                                var_name="Category_Region", 
                                value_name="Rent_Price")

melted_data[['Class', 'Region']] = melted_data['Category_Region'].str.extract(r'(Class_[A-E])_(.*)')
melted_data = melted_data.dropna(subset=["Rent_Price", "Class", "Region"])

try:
    melted_data['Month'] = pd.to_datetime(melted_data['Month'])
    melted_data = melted_data.sort_values(by='Month')
except:
    st.write("Could not convert Month to datetime. Check your date format.")

if st.checkbox("Show cleaned raw data"):
    st.write(cleaned_data.head())

# Sidebar filters
unique_classes = melted_data['Class'].unique()
unique_regions = melted_data['Region'].unique()

selected_class = st.sidebar.selectbox("Select Class:", sorted(unique_classes))
selected_region = st.sidebar.selectbox("Select Region:", sorted(unique_regions))

filtered_data = melted_data[(melted_data['Class'] == selected_class) & (melted_data['Region'] == selected_region)]

# Prepare first figure: Rent trend over time
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=filtered_data, x="Month", y="Rent_Price", ax=ax)
ax.set_title(f"Average Rent Price Over Time ({selected_class}, {selected_region})", fontsize=14)
ax.set_xlabel("Month")
ax.set_ylabel("Rent Price (HK$/m²)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

st.header("Regression Model on Selected Class & Region")

# Removed Lasso Regression from the options
model_type = st.sidebar.radio(
    "Select Model Type:",
    ("Linear Regression", "Polynomial Regression")
)

use_log_transform = st.sidebar.checkbox("Apply log transform to Rent_Price?", value=False)

if model_type == "Polynomial Regression":
    poly_degree = st.sidebar.slider("Select polynomial degree:", 1, 5, 2)

subset = filtered_data.copy()
subset = subset.sort_values(by='Month')
subset['Month_Numeric'] = np.arange(len(subset))
X = subset['Month_Numeric'].values.reshape(-1, 1)
y = subset['Rent_Price'].values

if use_log_transform:
    y = np.log(y + 1e-6)

if model_type == "Linear Regression":
    model = LinearRegression()
else:  # Polynomial Regression
    poly = PolynomialFeatures(degree=poly_degree)
    X = poly.fit_transform(X)
    model = LinearRegression()

model.fit(X, y)
y_pred = model.predict(X)

if use_log_transform:
    y = np.exp(y) - 1e-6
    y_pred = np.exp(y_pred) - 1e-6

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Prepare second figure: Regression fit
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.scatter(subset['Month_Numeric'], subset['Rent_Price'], color='blue', label="Actual Data", alpha=0.6)

if model_type == "Polynomial Regression":
    X_sorted = np.sort(subset['Month_Numeric'].values)
    X_line = X_sorted.reshape(-1, 1)
    X_line_poly = poly.transform(X_line)
    y_line_pred = model.predict(X_line_poly)
    if use_log_transform:
        y_line_pred = np.exp(y_line_pred) - 1e-6
    ax2.plot(X_line, y_line_pred, color='red', label="Regression Line")
else:
    ax2.plot(subset['Month_Numeric'], y_pred, color='red', label="Regression Line")

ax2.set_title(f"Regression Fit ({selected_class}, {selected_region})", fontsize=14)
ax2.set_xlabel("Time (Numeric)")
ax2.set_ylabel("Rent Price (HK$/m²)")
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Create two columns to place figures side by side
col1, col2 = st.columns((1,1))
col1.pyplot(fig)

col2.subheader("Regression Results")
col2.write(f"**Model:** {model_type}")
if model_type == "Polynomial Regression":
    col2.write(f"**Polynomial Degree:** {poly_degree}")
if use_log_transform:
    col2.write("**Target was log-transformed for regression.**")

col2.write(f"**MSE:** {mse:.2f}")
col2.write(f"**R² Score:** {r2:.2f}")
col2.pyplot(fig2)

st.markdown("""
**Note:**  
- Linear Regression: Fits a straight line.  
- Polynomial Regression: Allows fitting curves by adding polynomial terms.  
- Log Transform: Can help if data grows exponentially or if errors are multiplicative.
""")
