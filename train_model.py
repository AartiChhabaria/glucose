# Import important tools (libraries)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

# 1. Load dataset from CSV
data = pd.read_csv("fake_data.csv")   # ✅ check file name
print("✅ Dataset loaded successfully")
print(data.head())   # show first 5 rows

# 2. Split into input (X) and output (y)
X = data[["optical", "impedance", "pulse", "temperature"]]  # inputs
y = data["glucose"]  # output

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# 5. Test the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("📊 Mean Absolute Error:", mae)

# 6. Save the trained model
joblib.dump(model, "glucose_model.pkl")
print("✅ Model saved as glucose_model.pkl")

