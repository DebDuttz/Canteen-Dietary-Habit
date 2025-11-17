import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def parse_spending(x):
    """Convert '₹40–60' or '100–150' into numeric midpoint."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace("₹", "").replace(" ", "").replace("Rs.", "").replace("INR", "")
    s = s.replace("–", "-")
    if "-" in s:
        parts = [p for p in s.split("-") if p]
        if len(parts) >= 2:
            a = "".join(ch for ch in parts[0] if ch.isdigit())
            b = "".join(ch for ch in parts[1] if ch.isdigit())
            if a and b:
                return (float(a) + float(b)) / 2
    digits = "".join(ch for ch in s if ch.isdigit())
    return float(digits) if digits else np.nan


# 1. Load dataset
df = pd.read_excel(
    "Canteen_Dietary_Habits 12.xlsx",
    sheet_name="Canteen_Dietary_Habits_Analysis"
)
df.columns = df.columns.str.strip()

# 2. Visit hour from "Actual Time of Visit"
df["Visit_Time_Parsed"] = pd.to_datetime(df["Actual Time of Visit"], errors="coerce")
df["Visit_Hour"] = df["Visit_Time_Parsed"].dt.hour

# 3. Numeric visits per day (map text to numbers)
vis_map = {"Once": 1, "Twice": 2, "Thrice": 3, "More than 3": 4}
df["Visits_Num"] = df["Number of Visits per Day"].map(vis_map).fillna(1)

# 4. Spending midpoint
df["Spending_Mid"] = df["Average Spending (₹)"].apply(parse_spending)

# 5. Target: Peak hour label → binary
df["Peak_Label_Binary"] = df["Peak Hour Label"].map({"Yes": 1, "No": 0})

# 6. Drop rows with missing crucial fields
df = df.dropna(subset=["Visit_Hour", "Spending_Mid", "Peak_Label_Binary"])

# 7. Features we will use
feature_cols = [
    "Visits_Num",
    "Visit_Hour",
    "Duration of Stay (minutes)",
    "Preferred Food Type",
    "Meal Type",
    "Veg/Non-Veg/Vegan",
    "Spending_Mid",
    "Distance from Canteen",
    "Reason for Visit",
    "Day of Week",
]

X = df[feature_cols]
y = df["Peak_Label_Binary"]

numeric_features = ["Visits_Num", "Visit_Hour", "Duration of Stay (minutes)", "Spending_Mid"]
categorical_features = [c for c in feature_cols if c not in numeric_features]

# 8. Preprocessing + Model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("pre", preprocessor),
    ("model", model)
])

# 9. Train-test split and fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)

# 10. Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved to model.pkl")
