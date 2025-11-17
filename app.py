from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# ---------- Load ML model ----------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLS = [
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

# ---------- Load dataset for dashboard ----------
def parse_spending(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("₹", "").replace(" ", "").replace("Rs.", "").replace("INR", "")
    s = s.replace("–", "-")
    if "-" in s:
        a, b = s.split("-")[:2]
        a = "".join([c for c in a if c.isdigit()])
        b = "".join([c for c in b if c.isdigit()])
        if a and b:
            return (float(a) + float(b)) / 2
    d = "".join([c for c in s if c.isdigit()])
    return float(d) if d else np.nan

df = pd.read_excel(
    "Canteen_Dietary_Habits 12.xlsx",
    sheet_name="Canteen_Dietary_Habits_Analysis"
)
df.columns = df.columns.str.strip()

df["Visit_Time_Parsed"] = pd.to_datetime(df["Actual Time of Visit"], errors="coerce")
df["Visit_Hour"] = df["Visit_Time_Parsed"].dt.hour

vis_map = {"Once": 1, "Twice": 2, "Thrice": 3, "More than 3": 4}
df["Visits_Num"] = df["Number of Visits per Day"].map(vis_map).fillna(1)

df["Spending_Mid"] = df["Average Spending (₹)"].apply(parse_spending)
df["Peak_Label_Binary"] = df["Peak Hour Label"].map({"Yes": 1, "No": 0})

for col in ["Preferred Food Type", "Meal Type", "Veg/Non-Veg/Vegan", "Day of Week", "Reason for Visit"]:
    df[col] = df[col].fillna("Unknown")

# ---------- KPI COUNTERS ----------
total_visits = int(len(df))
avg_spend = float(df["Spending_Mid"].dropna().mean())
avg_crowd = float(df["Crowdedness Rating (1–5)"].dropna().mean())
peak_share = float(df["Peak_Label_Binary"].mean()) if df["Peak_Label_Binary"].notna().any() else 0.0

# ---------- Chart Data ----------
visits_by_hour = df.groupby("Visit_Hour").size().sort_index().to_dict()
peak_prob_by_hour = df.groupby("Visit_Hour")["Peak_Label_Binary"].mean().fillna(0).to_dict()
food_type_counts = df["Preferred Food Type"].value_counts().head(8).to_dict()
diet_counts = df["Veg/Non-Veg/Vegan"].value_counts().to_dict()

dashboard_data = {
    "total_visits": total_visits,
    "avg_spend": round(avg_spend, 2),
    "avg_crowd": round(avg_crowd, 2),
    "peak_share": round(peak_share * 100, 1),
    "visits_by_hour": visits_by_hour,
    "peak_prob_by_hour": {str(k): float(v) for k, v in peak_prob_by_hour.items()},
    "food_type_counts": food_type_counts,
    "diet_counts": diet_counts,
}


# ---------- PREDICTION HELPERS ----------
def classify_rush(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

def staffing_recommendation(rush):
    if rush == "High":
        return "Recommended: 3–4 counter staff + 3 kitchen staff."
    elif rush == "Medium":
        return "Recommended: 2 counter staff + 2–3 kitchen staff."
    return "Recommended: 1 counter staff + 2 kitchen staff is enough."

def prep_recommendation(meal, food, rush):
    if rush == "High":
        return f"For {meal} focusing on {food}, prepare 30–40% extra stock."
    elif rush == "Medium":
        return f"For {meal} and {food}, keep moderate extra portions ready."
    return f"For {meal}/{food}, keep basic stock to avoid wastage."


# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template(
        "index.html",
        dashboard_data=json.dumps(dashboard_data)
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    row = {
        "Visits_Num": float(data["visits_per_day"]),
        "Visit_Hour": float(data["visit_hour"]),
        "Duration of Stay (minutes)": float(data["stay_duration"]),
        "Preferred Food Type": data["food_type"],
        "Meal Type": data["meal_type"],
        "Veg/Non-Veg/Vegan": data["diet_type"],
        "Spending_Mid": float(data["spending"]),
        "Distance from Canteen": data["distance"],
        "Reason for Visit": data["reason"],
        "Day of Week": data["day"],
    }

    input_df = pd.DataFrame([row], columns=FEATURE_COLS)

    prob_peak = model.predict_proba(input_df)[0][1]
    pred_peak = model.predict(input_df)[0]

    rush = classify_rush(prob_peak)
    staff = staffing_recommendation(rush)
    prep = prep_recommendation(data["meal_type"], data["food_type"], rush)

    return jsonify({
        "prediction": int(pred_peak),
        "confidence": float(prob_peak),
        "rush_level": rush,
        "staffing": staff,
        "prep": prep
    })


if __name__ == "__main__":
    app.run(debug=True)
