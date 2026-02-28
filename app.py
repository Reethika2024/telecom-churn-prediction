
# coding: utf-8

import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load base dataset (used for column alignment)
df_1 = pd.read_csv("first_telc.csv")

# Load trained model
model = pickle.load(open("model.sav", "rb"))

# Create tenure groups for training data
labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
df_1["tenure_group"] = pd.cut(
    df_1["tenure"].astype(int),
    bins=range(1, 80, 12),
    right=False,
    labels=labels
)
df_1.drop(columns=["tenure"], inplace=True)

# Save training dummy columns (VERY IMPORTANT)
train_columns = pd.get_dummies(
    df_1[
        [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod",
            "tenure_group"
        ]
    ]
).columns


@app.route("/")
def loadPage():
    return render_template("home.html")


@app.route("/", methods=["POST"])
def predict():
    try:
        # Collect inputs
        data = [[
            request.form["query1"],
            request.form["query2"],
            request.form["query3"],
            request.form["query4"],
            request.form["query5"],
            request.form["query6"],
            request.form["query7"],
            request.form["query8"],
            request.form["query9"],
            request.form["query10"],
            request.form["query11"],
            request.form["query12"],
            request.form["query13"],
            request.form["query14"],
            request.form["query15"],
            request.form["query16"],
            request.form["query17"],
            request.form["query18"],
            request.form["query19"]
        ]]

        new_df = pd.DataFrame(
            data,
            columns=[
                "SeniorCitizen", "MonthlyCharges", "TotalCharges",
                "gender", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod", "tenure"
            ]
        )

        # Convert numeric columns
        new_df["SeniorCitizen"] = new_df["SeniorCitizen"].astype(int)
        new_df["MonthlyCharges"] = new_df["MonthlyCharges"].astype(float)
        new_df["TotalCharges"] = new_df["TotalCharges"].astype(float)
        new_df["tenure"] = new_df["tenure"].astype(int)

        # Combine with base data to keep consistency
        df_2 = pd.concat([df_1.copy(), new_df], ignore_index=True)

        # Create tenure group
        df_2["tenure_group"] = pd.cut(
            df_2["tenure"].astype(int),
            bins=range(1, 80, 12),
            right=False,
            labels=labels
        )
        df_2.drop(columns=["tenure"], inplace=True)

        # One-hot encode
        df_dummies = pd.get_dummies(
            df_2[
                [
                    "gender", "SeniorCitizen", "Partner", "Dependents",
                    "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod",
                    "tenure_group"
                ]
            ]
        )

        # ALIGN COLUMNS (CRITICAL FIX)
        df_dummies = df_dummies.reindex(columns=train_columns, fill_value=0)

        # Predict
        prediction = model.predict(df_dummies.tail(1))[0]
        probability = model.predict_proba(df_dummies.tail(1))[0][1] * 100

        if prediction == 1:
            output1 = "This customer is likely to be churned!!"
        else:
            output1 = "This customer is likely to continue!!"

        output2 = f"Confidence: {probability:.2f}%"

        return render_template("home.html", output1=output1, output2=output2)

    except Exception as e:
        return render_template(
            "home.html",
            output1="Prediction Error",
            output2=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)

