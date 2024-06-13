import numpy as np
import pandas as pd
import sys
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

app = Flask(__name__)

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        try:
            data = CustomData(
                age=int(request.form["age"]),
                occupation=request.form["occupation"],
                annual_income=float(request.form["annual_income"]),
                num_bank_accounts=int(request.form["num_bank_accounts"]),
                num_credit_card=int(request.form["num_credit_card"]),
                interest_rate=float(request.form["interest_rate"]),
                num_of_loan=int(request.form["num_of_loan"]),
                delay_from_due_date=int(request.form["delay_from_due_date"]),
                num_of_delayed_payment=int(request.form["num_of_delayed_payment"]),
                changed_credit_limit=int(request.form["changed_credit_limit"]),
                num_credit_inquiries=int(request.form["num_credit_inquiries"]),
                credit_mix=request.form["credit_mix"],
                outstanding_debt=float(request.form["outstanding_debt"]),
                credit_utilization_ratio=float(request.form["credit_utilization_ratio"]),
                payment_of_min_amount=request.form["payment_of_min_amount"],
                total_emi_per_month=float(request.form["total_emi_per_month"]),
                amount_invested_monthly=float(request.form["amount_invested_monthly"]),
                payment_behaviour=request.form["payment_behaviour"],
                monthly_balance=float(request.form["monthly_balance"])
            )
            preds_df = data.get_data_df()
            predict_pipeline = PredictPipeline()
            credit_score_encoder = {"Poor": 0, "Standard": 1, "Good": 2}
            prediction_results = predict_pipeline.predict(preds_df)
            prediction_results = [k for k, v in credit_score_encoder.items() if v == prediction_results]
            return render_template("predict.html", prediction_results=prediction_results[0])
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(debug=True)