import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor = load_object(file_path="artifact/preprocessor.pkl")
            model = load_object(file_path="artifact/model.pkl")
            features = preprocessor.transform(features)
            return model.predict(features)
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 age: int,
                 occupation: str,
                 annual_income: float,
                 num_bank_accounts: int,
                 num_credit_card: int,
                 interest_rate: float,
                 num_of_loan: int,
                 delay_from_due_date: int,
                 num_of_delayed_payment: int,
                 changed_credit_limit: int,
                 num_credit_inquiries: int,
                 credit_mix: str,
                 outstanding_debt: float,
                 credit_utilization_ratio: float,
                 payment_of_min_amount: str,
                 total_emi_per_month: float,
                 amount_invested_monthly: float,
                 payment_behaviour: str,
                 monthly_balance: float):

        self.age = age
        self.occupation = occupation
        self.annual_income = annual_income
        self.num_bank_accounts = num_bank_accounts
        self.num_credit_card = num_credit_card
        self.interest_rate = interest_rate
        self.num_of_loan = num_of_loan
        self.delay_from_due_date = delay_from_due_date
        self.num_of_delayed_payment = num_of_delayed_payment
        self.changed_credit_limit = changed_credit_limit
        self.num_credit_inquiries = num_credit_inquiries
        self.credit_mix = credit_mix
        self.outstanding_debt = outstanding_debt
        self.credit_utilization_ratio = credit_utilization_ratio
        self.payment_of_min_amount = payment_of_min_amount
        self.total_emi_per_month = total_emi_per_month
        self.amount_invested_monthly = amount_invested_monthly
        self.payment_behaviour = payment_behaviour
        self.monthly_balance = monthly_balance

    def get_data_dict(self):
        try:
            return {
                "Age": self.age,
                "Occupation": self.occupation,
                "Annual_Income": self.annual_income,
                "Num_Bank_Accounts": self.num_bank_accounts,
                "Num_Credit_Card": self.num_credit_card,
                "Interest_Rate": self.interest_rate,
                "Num_of_Loan": self.num_of_loan,
                "Delay_from_due_date": self.delay_from_due_date,
                "Num_of_Delayed_Payment": self.num_of_delayed_payment,
                "Changed_Credit_Limit": self.changed_credit_limit,
                "Num_Credit_Inquiries": self.num_credit_inquiries,
                "Credit_Mix": self.credit_mix,
                "Outstanding_Debt": self.outstanding_debt,
                "Credit_Utilization_Ratio": self.credit_utilization_ratio,
                "Payment_of_Min_Amount": self.payment_of_min_amount,
                "Total_EMI_per_month": self.total_emi_per_month,
                "Amount_invested_monthly": self.amount_invested_monthly,
                "Payment_Behaviour": self.payment_behaviour,
                "Monthly_Balance": self.monthly_balance
            } 
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_df(self):
        return pd.DataFrame([self.get_data_dict()])