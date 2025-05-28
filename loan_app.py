from flask import Flask, render_template ,request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open("regresseion_model.pkl",'rb'))
scaler=pickle.load(open("scaler.pkl",'rb'))

@app.route('/')
def home():
    return render_template('loan.html',prediction='')

@app.route('/predict',methods=['POST'])

def index():
    input_data={
        'Age': int(request.form['Age']),
        'CreditScore':int(request.form['CreditScore']),
        'EmploymentStatus':request.form['EmploymentStatus'],
        'EducationLevel':request.form['EducationLevel'],
        'LoanAmount':float(request.form['LoanAmount']),
        'LoanDuration':int(request.form['LoanDuration']),
        'CreditCardUtilizationRate': float(request.form['CreditCardUtilizationRate']),
        'BankruptcyHistory': int(request.form['BankruptcyHistory']),
        'PreviousLoanDefaults':int(request.form['PreviousLoanDefaults']),
        'LengthOfCreditHistory':int(request.form['LengthOfCreditHistory']),
        'MonthlyIncome': int(request.form['MonthlyIncome']),
        'NetWorth':float(request.form['NetWorth']),
        'InterestRate':float(request.form['InterestRate']),
    }
    input_df = pd.DataFrame([input_data])
    input_df['EmploymentStatus']=input_df['EmploymentStatus'].map({'Unemployed':0,'Self-Employed':1,'Employed':2})
    input_df['EducationLevel']=input_df['EducationLevel'].map({'High School':0,'Associate':1,'Bachelor':2,'Master':3,'Doctorate':4})
    
    features_to_log1p=['LoanAmount','MonthlyIncome','NetWorth']
    
    input_df[features_to_log1p]=input_df[features_to_log1p].apply (np.log1p)
    
    columns_to_standardize=['Age', 'CreditScore', 'LoanAmount', 'LoanDuration',
    'CreditCardUtilizationRate', 'LengthOfCreditHistory',
    'MonthlyIncome', 'NetWorth', 'InterestRate'
     ]
    
    input_df[columns_to_standardize]=scaler.transform(input_df[columns_to_standardize])
    riskscore=model.predict(input_df)[0]
    riskscore=round(riskscore,2)
    score= f"your riskscore is {riskscore}"
    return render_template('loan.html',prediction=score)

if __name__ == '__main__':
    app.run(debug=True)
