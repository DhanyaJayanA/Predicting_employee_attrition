# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        MonthlyIncome = float(request.form['MonthlyIncome'])
        NumCompaniesWorked = int(request.form['NumCompaniesWorked'])
        YearsSinceLastPromotion = int(request.form['YearsSinceLastPromotion'])
        JobSatisfaction = int(request.form['JobSatisfaction'])
        EnvironmentSatisfaction = int(request.form['EnvironmentSatisfaction'])
        WorkLifeBalance = int(request.form['WorkLifeBalance'])
        Education = int(request.form['Education'])
        PerformanceRating = int(request.form['PerformanceRating'])
        DailyRate = float(request.form['DailyRate'])
        RelationshipSatisfaction = int(request.form['RelationshipSatisfaction'])
        TotalWorkingYears = int(request.form['TotalWorkingYears'])
        OverTime = int(request.form['OverTime'])
        JobInvolvement = int(request.form['JobInvolvement'])
        
        data = np.array([[Age, MonthlyIncome, NumCompaniesWorked, YearsSinceLastPromotion, JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, Education,PerformanceRating,DailyRate,RelationshipSatisfaction,TotalWorkingYears,OverTime,JobInvolvement]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)