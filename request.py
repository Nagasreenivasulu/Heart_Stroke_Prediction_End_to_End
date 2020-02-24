import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'gender':1, 'age':3.0, 'hypertension':0, 'heart_disease':0, 'Residence_type':0, 'avg_glucose_level':95.12, 'bmi':18.0,'Govt_job':0,'Never_worked':0, 'Private':0, 'Self-employed':0,'formerly smoked':0,'never smoked':0})

print(r.json())