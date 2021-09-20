Here is the source codes and all the dependencies that enable the heroku app that does the prediction of whether a potential creditor would default based on these values stored on input format below. 

input format:
<pre> 
 {
   "person_age": 26,
  "person_income": 48000,
  "person_home_ownership": "RENT",
  "person_emp_length": 2.0,
  "loan_intent": "MEDICAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 12.21,
  "loan_percent_income": 0.21,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 3
 }
 </pre> 

expected test proba result from above input: .03851019963622093


expected output format:
<pre> 
{
 'model':'credit-risk-scorer-dari-modul-agustiar',
 'risky':False,
 'score_proba':.03851019963622093,
 'version': '1.0.Ordinal-punya-sendiri',
}
 </pre> 

http to access (do launch on test_app.py):
- http://127.0.0.1:5000/predict
with the method POST

The expected accuracy of the model is as follows:
![Alt text](development/model_accuracy.jpg?raw=true "Model Accuracy on New Data")

A lot of what was done and written here is thanks to Mas Agustiar (for the supervised learning materials/tricks/dataset) and especially to Mas Mamduh @mamduhmahfudzi: materials and ideas he shared, the exercise he is assigning--the one I am doing right now, AND his repository that makes this API and app possible: https://github.com/mamduhmahfudzi/credit-scorer.
