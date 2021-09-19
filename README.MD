input format:

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
 
expected test proba result from above input: .03851019963622093


expected output format:
{
 'model':'credit-risk-scorer-dari-modul-agustiar',
 'risky':False,
 'score_proba':.03851019963622093,
 'version': '1.0.Ordinal-punya-sendiri',
}

http to access(?):
http://127.0.0.1:5000/predict
with method POST
