Here the source codes and all the dependencies that enable the heroku app that does the prediction of whether a potential creditor would default based on these values stored on input format below. 

input format:
```json
{
  "person_age": 30,
  "person_income": 82000,
  "person_home_ownership": "MORTGAGE",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "A",
  "loan_amnt": 5000,
  "loan_int_rate": 6.76,
  "loan_percent_income": 0.06,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 8
 }
```

expected test proba result from above input: .004475969355553389

expected output format:
```bash
{
 'model':'credit-risk-scorer-dari-modul-agustiar',
 'risky':False,
 'score_proba':.004475969355553389,
 'version': '1.0.Ordinal-punya-sendiri',
}
```

to launch:
```bash
py app.py
```

and the http to access:
- http://127.0.0.1:5000/predict
```POST```

The expected accuracy of the model is as follows:


AUC | accuracy | precision Type | recall 
------|-------------|-------|---------
0.928285 | 0.864063 | 0.908163 | 0.749474 


A lot of what was done and written here is thanks to Mas Agustiar (for the supervised learning materials/tricks/dataset) and especially to Mas Mamduh @mamduhmahfudzi: materials and ideas he shared, the exercise he is assigning--the one I am doing right now, AND his repository that makes this API and app possible: https://github.com/mamduhmahfudzi/credit-scorer.
