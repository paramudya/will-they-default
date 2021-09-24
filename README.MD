Here the source codes and all the dependencies that enable the heroku app (online here: [will-they-default](https://will-they-default.herokuapp.com) that does the prediction of whether a potential creditor would default based on these values stored on input format below. 

**input format**:
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

expected test proba result from above input: .009186364710330963

**expected output format**:
```bash
{
    "model": "credit-risk-predictor",
    "risky": false,
    "score_proba": 0.009186364710330963,
    "version": "0.4.6"
}
```

to **launch**:
```bash
python test_app.py
```
notice it is **test_app**.py and not the usual app.py.

and the **http** to access:
- http://127.0.0.1:5000/predict
```POST```

The expected accuracy of the model is as follows:


AUC | accuracy | precision | recall 
------|-------------|-------|---------
0.0.93137 | 0.864023 | 0.929825 | 0.74386

**missing values handling**

Each category of the columns are handling this problem uniquely:
- Categorical columns **do not explicitly handle** it because it was actually already handle by the brilliant feature 'handle_unknown' offered by sklearn's OneHotEncoder and setting it to 'ignore': new values that weren't seen on train set will just not be assigned to any of the dummy columns of the feature (all 0s).
- Numerical columns handle it with **'median'** method as usual.
- Ordinal columns replace the missing values with **'mean'**, because the distribution of ordinal values are sometimes far from Gaussian; and when the first number already represents more than half of the data, then the value will be set to its first number, not counting at all the values on the other extreme that might also contribute. The 'mean' method seems fairer for this case (and other more usual cases too).

**outliers handling**

Only done on the numerical columns and no outlier handling is done in the development/training process, only on the inputted data daruing prediction--because during development (see development_notebook.ipynb under development dir) it was shown that eliminating potential outliers reduces the predictive power of the model.

The outliers are very loosely handled as the limit for the new data to be considered as outlier would be +- 1 * IQR of the maximum and minimum value of the column as found on the training set. The function is created from scratch utilizing TransformerMixin (check outlie.py).

**credits**

A lot of what was done and written here is thanks to Mas Mamduh @mamduhmahfudzi and the main reason this app/API is even here: https://github.com/mamduhmahfudzi/credit-scorer.
