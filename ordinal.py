from sklearn.pipeline import TransformerMixin

class Ordinal(TransformerMixin):
    def __init__(self,ordi):
        self.ordi=ordi

    def fit(self, X, y=None):
        loan_grades=X.loan_grade.value_counts().index.to_list()
        self.ordinal_ranking = {loan_grades[i]:i+1 for i in range(0, len(loan_grades))}
        return self

    def transform(self, X):
        for feat in self.ordi:
            X[feat] = X[feat].map(self.ordinal_ranking)
        return X