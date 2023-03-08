import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import pickle ,argparse,os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=Path)
parser.add_argument("--weight_path", type=Path)

def predict(data,model):
    housing = data.drop("median_house_value", axis=1)
    housing_labels = data["median_house_value"].copy()

    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median, inplace=True)

    housing_num = housing.drop("ocean_proximity", axis=1)

    # column index
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self  # nothing else to do
        def transform(self, X):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                            bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]
            
    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    #housing_num_tr = num_pipeline.fit_transform(housing_num)


    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    data_preparation = full_pipeline.transform(housing)
    return model.predict(data_preparation)

if __name__ == "__main__":
    arg_parser = parser.parse_args()
    model_path=os.path.join(arg_parser.weight_path,'model_pkl')
    data=pd.read_csv(os.path.join(arg_parser.input_path,"test.csv"))
    with open(model_path , 'rb') as f:
        model = pickle.load(f)
    predict(data,model)