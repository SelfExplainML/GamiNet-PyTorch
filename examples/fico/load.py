import json
import numpy as np
import pandas as pd

def load_fico_challange(path="./"):
    
    data = pd.read_csv(path + "heloc_dataset_v1.csv")
    meta_info = json.load(open(path + "data_types.json"))
    data = data.replace(-9, np.nan).replace(-8, np.nan).replace(-7, np.nan)

    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    imp.fit(data.iloc[:,1:])  
    data.iloc[:,1:] = imp.transform(data.iloc[:,1:])
    x, y = data.iloc[:,1:].values, data.iloc[:,[0]].values
    return x, y, "Regression", meta_info
