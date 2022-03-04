import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label)**2))


def get_bike_share(random_state=0):
    
    # we use hour dataset and predict total cnt
    task_type = "Regression"
    data = pd.read_csv("./bike_share_hour/bike_share_hour.csv", index_col=[0])
    meta_info = json.load(open("./bike_share_hour/data_types.json"))
    x, y = data.iloc[:,1:-3].values, data.iloc[:,[-1]].values
    xx = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            meta_info[key]['scaler'] = sy
        elif item['type'] == 'categorical':
            enc = OrdinalEncoder()
            xx[:,[i]] = enc.fit_transform(x[:,[i]])
            meta_info[key]['values'] = []
            for item in enc.categories_[0].tolist():
                try:
                    if item == int(item):
                        meta_info[key]['values'].append(str(int(item)))
                    else:
                        meta_info[key]['values'].append(str(item))
                except ValueError:
                    meta_info[key]['values'].append(str(item))
        else:
            sx = MinMaxScaler((0, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]['scaler'] = sx
    selected_features = ['season', 'mnth', 'hr', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
    meta_info = {key: meta_info[key] for key in selected_features}

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32)[:, [0, 2, 3, 5, 6, 7, 8, 10, 11]],
                                                        y.astype(np.float32),
                                                        test_size=0.2, random_state=random_state)
    return train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse,sy)


def get_credit_default(random_state=0):

    meta_info = {'LIMIT_BAL':{'type':'continuous'},
                 'PAY_0':{'type':'continuous'},
                 'PAY_2':{'type':'continuous'},
                 'PAY_3':{'type':'continuous'},
                 'PAY_4':{'type':'continuous'},
                 'PAY_5':{'type':'continuous'},
                 'PAY_6':{'type':'continuous'},
                 'BILL_AMT1':{'type':'continuous'},
                 'BILL_AMT2':{'type':'continuous'},
                 'BILL_AMT3':{'type':'continuous'},
                 'BILL_AMT4':{'type':'continuous'},
                 'BILL_AMT5':{'type':'continuous'},
                 'BILL_AMT6':{'type':'continuous'},
                 'PAY_AMT1':{'type':'continuous'},
                 'PAY_AMT2':{'type':'continuous'},
                 'PAY_AMT3':{'type':'continuous'},
                 'PAY_AMT456':{'type':'continuous'},
                 'FLAG_UTIL_RAT1':{'type':'categorical'},
                 'UTIL_RAT1':{'type':'continuous'},
                 'UTIL_RAT_AVG':{'type':'continuous'},
                 'UTIL_RAT_RANGE':{'type':'continuous'},
                 'UTIL_RAT_MAX':{'type':'continuous'},
                 'FLAG_PAY_RAT1':{'type':'categorical'},
                 'PAY_RAT1':{'type':'continuous'},
                 'PAY_RAT_AVG':{'type':'continuous'},
                 'PAY_RAT_RANGE':{'type':'continuous'},
                 'PAY_RAT_MAX':{'type':'continuous'},
                 'Default Payment':{'type':'target'}}

    data = pd.read_csv('./credit_default/credit_data_processed.csv', index_col=[0])
    x, y = data.loc[:,list(meta_info.keys())[:-1]].values, data.loc[:,['default.payment.next.month']].values

    xx = np.zeros(x.shape)
    task_type = 'Classification'
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            enc = OrdinalEncoder()
            enc.fit(y)
            y = enc.transform(y)
            meta_info[key]['values'] = enc.categories_[0].tolist()
        elif item['type'] == 'categorical':
            enc = OrdinalEncoder()
            enc.fit(x[:,[i]])
            ordinal_feature = enc.transform(x[:,[i]])
            xx[:,[i]] = ordinal_feature
            meta_info[key]['values'] = enc.categories_[0].tolist()
        else:
            sx = MinMaxScaler((0, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]['scaler'] = sx

    train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y, test_size=0.2, random_state=random_state)
    return train_x, test_x, train_y, test_y, task_type, meta_info


def get_california_housing(random_state=0):

    task_type = "Regression"
    cal_housing = fetch_california_housing()
    sx = MinMaxScaler((0, 1))
    sy = MinMaxScaler((0, 1))
    xx = sx.fit_transform(cal_housing.data)
    yy = sy.fit_transform(cal_housing.target.reshape(-1, 1))

    get_metric = metric_wrapper(rmse, sy)
    meta_info = {name: {"type": "continuous"} for name in cal_housing.feature_names}
    meta_info.update({cal_housing.target_names[0]: {"type": "target"}})
    train_x, test_x, train_y, test_y = train_test_split(xx, yy, test_size=0.2, random_state=random_state)
    return train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse,sy)
