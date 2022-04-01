import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def load_credit_default(random_state=0):

    data = pd.read_excel('./data/credit_default/default of credit card clients.xls', header=1)
    meta_info = json.load(open('./data/credit_default/data_types.json'))
    payment_list = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
               'PAY_AMT1', 'PAY_AMT2','PAY_AMT3', 'PAY_AMT4','PAY_AMT5', 'PAY_AMT6']
    data.loc[:,payment_list] = (np.sign(data.loc[:,payment_list]).values * np.log10(np.abs(data.loc[:,payment_list]) + 1))
    x, y = data.iloc[:,1:-1].values, data.iloc[:,[-1]].values

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

    data = pd.read_csv('./data/credit_default/credit_data_processed.csv', index_col=[0])
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