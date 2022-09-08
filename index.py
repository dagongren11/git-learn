import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
# # HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
# # def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
# #     if not os.path.isdir(housing_path):
# #         os.makedirs(housing_path)
# #     tgz_path = os.path.join(housing_path,"housing.tgz")
# #     urllib.request.urlretrieve(housing_url,tgz_path)
# #     housing_tgz=tarfile.open(tgz_path)
# #     housing_tgz.extractall(path=housing_path)
# #     housing_tgz.close()
# # fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,'housing.csv')
    return pd.read_csv(csv_path)
housing = load_housing_data()
print(housing.head())
# housing.hist(bins=50,figsize=(20,15))
# plt.show()

housing['income_cat'] = pd.cut(housing['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])
# housing['income_cat'].hist()
# plt.show()
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, text_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[text_index]
print(strat_test_set['income_cat'].value_counts()/len(strat_test_set))
for set_ in (strat_train_set,strat_test_set):
    set_.drop('income_cat',axis = 1,inplace = True)
housing1 = strat_train_set.copy()
# housing1.plot(kind = 'scatter',x = 'longitude',y = 'latitude',alpha = 0.1,
#               s=housing1['population']/100,lable = 'population',figsize = (10,7),
#               c='median_housing_value',cmap=plt.get_cmap('jet'),colorbar = True,)
# plt.legend()
#
# plt.show()
housing1_num = housing1.drop('ocean_proximity',axis = 1)

num_attribs = list(housing1_num)
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([
    ('num',)
])

train_set, test_set = train_test_split(housing,test_size=0.2,random_state=42)
