import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

from config import DefaultConfig
from dataset import get_sample, get_train_data
import warnings
warnings.filterwarnings('ignore')
from feature.user import get_user_count, get_user_eloc_count, get_user_eloc_hour_count, get_user_sloc_eloc_hour_count, \
    get_user_sloc_hour_count



opt = DefaultConfig()
opt.update()


class BikePrediction:
    def __init__(self):
        self.train, self.test= get_train_data(opt)

        self.rf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42, warm_start=True)
        self.opt = DefaultConfig()
        self.sample = get_sample(self.train, self.test)
        self.filters = {'orderid', 'userid', 'biketype', 'geohashed_start_loc', 'bikeid', 'starttime', 'geohashed_end_loc',
                   'label', 'category', 'distance', 'eloc_lat', 'eloc_lon', 'eloc_sloc_lat_sub', 'eloc_sloc_lon_sub', "manhattan", "sloc_lat"
                        , "sloc_lon"}


    def filter_peak_hours(self, data):
        data['hour'] = pd.to_datetime(data['starttime']).dt.hour
        morning_peak = data[(data['hour'] >= 7) & (data['hour'] <= 9)]
        evening_peak = data[(data['hour'] >= 17) & (data['hour'] <= 19)]
        return morning_peak, evening_peak

    def get_hot_blocks(self, data):
        return data['geohashed_start_loc'].value_counts().head(10).index.tolist()

    def add_features(self, result):
        result['day_of_week'] = pd.to_datetime(result['starttime']).dt.dayofweek
        result = get_user_count(self.train, result)  # user_count # dui 90 wc
        result = get_user_eloc_count(self.train, result)  # user_eloc_count # dui 90 wc

        result = get_user_eloc_hour_count(self.train, result)
        result = get_user_sloc_hour_count(self.train, result)  # user_sloc_hour_fre
        result = get_user_sloc_eloc_hour_count(self.train, result)  # user_sloc_eloc_hour_fre
        return result

    def predict_destination(self):
        morning_train, evening_train = self.filter_peak_hours(self.train)
        hot_morning_blocks = self.get_hot_blocks(morning_train)
        hot_evening_blocks = self.get_hot_blocks(evening_train)
        morning_train = morning_train[morning_train['geohashed_start_loc'].isin(hot_morning_blocks)]
        morning_train = self.add_features(morning_train)


        # Define predictors and target
        predictors = ['geohashed_start_loc_Latitude', 'geohashed_start_loc_Longitude', 'day_of_week', 'distance',
                      'end_loc_popularity']
        predictors = list(filter(lambda x: x not in self.filters, morning_train.columns.tolist()))



        target = 'category'

        X_train, X_val, y_train, y_val = train_test_split(morning_train[predictors], morning_train[target],
                                                          test_size=0.2, random_state=42)

        # Create a random forest classifier
        rf = RandomForestClassifier(n_estimators=2, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        accuracy = rf.score(X_val, y_val)
        print(f"Validation accuracy: {accuracy * 100:.2f}%")

        for i in range(5):  # train in 5 steps
            self.rf.n_estimators += 1  # add 10 more trees in each iteration
            self.rf.fit(X_train, y_train)
            accuracy = self.rf.score(X_val, y_val)
            print(f"After {self.rf.n_estimators} trees, validation accuracy: {accuracy * 100:.2f}%")

            # save the model
            with open(f"random_forest_model_step_{i}.pkl", "wb") as file:
                pickle.dump(self.rf, file)

        return self.rf


    def predict_with_model(self):

        morning_train, evening_train = self.filter_peak_hours(self.test)
        hot_morning_blocks = self.get_hot_blocks(morning_train)
        hot_evening_blocks = self.get_hot_blocks(evening_train)
        morning_train = morning_train[morning_train['geohashed_start_loc'].isin(hot_morning_blocks)]
        morning_train = self.add_features(morning_train)
        # 假设你的测试数据叫test_data
        # 1. 添加和处理特征

        print(morning_train.columns)

        predictors = list(filter(lambda x: x not in self.filters, morning_train.columns.tolist()))

        imputer = SimpleImputer(strategy='mean')
        morning_train[predictors] = imputer.fit_transform(morning_train[predictors])

        # 2. 使用模型进行预测
        predictions = self.rf.predict(morning_train[predictors])  # 使用模型预测

        # 3. 如果需要，可以将预测结果保存到文件
        # 这部分代码可以根据你的实际需求进行调整
        morning_train['predicted_end_loc'] = predictions
        morning_train[['orderid', 'predicted_end_loc']].to_csv('predictions.csv', index=False)

        return predictions

# Example
model = BikePrediction()
forest = model.predict_destination()
model.predict_with_model()
