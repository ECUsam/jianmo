import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

from config import DefaultConfig


class BikePrediction:
    def __init__(self):
        self.train = pd.read_csv('output_train.csv')
        self.test = pd.read_csv('output_test.csv')
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, warm_start=True)
        self.opt = DefaultConfig()

    def filter_peak_hours(self, data):
        data['hour'] = pd.to_datetime(data['starttime']).dt.hour
        morning_peak = data[(data['hour'] >= 7) & (data['hour'] <= 9)]
        evening_peak = data[(data['hour'] >= 17) & (data['hour'] <= 19)]
        return morning_peak, evening_peak

    def get_hot_blocks(self, data):
        return data['geohashed_start_loc'].value_counts().head(10).index.tolist()

    def add_features(self, data):
        # Add day of the week
        data['day_of_week'] = pd.to_datetime(data['starttime']).dt.dayofweek

        # Add distance (Euclidean for simplicity)
        data['distance'] = np.sqrt((data['geohashed_start_loc_Latitude'] - data['geohashed_end_loc_Latitude']) ** 2 +
                                   (data['geohashed_start_loc_Longitude'] - data['geohashed_end_loc_Longitude']) ** 2)

        # Popularity of ending location
        end_loc_counts = data['geohashed_end_loc'].value_counts().to_dict()
        data['end_loc_popularity'] = data['geohashed_end_loc'].map(end_loc_counts)

        return data

    def predict_destination(self):
        morning_train, evening_train = self.filter_peak_hours(self.train)
        hot_morning_blocks = self.get_hot_blocks(morning_train)
        hot_evening_blocks = self.get_hot_blocks(evening_train)
        morning_train = morning_train[morning_train['geohashed_start_loc'].isin(hot_morning_blocks)]
        morning_train = self.add_features(morning_train)

        filters = {'orderid', 'userid', 'biketype', 'geohashed_start_loc', 'bikeid', 'starttime', 'geohashed_end_loc',
                   'label'}
        # Define predictors and target
        predictors = ['geohashed_start_loc_Latitude', 'geohashed_start_loc_Longitude', 'day_of_week', 'distance',
                      'end_loc_popularity']
        target = 'geohashed_end_loc'

        X_train, X_val, y_train, y_val = train_test_split(morning_train[predictors], morning_train[target],
                                                          test_size=0.2, random_state=42)

        # Create a random forest classifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)

        accuracy = rf.score(X_val, y_val)
        print(f"Validation accuracy: {accuracy * 100:.2f}%")

        for i in range(5):  # train in 5 steps
            self.rf.n_estimators += 100  # add 10 more trees in each iteration
            self.rf.fit(X_train, y_train)
            accuracy = self.rf.score(X_val, y_val)
            print(f"After {self.rf.n_estimators} trees, validation accuracy: {accuracy * 100:.2f}%")

            # save the model
            with open(f"random_forest_model_step_{i}.pkl", "wb") as file:
                pickle.dump(self.rf, file)

        return self.rf


# Example
model = BikePrediction()
forest = model.predict_destination()
