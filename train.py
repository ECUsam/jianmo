import numpy as np
import pandas as pd
# from utils.tools import *
import warnings
warnings.filterwarnings('ignore')


class solve_class:
    def __init__(self):
        self.train = pd.read_csv('output_train.csv')
        self.test = pd.read_csv('output_test.csv')
        self.top_10_morning, self.top_10_evening, self.morning_peak, self.evening_peak = top_10_hotspots(self.train)
        print(self.morning_peak)





def top_10_hotspots(df):
    df['hour'] = pd.to_datetime(df['starttime']).dt.hour
    morning_peak = df[(df['hour'] >= 7) & (df['hour'] < 9)]
    evening_peak = df[(df['hour'] >= 17) & (df['hour'] < 19)]
    top_10_morning = morning_peak['geohashed_start_loc'].value_counts().head(10)
    top_10_evening = evening_peak['geohashed_start_loc'].value_counts().head(10)
    return top_10_morning, top_10_evening, morning_peak, evening_peak




if __name__ == "__main__":
    solve = solve_class()
