from pandas import array


def make_geo_bigger(lat, ton):
    return (lat - 38) * 100, (ton - 115) * 100


def make_big_recover(lat_, ton_):
    return (lat_ / 100) + 38, (ton_ / 100) + 115


def find_nearest_category(point, categories):
    min_distance = float('inf')
    nearest_category = None

    for category, coord in categories.items():
        distance = ((point[0] - coord[0]) ** 2 + (point[1] - coord[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_category = category

    return nearest_category


if __name__ == "__main__":
    categories = {0: array([116.23654607,  40.06645945]), 1: array([116.43272195,  39.84894588]), 2: array([116.26646715,  39.90685922]), 3: array([116.49430804,  39.96554656]), 4: array([116.37064823,  39.99583434]), 5: array([116.49982542,  39.7831434 ]), 6: array([116.59244192,  39.91314018]), 7: array([116.33322063,  39.84001663]), 8: array([116.42048452,  40.05743432]), 9: array([116.33677512,  40.03636627]), 10: array([116.28386409,  40.01741315]), 11: array([116.19290635,  39.92113804]), 12: array([116.51861185,  39.85078378]), 13: array([116.31396954,  40.07147981]), 14: array([116.40334947,  39.84758123]), 15: array([116.42243862,  39.80094574]), 16: array([116.47438245,  40.00084243]), 17: array([116.29418397,  39.80897552]), 18: array([116.23762809,  39.95757263]), 19: array([116.5669972 ,  39.97751713]), 20: array([116.20394328,  39.79110736]), 21: array([116.55448191,  39.76731656]), 22: array([116.35842057,  40.07834357]), 23: array([116.42033351,  39.99394713]), 24: array([116.55654704,  39.91166583]), 25: array([116.51487954,  39.91663412]), 26: array([116.33076084,  39.77525418]), 27: array([116.48454608,  39.87955972]), 28: array([116.32792799,  39.99173771]), 29: array([116.22533303,  39.90246973]), 30: array([116.37600606,  39.84779485]), 31: array([116.24146884,  39.85517165]), 32: array([116.63287946,  39.90915351]), 33: array([116.54378932,  40.05007755]), 34: array([116.28316253,  39.94684166]), 35: array([116.46456146,  39.83778205]), 36: array([116.29373189,  39.8501107 ]), 37: array([116.48552121,  39.91635566]), 38: array([116.37152579,  39.80350946]), 39: array([116.57001405,  39.8122047 ])}

    print(find_nearest_category((4, 5), categories))  # 一个测试坐标，你可以根据需要更改
