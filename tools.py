def make_geo_bigger(lat, ton):
    return (lat - 38) * 100, (ton - 115) * 100


def make_big_recover(lat_, ton_):
    return (lat_ / 100) + 38, (ton_ / 100) + 115
