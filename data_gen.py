from faker import Faker
import pandas as pd
import numpy as np
import datetime
from time import perf_counter as pc
import warnings

warnings.filterwarnings("ignore")


def generate(x):
    loc = []
    dates = []
    cpo = []
    d1 = datetime.date(2018, 8, 1)
    d2 = datetime.date(2020, 12, 31)
    for _ in range(x):
        loc.append(list(f.location_on_land()))
        dates.append(
            f.date_between_dates(
                date_start=d1,
                date_end=d2,
            )
        )

    for _ in range(500):
        cpo.append(f.company())

    df = pd.DataFrame(loc, columns=["lat", "lng", "location", "country", "timezone"])
    df["endtime"] = dates
    df["cdrId"] = np.random.randint(low=1000000, high=9999999, size=x)
    df["duration"] = np.random.randint(low=10, high=1000, size=x)
    df["grossAmount"] = np.random.uniform(low=1.0, high=1000.0, size=x)
    df["kWh"] = np.random.uniform(low=1.0, high=500.0, size=x)
    df["typeOfCurrent"] = np.random.choice(["AC", "DC", "Unknown"], size=x)
    df["CPO_Name"] = np.random.choice(cpo, size=x)
    df["connector_id"] = np.random.choice(
        np.random.uniform(low=1000000, high=9999999, size=int(x / 5)), size=x
    )
    df['city'] = df['timezone'].apply(lambda x: str(x).split('/')[1])
    df['continent'] = df['timezone'].apply(lambda x: str(x).split('/')[0])
    return df


f = Faker()
l = 100
s = pc()
print("Generating {} data points ".format(l))
df = generate(l)
print(df.head())
print(df.columns)
print("Time taken to generate {} data points : {}".format(l, pc() - s))

df.to_csv("data/data.csv")
