import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from styling import fetch_style as fs
import re
import numpy as np
from datetime import datetime
import os

class DataLoad:
    data = None
    path = None

    def __init__(self):
        self.data = pd.DataFrame()
        self.path = (
            "data/data.csv"
        )
        print("Size of csv file : {} MB".format(os.stat(self.path).st_size/1000000))

    def __load_data__(self):
        self.data = pd.read_csv(self.path)
        self.data["country"] = self.data["country"].apply(
            lambda i: re.sub("[^A-Za-z0-9, ]+", " ", str(i))
        )
        self.data["city"] = self.data["city"].apply(
            lambda i: re.sub("[^A-Za-z0-9, ]+", " ", str(i))
        )
        self.data["location"] = self.data["location"].apply(
            lambda i: re.sub("[^A-Za-z0-9, ]+", " ", str(i))
        )
        self.data["CPO_Name"] = self.data["CPO_Name"].apply(
            lambda i: re.sub("[^A-Za-z0-9, ]+", " ", str(i))
        )
        return self.data

    def __get_all_unique__(self):
        return {
            "country": list(sorted(self.data["country"].unique().astype(str))),
            "city": list(sorted(self.data["city"].unique().astype(str))),
            "location": list(sorted(self.data["location"].unique().astype(str))),
            "cpo": list(sorted(self.data["CPO_Name"].unique().astype(str))),
        }


d = DataLoad()
data = d.__load_data__()
all_unique = d.__get_all_unique__()


def format_options(df, col_name):
    if col_name == "custom_period":
        x = [{"label": i, "value": i} for i in [1, 3, 6, 9, 12, 18, 24, 48]]
        return x
    else:
        x = [
            {
                "label": re.sub("[^A-Za-z0-9, ]+", " ", str(i)),
                "value": re.sub("[^A-Za-z0-9, ]+", " ", str(i)),
            }
            for i in sorted(df[col_name].unique().astype(str))
        ]
        return x


def get_options(col_name):
    return format_options(data, col_name)


def get_default(country=[], city=[], loc=[], cpo=[]):
    if len(cpo) == 0 or cpo == "All":
        cpo = all_unique["cpo"]
    if len(loc) == 0 or loc == "All":
        loc = all_unique["location"]
    if len(city) == 0 or city == "All":
        city = all_unique["city"]
    if len(country) == 0 or country == "All":
        country = all_unique["country"]

    return country, city, loc, cpo


def get_figure(country=[], city=[], loc=[], cpo=[]):
    if len(country) == 0:
        country, city, loc, cpo = get_default(country=country)
    if len(city) == 0:
        _, city, loc, cpo = get_default(country=country)
    if len(loc) == 0:
        _, _, loc, cpo = get_default(country=country, city=city)
    if len(cpo) == 0:
        _, _, _, cpo = get_default(country=country, city=city, loc=loc)

    df = data.copy(deep=True)
    print("figure :", country)
    df = df.loc[df["country"].isin(country)]
    df = df.loc[df["city"].isin(city)]
    df = df.loc[df["location"].isin(loc)]
    df = df.loc[df["CPO_Name"].isin(cpo)]

    token = "pk.eyJ1IjoiYW5pbWVzaHNhcnJhZiIsImEiOiJja29taHZ1bDIwNDc0MndrN2RtbGxnZ3ZjIn0.edXeqWWxgs4l0ypXGwLn4Q"
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lng",
        color="typeOfCurrent",
        hover_name="location",
        hover_data=["city"],
        zoom=1,
        color_discrete_sequence=["#6FA6D8", "#FC9407", "#63DF66", "#F1EE39"],
        opacity=0.2,
    )
    # Now using Mapbox
    fig.update_layout(
        mapbox_style="dark",
        mapbox_accesstoken=token,
        margin=dict(l=5, r=0, t=5, b=5),
        paper_bgcolor="#727272",
        legend=dict(font_color="#CACACA", borderwidth=2)

    )
    return fig


def get_final_values(df):
    if len(df) == 0:
        return 0, 0, 0, 0, 0
    else:
        session_count = len(df)
        cpo_count = len(df["CPO_Name"].unique())
        gross_amt = np.round(sum(df["grossAmount"]), 0)
        avg_power = np.round(np.mean(df["kWh"]), 2)
        total_charge_time = np.round((sum(df["duration"]) / 60), 1)
        print(
            "final val : ",
            session_count,
            cpo_count,
            gross_amt,
            avg_power,
            total_charge_time,
        )
        return session_count, cpo_count, gross_amt, avg_power, total_charge_time


def get_updates(country=None, city=None, loc=None, cpo=None, period="None"):
    country, city, loc, cpo = get_default(country, city, loc, cpo)

    df = data.copy(deep=True)

    df = df.loc[df["country"].isin(country)]
    df = df.loc[df["city"].isin(city)]
    df = df.loc[df["location"].isin(loc)]
    df = df.loc[df["CPO_Name"].isin(cpo)]

    # TODO : Convert timestamp to datetime and add date filter for period
    if period is not None:
        df["months_from_today"] = (
            datetime.today() - pd.to_datetime(df["endtime"])
        ) / np.timedelta64(1, "M")
        df = df.loc[df["months_from_today"] <= int(period)]
    return get_final_values(df)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container(
    [
        # Row 1
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [dbc.CardBody(html.H2("EV Analytics Dashboard"))],
                            style={"textAlign": "center"},
                        ),
                    ],
                    width=7,
                ),
            ],
            className="mb-3 mt-3",
            justify="center",
        ),
        # Row 2
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H6(
                            "Country",
                            className="above-dropdown",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "City",
                            className="above-dropdown",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "Location",
                            className="above-dropdown",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "CPO Name",
                            className="above-dropdown",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "Time Period (in months)",
                            className="above-dropdown",
                        ),
                    ],
                    width=2,
                ),
            ],
            className="mt-1",
            justify="center",
        ),
        # Row 3
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dd_country",
                                    options=get_options(col_name="country"),
                                    multi=True,
                                    value=[],
                                    placeholder="Country(Default - All selected)",
                                    style=fs("dropdowns"),
                                ),
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dd_city",
                                    options=[],
                                    multi=True,
                                    value=[],
                                    placeholder="City(Default - All selected)",
                                    style=fs("dropdowns"),
                                ),
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dd_location",
                                    options=[],
                                    multi=True,
                                    value=[],
                                    placeholder="Location(Default - All selected)",
                                    style=fs("dropdowns"),
                                ),
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dd_cpo",
                                    options=[],
                                    multi=True,
                                    value=[],
                                    placeholder="CPO (Default - All selected)",
                                    style=fs("dropdowns"),
                                ),
                            ]
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="dd_period",
                                    options=get_options("custom_period"),
                                    multi=False,
                                    value=None,
                                    placeholder="Period(Months, Default - All data)",
                                    style=fs("dropdowns"),
                                ),
                            ]
                        ),
                    ],
                    width=2,
                ),
            ],
            className="mb-2",
            justify="center",
        ),
        # Row 4
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H6("Total number of sessions", style=fs("above-values")),
                        html.H4(
                            id="id_session", children=[], style=fs("actual-values")
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "Total number of unique CPOs",
                            style=fs("above-values"),
                        ),
                        html.H4(
                            id="id_cpo_count", children=[], style=fs("actual-values")
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "Gross Amount (€)",
                            style=fs("above-values"),
                        ),
                        html.H4(
                            id="id_gross_amount",
                            children=[],
                            style=fs("actual-values"),
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "Average Power Usage (kWh)",
                            style=fs("above-values"),
                        ),
                        html.H4(
                            id="id_avg_power",
                            children=[],
                            style=fs("actual-values"),
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        html.H6(
                            "Total Charging Time (hrs) ",
                            style=fs("above-values"),
                        ),
                        html.H4(
                            id="id_total_charge_time",
                            children=[],
                            style=fs("actual-values"),
                        ),
                    ],
                    width=2,
                ),
            ],
            className="mb-2 mt-3",
            justify="center",
        ),
        # Row 5
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="map", figure={}),
                    ],
                    width=10,
                ),
            ],
            className="mb-2 mt-2",
            justify="center",
        ),
        # Row 6
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id="graph2", figure={}),
                    ],
                    width=5,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="graph3", figure={}),
                    ],
                    width=5,
                ),
            ],
            className="mb-2 mt-2",
            justify="center",
        ),
    ],
    fluid=True,
)


# callback for changing filter options
@app.callback(
    [
        Output(component_id="dd_city", component_property="options"),
    ],
    [
        Input(component_id="dd_country", component_property="value"),
    ],
)
def update(country):
    df = data.copy(deep=True)
    if len(country) == 0 or (country == "All"):
        country = all_unique["country"]

    print("country changed: ", len(country), country)
    return [
        format_options(
            df.loc[df["country"].isin(country)],
            "city",
        )
    ]


@app.callback(
    [
        Output(component_id="dd_location", component_property="options"),
    ],
    [
        Input(component_id="dd_city", component_property="value"),
    ],
)
def update(city):
    df = data.copy(deep=True)
    if len(city) == 0 or (city == "All"):
        city = all_unique["city"]

    return [format_options(df.loc[df["city"].isin(city)], "location")]


@app.callback(
    [
        Output(component_id="dd_cpo", component_property="options"),
    ],
    [
        Input(component_id="dd_location", component_property="value"),
    ],
)
def update(loc):
    df = data.copy(deep=True)
    if len(loc) == 0 or (loc == "All"):
        loc = all_unique["location"]
    print("loc changed: ", len(loc))
    return [format_options(df.loc[df["location"].isin(loc)], "CPO_Name")]


# callback for changing children and figures
@app.callback(
    [
        Output(component_id="id_session", component_property="children"),
        Output(component_id="id_cpo_count", component_property="children"),
        Output(component_id="id_gross_amount", component_property="children"),
        Output(component_id="id_avg_power", component_property="children"),
        Output(component_id="id_total_charge_time", component_property="children"),
    ],
    [
        Input(component_id="dd_country", component_property="value"),
        Input(component_id="dd_city", component_property="value"),
        Input(component_id="dd_location", component_property="value"),
        Input(component_id="dd_cpo", component_property="value"),
        Input(component_id="dd_period", component_property="value"),
    ],
)
def update_dash(country, city, loc, cpo, period):
    print("UPDATING VALUES")
    return get_updates(country, city, loc, cpo, period)


@app.callback(
    [
        Output(component_id="map", component_property="figure"),
    ],
    [
        Input(component_id="dd_country", component_property="value"),
        Input(component_id="dd_city", component_property="value"),
        Input(component_id="dd_location", component_property="value"),
        Input(component_id="dd_cpo", component_property="value"),
    ],
)
def update_dash(country, city, loc, cpo):
    return [get_figure(country, city, loc, cpo)]


if __name__ == "__main__":
    app.run_server(debug=True, port=8911)
