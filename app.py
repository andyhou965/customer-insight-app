import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.graph_objs as go
import dash_daq as daq

import pandas as pd

from utils import *

from modules.metrics import generate_metrics_fig

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app._favicon = os.path.join(APP_PATH, os.path.join("assets", "favicon.ico"))
app.title = "Customer Insight APP"
server = app.server
app.config["suppress_callback_exceptions"] = True

data_file = os.path.join(APP_PATH, os.path.join("data", "OnlineRetail.csv"))
df = pd.read_csv(data_file, encoding='unicode_escape')


def build_tab_1():
    (
        monthly_revenue_fig,
        monthly_growth_fig,
        monthly_active_customers_fig,
        monthly_order_number_fig,
        monthly_avg_order_fig,
    ) = generate_metrics_fig(df)
    return (
        html.Div(
            className="main-content-container",
            children=[
                build_side_panel(),
                html.Div(
                    id="graphs-container",
                    children=[
                        build_double_panel(
                            "Monthly Revenue",
                            dcc.Graph(
                                figure=monthly_revenue_fig,
                                config={'displaylogo': False},
                            ),
                            'Monthly Growth Rate',
                            dcc.Graph(
                                figure=monthly_growth_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                        html.Br(),
                        build_double_panel(
                            "Left Title", html.Div(), "Right Title", html.Div()
                        ),
                        html.Br(),
                        build_double_panel(
                            "Left Title", html.Div(), "Right Title", html.Div()
                        ),
                        build_single_panel("Title", html.Div()),
                    ],
                ),
            ],
        ),
    )


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_top_banner(),
        html.Div(
            id="app-container",
            children=[
                build_menu_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
    ],
)


@app.callback([Output("app-content", "children")], [Input("app-tabs", "value")])
def render_tab_content(tab_switch):
    if tab_switch == "tab1":
        return build_tab_1()
    return (
        html.Div(
            className="main-content-container",
            children=[],
        ),
    )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
