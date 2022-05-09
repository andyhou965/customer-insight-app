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

from modules.metrics import (
    monthly_revenue_fig,
    monthly_growth_fig,
    monthly_active_customers_fig,
)

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app._favicon = os.path.join(APP_PATH, os.path.join("assets", "favicon.ico"))
app.title = "Customer Insight APP"
server = app.server
app.config["suppress_callback_exceptions"] = True


def build_tab_1():
    return [
        #     # Manually select metrics
        #     html.Div(
        #         id="set-specs-intro-container",
        #         # className='twelve columns',
        #         children=html.P(
        #             "Use historical control limits to establish a benchmark, or set new values."
        #         ),
        #     ),
        #     html.Div(
        #         id="settings-menu",
        #         children=[
        #             html.Div(
        #                 id="metric-select-menu",
        #                 # className='five columns',
        #                 children=[
        #                     html.Label(id="metric-select-title", children="Select Metrics"),
        #                     html.Br(),
        #                     dcc.Dropdown(
        #                         id="metric-select-dropdown",
        #                         options=list(
        #                             {"label": param, "value": param} for param in params[1:]
        #                         ),
        #                         value=params[1],
        #                     ),
        #                 ],
        #             ),
        #             html.Div(
        #                 id="value-setter-menu",
        #                 # className='six columns',
        #                 children=[
        #                     html.Div(id="value-setter-panel"),
        #                     html.Br(),
        #                     html.Div(
        #                         id="button-div",
        #                         children=[
        #                             html.Button("Update", id="value-setter-set-btn"),
        #                             html.Button(
        #                                 "View current setup",
        #                                 id="value-setter-view-btn",
        #                                 n_clicks=0,
        #                             ),
        #                         ],
        #                     ),
        #                     html.Div(
        #                         id="value-setter-view-output", className="output-datatable"
        #                     ),
        #                 ],
        #             ),
        #         ],
        #     ),
    ]


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
    return (
        html.Div(
            className="main-content-container",
            children=[
                build_side_panel(),
                html.Div(
                    id="graphs-container",
                    children=[
                        build_double_panel(
                            "Left Title", html.Div(), "Right Title", html.Div()
                        ),
                        build_double_panel(
                            "Left Title", html.Div(), "Right Title", html.Div()
                        ),
                        build_double_panel(
                            "Left Title", html.Div(), "Right Title", html.Div()
                        ),
                        build_single_panel("Title", html.Div()),
                    ],
                ),
            ],
        ),
    )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
