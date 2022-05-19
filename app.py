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
from modules.customer_segmentation import generate_segmentation_fig
from modules.clv_prediction import generate_clv_fig

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

fig_layout = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "yaxis": dict(showgrid=True, showline=False, zeroline=True),
    "autosize": True,
    "height": 400,
    "margin": dict(l=20, r=20, t=20),
    # "title_font_color": "yellow",
    # "font_color": "yellow",
    "template": "plotly_dark",
}


def build_tab_1():
    (
        monthly_revenue_fig,
        monthly_growth_fig,
        monthly_active_customers_fig,
        monthly_order_number_fig,
        new_existing_fig,
    ) = generate_metrics_fig(df, fig_layout)
    return (
        html.Div(
            className="main-content-container",
            children=[
                # build_side_panel(),
                html.Div(
                    className="graphs-container",
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
                            "Monthly Active Customers",
                            dcc.Graph(
                                figure=monthly_active_customers_fig,
                                config={'displaylogo': False},
                            ),
                            'Monthly Total # of Order',
                            dcc.Graph(
                                figure=monthly_order_number_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                        html.Br(),
                        # build_double_panel(
                        #     "Monthly Revenue",
                        #     dcc.Graph(
                        #         figure=monthly_revenue_fig,
                        #         config={'displaylogo': False},
                        #     ),
                        #     'Monthly Growth Rate',
                        #     dcc.Graph(
                        #         figure=monthly_growth_fig,
                        #         config={'displaylogo': False},
                        #     ),
                        # ),
                        build_single_panel(
                            "New vs Existing",
                            dcc.Graph(
                                figure=new_existing_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )


def build_tab_2():
    (
        recency_fig,
        frequency_fig,
        monetary_fig,
        freq_revenue_fig,
        recency_revenue_fig,
        recency_freq_fig,
    ) = generate_segmentation_fig(df, fig_layout)
    return (
        html.Div(
            className="main-content-container",
            children=[
                build_side_panel(),
                html.Div(
                    className="graphs-container",
                    children=[
                        build_triple_panel(
                            "Recency",
                            dcc.Graph(
                                figure=recency_fig,
                                config={'displaylogo': False},
                            ),
                            "Frequency",
                            dcc.Graph(
                                figure=frequency_fig,
                                config={'displaylogo': False},
                            ),
                            "Monetary Value",
                            dcc.Graph(
                                figure=monetary_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                        html.Br(),
                        build_single_panel(
                            "Segments(Frequency vs Revenue)",
                            dcc.Graph(
                                figure=freq_revenue_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                        html.Br(),
                        build_single_panel(
                            "Segments(Recency vs Revenue)",
                            dcc.Graph(
                                figure=recency_revenue_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                        html.Br(),
                        build_single_panel(
                            "Segments(Recency vs Frequency)",
                            dcc.Graph(
                                figure=recency_freq_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )


def build_tab_3():
    (hist_fig, clv_fig) = generate_clv_fig(df, fig_layout)
    return (
        html.Div(
            className="main-content-container",
            children=[
                build_side_panel(),
                html.Div(
                    className="graphs-container",
                    children=[
                        build_single_panel(
                            "6m Revenue",
                            dcc.Graph(
                                figure=hist_fig,
                                config={'displaylogo': False},
                            ),
                        ),
                        html.Br(),
                        build_single_panel(
                            "LTV",
                            dcc.Graph(
                                figure=clv_fig,
                                config={'displaylogo': False},
                            ),
                        ),
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
                # build_menu_tabs(),
                # Main app
                html.Div(
                    id="app-content",
                    className="tabs",
                    children=[
                        dcc.Tabs(
                            id="app-tabs",
                            value="tab1",
                            className="custom-tabs",
                            children=[
                                dcc.Tab(
                                    id="metrics-tab",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    value="tab1",
                                    label="The Metrics",
                                    children=build_tab_1(),
                                ),
                                dcc.Tab(
                                    id="segmentation-tab",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    value="tab2",
                                    label="Customer Segmentation",
                                    children=build_tab_2(),
                                ),
                                dcc.Tab(
                                    id="clv-tab",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    value="tab3",
                                    label="Customer Lifetime Value",
                                    children=build_tab_3(),
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


# @app.callback([Output("app-content", "children")], [Input("app-tabs", "value")])
# def render_tab_content(tab_switch):
#     # The Metrics
#     if tab_switch == "tab1":
#         return build_tab_1()
#     # Customer Segmentation
#     elif tab_switch == "tab2":
#         return build_tab_2()
#     # CLV - Customer Lifetime Value
#     elif tab_switch == "tab3":
#         return build_tab_3()

#     return (
#         html.Div(
#             className="main-content-container",
#             children=[],
#         ),
#     )


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050, dev_tools_ui=False)
