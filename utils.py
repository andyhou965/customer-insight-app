import dash_html_components as html
import dash_core_components as dcc


def build_top_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("Customer Insight APP"),
                    html.H6(
                        "Fuel the Company's Growth by Applying the Predictive Approach to All the Actions"
                    ),
                ],
            ),
            html.Div(
                id="banner-logo",
                children=[
                    html.A(
                        html.Button(children="Personal Website"),
                        href="#",
                    ),
                    html.Button(
                        id="learn-more-button", children="LEARN MORE", n_clicks=0
                    ),
                ],
            ),
        ],
    )


def build_menu_tabs():
    tab_information = [
        ("metrics-tab", "tab1", "The Metrics"),
        ("segmentation-tab", "tab2", "Customer Segmentation"),
        ("clv-tab", "tab3", "Customer Lifetime Value"),
        # ("churn-tab", "tab4", "Churn Prediction"),
        # ("npd-tab", "tab5", "Next Purchase Day"),
        # ("sales-tab", "tab6", "Predicting Sales"),
        # ("market-tab", "tab7", "Market Response Models"),
        # ("uplift-tab", "tab8", "Uplift Modeling"),
        # ("ab-tab", "tab9", "A/B Testing"),
    ]
    tabs = [
        dcc.Tab(
            id=id,
            label=label,
            value=value,
            className="custom-tab",
            selected_className="custom-tab--selected",
        )
        for (id, value, label) in tab_information
    ]
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs", value="tab1", className="custom-tabs", children=tabs
            )
        ],
    )


def generate_title_banner(title):
    return html.Div(className="section-banner", children=title)


def build_side_panel(children=[]):
    return html.Div(
        # id="quick-stats",
        className="quick-stats row",
        children=children,
    )


def build_single_panel(title, child):
    return html.Div(
        # id="control-chart-container",
        className="control-chart-container twelve columns",
        children=[generate_title_banner(title), child],
    )


def build_double_panel(left_title, left_child, right_title, right_child):
    return html.Div(
        # id="top-section-container",
        className="top-section-container row",
        children=[
            # Left Container
            html.Div(
                className="metric-summary-session six columns",
                children=[
                    generate_title_banner(left_title),
                    html.Div(className="metric-div", children=[left_child]),
                ],
            ),
            # Right Container
            html.Div(
                className="metric-summary-session six columns",
                children=[
                    generate_title_banner(right_title),
                    html.Div(className="metric-div", children=[right_child]),
                ],
            ),
        ],
    )


def build_triple_panel(
    left_title, left_child, mid_title, mid_child, right_title, right_child
):
    return html.Div(
        # id="top-section-container",
        className="top-section-container row",
        children=[
            # Left Container
            html.Div(
                className="metric-summary-session four columns",
                children=[
                    generate_title_banner(left_title),
                    html.Div(className="metric-div", children=[left_child]),
                ],
            ),
            # Mid Container
            html.Div(
                className="metric-summary-session four columns",
                children=[
                    generate_title_banner(mid_title),
                    html.Div(className="metric-div", children=[mid_child]),
                ],
            ),
            # Right Container
            html.Div(
                className="metric-summary-session four columns",
                children=[
                    generate_title_banner(right_title),
                    html.Div(className="metric-div", children=[right_child]),
                ],
            ),
        ],
    )
