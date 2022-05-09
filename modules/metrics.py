#%%
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import pathlib

# import plotly.offline as pyoff

# pyoff.init_notebook_mode()

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
data_file = os.path.join(APP_PATH, os.path.join("OnlineRetail.csv"))

#%%
plot_layout = {
    # "paper_bgcolor": "rgba(0,0,0,0)",
    # "plot_bgcolor": "rgba(0,0,0,0)",
    "xaxis": dict(showline=False, showgrid=False, zeroline=True, type="category"),
    "yaxis": dict(showgrid=True, showline=False, zeroline=True),
    "autosize": True,
    # "title_text": "Montly Revenue",
    # "title_font_color": "yellow",
    # "font_color": "yellow",
    "template": "plotly_dark",
}

tx_data = pd.read_csv(data_file, encoding='unicode_escape')
# converting the type of Invoice Date Field from string to datetime.
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
# creating YearMonth field for the ease of reporting and visualization
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(
    lambda date: 100 * date.year + date.month
)
# calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']
tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()

# Monthly Revenue Plot
monthly_revenue_fig = go.Figure(
    data=[
        go.Scatter(
            x=tx_revenue['InvoiceYearMonth'],
            y=tx_revenue['Revenue'],
        )
    ],
    layout=plot_layout,
)
monthly_revenue_fig.update_layout(title="Monthly Revenue")
# dcc.Graph(figure=fig)

# Monthly Growth Rate Plotly
tx_revenue['MonthlyGrowth'] = tx_revenue['Revenue'].pct_change()

monthly_growth_fig = go.Figure(
    data=[
        go.Scatter(
            x=tx_revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
            y=tx_revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
        )
    ],
    layout=plot_layout,
)
monthly_growth_fig.update_layout(title='Monthly Growth Rate')

# Monthly Active Customers Plot
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
# creating monthly active customers dataframe by counting unique Customer IDs
tx_monthly_active = (
    tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
)

monthly_active_customers_fig = go.Figure(
    data=[
        go.Bar(
            x=tx_monthly_active['InvoiceYearMonth'],
            y=tx_monthly_active['CustomerID'],
        )
    ],
    layout=plot_layout,
)
monthly_active_customers_fig.update_layout(title='Monthly Active Customers')

# Monthly Total # of Order Plot
tx_monthly_sales = tx_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()
monthly_order_number_fig = go.Figure(
    data=[
        go.Bar(
            x=tx_monthly_sales['InvoiceYearMonth'],
            y=tx_monthly_sales['Quantity'],
        )
    ],
    layout=plot_layout,
)
monthly_order_number_fig.update_layout(title='Monthly Total # of Order')

# Monthly Order Average Plot
tx_monthly_order_avg = tx_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()
monthly_avg_order_fig = go.Figure(
    data=[
        go.Bar(
            x=tx_monthly_order_avg['InvoiceYearMonth'],
            y=tx_monthly_order_avg['Revenue'],
        )
    ],
    layout=plot_layout,
)
monthly_avg_order_fig.update_layout(title='Monthly Order Average')
# # %%
# # create a dataframe contaning CustomerID and first purchase date
# tx_min_purchase = tx_uk.groupby('CustomerID').InvoiceDate.min().reset_index()
# tx_min_purchase.columns = ['CustomerID', 'MinPurchaseDate']
# tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(
#     lambda date: 100 * date.year + date.month
# )

# # merge first purchase date column to our main dataframe (tx_uk)
# tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')

# tx_uk.head()

# # create a column called User Type and assign Existing
# # if User's First Purchase Year Month before the selected Invoice Year Month
# tx_uk['UserType'] = 'New'
# tx_uk.loc[
#     tx_uk['InvoiceYearMonth'] > tx_uk['MinPurchaseYearMonth'], 'UserType'
# ] = 'Existing'

# # calculate the Revenue per month for each user type
# tx_user_type_revenue = (
#     tx_uk.groupby(['InvoiceYearMonth', 'UserType'])['Revenue'].sum().reset_index()
# )

# # filtering the dates and plot the result
# tx_user_type_revenue = tx_user_type_revenue.query(
#     "InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112"
# )
# plot_data = [
#     go.Scatter(
#         x=tx_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
#         y=tx_user_type_revenue.query("UserType == 'Existing'")['Revenue'],
#         name='Existing',
#     ),
#     go.Scatter(
#         x=tx_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
#         y=tx_user_type_revenue.query("UserType == 'New'")['Revenue'],
#         name='New',
#     ),
# ]

# plot_layout = go.Layout(xaxis={"type": "category"}, title='New vs Existing')
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)
# # %%
# # create a dataframe that shows new user ratio - we also need to drop NA values (first month new user ratio is 0)
# tx_user_ratio = (
#     tx_uk.query("UserType == 'New'")
#     .groupby(['InvoiceYearMonth'])['CustomerID']
#     .nunique()
#     / tx_uk.query("UserType == 'Existing'")
#     .groupby(['InvoiceYearMonth'])['CustomerID']
#     .nunique()
# )
# tx_user_ratio = tx_user_ratio.reset_index()
# tx_user_ratio = tx_user_ratio.dropna()

# # print the dafaframe
# tx_user_ratio

# # plot the result

# plot_data = [
#     go.Bar(
#         x=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")[
#             'InvoiceYearMonth'
#         ],
#         y=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")[
#             'CustomerID'
#         ],
#     )
# ]

# plot_layout = go.Layout(xaxis={"type": "category"}, title='New Customer Ratio')
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)
# # %%
# # identify which users are active by looking at their revenue per month
# tx_user_purchase = (
#     tx_uk.groupby(['CustomerID', 'InvoiceYearMonth'])['Revenue'].sum().reset_index()
# )

# # create retention matrix with crosstab
# tx_retention = pd.crosstab(
#     tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']
# ).reset_index()

# tx_retention.head()

# # create an array of dictionary which keeps Retained & Total User count for each month
# months = tx_retention.columns[2:]
# retention_array = []
# for i in range(len(months) - 1):
#     retention_data = {}
#     selected_month = months[i + 1]
#     prev_month = months[i]
#     retention_data['InvoiceYearMonth'] = int(selected_month)
#     retention_data['TotalUserCount'] = tx_retention[selected_month].sum()
#     retention_data['RetainedUserCount'] = tx_retention[
#         (tx_retention[selected_month] > 0) & (tx_retention[prev_month] > 0)
#     ][selected_month].sum()
#     retention_array.append(retention_data)

# # convert the array to dataframe and calculate Retention Rate
# tx_retention = pd.DataFrame(retention_array)
# tx_retention['RetentionRate'] = (
#     tx_retention['RetainedUserCount'] / tx_retention['TotalUserCount']
# )

# # plot the retention rate graph
# plot_data = [
#     go.Scatter(
#         x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
#         y=tx_retention.query("InvoiceYearMonth<201112")['RetentionRate'],
#         name="organic",
#     )
# ]

# plot_layout = go.Layout(xaxis={"type": "category"}, title='Monthly Retention Rate')
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.iplot(fig)
# # %%
# # create our retention table again with crosstab() and add firs purchase year month view
# tx_retention = pd.crosstab(
#     tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']
# ).reset_index()
# tx_retention = pd.merge(
#     tx_retention,
#     tx_min_purchase[['CustomerID', 'MinPurchaseYearMonth']],
#     on='CustomerID',
# )
# new_column_names = ['m_' + str(column) for column in tx_retention.columns[:-1]]
# new_column_names.append('MinPurchaseYearMonth')
# tx_retention.columns = new_column_names

# # create the array of Retained users for each cohort monthly
# retention_array = []
# for i in range(len(months)):
#     retention_data = {}
#     selected_month = months[i]
#     prev_months = months[:i]
#     next_months = months[i + 1 :]
#     for prev_month in prev_months:
#         retention_data[prev_month] = np.nan

#     total_user_count = tx_retention[
#         tx_retention.MinPurchaseYearMonth == selected_month
#     ].MinPurchaseYearMonth.count()
#     retention_data['TotalUserCount'] = total_user_count
#     retention_data[selected_month] = 1

#     query = "MinPurchaseYearMonth == {}".format(selected_month)

#     for next_month in next_months:
#         new_query = query + " and {} > 0".format(str('m_' + str(next_month)))
#         retention_data[next_month] = np.round(
#             tx_retention.query(new_query)['m_' + str(next_month)].sum()
#             / total_user_count,
#             2,
#         )
#     retention_array.append(retention_data)

# tx_retention = pd.DataFrame(retention_array)
# tx_retention.index = months

# # showing new cohort based retention table
# tx_retention
# %%
