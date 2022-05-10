# import libraries
from __future__ import division
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import plotly.graph_objs as go
from sklearn.cluster import KMeans


def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(
        drop=True
    )
    df_new['index'] = df_new.index
    df_final = pd.merge(
        df, df_new[[cluster_field_name, 'index']], on=cluster_field_name
    )
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field_name})
    return df_final


def generate_segmentation_fig(df, fig_layout):
    tx_data = df.copy()
    plot_layout = fig_layout
    plot_layout["xaxis"] = dict(showline=False, showgrid=False, zeroline=True)
    # convert the string date field to datetime
    tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

    # we will be using only UK data
    tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)

    # create a generic user dataframe to keep CustomerID and new segmentation scores
    tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
    tx_user.columns = ['CustomerID']

    # get the max purchase date for each customer and create a dataframe with it
    tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID', 'MaxPurchaseDate']

    # we take our observation point as the max invoice date in our dataset
    tx_max_purchase['Recency'] = (
        tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']
    ).dt.days

    # merge this dataframe to our new user dataframe
    tx_user = pd.merge(
        tx_user, tx_max_purchase[['CustomerID', 'Recency']], on='CustomerID'
    )
    # Recency
    recency_fig = go.Figure(
        data=[go.Histogram(x=tx_user['Recency'])], layout=plot_layout
    )

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Recency']])
    tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
    tx_user = order_cluster('RecencyCluster', 'Recency', tx_user, False)

    # get order counts for each user and create a dataframe with it
    tx_frequency = tx_uk.groupby('CustomerID').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['CustomerID', 'Frequency']

    # add this data to our main dataframe
    tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

    # Frequency
    frequency_fig = go.Figure(
        data=[go.Histogram(x=tx_user.query('Frequency < 1000')['Frequency'])],
        layout=plot_layout,
    )

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Frequency']])
    tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
    tx_user = order_cluster('FrequencyCluster', 'Frequency', tx_user, True)

    # calculate revenue for each customer
    tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']
    tx_revenue = tx_uk.groupby('CustomerID').Revenue.sum().reset_index()

    # merge it with our main dataframe
    tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

    # Monetary
    monetary_fig = go.Figure(
        data=[go.Histogram(x=tx_user.query('Revenue < 10000')['Revenue'])],
        layout=plot_layout,
    )
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(tx_user[['Revenue']])
    tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
    tx_user = order_cluster('RevenueCluster', 'Revenue', tx_user, True)

    # Segmentation
    tx_user['OverallScore'] = (
        tx_user['RecencyCluster']
        + tx_user['FrequencyCluster']
        + tx_user['RevenueCluster']
    )
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore'] > 2, 'Segment'] = 'Mid-Value'
    tx_user.loc[tx_user['OverallScore'] > 4, 'Segment'] = 'High-Value'
    tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")
    # Segments(Frequency vs Revenue)
    freq_revenue_fig = go.Figure(
        data=[
            go.Scatter(
                x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
                y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
                mode='markers',
                name='Low',
                marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8),
            ),
            go.Scatter(
                x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
                y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
                mode='markers',
                name='Mid',
                marker=dict(size=9, line=dict(width=1), color='green', opacity=0.5),
            ),
            go.Scatter(
                x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
                y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
                mode='markers',
                name='High',
                marker=dict(size=11, line=dict(width=1), color='red', opacity=0.9),
            ),
        ],
        layout=plot_layout,
    )
    freq_revenue_fig.update_layout(
        yaxis={'title': "Revenue"},
        xaxis={'title': "Frequency"},
    )

    # Segments(Recency vs Revenue)
    recency_revenue_fig = go.Figure(
        data=[
            go.Scatter(
                x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
                y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
                mode='markers',
                name='Low',
                marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8),
            ),
            go.Scatter(
                x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
                y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
                mode='markers',
                name='Mid',
                marker=dict(size=9, line=dict(width=1), color='green', opacity=0.5),
            ),
            go.Scatter(
                x=tx_graph.query("Segment == 'High-Value'")['Recency'],
                y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
                mode='markers',
                name='High',
                marker=dict(size=11, line=dict(width=1), color='red', opacity=0.9),
            ),
        ],
        layout=plot_layout,
    )
    recency_revenue_fig.update_layout(
        yaxis={'title': "Revenue"},
        xaxis={'title': "Recency"},
    )

    # Segments(Recency vs Frequency)
    recency_freq_fig = go.Figure(
        data=[
            go.Scatter(
                x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
                y=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
                mode='markers',
                name='Low',
                marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8),
            ),
            go.Scatter(
                x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
                y=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
                mode='markers',
                name='Mid',
                marker=dict(size=9, line=dict(width=1), color='green', opacity=0.5),
            ),
            go.Scatter(
                x=tx_graph.query("Segment == 'High-Value'")['Recency'],
                y=tx_graph.query("Segment == 'High-Value'")['Frequency'],
                mode='markers',
                name='High',
                marker=dict(size=11, line=dict(width=1), color='red', opacity=0.9),
            ),
        ],
        layout=plot_layout,
    )
    recency_freq_fig.update_layout(
        yaxis={'title': "Frequency"},
        xaxis={'title': "Recency"},
    )
    return (
        recency_fig,
        frequency_fig,
        monetary_fig,
        freq_revenue_fig,
        recency_revenue_fig,
        recency_freq_fig,
    )
