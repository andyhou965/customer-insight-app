import plotly.graph_objs as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


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


def generate_churn_fig(df, fig_layout):
    df_data = df.copy()
    plot_layout = fig_layout
    plot_layout["xaxis"] = dict(
        showline=False, showgrid=False, zeroline=True, type="category"
    )
    plot_layout["yaxis"] = dict(title="Churn Rate")

    df_data.loc[df_data.Churn == 'No', 'Churn'] = 0
    df_data.loc[df_data.Churn == 'Yes', 'Churn'] = 1

    df_plot = df_data.groupby('gender').Churn.mean().reset_index()
    gender_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['gender'],
                y=df_plot['Churn'],
                width=[0.5, 0.5],
                marker=dict(color=['green', 'blue']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('Partner').Churn.mean().reset_index()
    partner_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['Partner'],
                y=df_plot['Churn'],
                width=[0.5, 0.5],
                marker=dict(color=['green', 'blue']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('PhoneService').Churn.mean().reset_index()
    phone_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['PhoneService'],
                y=df_plot['Churn'],
                width=[0.5, 0.5],
                marker=dict(color=['green', 'blue']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('MultipleLines').Churn.mean().reset_index()
    multiplelines_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['MultipleLines'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('InternetService').Churn.mean().reset_index()
    internet_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['InternetService'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('OnlineSecurity').Churn.mean().reset_index()
    security_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['OnlineSecurity'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('OnlineBackup').Churn.mean().reset_index()
    backup_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['OnlineBackup'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('DeviceProtection').Churn.mean().reset_index()
    protection_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['DeviceProtection'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('TechSupport').Churn.mean().reset_index()
    support_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['TechSupport'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('StreamingTV').Churn.mean().reset_index()
    streamingTV_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['StreamingTV'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('StreamingMovies').Churn.mean().reset_index()
    streamingMovies_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['StreamingMovies'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('Contract').Churn.mean().reset_index()
    contract_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['Contract'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('PaperlessBilling').Churn.mean().reset_index()
    billing_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['PaperlessBilling'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('PaymentMethod').Churn.mean().reset_index()
    payment_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['PaymentMethod'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange', 'red']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.groupby('tenure').Churn.mean().reset_index()
    tenure_fig = go.Figure(
        data=[
            go.Scatter(
                x=df_plot['tenure'],
                y=df_plot['Churn'],
                mode='markers',
                name='Low',
                marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8),
            )
        ],
        layout=plot_layout,
    )

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_data[['tenure']])
    df_data['TenureCluster'] = kmeans.predict(df_data[['tenure']])

    df_data = order_cluster('TenureCluster', 'tenure', df_data, True)
    df_data['TenureCluster'] = df_data["TenureCluster"].replace(
        {0: 'Low', 1: 'Mid', 2: 'High'}
    )

    df_plot = df_data.groupby('TenureCluster').Churn.mean().reset_index()
    plot_layout["xaxis"] = {"type": "category", "categoryarray": ['Low', 'Mid', 'High']}
    tenurecluster_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['TenureCluster'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange', 'red']),
            )
        ],
        layout=plot_layout,
    )

    df_plot = df_data.copy()
    df_plot['MonthlyCharges'] = df_plot['MonthlyCharges'].astype(int)
    df_plot = df_plot.groupby('MonthlyCharges').Churn.mean().reset_index()
    plot_layout["xaxis"] = {'title': "Monthly Charges"}
    plot_layout["yaxis"] = {'title': "Churn Rate"}
    monthlyCharges_fig = go.Figure(
        data=[
            go.Scatter(
                x=df_plot['MonthlyCharges'],
                y=df_plot['Churn'],
                mode='markers',
                name='Low',
                marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8),
            )
        ],
        layout=plot_layout,
    )

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_data[['MonthlyCharges']])
    df_data['MonthlyChargeCluster'] = kmeans.predict(df_data[['MonthlyCharges']])
    df_data = order_cluster('MonthlyChargeCluster', 'MonthlyCharges', df_data, True)
    df_data['MonthlyChargeCluster'] = df_data["MonthlyChargeCluster"].replace(
        {0: 'Low', 1: 'Mid', 2: 'High'}
    )
    df_plot = df_data.groupby('MonthlyChargeCluster').Churn.mean().reset_index()
    plot_layout["xaxis"] = {"type": "category", "categoryarray": ['Low', 'Mid', 'High']}
    plot_layout["yaxis"] = dict(showgrid=True, showline=False, zeroline=True)
    monthlyChargesCluster_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['MonthlyChargeCluster'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    df_data.loc[
        pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull(), 'TotalCharges'
    ] = np.nan
    df_data = df_data.dropna()
    df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')

    df_plot = df_data.copy()
    df_plot['TotalCharges'] = df_plot['TotalCharges'].astype(int)
    df_plot = df_plot.groupby('TotalCharges').Churn.mean().reset_index()
    plot_layout["yaxis"] = {'title': "Churn Rate"}
    plot_layout["xaxis"] = {'title': "Total Charges"}
    totalCharges_fig = go.Figure(
        data=[
            go.Scatter(
                x=df_plot['TotalCharges'],
                y=df_plot['Churn'],
                mode='markers',
                name='Low',
                marker=dict(size=7, line=dict(width=1), color='blue', opacity=0.8),
            )
        ],
        layout=plot_layout,
    )

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_data[['TotalCharges']])
    df_data['TotalChargeCluster'] = kmeans.predict(df_data[['TotalCharges']])
    df_data = order_cluster('TotalChargeCluster', 'TotalCharges', df_data, True)
    df_data['TotalChargeCluster'] = df_data["TotalChargeCluster"].replace(
        {0: 'Low', 1: 'Mid', 2: 'High'}
    )
    df_plot = df_data.groupby('TotalChargeCluster').Churn.mean().reset_index()
    plot_layout["xaxis"] = {"type": "category", "categoryarray": ['Low', 'Mid', 'High']}
    plot_layout["yaxis"] = dict(showgrid=True, showline=False, zeroline=True)
    totalChargeCluster_fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot['TotalChargeCluster'],
                y=df_plot['Churn'],
                width=[0.5, 0.5, 0.5],
                marker=dict(color=['green', 'blue', 'orange']),
            )
        ],
        layout=plot_layout,
    )

    # import Label Encoder
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    dummy_columns = []  # array for multiple value columns

    for column in df_data.columns:
        if df_data[column].dtype == object and column != 'customerID':
            if df_data[column].nunique() == 2:
                # apply Label Encoder for binary ones
                df_data[column] = le.fit_transform(df_data[column])
            else:
                dummy_columns.append(column)

    # apply get dummies for selected columns
    df_data = pd.get_dummies(data=df_data, columns=dummy_columns)

    all_columns = []
    for column in df_data.columns:
        column = (
            column.replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("-", "_")
        )
        all_columns.append(column)

    df_data.columns = all_columns
    glm_columns = 'gender'

    for column in df_data.columns:
        if column not in ['Churn', 'customerID', 'gender']:
            glm_columns = glm_columns + ' + ' + column

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    glm_model = smf.glm(
        formula='Churn ~ {}'.format(glm_columns),
        data=df_data,
        family=sm.families.Binomial(),
    )
    res = glm_model.fit()

    # create feature set and labels
    X = df_data.drop(['Churn', 'customerID'], axis=1)
    y = df_data.Churn
    # train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=56
    )
    xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.08, objective='binary:logistic', n_jobs=-1
    ).fit(X_train, y_train)

    df_data['proba'] = xgb_model.predict_proba(df_data[X_train.columns])[:, 1]

    return (
        gender_fig,
        partner_fig,
        phone_fig,
        multiplelines_fig,
        internet_fig,
        security_fig,
        backup_fig,
        protection_fig,
        support_fig,
        streamingTV_fig,
        streamingMovies_fig,
        contract_fig,
        billing_fig,
        payment_fig,
        tenure_fig,
        tenurecluster_fig,
        monthlyCharges_fig,
        monthlyChargesCluster_fig,
        totalCharges_fig,
        totalChargeCluster_fig,
        df_data,
    )
