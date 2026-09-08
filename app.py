import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent / 'winequality-red.csv')
df.columns = df.columns.str.title()

report_content = """
## Explore the wine quality dataset

Choose two measurements to inspect their distributions and relationship to the
quality rating. Each point represents a record in the supplied red-wine dataset.

This is an educational analysis of observational data. Associations in this sample
do not establish causes or support recommendations for changing a wine's alcohol
content. The notebook contains the statistical modelling work; this dashboard
explores the recorded measurements and does not generate predictions.
"""

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Wine Quality Analysis Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        dcc.Tab(label='Analysis Overview', children=[
            html.Div([
                dcc.Markdown(report_content, style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc'}),

                html.Div([
                    html.Div([
                        html.Label('Select Feature 1:'),
                        dcc.Dropdown(
                            id='feature-dropdown-1',
                            options=[{'label': feature, 'value': feature} for feature in df.columns[:-1]],
                            clearable=False,
                            value=df.columns[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
                    
                    html.Div([
                        html.Label('Select Feature 2:'),
                        dcc.Dropdown(
                            id='feature-dropdown-2',
                            options=[{'label': feature, 'value': feature} for feature in df.columns[:-1]],
                            clearable=False,
                            value=df.columns[1]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'})
                ], style={'padding': '20px 0'}),

                
                html.Div([
                    html.Div([
                        dcc.Graph(id='histogram-plot')
                    ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
                    
                    html.Div([
                        dcc.Graph(id='scatter-plot')
                    ], style={'width': '48%', 'display': 'inline-block'})
                ]),
            ])
        ]),
        
        dcc.Tab(label='Data & Additional Insights', children=[
            html.Div([
                html.H4("Sample data"),
                dcc.Markdown(df.head().to_markdown(), style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc'}),

                # Additional meaningful graph from the notebook
                html.Div([
                    html.H4("Volatile acidity and alcohol by quality rating"),
                    dcc.Graph(id='additional-plot')
                ], style={'padding': '20px 0'})
            ])
        ])
    ])
])

@app.callback(
    [Output('histogram-plot', 'figure'),
     Output('scatter-plot', 'figure')],
    [Input('feature-dropdown-1', 'value'),
     Input('feature-dropdown-2', 'value')]
)
def update_plots(feature1, feature2):
    feature1 = feature1 if feature1 in df.columns[:-1] else df.columns[0]
    feature2 = feature2 if feature2 in df.columns[:-1] else df.columns[1]
    hist_fig = px.histogram(
        df, x=feature1, color='Quality', barmode='overlay',
        title=f'Histogram of {feature1}',
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    
    scatter_fig = px.scatter(
        df, x=feature1, y=feature2, color='Quality',
        title=f'{feature1} vs {feature2}',
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    
    return hist_fig, scatter_fig

@app.callback(
    Output('additional-plot', 'figure'),
    Input('additional-plot', 'id')
)
def generate_additional_plot(_):
    additional_fig = px.scatter(
        df, x='Volatile Acidity', y='Alcohol', color='Quality',
        title='Volatile acidity and alcohol by quality rating',
        labels={'Volatile Acidity': 'Volatile Acidity', 'Alcohol': 'Alcohol'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    return additional_fig

if __name__ == '__main__':
    app.run(debug=False)
