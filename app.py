import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv('winequality-red.csv')
df.columns = df.columns.str.title()

report_content = """
## Insights from the Data

- **Key Indicators**: Alcohol, volatile acidity, and sulphates are among the most important features influencing wine quality.
- **Correlations**: Alcohol shows a strong positive correlation with quality, while volatile acidity shows a negative correlation.
- **Outliers and Skewness**: Some features exhibit significant skewness and outliers, impacting the correlation and regression analysis.

---

### Conclusion

- **Statistical Significance**: Significant differences in alcohol content between high-quality and low-quality wines.
- **Consumer Advice**: Prefer wines with 11.4%-11.7% alcohol for higher quality.
- **Winemaking Tip**: Target the 11-11.5% alcohol range for potentially higher-quality wines.
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
                            value=df.columns[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
                    
                    html.Div([
                        html.Label('Select Feature 2:'),
                        dcc.Dropdown(
                            id='feature-dropdown-2',
                            options=[{'label': feature, 'value': feature} for feature in df.columns[:-1]],
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
                html.H4("Sample Data:"),
                dcc.Markdown(df.head().to_markdown(), style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ccc'}),

                # Additional meaningful graph from the notebook
                html.Div([
                    html.H4("Volatile Acidity vs Alcohol (Impact on Quality)"),
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
        title='Volatile Acidity vs Alcohol (Impact on Quality)',
        labels={'Volatile Acidity': 'Volatile Acidity', 'Alcohol': 'Alcohol'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    return additional_fig

if __name__ == '__main__':
    app.run_server(debug=True)