import dash
from dash import html, dcc, Input, Output, dash_table
import pandas as pd
import plotly.express as px

# Ler o arquivo CSV
df = pd.read_csv('database/data.csv', parse_dates=['Date'])
df['Filter Data'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# df['Formatted Date'] = df['Date'].dt.strftime('%d/%m/%Y')

# Remover a coluna 'Hash' do DataFrame
df = df.drop(columns=['Hash'])


# Inicializar a aplicação Dash
app = dash.Dash(__name__)

# Layout da aplicação
app.layout = html.Div([
    # Seção dos filtros
    html.Div([
        # Filtro para a Categoria
        html.Div([
            html.Label('Categoria:'),
            dcc.Dropdown(
                id='filter-category',
                options=[{'label': i, 'value': i}
                         for i in df['Category'].unique()],
                value=None,
                multi=True
            )
        ]),
        # Filtro para o Tipo de Cartão
        html.Div([
            html.Label('Tipo de Cartão:'),
            dcc.Dropdown(
                id='filter-card-type',
                options=[{'label': i, 'value': i}
                         for i in df['Card type'].unique()],
                value=None,
                multi=True
            )
        ]),
        # Filtro para o Tipo (Type)
        html.Div([
            html.Label('Tipo:'),
            dcc.Dropdown(
                id='filter-type',
                options=[{'label': i, 'value': i}
                         for i in df['Type'].unique()],
                value=None,
                multi=True
            )
        ]),
        # Filtro para a Data
        html.Div([
            html.Label('Data:'),
            dcc.DatePickerRange(
                id='filter-date',
                start_date=df['Filter Data'].min(),
                end_date=df['Filter Data'].max()
            )
        ]),
    ]),
    # Gráfico de barras para Total por Categoria
    dcc.Graph(id='category-bar-chart'),
    # Gráfico de barras para Total por Tipo (Receita e Despesa)
    dcc.Graph(id='type-bar-chart'),
    # Tabela para exibir os dados
    dash_table.DataTable(
        id='table',
        columns=[{"name": "Data", "id": "Date"},
                 {"name": "Descrição", "id": "Description"},
                 {"name": "Preço", "id": "Price"},
                 {"name": "Categoria", "id": "Category"},
                 {"name": "Cartão", "id": "Card type"},
                 {"name": "Tipo", "id": "Type"}
                 ],
        data=df.to_dict('records'),
        page_size=10,
    )
])

# Callback para atualizar a tabela e os gráficos com base nos filtros


@app.callback(
    [Output('table', 'data'), Output('category-bar-chart',
                                     'figure'), Output('type-bar-chart', 'figure')],
    [Input('filter-category', 'value'),
     Input('filter-card-type', 'value'),
     Input('filter-type', 'value'),
     Input('filter-date', 'start_date'),
     Input('filter-date', 'end_date')]
)
def update_content(selected_category, selected_card_type, selected_type, start_date, end_date):
    filtered_df = df
    if selected_category:
        filtered_df = filtered_df[filtered_df['Category'].isin(
            selected_category)]
    if selected_card_type:
        filtered_df = filtered_df[filtered_df['Card type'].isin(
            selected_card_type)]
    if selected_type:
        filtered_df = filtered_df[filtered_df['Type'].isin(selected_type)]
    if start_date and end_date:
        # As datas de início e fim são strings no formato ISO, então convertemos para datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # Filtrar baseado na coluna 'Filter Data'
        filtered_df = filtered_df[(filtered_df['Filter Data'] >= start_date) &
                                  (filtered_df['Filter Data'] <= end_date)]

    table_data = filtered_df.to_dict('records')

    # Criar o gráfico de barras para as categorias
    category_sum = filtered_df.groupby('Category')['Price'].sum().reset_index()
    category_fig = px.bar(category_sum, x='Category', y='Price',
                          color='Category', title='Total por Categoria')

    # Criar o gráfico de barras para Receita e Despesa com cores diferentes
    type_sum = filtered_df.groupby('Type')['Price'].sum().reset_index()
    type_fig = px.bar(type_sum, x='Type', y='Price',
                      color='Type', title='Total por Tipo')

    return table_data, category_fig, type_fig


# Rodar a aplicação
if __name__ == '__main__':
    app.run_server(debug=True)
