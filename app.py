import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np

import nltk
# RUN python -m nltk.downloader popular
nltk.download('popular')
import os

import plotly.express as px
import plotly.graph_objects as go

from covanalysis import Covid

# import sys
# import warnings
#
# if not sys.warnoptions:  # allow overriding with `-W` option
#     warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')


# # nltk_path=os.path.abspath('./nltk_data')
# # #print('My Path: ', path)
# nltk.data.path.append('./nltk_data')


external_stylesheets =['https://codepen.io/amyoshino/pen/jzXypZ.css']
#['https://codepen.io/chriddyp/pen/bWLwgP.css']

theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server


markdown_text='''
#### Project Description

In this project we analyze COVID-19 data collected during the early stages of
the virus.  

<font size='4'><b>Data Source:</b></font>
<a href="https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset" >www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset</a>
<br>
<b>File Name:</b> COVID19_line_list_data.csv

'''

#read in the data

df=pd.read_csv('COVID19_data.csv')
# countries=pd.read_csv('countries.csv')
# china_cities=pd.read_csv('china_cities.csv')


cov=Covid(df)
df=cov.create_master_frame(df)


sym = cov.symptom_fequency(df, 'symptom', 34)
sym_freq = nltk.FreqDist(sym)

perc_sym=cov.perc_symptoms(sym)
#print(perc_sym)
group_sym=df['symptom'].dropna()
group_sym=[', '.join(set([x.strip() for x in item.split(',')])) for item in group_sym]
group_freq=nltk.FreqDist(group_sym)

group_perc_sym=cov.group_perc_symptoms(group_freq)

sym_to_death=cov.symptom_to_death(df)
sym_to_rec=cov.symptom_to_recovered(df)


colors={
    'background':'#111111',
    'text':'#2FDBFF'
}


options={
    'Symptom Frequency':['Individual','Group'],
    'Timeline':['Death Timeline','Recovery Timeline'],
    'Map Trace':['Main Map'],
    'Predictions':['Feature Importance']
        }

symp_freq_data={
    'Individual':{'x':[sym[0] for sym in sym_freq.most_common(15)],
                  'y':[sym[1] for sym in sym_freq.most_common(15)]},
    'Group':{'x':[sym[0] for sym in group_freq.most_common(15)],
             'y':[sym[1] for sym in group_freq.most_common(15)]}
                }

mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(
        l=30,
        r=30,
        b=20,
        t=40
    ),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Satellite Overview',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="dark",
        center=dict(
            lon=-78.05,
            lat=42.54
        ),
        zoom=7,
    )
),





rootLayout = html.Div([
    daq.BooleanSwitch(
        on=True,
        id='darktheme-daq-booleanswitch',
        className='dark-theme-control'
    ), html.Br()
    ])


#creating the layout of the app

app.layout=html.Div(id='dark-theme-container',
                    style={'backgroundColor':'#3E3B3A'},
                    children=[

    #we define the FIRST ROW
    html.Div(
        children=[
        html.H2(
            children='COVID-19 Dashboard!',
            style={'padding': '10px', 'backgroundColor': "#312E2D",#'#2a3f5f',
                        'box-shadow': '2px 4px #090909',
                   "color": "white", "font-family": "Helvetica",
                   "font-weight": "bolder",'textAlign':'center'}
        ),
            dcc.Markdown(dangerously_allow_html=True,
                children=markdown_text,
                         style={'box-shadow': '2px 7px #090909',
                                'padding':'3px',
                                'margin-left':'1%',
                                'margin-right':'1%',
                                'font-size':'16px',
                                'backgroundColor':'#312E2D',
                                'color':'white'
                                #'margin-bottom':'2%'
                                })

    ],className='row'),
        html.Hr(),

    #we create ROW TWO
    html.Div(
             children=[
        #we create COLUMN ONE of ROW TWO
        html.Div(style={'backgroundColor':'#3E3B3A'},
                 children=[
            #Sub Row 1

            html.Label('Main Information',
                       style={'font-size': '22px',
                              #'backgroundColor': '#2a3f5f',
                              'color':'white',
                                'box-shadow': '2px 7px #090909',
                              'padding':'15px',
                                "font-family": "Helvetica",
                                "font-weight": "bolder",
                              'backgroundColor': '#312E2D',
                              'margin-bottom':'3%'


                              }),
            dcc.RadioItems(
                id='main_info-radio',
                options=[{'label':k,'value':k} for k in options.keys()
                         ],
                value='Symptom Frequency',
                style={'box-shadow': '3px 7px #090909',
                       'font-size': '18px',
                       'backgroundColor':'#312E2D',
                       'padding':'5px',
                       'color':'white'},

                #labelStyle={'padding':'10px','display': 'inline-block'}

            ),
            html.Hr(),
            html.Label('Select Type',
                       style={'font-size': '20px',
                              'backgroundColor': '#312E2D',
                                'box-shadow': '3px 7px #090909',
                              'margin-bottom':'3%',
                              'color': 'white',
                              'padding': '15px',
                              "font-family": "Helvetica",
                              "font-weight": "bolder",

                              }),
            dcc.RadioItems(id='type_info-dropdown',
                         style={'width': '97%',
                                'backgroundColor': '#312E2D',
                                'box-shadow': '2px 10px #090909',
                                'padding': '5px',
                                'color': 'white',
                                #'font-weight':'bolder',
                                #'box-shadow': '3px 4px #2a3f5f',
                                'font-size': '18px',
                                'margin-bottom':'2%'
                                #"border": "2px black solid",
                                # 'float': 'top-center'
                             }
                         ),
             #html.Hr()
            # html.Label('Type of Graph'),
            # dcc.RadioItems(
            #     id='subgraph-types',
            #     options=[
            #         {'label': 'Histogram', 'value': 'histogram'},
            #         {'label': 'Box Plot', 'value': 'box'},
            #         {'label': 'Scatter Plot', 'value': 'scatter'}
            #     ],
            #     value='histogram',
            #     # labelStyle={'padding':'10px','display': 'inline-block'}
            #
            # )

        ],className='two columns'),


        #COLUMN TWO of ROW ONE
        html.Div([

            #html.Label('City'),


            dcc.Graph(id='main-graph',
                      hoverData={'points':[{'label':'fever'},{'label':'Germany'}]},
                      figure={"layout": {
                                    #"title": "Main Screen",
                                    "height": 500,  # px
                                    "width":650
                                },
                            },
                      style={'padding': '3px',
                             'backgroundColor': '#888888',
                                'box-shadow': '7px 13px #594747',
                             "color": "grey",
                             "font-weight": "bolder",
                             'margin-left':'2%'},

                      ),

           html.Hr(),

        ],className='pretty_container six columns'),

        #Column 3 of ROW 2
        html.Div(
            children=[
            dcc.Graph(id='side-graph',
                      figure={"layout": {
                          #"title": "Main Screen",
                          "height": 450,  # px
                          "width": 350
                                },
                            },
                      style={'padding': '3px',
                                'box-shadow': '10px 12px #533B36',
                             'backgroundColor': '#888888',
                             "color": "white",
                             "font-weight": "bolder",
                             'margin-left':'3%',
                             'margin-right':'7%',
                             'margin-top':'5%'},

                      ),
        ],className='four columns')

    ]),

])

# html.Div(id='dark-theme-components', children=[
#         daq.DarkThemeProvider(theme=theme, children=rootLayout)
#     ], style={'border': 'solid 1px #A2B1C6', 'border-radius': '5px', 'padding': '50px', 'margin-top': '20px'})
# ], style={'padding': '50px'})


@app.callback(
    Output('type_info-dropdown','options'),
    [Input('main_info-radio','value')]
)

def update_options(selected_info):
    return [{'label':i,'value':i} for i in options[selected_info]]


@app.callback(
    Output('type_info-dropdown','value'),
    [Input('type_info-dropdown','options')]
)

def set_value(available_options):
    return available_options[0]['value']


@app.callback(
    Output('main-graph','figure'),
    [Input('type_info-dropdown','value')]
)

def update_graph(selected_type):


    if selected_type=='Individual':


        perc_sym.loc[perc_sym['count'] < 5, 'symptom'] = 'other symptoms'
        #print(perc_sym)
        #figure = px.pie(perc_sym, values='count', names='symptom', title='Symptoms')
        data = [dict(type='pie',
                    values=perc_sym['count'],
                     labels=perc_sym['symptom'],
                     customdata=perc_sym['symptom'],
                     textposition='inside',
                     uniformtext_minsize=12,
                     uniformtext_mode='hide',
                     ),
                ]
        figure={'data':data,
                'layout':{'margin':{'b':50, 'r':30},
                          'title':{'text':"Frequency of Symptoms",
                                   'font': dict(family='Sherif',
                                               size=22,
                                               color='white')
                                                        },
                          'paper_bgcolor':'#201D1C',
                          'text_color':'white',
                          'marker':{'color':'white'},
                          'uniformtext_mode':'hide',
                          'legend':{'font':dict(
                                        color='white'
                                                )}
                          }
                }

    elif selected_type=='Group':
        group_perc_sym.loc[group_perc_sym['count']<3,'symptom']='other symptoms'

        data = [dict(type='pie',
                    values=group_perc_sym['count'],
                     labels=group_perc_sym['symptom'],
                     customdata=group_perc_sym['symptom'],
                     color='white',
                     textposition='inside',
                     uniformtext_minsize=12,
                     uniformtext_mode='hide',
                     textcolor='white'

                     )]
        figure={'data':data,
                'layout':{'margin':{'b':50, 'r':30},
                          'title':{'text':"Frequency of Grouped Symptoms",
                                   'font': dict(family='Sherif',
                                               size=22,
                                               color='white')
                                                        },
                          'paper_bgcolor':'#201D1C',
                          'text_color':'white',
                          'marker':{'color':'white'},
                          'uniformtext_mode':'hide',
                          'legend':{'font':dict(
                                        color='white'
                                                )}
                          }
                }
        #figure.update_traces(textposition='inside')
        #figure = px.pie(group_perc_sym, values='count', names='symptom', title='Grouped Symptoms')

    elif selected_type=='Death Timeline':
        df = pd.melt(sym_to_death[['symptom_to_death', 'symptom_to_hosp', 'hosp_to_death']])

        gender = sym_to_death['gender'].tolist() * 3
        df['gender'] = gender
        symptom = sym_to_death['symptom'].tolist() * 3
        df['symptom'] = symptom
        df.rename(columns={'value': 'Days'}, inplace=True)

        figure = px.box(data_frame=df, x='variable', y='Days', color='gender', notched=True, points='all',
                     hover_data=['symptom'],template='plotly_dark',
                        title='Timeline to Death')
        # fig.update_yaxes(title='Days')
        figure.update_xaxes(title=None, tickvals=[-0.5, 0.5, 1.6],
                         ticktext=['Symptom Onset to Death', 'Symptom Onset to Hospitalization',
                                   'Hospitalization to Death'], tickangle=10)

    elif selected_type=='Recovery Timeline':
        df = pd.melt(sym_to_rec[['symptom_to_recovered', 'symptom_to_hosp', 'hosp_to_recovered']])
        gender = sym_to_rec['gender'].tolist() * 3
        symptom = sym_to_rec['symptom'].tolist() * 3
        df['symptom'] = symptom
        df['gender'] = gender
        df.rename(columns={'value': 'Days'}, inplace=True)
        figure = px.box(data_frame=df, x='variable', y='Days', color='gender',
                     notched=True, points='all', hover_data=['symptom'],
                        template='plotly_dark', title='Timeline to Recovery')
        # fig.update_yaxes(title='Days')
        figure.update_xaxes( title=None,
                        tickvals=[-0.5, 0.5, 1.6],
                         ticktext=['Symptom Onset to Recovery', 'Symptom Onset to Hospitalization',
                                   'Hospitalization to Recovery'], tickangle=10)

    elif selected_type == 'Main Map':

        df = pd.read_csv('COVID19_data.csv')
        countries = pd.read_csv('countries.csv')
        china_cities = pd.read_csv('china_cities.csv')

        cov = Covid(df)
        df = cov.create_master_frame(df)

        vis_wuhan = df[(df['visiting Wuhan'] == 1) & (df['country'] == 'China')]['location'].unique().tolist()
        countries_names = countries['name'].tolist()

        figure = go.Figure(go.Scattermapbox(
                    mode="markers+lines",
                    marker={'size': 12},
                    customdata=df['country']
                    ))

        wuhan_lon = 114.283333
        wuhan_lat = 30.583332

        for country in df['country'].unique():
            if country in countries_names:
                latitude = countries[countries['name'] == country]['latitude'].tolist()[0]
                longitude = countries[countries['name'] == country]['longitude'].tolist()[0]
                #         print(latitude)
                size = df[df['country'] == country]['pop'].tolist()[0] * .2

                figure.add_trace(go.Scattermapbox(
                    mode="markers+lines",
                    lon=[wuhan_lon, longitude],
                    lat=[wuhan_lat, latitude],
                    marker={'size': size,'color':'red'},
                    hoverinfo='text',
                    text='<br>Country: {}<br>'.format(country)+'Confirmed Cases: {}'.format(size//0.2),
                    name=country

                     ))


        for city in vis_wuhan[1:]:
            if city in china_cities['city_name'].tolist():
                #         print(city)

                city_lon = china_cities[china_cities['city_name'] == city]['lon'].tolist()[0]
                city_lat = china_cities[china_cities['city_name'] == city]['lat'].tolist()[0]

                figure.add_trace(go.Scattermapbox(
                    mode="markers+lines",
                    lon=[wuhan_lon, city_lon],
                    lat=[wuhan_lat, city_lat],
                    marker={'size': 8,'color':'black'},
                    hoverinfo='text',
                    text=city,
                    name=city

                ))

        figure.update_layout(
            margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
            mapbox={
                'center': {'lon': wuhan_lon, 'lat': wuhan_lat},
                'style': "stamen-terrain",
                'center': {'lon': wuhan_lon, 'lat': wuhan_lat},
                'zoom': 1})


    elif selected_type=='Feature Importance':

        from sklearn.ensemble import RandomForestClassifier

        X=pd.read_csv('covid_tree.csv',index_col=[0])
        y=pd.read_csv('covid_tree_y.csv')
        #y=np.array(y)

        rf_model = RandomForestClassifier(n_estimators=150, max_features=6)
        rf_model.fit(X,y)
        #rf_model=joblib.load('rf_model.pkl')

        feat_imp = pd.DataFrame(rf_model.feature_importances_,
                                columns=['Importance'], index=X.columns)
        feat_imp=feat_imp[feat_imp['Importance']>0].sort_values(by='Importance',ascending=False)
        #print(feat_imp)

        figure = px.bar(feat_imp, y='Importance', template='plotly_dark',
                        title='Feature Importance',
                        custom_data=feat_imp.columns)

        figure.update_xaxes(tickvals=list(range(len(feat_imp.index))),
                            ticktext=feat_imp.index)
        figure.update_yaxes(title_text='Average decrease in impurity')



    return figure

@app.callback(
    dash.dependencies.Output('side-graph','figure'),
    [dash.dependencies.Input('main-graph','hoverData'),
    dash.dependencies.Input('type_info-dropdown','value')]
)

def update_graph(hoverData,selected_type):
    #print(hoverData['points'][0]['label'])


    if selected_type=='Individual':
        symptom = hoverData['points'][0]['label']
        key = {'1': 'Died', '0': 'Recovered'}
        # print(df[])
        df1 = df.copy()
        if symptom == 'other symptoms' or symptom == 'no symptoms':
            symptom = 'fever'

        figure={
                'data': [
                    dict(
                        x=df1[df1['final_status'] == i][symptom],
                        y=df1[df1['final_status'] == i]['age'],
                        #text=df[df['final_status']==i]['gender'],
                        mode='markers',
                        opacity=1,
                        type='box',
                        boxpoints='all',
                        notched=False,
                        jitter='jitter',
                        name=key[str(i)],
                    ) for i in [0,1]
                        ],
                'layout':dict(
                        xaxis={'title':{'text':symptom,
                               'font':{'color':'white'},
                                        },
                               'tickfont':dict(color="white"),
                               'gridcolor':'grey',
                               'showgrid':True,
                               'tick0':0

                               },
                        yaxis={'title':{ 'text':'Age',
                                         'font':{'color':'white'}},
                               'tickfont':dict(color='white'),
                               'gridcolor':'grey',
                               'showgrid':True
                               },
                        title = {'text':'Distribution of age by {} and status'.format(symptom),
                                 'font':dict(family='Sherif',
                                          size=18,
                                          color='white')
                                 },
                        paper_bgcolor= '#201D1C',
                        plot_bgcolor='#201D1C',
                        text_color= 'white',
                        marker ={'color': 'white'},
                        legend= {'font': dict(
                            color='white'
                                    )},
                        margin = {'l': 80, 'b': 100, 't': 100, 'r': 50},

                )

                    }

    elif selected_type=='Group':
        symptom = hoverData['points'][0]['label']
        key = {'1': 'Died', '0': 'Recovered'}
        # print(df[])
        df1 = df.copy()
        if symptom == 'other symptoms' or symptom == 'no symptoms':
            symptom = 'fever'
        # df1[['age','final_status']].dropna(inplace=True)
        #
        # figure=px.box(data_frame=df1[df1['symptom']==symptom],
        #               x='final_status',y='age',points='all')
        figure = {
            'data': [
                dict(
                    #x=df1[df1['final_status'] == i][symptom],
                    y=df1[(df1['gender'] == i) & (df1['symptom']==symptom)]['age'],
                    text= 'Status (0=Died, 1=Recovered, nan=Unknown): '+df[df['gender']==i]['final_status'].astype(str),
                    mode='markers',
                    opacity=5,
                    type='box',
                    boxpoints='all',
                    notched=False,
                    jitter='jitter',
                    name=i,
                ) for i in ['male','female']
            ],
            'layout': dict(
                xaxis={'title': {'text': symptom,
                                 'font': {'color': 'white'},
                                 },
                       'tickfont': dict(color="white"),
                       'gridcolor': 'grey',
                       'showgrid': True,
                       'tick0': 0

                       },
                yaxis={'title': {'text': 'Age',
                                 'font': {'color': 'white'}},
                       'tickfont': dict(color='white'),
                       'gridcolor': 'grey',
                       'showgrid': True
                       },
                title={'text': 'Distribution of age by {} and status'.format(symptom),
                       'font': dict(family='Sherif',
                                    size=18,
                                    color='white')
                       },
                paper_bgcolor='#201D1C',
                plot_bgcolor='#201D1C',
                text_color='white',
                marker={'color': 'white'},
                legend={'font': dict(
                    color='white'
                )},
                margin={'l': 80, 'b': 100, 't': 100, 'r': 50},

            )

        }

    elif selected_type=='Death Timeline':
        figure={
                'data': [
                    dict(
                        x=sym_to_death[sym_to_death['gender'] == i]['symptom_to_death'],
                        y=sym_to_death[sym_to_death['gender'] == i]['age'],
                        text='Symptom Onset: '+sym_to_death[sym_to_death['gender']==i]['symptom_onset'].astype(str)+'<br>'+\
                        'Recovered On: '+sym_to_death[sym_to_death['gender']==i]['death'].astype(str)+'<br>'+\
                            'Symptoms:'+sym_to_death[sym_to_death['gender']==i]['symptom'],
                        mode='markers',
                        opacity=1,
                        type='scatter',
                        marker={
                        'colorscale':'Viridis',
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'black'},
                        },
                        # meanline_visible='True',
                        # hover_data=df.columns,
                        name=i,
                    ) for i in ['male', 'female']
                ],

            'layout': dict(
                xaxis={'title': {'text': 'Number of Days From Symptom Onset to Death',
                                 'font': {'color': 'white'},
                                 },
                       'tickfont': dict(color="white"),
                       'gridcolor': 'grey',
                       'showgrid': True,
                       'tick0': 0

                       },
                yaxis={'title': {'text': 'Age',
                                 'font': {'color': 'white'}},
                       'tickfont': dict(color='white'),
                       'gridcolor': 'grey',
                       'showgrid': True
                       },
                title={'text': 'Timeline to Death by Age and Gender',
                       'font': dict(family='Sherif',
                                    size=18,
                                    color='white')
                       },
                paper_bgcolor='#201D1C',
                plot_bgcolor='#201D1C',
                text_color='white',
                marker={'color': 'white'},
                legend={'font': dict(
                    color='white'
                )},
                margin={'l': 80, 'b': 100, 't': 100, 'r': 50},

            )

            }
    elif selected_type=='Recovery Timeline':

        figure={
                'data': [
                    dict(
                        x=sym_to_rec[sym_to_rec['gender'] == i]['symptom_to_recovered'],
                        y=sym_to_rec[sym_to_rec['gender'] == i]['age'],
                        text='Symptom Onset: '+sym_to_rec[sym_to_rec['gender']==i]['symptom_onset'].astype(str)+'<br>'+\
                        'Recovered On: '+sym_to_rec[sym_to_rec['gender']==i]['recovered'].astype(str)+'<br>'+\
                            'Symptoms:'+sym_to_rec[sym_to_rec['gender']==i]['symptom'],
                        mode='markers',
                        opacity=1,
                        type='scatter',
                        marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'black'},
                        },
                        name=i,
                    ) for i in ['male', 'female']
                ],
                'layout': dict(
                xaxis={'title': {'text': 'Number of Days From Symptom Onset to Recovery',
                                 'font': {'color': 'white'},
                                 },
                       'tickfont': dict(color="white"),
                       'gridcolor': 'grey',
                       'showgrid': True,
                       'tick0': 0

                       },
                yaxis={'title': {'text': 'Age',
                                 'font': {'color': 'white'}},
                       'tickfont': dict(color='white'),
                       'gridcolor': 'grey',
                       'showgrid': True
                       },
                title={'text': 'Timeline to Recovery by Age and Gender',
                       'font': dict(family='Sherif',
                                    size=18,
                                    color='white')
                       },
                paper_bgcolor='#201D1C',
                plot_bgcolor='#201D1C',
                text_color='white',
                marker={'color': 'white'},
                legend={'font': dict(
                    color='white'
                )},
                margin={'l': 80, 'b': 100, 't': 100, 'r': 50},

            )

            }
    elif selected_type=='Main Map':
        try:
            longitude= hoverData['points'][0]['lon']

            latitude=hoverData['points'][0]['lat']

            #print(round(longitude,6), round(latitude,6))
            countries = pd.read_csv('countries.csv')
            china_cities = pd.read_csv('china_cities.csv')

            if latitude in countries['latitude'].tolist():
                #print(latitude)

                location_name=countries[countries['latitude']==latitude]['name'].tolist()[0]
                #print(country_name)
                location=df[df['country']==location_name]
                location.loc[location.index, 'reporting date'] = pd.to_datetime(location.loc[location.index, 'reporting date'])
            elif latitude in china_cities['lat'].tolist():
                print(latitude)

                location_name=china_cities[china_cities['lat']==latitude]['city_name'].tolist()[0]
                #print(country_name)
                location=df[df['location']==location_name]
                location.loc[location.index, 'reporting date'] = pd.to_datetime(location.loc[location.index, 'reporting date'])


            figure = px.scatter(data_frame=location, x='reporting date', y='location',
                                title='Reported Cases Location in '+location_name,
                         hover_data=['location', 'age', 'gender'],
                                template='plotly_dark')
            figure.update_xaxes(tickformat='%m/%d')
        except:
            country = df[df['country'] == 'Germany']
            country.loc[:, 'reporting date'] = pd.to_datetime(country.loc[:, 'reporting date'])
            figure = px.scatter(data_frame=country, x='reporting date', y='location',
                                title='Reported Cases by Region in Germany',
                                hover_data=['location', 'age', 'gender'],
                                template='plotly_dark')
            figure.update_xaxes(tickformat='%m/%d')

    elif selected_type=='Feature Importance':
        to_hospital=pd.read_csv('to_hospital.csv')
        to_hospital['final_status']=to_hospital['final_status'].apply(lambda x:
                                                                      'Dead' if x==1 else 'Recovered' if x==0
                                                                      else x)
        figure=px.box(data_frame=to_hospital,x='symptom_to_hosp',y='age',color='final_status',points='all',
      template='plotly_dark', labels={'final_status':'Status'})
        figure.update_xaxes(title_text='Days from symptom onset to hospitalization')
        figure.update_yaxes(title_text='Age')




    return figure




#
# @app.callback(
#     Output('side-graph','figure'),
#     [Input('type_info-dropdown','value')]
# )
#
# def update_graph(selected_city):
#     data=[]
#     data.append({'x':city_data[selected_city]['x'],
#                  'y':city_data[selected_city]['y'],
#                  'type':'line',
#                 'name':selected_city
#                  })
#     figure={'data':data}
#     return figure



















if __name__ == "__main__":
    # import warnings
    # warnings.warn("use 'nltk', not 'python -m nltk.downloader'", DeprecationWarning)
    app.run_server(debug=True)