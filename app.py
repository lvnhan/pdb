"""
HealthCare Dasboard Template with the ThemeSwitchAIO component
Note - this requires dash-bootstrap-components>=1.0.0 and dash>=2.0 dash-bootstrap-templates>=1.0.4.
NhanVietLe

"""
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO

import dash_daq as daq
from datetime import timedelta, date
from dateutil import parser #use to parse date string to date object
import numpy as np


#############################################
#local style HTML
#############################################
colors = {
    'pl.bkground': 'rgba(0, 0, 0, 0)',
    'pg.bkground': 'rgba(0, 0, 0, 0)',
    'drop.bkg': '#1E1E1E',
    'text.light': '#B6B6B6',
    'text.note': '#989898'
}
fontnames = {
    'u': 'Segoe UI',
    'l': 'Segoe UI Light',
    's': 'Segoe UI Semibold',
    'b': 'Segoe UI Black'
    }

fontsz = {
    'f40': '2.5em'  ,
    'f30': '1.875em',
    'f14': '0.875em',
    'f16': '1em',
    'f12': '0.75em' ,
    'f10': '0.625em' #10px = 0.625rem. 12px = 0.75rem. 14px = 0.875rem. 16px = 1rem (base)
    }
#############################################
# pass variables
#############################################
title = ['Bảng đồ cấp độ dịch trong cộng đồng', 'STATISTIC DATA']
stitle= ['Dữ liệu ghi nhận từ OWID','Visualising data with Plotly - Dash']
LTitleChart = 'Biểu đồ thống kê dịch bệnh theo ngày'
STitleChart = 'Biểu đồ thống kê số ca nhiễm tích lũy & tử vong'

fnote = ['7-day rolling average. Due to limited testing, the number of confirmed cases is lower than the true number of infections.',
         '(Dữ liệu về các ca nhiễm được cập nhật thường xuyên từ OWID)',
         'Dữ liệu sẽ hiển thị giá trị 0 khi số liệu chưa được cập nhật.']

cardtitle =['Ca nhiễm tích lũy', 'Ca nhiễm/ngày', 'Liều vaccine/ngày', 'Ca tử vong tích lũy']
cardcontent =['Mức tăng giảm: ', 'Tỷ lệ/1M dân: ', 'Tiêm đủ liều: ', 'Tỷ lệ/1M dân: ']
cardclass ={
    'h':'bg-primary bg-gradient fw-light text-white', #bg-info bg-primary bg-secondary
    'b':'card-title fw-bold',
    's':'card-text',
    'g':'text-center shadow-sm rounded'
    }
defCountries = ['France','Vietnam','Germany','United Kingdom','Italy','Japan','Canada','United States']
metrics = {
    'new_cases_smoothed':'Ca nhiễm', 
    'new_deaths_smoothed':'Ca tử vong', 
    'people_vaccinated': 'Số người được tiêm', 
    'people_fully_vaccinated': 'Số người được tiêm đủ' 
    }
color_scale = {
    'new_cases_smoothed':'redor', 
    'new_deaths_smoothed':'greys', 
    'people_vaccinated': 'blues', 
    'people_fully_vaccinated': 'greens' 
    }

#############################################
#Initialize empty array of given length
#############################################
sd = [0] * 4

#############################################
#select the Bootstrap stylesheets and figure templates for the theme toggle here:
    #############################################
template_theme1 = "simplex" #"flatly"
template_theme2 = "darkly"
#template_theme2 = "superhero"
url_theme1 = dbc.themes.SIMPLEX #dbc.themes.FLATLY
url_theme2 = dbc.themes.DARKLY #SUPERHERO
#url_theme2 = dbc.themes.SUPERHERO #Some themes: LUMEN SIMPLEX FLATLY

dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css")
app = Dash(__name__, external_stylesheets=[url_theme1, dbc_css])
server = app.server
app.title = "Plotly.Dash Coronavirus Pandemic"


#############################################
# Loading & preprocessing data
#############################################
df = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv', parse_dates=True)

df.dropna(subset = ['continent'], inplace = True) 
# Replace NaN Values with Zeros in Pandas DataFrame
df = df.replace(np.nan, 0)

# Copy and select only the columns to need for map
mapdf = df.copy(deep=True)
# change 'iso_code' column to 'iso_alpha'
mapdf.rename(columns = {'iso_code':'iso_alpha'}, inplace = True)
mapdf = mapdf[['iso_alpha', 'continent','location','date','total_cases','new_cases', 'total_deaths', 
               'new_cases_smoothed','new_deaths_smoothed',
               'people_vaccinated', 'people_fully_vaccinated']]
# Extract and add more the year from date column
df['year'] = pd.DatetimeIndex(df['date']).year

dfc = df.copy(deep=True)
df.index = pd.to_datetime(df['date'])
df.sort_index(ascending = False, inplace=True)
dmax = date.today() + timedelta(days=-1)
diff = date.fromisoformat(dmax.strftime('%Y-%m-%d')) - date.fromisoformat('2020-01-22')

#############################################
# preprocess data for scatter fig
# Remove all NULL value from continent
# get date max from dataframe
#############################################
dfc.index = pd.to_datetime(dfc['date'])

date_req = dfc.index.max() + timedelta(days=-1)
filterMask = dfc['date'] >= date_req.strftime('%Y-%m-%d') #<= apply for animation
dfmc = dfc[filterMask]

#############################################
# Data statistic
#############################################
befDate = dmax + timedelta(days=-1)
#############################################
# Calculate total cumulative confirmed cases
#############################################
tcases_p  = 0
tcases_c = 0
deviant = 0
tcases_percent = 0.0
def update_cardsinfo(adf, max_dt, bef_dt):
    befDate = bef_dt
    dmax = max_dt
    #############################################
    # Creates a list containing 4 lists, each of 2 items, all set to empty
    # each card includes: 0- body | 1-sup info
    #############################################
    w, h = 2, 4
    card_rstl = [['' for x in range(w)] for y in range(h)]
    
    #############################################
    # Calculate total cumulative confirmed cases
    #############################################    
    tcases_p = adf[adf['date'] == befDate.strftime('%Y-%m-%d')].sum(numeric_only = True)['total_cases']
    tcases_c = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['total_cases']  
    sd[0] = tcases_c
    deviant = tcases_c - tcases_p
    tcases_percent = deviant/tcases_p 

    card_rstl[0][0] = "{:,.0f}".format(sd[0])
    card_rstl[0][1]  = [#cardcontent[0],
                         (lambda x: html.Label('\u25B2 ', style={'color': '#FF0000'}) if x>0 else html.Label("\u25BC ", style={'color': '#228B22'}))(deviant),
                         ("{:,.0f}".format(deviant)+" ({:.2%})").format(tcases_percent)
                         ]
    #############################################
    # Total new cases per 1,000,000 people
    #############################################
    ncases_c = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['new_cases']
    sd[1] = ncases_c
    
    ncases_pm = adf[adf['date'] == befDate.strftime('%Y-%m-%d')].sum(numeric_only = True)['new_cases_per_million']
    ncases_cm = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['new_cases_per_million']
    
    deviantn = ncases_cm - ncases_pm
    ncases_percent = deviantn/ncases_pm

    card_rstl[1][0] = "{:,.0f}".format(sd[1])
    card_rstl[1][1] = [cardcontent[1],
                       "{:,.0f}".format(ncases_cm) + " (",
                       (lambda x: html.Label('\u25B2 ', style={'color': '#FF0000'}) if x>0 else html.Label("\u25BC ", style={'color': '#7CFC00'}))(deviantn),
                       #forestgreen	#228B22 | lawngreen 	#7CFC00
                       "{:,.2%}".format(ncases_percent) + ")"
        ]
    
    #############################################
    # Total number of people who received at least one vaccine dose
    #############################################
    pvacc_c = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['people_vaccinated']
    pfvacc_c = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['people_fully_vaccinated']
    
    sd[2] = pvacc_c                            
    card_rstl[2][0] = "{:,.0f}".format(sd[2])
    card_rstl[2][1] = [cardcontent[2] + "{:,.0f}".format(pfvacc_c)]
    
    #############################################
    # Total deaths
    #############################################
    tdeaths_c = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['total_deaths']
    tdeaths_percent = adf[adf['date'] == dmax.strftime('%Y-%m-%d')].sum(numeric_only = True)['total_deaths_per_million']

    sd[3] = tdeaths_c
    card_rstl[3][0] = "{:,.0f}".format(sd[3])
    card_rstl[3][1] = [cardcontent[3] + "{:,.0f}".format(tdeaths_percent)]
 
    return card_rstl

#############################################
header = html.Div(children=[
    dbc.Row([
        dbc.Col([
            html.H2(title[0], className="text-white",
                    style={'font-family': fontnames['l']})
            ], width=12)
        ]),
    dbc.Row([
        dbc.Col([html.Sup(stitle[0], style={'color':'#E6E6E6'})], width=12)
        ])
    ], className="bg-primary bg-gradient p-3 mb-2")


SubHeader = html.Div(children=[
            html.H5("Biểu đồ", style={'font-family': fontnames['l']})
            ])
#############################################
# Define top four cards
#############################################
card = {
        '0':
            dbc.Card([ #{:.9f}
                dbc.CardHeader(id="c0", children=[cardtitle[0]], className=cardclass['h']),
                dbc.CardBody([html.H3(className=cardclass['b'], style={'color':'#FF0000'}, id="c0b"),
                              html.Sup(id="c0s", children=[], className=cardclass['s'])
                              ]) 
                ],className=cardclass['g'], style={"height": 150}
                ),      
        '1':
            dbc.Card([
                dbc.CardHeader(id="c1", children=[cardtitle[1]], className=cardclass['h']),
                dbc.CardBody([html.H3(className=cardclass['b'], style={'color':'#FF8C00'}, id="c1b"), #orangered:#FF4500 darkorange: #FF8C00
                              html.Sup(id="c1s", children=[], className=cardclass['s'])
                              ])
                ],className=cardclass['g'], style={"height": 150}
                ),
        '2':
            dbc.Card([
                dbc.CardHeader(id="c2", children=[cardtitle[2]], className=cardclass['h']),
                dbc.CardBody([html.H3(className=cardclass['b'], style={'color':'#409602'}, id="c2b"),
                              html.Sup(id="c2s", children =[], className=cardclass['s'])
                              ])
                ],className=cardclass['g'], style={"height": 150}
                ),
        '3':
            dbc.Card([
                dbc.CardHeader(id="c3", children=[cardtitle[3]], className=cardclass['h']),
                dbc.CardBody([html.H3(className=cardclass['b'], id="c3b"),
                              html.Sup(id="c3s", children=[], className=cardclass['s'])
                              ])
                ],className=cardclass['g'], style={"height": 150}
                ),
        }
cards = html.Div([
        dbc.Row(
            [
                dbc.Col(dbc.Card(card['0'])),
                dbc.Col(dbc.Card(card['1'])),
                dbc.Col(dbc.Card(card['2'])),
                dbc.Col(dbc.Card(card['3'])),
            ],
            className="mb-0",
        )])
#############################################
# Define switching btn for dark-light screen
#############################################
switch = html.Div([dbc.Row([dbc.Col(children=[html.Sup("Coronavirus Pandemic (COVID-19).", style={'font-weight': 'bold'}),
                                              html.Sup(" Dữ liệu đã được cập nhật đến ngày " + dmax.strftime('%d-%m-%Y') + " (UTC) | "),
                                              html.Sup("Đợt bùng phát từ ngày 22-01-2020 đến nay: {}".format(diff.days)+" ngày.")]
                                    ,width=9),
                            dbc.Col(html.Div(id="switch-result",children=["Light screen"],
                                             style={'textAlign':'right',
                                                    'font-size':'12'}),
                                              width=2),
                            dbc.Col(
                                ThemeSwitchAIO(aio_id="theme",
                                               themes=[url_theme1, url_theme2],
                                               icons={"left":"{%}", "right":"{%}"}
                                               #icons={"left":"fa fa-moon", "right":"fa fa-sun"}
                                               ), width=1)])])

#############################################
# Define pg footer 
#############################################
bottom = html.P("© 2022 LAC VIET Computing Corp.")
#############################################
# Define criterias
#############################################
# year slider
yrslider = dcc.RangeSlider(
    min = df['year'].min(),
    max = df['year'].max(),
    step = 1,
    value = [df['year'].min(), df['year'].max()],
    marks={str(year): str(year) for year in df['year'].unique()},
    id='year-slider'
    )

# Creating a Dropdown Menu
# Creates a list of dictionaries, which have the keys 'label' and 'value'.
def get_options(locations):
    dict_list = [{'label': 'Select all', 'value': 'all_values'}]
    for i in locations:
        dict_list.append({'label': i, 'value': i})
    return dict_list

def get_days(days):
    dict_list = []
    for i in days:
        dict_list.append({'label': i, 'value': i})
    return dict_list

crictrl = html.Div(className='div-user-controls',
                     children=[
                         html.H6('Plotly.Dash',style={'color': colors['text.light'], 'font-family': fontnames['l']}, className="mt-5"),
                         html.H5(title[1]),
                         html.Div(children=[html.Sup('Năm bùng phát dịch bệnh'), yrslider], className="mt-3"),
                         html.Div(className='div-for-dropdown mt-3',
                                  children=[
                                      html.Sup('Chọn ngày'),
                                      dcc.Dropdown(id='dateselector', multi=False,
                                                   value=df['date'].unique().max(),
                                                   style={'backgroundColor': colors['drop.bkg'],
                                                   'font-family': fontnames['u'],
                                                   'font-size': fontsz['f16']},
                                                   className='dateselector'
                                      )
                                      ]
                                  ),                         
                         html.Div(className='div-for-dropdown mt-3',
                                  children=[
                                      html.Sup('Chọn quốc gia'),
                                      dcc.Dropdown(id='locselector',
                                                   options=get_options(df['location'].unique()), multi=True,
                                                   value=defCountries,
                                                   style={'backgroundColor': colors['drop.bkg'],
                                                   'font-family': fontnames['u'],
                                                   'font-size': fontsz['f16']},
                                                   className='locselector'
                                      )
                                      ]
                                  ),
                         html.Div(className='div-for-dropdown mt-3',
                                  children=[
                                      html.Sup('Chỉ số theo dõi'),
                                      dcc.Dropdown(id='crossfilter_metricselector',
                                                   options=metrics, multi=False,
                                                   value='new_cases_smoothed',
                                                   style={'backgroundColor': colors['drop.bkg'],
                                                   'font-family': fontnames['u'],
                                                   'font-size': fontsz['f16']},
                                                   className='locselector'
                                      )
                                      ]
                                  ),
                         html.Div(className='div-for-dropdown mt-3',
                                  children=[html.Div(children=[fnote[2],html.Br(),'Chọn đồ thị'], 
                                                     className='mt-5 mx-1'),
                                            html.Div(children=[
                                                daq.ToggleSwitch(id='ChartTypeToggle-switch', size=25,
                                                                 color='grey',label=['Dạng line', 'Dạng log'],
                                                                 value=False)], className='mb-3'
                                                     ) 
                                            ], style={'font-size': fontsz['f12'],})
                         ], style={'font-family': fontnames['u']}
    )
#############################################
ggraph = html.Div(dcc.Graph(id="ggraph"), className="m-4")
cgraph = {
    'line':html.Div(dcc.Graph(id="cgraph_line"), className="m-4"),
    'scatter':html.Div(dcc.Graph(id="cgraph_scatter"), className="m-4"),
    'map':html.Div(dcc.Graph(id="cgraph_map"), className="m-4"),
    }

#############################################
# layout
#############################################
app.layout = dbc.Container(children=[
    dbc.Row(
        [
            dbc.Col(width=1),
            dbc.Col(
                [
                    header,
                    switch,
                ], width=10
            ),
            dbc.Col(width=1)
        ]
    ),
    dbc.Row(
        [
        dbc.Col(width=1),
        dbc.Col(cards, width=10),
        dbc.Col(width=1)
        ]
        ),
    dbc.Row(
        [
        dbc.Col(width=1),
        dbc.Col(crictrl, width=2, className="shadow-sm rounded m-1"),
        dbc.Col([cgraph['line'], cgraph['map'], cgraph['scatter'],ggraph], width=8, className="shadow-sm rounded m-1"), 
        dbc.Col(width=1)
        ],className="m-1"
        ),
    dbc.Row(
        [
        dbc.Col(width=1),
        dbc.Col(bottom),
        dbc.Col(width=1)
        ]
        )],
    className="m-1 dbc", #m-4
    fluid=True,
)

@app.callback(
    Output('c0b', 'children'),
    Output('c0s', 'children'),
    Output('c1b', 'children'),
    Output('c1s', 'children'),
    Output('c2b', 'children'),
    Output('c2s', 'children'),
    Output('c3b', 'children'),
    Output('c3s', 'children'),
    Input('dateselector', 'value'),
    Input('locselector', 'value')
    )
def update_cards(selected_date, selected_loc):
    dmax = parser.parse(selected_date)
    befDate = dmax + timedelta(days=-1)
    if selected_loc==['all_values']:
        dff = df
    else:
        dff = df[df['location'].isin(selected_loc)]

    #############################################
    # Calculate total cumulative confirmed cases
    #############################################
    cardinfo = update_cardsinfo(dff, dmax, befDate)

    return  cardinfo[0][0], cardinfo[0][1], cardinfo[1][0], cardinfo[1][1], cardinfo[2][0], cardinfo[2][1], cardinfo[3][0], cardinfo[3][1]

@app.callback(  
    Output('dateselector', 'options'),
    Output('dateselector', 'value'),
    Input('year-slider', 'value')
    )

def set_date_range(selected_year):
    yrmin = int(selected_year[0])
    yrmax = int(selected_year[1])
    dff = df[(df.year >= yrmin) & (df.year <= yrmax)]

    dmax = dff['date'].unique().max()
    return get_days(dff['date'].unique()), dmax

@app.callback(
    Output("switch-result", 'children'),
    Output("cgraph_line", "figure"),
    Output("cgraph_scatter", "figure"),
    Output("ggraph", "figure"),
    Output("cgraph_map", "figure"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
    Input('year-slider', 'value'),
    Input('dateselector', 'value'),
    Input('locselector', 'value'),
    Input('crossfilter_metricselector','value'),
    Input('ChartTypeToggle-switch', 'value')
    
)
def update_theme(toggle, selected_year, selected_date, selected_loc, xaxis_name, selected_chart_type):
    yrmin = int(selected_year[0])
    yrmax = int(selected_year[1])

    template = template_theme1 if toggle else template_theme2
    s='Nền sáng' if toggle else 'Nền tối'
    
    if selected_loc==['all_values']:
        dff = df[(df.year >= yrmin) & (df.year <= yrmax)]
        dffmc = dfmc[(dfmc.year >= yrmin) & (dfmc.year <= yrmax)]
    else:
        dff = df[(df.year >= yrmin) & (df.year <= yrmax)]
        dffmc = dfmc[(dfmc.year >= yrmin) & (dfmc.year <= yrmax)]
        
        dff = dff[dff['location'].isin(selected_loc)]
        #dffmc = dffmc[dffmc['location'].isin(selected_loc)]
    

    # Charting in Dash – Displaying a Plotly-Figure
    #############################################
    # reload dataframe by criteria
    #############################################
    dmax = parser.parse(selected_date)
    
    cfigline = px.line(dff, x='date', y=dff[xaxis_name], color='location',  markers=False, height=500, #width=700,
                  labels={
                  "date": "Ngày ghi nhận",
                  "new_cases_smoothed": "Số ca nhiễm bình quân trong tuần",
                  "new_deaths_smoothed": "Số tử vong bình quân trong tuần",
                  "people_vaccinated":"Số người đã tiêm ít nhất một liều vaccine",
                  "people_fully_vaccinated":"Số người tiêm đủ liều",
                  "location": "Quốc gia"
                  }, #type=log,
                  #log_y=True,
                  template=template,
                  #animation_frame="date", animation_group="location",
                  title= LTitleChart + "<br><sup>Cập nhật đến ngày: " + dmax.strftime('%d-%m-%Y') +"</sup>"
                  )
    #show and hide legend on chart
    cfigline.update_layout(showlegend=False)
    # legend position
    #cfigline.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    # horizontal legend
    #cfigline.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    
    #switching chart type by toggle button
    if(selected_chart_type==False):
        cfigline.update_yaxes(type="linear")
    else:
        cfigline.update_yaxes(type="log")

    #Modify hover label    
    ##cfigline.update_traces(mode="markers+lines", hovertemplate=None)
    cfigline.update_traces(hovertemplate="%{y} ca")#"%{x}<br>Ca nhiễm: %{y}<br>"  #%{ <variable>} 
    cfigline.update_xaxes(spikecolor="grey", spikedash="dot" ) #"solid", "dot", "dash", "longdash", "dashdot", or "longdashdot"
    #cfigline.update_traces(marker_line_color('rgb(0, 2, 1)'))        
    # styling hover
    if toggle:
        cfigline.update_layout(hovermode="x unified", hoverlabel=dict(bgcolor='rgba(255,255,255,0.75)'),
                               showlegend=False)#'x', 'x unified', 'closest' (default) 
    else:
        cfigline.update_layout(hovermode="x unified", hoverlabel=dict(bgcolor='rgba(0,0,0,0.5)'),
                               showlegend=False) 
                                
        #Scatter with play x-axis
    cfigscatter = px.scatter(dffmc, x="total_cases", y="total_deaths",
                              #animation_frame="date", animation_group="location",
                              size="population",
                              color="continent", hover_name="location",
                              #log_x=True, #log_y=True,
                              size_max=60, opacity=0.7,
                              range_x=[100,10000000], range_y=[25,140000],
                              labels={
                                  "total_cases": "Tổng ca nhiễm tích lũy",
                                  "total_deaths":"Tổng ca tử vong",
                                  "total_cases_per_million": "Tỷ lệ nhiễm/1000000 dân",
                                  "location": "Quốc gia",
                                  "continent": "Châu lục"},
                              template=template,
                              title= STitleChart + "<br><sup>Cập nhật đến ngày: " + date_req.strftime('%d-%m-%Y') +"</sup>"
                              )

    cfigscatter.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))

    gfig = px.scatter(
        df[df['date'] == dmax.strftime('%Y-%m-%d')],
        x="gdp_per_capita", #Gross domestic product at purchasing power parity (constant 2011 international dollars), most recent year available
        y="life_expectancy", # Life expectancy at birth in 2019
        size="population",
        color="continent",
        log_x=True,
        size_max=60,
        template=template,
        labels={
            "gdp_per_capita": "Tổng sản phẩm Quốc nội GDP",
            "life_expectancy":"Tuổi thọ bình quân",
            "population": "Dân số",
            "location": "Quốc gia",
            "continent": "Châu lục"
                         },
        title="Gapminder" + 
        "<br><sup>Số liệu thống kê theo các thông tin khác về tuổi thọ bình quân và tổng sản phẩm Quốc nội.</sup>"
    )
    gfig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    
    ####################################################
    #map
    mmapdf = mapdf[mapdf['date'] == dmax.strftime('%Y-%m-%d')]
    mfig = px.choropleth(mmapdf, locations="iso_alpha",
                    color=mmapdf[xaxis_name], 
                    hover_name="location", 
                    #color_continuous_scale=px.colors.sequential.Plasma
                    color_continuous_scale=color_scale[xaxis_name],# "redor",#"orrd",#"reds"
                    #"blues", "brwnyl","burgyl", "hot" #"rdylbu_r",# "rdpu_r","hot_r"
                    #hover_data=['covtrack'],
                    template=template,
                    labels={
                        'iso_alpha': 'iso',
                        'total_cases':'Ca nhiễm TL',
                        'new_cases': 'Ca nhiễm mới',
                        'total_deaths': 'Ca tử vong TL',
                        "new_cases_smoothed": "Ca nhiễm BQ/tuần",
                        "new_deaths_smoothed": "Ca tử vong BQ/tuần",
                        'people_vaccinated': 'Liều vacc',
                        'people_fully_vaccinated': 'Số người tiêm đủ vacc'
                        }
                    )
    mfig.update_layout(title_text = "Bảng đồ cấp độ theo chỉ số - " + metrics[xaxis_name] + "<br><sup>Cập nhật đến ngày: " + dmax.strftime('%d-%m-%Y') +"</sup>")
    mfig.update_geos(showcountries=True)
    
    return s, cfigline, cfigscatter, gfig, mfig

if __name__ == "__main__":
    app.run_server(debug=True)