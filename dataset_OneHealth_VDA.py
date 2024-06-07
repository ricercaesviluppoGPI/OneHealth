import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import calendar
from datetime import date
from matplotlib import cm
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
from datetime import date
import folium
import json
import geopandas
import branca
from folium import GeoJson
from folium import GeoJsonTooltip

st.set_page_config(
    page_title="OneHealth VDA",
    page_icon="⚕️", #📊
)

with st.sidebar:
    st.image("Gpi_CMYK_payoff.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="PNG")

st.write("# Dataset OneHealth Valle d'Aosta")

df_2017=pd.read_csv('Dataset completi/Dataset completo 2017.csv', sep=';')
df_2017['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)

df_2018=pd.read_csv('Dataset completi/Dataset completo 2018.csv', sep=';')
df_2018['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)

df_2019=pd.read_csv('Dataset completi/Dataset completo 2019.csv', sep=';')
df_2019['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)

df_2020=pd.read_csv('Dataset completi/Dataset completo 2020.csv', sep=';')
df_2020['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)

df_2021=pd.read_csv('Dataset completi/Dataset completo 2021.csv', sep=';')
df_2021['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)

istat_codes_dict={}
istat_codes_dict={
'ALLEIN':7001, 
'ANTEY-SAINT-ANDRÉ':7002,
'AOSTA':7003, 
'ARNAD':7004, 
'ARVIER':7005, 
'AVISE':7006, 
'AYAS':7007, 
'AYMAVILLES':7008, 
'BARD':7009, 
'BIONAZ':7010, 
'BRISSOGNE':7011, 
'BRUSSON':7012, 
'CHALLAND-SAINT-ANSELME':7013, 
'CHALLAND-SAINT-VICTOR':7014, 
'CHAMBAVE':7015, 
'CHAMOIS':7016, 
'CHAMPDEPRAZ':7017, 
'CHAMPORCHER':7018, 
'CHARVENSOD':7019, 
'CHÂTILLON':7020, 
'COGNE':7021, 
'COURMAYEUR':7022, 
'DONNAS':7023, 
'DOUES':7024, 
'ÉMARÈSE':7025, 
'ETROUBLES':7026, 
'FÉNIS':7027, 
'FONTAINEMORE':7028, 
'GABY':7029, 
'GIGNOD':7030, 
'GRESSAN':7031,
'GRESSONEY-LA-TRINITÉ':7032, 
'GRESSONEY-SAINT-JEAN':7033,
'HÔNE':7034, 
'INTROD':7035, 
'ISSIME':7036, 
'ISSOGNE':7037, 
'JOVENÇAN':7038, 
'LA MAGDELEINE':7039, 
'LA SALLE':7040, 
'LA THUILE':7041, 
'LILLIANES':7042,
'MONTJOVET':7043, 
'MORGEX':7044, 
'NUS':7045, 
'OLLOMONT':7046, 
'OYACE':7047, 
'PERLOZ':7048, 
'POLLEIN':7049, 
'PONTBOSET':7050, 
'PONTEY':7051, 
'PONT-SAINT-MARTIN':7052, 
'PRÉ-SAINT-DIDIER':7053, 
'QUART':7054, 
'RHÊMES-NOTRE-DAME':7055, 
'RHÊMES-SAINT-GEORGES':7056, 
'ROISAN':7057, 
'SAINT-CHRISTOPHE':7058, 
'SAINT-DENIS':7059, 
'SAINT-MARCEL':7060, 
'SAINT-NICOLAS':7061, 
'SAINT-OYEN':7062, 
'SAINT-PIERRE':7063, 
'SAINT-RHÉMY-EN-BOSSES':7064, 
'SAINT-VINCENT':7065, 
'SARRE':7066, 
'TORGNON':7067, 
'VALGRISENCHE':7068, 
'VALPELLINE':7069, 
'VALSAVARENCHE':7070, 
'VALTOURNENCHE':7071, 
'VERRAYES':7072, 
'VERRÈS':7073, 
'VILLENEUVE':7074} 

dizionario_istat = {'Nome comune':'ISTAT code'}
dizionario_istat.update(istat_codes_dict)
dizionario_istat_invertito = {v: k for k, v in dizionario_istat.items()}

with open("comuni.geojson") as f:
    comuni_geojson = json.load(f)

df_comuni = geopandas.GeoDataFrame.from_features(comuni_geojson, crs="EPSG:4326")
df_comuni=df_comuni[df_comuni['com_istat_code_num'].isin(dizionario_istat_invertito)]
df_comuni=df_comuni[['geometry','name', 'com_istat_code_num']]
df_comuni['name']=df_comuni['name'].str.upper()
df_comuni.head()

comunimerge_2017 = df_comuni.merge(df_2017, how="left", left_on="name", right_on="Territory")
comunimerge_2018 = df_comuni.merge(df_2018, how="left", left_on="name", right_on="Territory")
comunimerge_2019 = df_comuni.merge(df_2019, how="left", left_on="name", right_on="Territory")
comunimerge_2020 = df_comuni.merge(df_2020, how="left", left_on="name", right_on="Territory")
comunimerge_2021 = df_comuni.merge(df_2021, how="left", left_on="name", right_on="Territory")

anno = st.selectbox(
   "Seleziona un anno:",
   (2017, 2018, 2019, 2020, 2021),
   placeholder="Seleziona anno"
)

if anno==2017:
    comunimerge=comunimerge_2017
    df=df_2017
elif anno==2018:
    comunimerge=comunimerge_2018
    df=df_2018
elif anno==2019:
    comunimerge=comunimerge_2019
    df=df_2019
elif anno==2020:
    comunimerge=comunimerge_2020
    df=df_2020
elif anno==2021:
    comunimerge=comunimerge_2021  
    df=df_2021

# Import necessary functions from branca library
from branca.element import Template, MacroElement

# Create the legend template as an HTML element
legend_template = """
{% macro html(this, kwargs) %}
<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index: 9999; background-color: rgba(255, 255, 255, 0.5);
     border-radius: 6px; padding: 10px; font-size: 10.5px; right: 20px; top: 20px;'>     
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background: #FFFFBF; opacity: 0.75;'></span>Bassissimi accessi al PS</li>
    <li><span style='background: #FDCEA0; opacity: 0.75;'></span>Bassi accessi al PS</li>
    <li><span style='background: #FDAE61; opacity: 0.75;'></span>Medi accessi al PS</li>
    <li><span style='background: #D7191C; opacity: 0.75;'></span>Alti accessi al PS</li>
    <li><span style='background: #543005; opacity: 0.75;'></span>Altissimi accessi al PS</li>
  </ul>
</div>
</div> 
<style type='text/css'>
  .maplegend .legend-scale ul {margin: 0; padding: 0; color: #0f0f0f;}
  .maplegend .legend-scale ul li {list-style: none; line-height: 18px; margin-bottom: 1.5px;}
  .maplegend ul.legend-labels li span {float: left; height: 16px; width: 16px; margin-right: 4.5px;}
</style>
{% endmacro %}

"""


m=folium.Map(location=[45.73497302394166, 7.312586537423167], control_scale = True, zoom_start=9)
comunimerge['Densità accessi PS']=comunimerge['N.Accessi al PS']/comunimerge['Popolazione TOT']
variabile1='Densità accessi PS'
variabile2='PM2p5_DAILY_ugm-3'
variabile3='PM10_DAILY_ugm-3'
variabile4='NO2_DAILY_ugm-3' 
variabile5='TOT'
variabile6='N.Accessi al PS'

colormap = branca.colormap.LinearColormap(
    vmin=comunimerge[variabile1].quantile(0.0),
    vmax=comunimerge[variabile1].quantile(1),
    colors=["#ffffbf", "#fdcea0", "#fdae61", "#d7191c", "#543005"],
    caption="")

tooltip = folium.GeoJsonTooltip(
    fields=["name", variabile6, variabile2, variabile3, variabile4, variabile5],
    aliases=["Comune:", variabile6, variabile2, variabile3, variabile4, variabile5],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border: 2px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """,
    max_width=800,
)

g = folium.GeoJson(
    comunimerge,
    style_function=lambda x: {
        "fillColor": colormap(x["properties"][variabile1])
        if x["properties"][variabile1] is not None
        else "transparent",
        "color": "black",
        "fillOpacity": 0.6,
    },
    tooltip=tooltip).add_to(m)

# Add the legend to the map
macro = MacroElement()
macro._template = Template(legend_template)
m.get_root().add_child(macro)


colormap.add_to(m)
folium.LayerControl().add_to(m)

st_data = st_folium(m, height=400, width=725)

col=st.selectbox(
   "Seleziona una variabile:",
   ('Densità di Popolazione', 'N.Farmaci prescritti',
       'N.Prestazioni erogate', 'N.Assistibili', 'N.Ospedali pubblici',
       'N.Accessi al PS', 'N.Dimessi Codice Bianco', 'N.Dimessi Codice Rosso', 'N.Prestazioni PS',
       'Esenzione per invalidità', 'Esenzione per malattia cronica',
       'Esenzione per reddito', 'N.Prestazioni ADI',  
       'Dimissione standard','Pagamenti SS', 'Incassi SS',
       'Valore Nic comuni'), 
   placeholder="Seleziona variabile"
)
if col=='N.Ospedali pubblici':
    col='PUBHOSP'

## Correlation Matrix
df.drop(['Superficie totale (ettari)', 'LOWFLHZ', 'MEDFLHZ', 'HIGHFLHZ',
'LANDATTZAA', 'MODLHZP1', 'MEDLHZP2', 'HIGHLHZP3', 'VHIGHLHZP4','N.Interventi Altro ADI', 'Trasferimento clinico'], axis=1, inplace=True)
correlation_matrix=df.corr(numeric_only=True, method='kendall')


fig=plt.figure(figsize=(8, 20), edgecolor='#b4d5a0', linewidth=12)
heatmap = sns.heatmap(correlation_matrix[[col]].sort_values(by=col, ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlazione %s vs altre variabili %s' %(col,anno), fontdict={'fontsize':15}, pad=5);
st.write(fig)

fig2=plt.figure(figsize=(8, 20), edgecolor='#b4d5a0', linewidth=12)
feature_importances=pd.read_csv('feature_importances_%s.csv' %anno, sep=';')
sns.set_theme(rc={'figure.figsize':(8,20)})
plot=sns.barplot(y=feature_importances['Variabili'], x=feature_importances['Importanza'], orient="h", palette = "rainbow").set_title("Feature Importance %s" %anno, fontdict={'fontsize':15})
st.write(fig2)
