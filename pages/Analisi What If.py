import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import plotly.express as px
import math
## NOTA: AL MOMENTO LA RF VIENE ALLENATA SUI DATI STORICI (2021) E POI LA PREVISIONE VIENE 
## FATTA SU QUELLI MODIFICATI (VA BENE? SAREBBE MEGLIO ALLENARLA DIRETTAMENTE SU QUELLI MODIFICATI?

st.write("# Analisi What-If")

#width = st.number_input("Width", 0, 300, 128)
st.markdown(
    f"""
    <style>
        .st-e4 {{
            max-width: {220}px
        }}
    </style>""",
    unsafe_allow_html=True,
)
col = st.multiselect(
    "Seleziona la/e variabile/i che vuoi modificare",
['N.Farmaci prescritti',
'N.Prestazioni erogate',
'Personale Sanitario SSN',
'Personale Professionale SSN',
'Personale Tecnico SSN',
'Personale Amministrativo SSN',
'Personale Sanitario SRC',
'Personale Professionale SRC',
'Personale Tecnico SRC',
'Personale Amministrativo SRC',
'N.Accessi al PS',
'N.Dimessi Codice Bianco',
'N.Dimessi Codice Giallo',
'N.Dimessi Codice Verde',
'N.Dimessi Codice Rosso',
'N.Prestazioni PS',
'Esenzione per invalidità',
'Esenzione per malattia cronica',
'Esenzione per malattia rara',
'Esenzione per reddito',
'Altre esenzioni',
'N.Prestazioni ADI',
'N.Interventi Medico ADI',
'N.Interventi Infermiere ADI',
'N.Interventi OSS ADI',
'N.Interventi Altro ADI',
'N.Interventi Specialista Sanitario ADI',
'Uscita per decesso',
'Dimissione protetta',
'Dimissione standard',
'N.Letti Degenza',
'N.Letti DH',
'N.Fumatori'], label_visibility='hidden'
)
#if col=='N.Femmine 80+ anni':
#    col='F-5'
#if col=='N.Maschi 80+ anni':
#    col='M-5'

#if 'N.Femmine 80+ anni' isin col:
#    col.insert('F-5)'
#if 'N.Maschi 80+ anni' isin col:
#    col.insert('M-5')

moltiplicatore = st.slider(
    "Imposta un fattore moltiplicativo per le variabili selezionate:",
    1.0, 3.0)
## facciamo partire il calcolo solo dal click sul bottone altrimenti 
## per ogni variabie che si aggiunge inizia il calcolo
if st.button(':chart_with_upwards_trend: Calcola la previsione', type="primary"):
    
    #df_2017=pd.read_csv('Dataset completi/Dataset completo 2017.csv', sep=';')
    #df_2017['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)
    #df_2017['Anno']='2017'
    df_2018=pd.read_csv('Dataset completi/Dataset completo 2018.csv', sep=';')
    df_2018['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)
    df_2018['Anno']='2018'
    df_2019=pd.read_csv('Dataset completi/Dataset completo 2019.csv', sep=';')
    df_2019['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)
    df_2019['Anno']='2019'
    df_2020=pd.read_csv('Dataset completi/Dataset completo 2020.csv', sep=';')
    df_2020['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)
    df_2020['Anno']='2020'
    df_2021=pd.read_csv('Dataset completi/Dataset completo 2021.csv', sep=';')
    df_2021['Territory'].replace('ÉMARÈSE', 'EMARÈSE', inplace=True)
    df_2021['Anno']='2021'

    dataset=pd.concat([df_2018,df_2019,df_2020,df_2021])
    dataset=dataset[['Territory', 'F-1', 'F-2', 'F-3', 'F-4', 'F-5', 'M-1', 'M-2', 'M-3',
       'M-4', 'M-5', 'Popolazione TOT', 
       'Superficie totale (Km2)', 'Densità di Popolazione',
       'Altitudine del centro (metri)', 
       'KILLINJ', 'ROADACC', 'N.Veicoli', 'N.Strutture Ricettive',
       'TAXABINCR', 'PENSINCR', 'N.Farmacie', 'N.Farmaci prescritti',
       'N.Prestazioni erogate', 'N.Assistibili', 'Personale Totale SSN',
       'Personale Sanitario SSN', 'Personale Professionale SSN',
       'Personale Tecnico SSN', 'Personale Amministrativo SSN',
       'Personale Sanitario SRC', 'Personale Professionale SRC',
       'Personale Tecnico SRC', 'Personale Amministrativo SRC',
       'N.Accessi al PS', 'N.Dimessi Codice Bianco', 'N.Dimessi Codice Giallo',
       'N.Dimessi Codice Verde', 'N.Dimessi Codice Rosso', 'N.Prestazioni PS',
       'Esenzione per invalidità', 'Esenzione per malattia cronica',
       'Esenzione per malattia rara', 'Esenzione per reddito',
       'Altre esenzioni', 'N.Prestazioni ADI', 'N.Interventi Medico ADI',
       'N.Interventi Infermiere ADI', 'N.Interventi OSS ADI',
       'N.Interventi Altro ADI', 'N.Interventi Specialista Sanitario ADI',
       'Uscita per decesso', 'Dimissione protetta', 'Dimissione standard',
       'Trasferimento clinico', 'Pagamenti SS',
       'Valore Nic comuni', 'PRVHOSPNHS', 'PUBHOSP', 'N.Letti Degenza',
       'N.Letti DH', 'N.Fumatori', 'NO2_DAILY_ugm-3', 'PM2p5_DAILY_ugm-3',
       'PM10_DAILY_ugm-3', 'TMIN_DAILY_C', 'TMAX_DAILY_C', 'RH2MIN_DAILY_%',
       'RH2MAX_DAILY_%', 'PREC_DAILY_mm', 'SNOW_DAILY_cm', 'WMAX_DAILY_ms-1',
       'LIGHTNING_DAILY_km-2', 'SSWTOT_DAILY_Whm-2', 'TSHOCK_DAILY_C', 'TOT',
       'Anno']] #'Incassi SS'
    
    #Previsione basata su dati reali
    features=dataset.copy()
    Territory=features['Territory'].reset_index()
    Territory.drop('index', axis=1, inplace=True)
    labels = np.array(features['Pagamenti SS'])
    features= features.drop(['Pagamenti SS', 'Territory'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)
    train_features = features[:-74]
    test_features = features[-74:]
    train_labels = labels[:-74]
    test_labels = labels[-74:]
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    fit = rf.fit(train_features, train_labels);
    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)
    mean_abs_err = round(np.mean(errors), 2)
    print('Mean Absolute Error:', mean_abs_err)
    mape = 100 * (errors / test_labels)
    accuracy =  100 - np.mean(mape)
    accuracy_tot = round(accuracy, 2)
    print('Accuracy:', accuracy_tot, '%.')
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    true_data = pd.DataFrame(data = {'Territory': Territory['Territory'][-74:], 'actual': labels[-74:]})
    predictions_data = pd.DataFrame(data = {'Territory': Territory['Territory'][-74:], 'prediction': predictions})

    if moltiplicatore!=1.0:
        # Previsione basata su dati modificati
        features_mod=dataset.copy()
        Territory=features_mod['Territory'].reset_index()
        Territory.drop('index', axis=1, inplace=True)
        for col in col:
            features_mod[col] = features_mod[col]*moltiplicatore
        labels_mod = np.array(features_mod['Pagamenti SS'])
        features_mod= features_mod.drop(['Pagamenti SS', 'Territory'], axis = 1)
        feature_list_mod = list(features_mod.columns)
        features_mod = np.array(features_mod)
        train_features_mod = features_mod[:-74]
        test_features_mod = features_mod[-74:]
        train_labels_mod = labels_mod[:-74]
        test_labels_mod = labels_mod[-74:]
        #rf_mod = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        #fit_mod = rf_mod.fit(train_features_mod, train_labels_mod);
        predictions_mod = rf.predict(test_features_mod)
        errors_mod = abs(predictions_mod - test_labels_mod)
        mean_abs_err_mod = round(np.mean(errors_mod), 2)
        print('Mean Absolute Error:', mean_abs_err_mod)
        mape_mod = 100 * (errors_mod / test_labels_mod)
        accuracy_mod =  100 - np.mean(mape_mod)
        accuracy_tot_mod = round(accuracy_mod, 2)
        print('Accuracy:', accuracy_tot_mod, '%.')
        importances_mod = list(rf.feature_importances_)
        feature_importances_mod = [(feature_mod, round(importance_mod, 2)) for feature_mod, importance_mod in zip(feature_list_mod, importances_mod)]
        feature_importances_mod = sorted(feature_importances_mod, key = lambda x: x[1], reverse = True)
        #[print('Variable: {:20} Importance: {}'.format(*pair_mod)) for pair_mod in feature_importances_mod];
        true_data_mod = pd.DataFrame(data = {'Territory': Territory['Territory'][-74:], 'actual': labels[-74:]})
        predictions_data_mod = pd.DataFrame(data = {'Territory': Territory['Territory'][-74:], 'prediction': predictions_mod})
        predictions_data_mod=predictions_data_mod[-74:]
    
        fig = px.line(true_data[-74:], x="Territory", y="actual", color_discrete_sequence=["#962086"], labels='Dati storici', markers=True) #4CC005  #0514C0  #F63366
        fig.add_scatter(x=true_data['Territory'][-74:], y=true_data['actual'][-74:], mode='lines', name='Dati storici', line=dict(color='#962086')) 
        fig.add_scatter(x=predictions_data_mod['Territory'], y=predictions_data_mod['prediction'], mode='lines', name='Simulazione', line=dict(color='#f48918'))
        fig.update_layout(title='Pagamenti del Sistema Sanitario: valori effettivi e previsti', xaxis_title='Comuni VDA', yaxis_title='Valori reali e previsti')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        millnames = ['Mila',' Milioni',' Miliardi',' Trillion']
        def millify(n):
            n = float(n)
            millidx = max(0,min(len(millnames)-1,
                            int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
            return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

        somma_pagamenti_reali=true_data['actual'][-74:].sum()
        st.write("Somma pagamenti del sistema sanitario (reali) per i comuni sopra elencati:", "%20s" %millify(somma_pagamenti_reali),"€")
        somma_pagamenti_previsti=predictions_data_mod['prediction'].sum()
        st.write("Somma pagamenti del sistema sanitario (previsti) per i comuni sopra elencati:",  "%20s" %millify(somma_pagamenti_previsti),"€")
   
    else:
        fig = px.line(true_data[-74:], x="Territory", y="actual", color_discrete_sequence=["#962086"], labels='Dati storici', markers=True) #4CC005  #0514C0  #F63366
        #fig.update_traces(hoverinfo='none')
        fig.add_scatter(x=true_data['Territory'][-74:], y=true_data['actual'][-74:], mode='lines', name='Dati storici', line=dict(color='#962086')) 
        fig.add_scatter(x=predictions_data['Territory'], y=predictions_data['prediction'], mode='lines', name='Previsione', line=dict(color='#f48918'))
        fig.update_layout(title='Pagamenti del Sistema Sanitario: valori effettivi e previsti', xaxis_title='Comuni VDA', yaxis_title='Valori reali e previsti')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        millnames = ['Mila',' Milioni',' Miliardi',' Trillion']
        def millify(n):
            n = float(n)
            millidx = max(0,min(len(millnames)-1,
                            int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
            return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

        somma_pagamenti_reali=true_data['actual'][-74:].sum()
        st.write("Somma pagamenti del sistema sanitario (reali) per i comuni sopra elencati:", "%20s" %millify(somma_pagamenti_reali),"€")
        somma_pagamenti_previsti=predictions_data['prediction'].sum()
        st.write("Somma pagamenti del sistema sanitario (previsti) per i comuni sopra elencati:",  "%20s" %millify(somma_pagamenti_previsti),"€")
   
    #st.write("Somma pagamenti del sistema sanitario (reali) per i comuni sopra elencati:", "**{:e}**".format(round(somma_pagamenti_reali,2)),"€")
    #st.write("Somma pagamenti del sistema sanitario (previsti) per i comuni sopra elencati:", "**{:e}**".format(round(somma_pagamenti_previsti,2)),"€")
    #somma_pagamenti=pd.DataFrame({"Pagamenti reali":somma_pagamenti_reali, "Pagamenti previsti":somma_pagamenti_previsti}, index=[0])
    #fig3 = dict({
    #    "data": [{"type": "bar",
    #              "x": ["Pagamenti reali", "Pagamenti previsti"],
    #              "y": [somma_pagamenti_reali, somma_pagamenti_previsti]}],
    #    "layout": {"title": {"text": "Confronto tra pagamenti reali e previsti"}}
    #})
    #st.plotly_chart(fig3, use_container_width=True)
