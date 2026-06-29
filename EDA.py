import pandas as pd
import numpy as np


nome_file_originale = 'US_Accidents.csv'
citta_target = 'Los Angeles'

# Lista vuota per accumulare solo i pezzi di dati relativi a Los Angeles
pezzi_citta = []
dimensione_blocco = 100000

# Estrazione dati riguardanti Los Angles
try:
    for chunk in pd.read_csv(nome_file_originale, chunksize=dimensione_blocco, low_memory=False):
        filtro_chunk = chunk[chunk['City'] == citta_target]
        
        if not filtro_chunk.empty:
            pezzi_citta.append(filtro_chunk)

    df_filtered = pd.concat(pezzi_citta, ignore_index=True)
    print(f"Caricamento completato")

except FileNotFoundError:
    print(f"\n file '{nome_file_originale}' non trovato")
    exit()

# Rimuoviamo i record che non hanno le coordinate spaziali (fondamentali per i nodi)
df_filtered = df_filtered.dropna(subset=['Start_Lat', 'Start_Lng'])

# Conversione della colonna temporale con il formato 'mixed'
df_filtered['Start_Time'] = pd.to_datetime(df_filtered['Start_Time'], format='mixed')

# Creo colonna del tempo aggregata per Mese
df_filtered['Mese_Anno'] = df_filtered['Start_Time'].dt.to_period('M')

# Nodi creati arrotondando LAT e LON a 2 cifre decimali (quadrati 1.1 km per quartiere)
df_filtered['Lat_Nodo'] = df_filtered['Start_Lat'].round(2)
df_filtered['Lng_Nodo'] = df_filtered['Start_Lng'].round(2)

# Definiamo un ID univoco per ogni nodo
df_filtered['Nodo_ID'] = df_filtered['Lat_Nodo'].astype(str) + "_" + df_filtered['Lng_Nodo'].astype(str)

nodi_unici = df_filtered['Nodo_ID'].nunique()
#print(f"Numero di nodi unici trovati a {citta_target}: {nodi_unici}")


# Tabella pivot: righe = Nodi, colonne = Mesi, valore = conteggio incidenti
matrix_signal = df_filtered.pivot_table(
    index='Nodo_ID', 
    columns='Mese_Anno', 
    values='ID', 
    aggfunc='count', 
    fill_value=0
)

# Eliminazioni quartieri meno rilevanti
soglia_minima_incidenti = 15
matrix_signal_cleaned = matrix_signal[matrix_signal.sum(axis=1) >= soglia_minima_incidenti]

coords_nodi = df_filtered.drop_duplicates('Nodo_ID').set_index('Nodo_ID')[['Lat_Nodo', 'Lng_Nodo']]
coords_nodi = coords_nodi.loc[matrix_signal_cleaned.index]

print("\n Salvataggio del nuovo file preprocessato")

# Uniamo le coordinate spaziali con la storia dei segnali mensili
df_final_dataset = pd.concat([coords_nodi, matrix_signal_cleaned], axis=1)

df_final_dataset = df_final_dataset.reset_index()

df_final_dataset.columns = [str(col) for col in df_final_dataset.columns]

# Salviamo il file in un nuovo CSV leggerissimo
nome_file_uscita = f"US_Accidents_{citta_target.replace(' ', '_')}_Signals.csv"
df_final_dataset.to_csv(nome_file_uscita, index=False)

print(df_final_dataset.head())
print(f"Nuovo dataset salvato")