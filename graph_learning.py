import pandas as pd
import numpy as np
import cvxpy as cp
import scipy.linalg as la
import matplotlib.pyplot as plt
import networkx as nx
import contextily as ctx
from shapely.geometry import Point
import geopandas as gpd

from graph_learning_ADMM import graph_learning_admm

print("Caricamento dati")
df = pd.read_csv('US_Accidents_Los_Angeles_Signals.csv')

colonne_segnali = df.columns[3:]
X = df[colonne_segnali].values

N_nodi = X.shape[0]
M_mesi = X.shape[1]

latitudini = df['Lat_Nodo'].values
longitudini = df['Lng_Nodo'].values

#print(f"Numero di nodi (quartieri): {N_nodi}")
#print(f"Mesi di osservazione storica: {M_mesi}")

# Adicenza geometrica
print("\n Costruzione del Grafo basato su distanza fisica")
Dist_geo = np.zeros((N_nodi, N_nodi))
for i in range(N_nodi):
    for j in range(N_nodi):
        Dist_geo[i, j] = np.sqrt((latitudini[i] - latitudini[j])**2 + (longitudini[i] - longitudini[j])**2)

beta = np.mean(Dist_geo)
A_geo = np.exp(-(Dist_geo**2) / (2 * beta**2))
np.fill_diagonal(A_geo, 0)

D_geo = np.diag(np.sum(A_geo, axis=1))
L_geo = D_geo - A_geo
autovalori_geo, U_geo = la.eigh(L_geo) 

# Adiacenza ottimale con ADMM
print("\n--- Ottimizzazione: Calcolo della Matrice di Adiacenza Ottimale ---")
Z = np.zeros((N_nodi, N_nodi))
for i in range(N_nodi):
    for j in range(N_nodi):
        Z[i, j] = np.sum((X[i, :] - X[j, :])**2)

if np.max(Z) > 0:
    Z = Z / np.max(Z)

alpha = 0.5 
A_ottimale = graph_learning_admm(Z, alpha, rho=1.5, max_iters=3000, tol=1e-5)

D_learned = np.diag(np.sum(A_ottimale, axis=1))
L_learned = D_learned - A_ottimale
autovalori_learned, U_learned = la.eigh(L_learned)


# Plot per il confronto
print("\n--- Studio Comparativo dell'Errore al variare di K ---")
x_originale = X[:, -1]

x_hat_geo = U_geo.T @ x_originale 
x_hat_learned = U_learned.T @ x_originale

valori_k = np.arange(5, 150, 5)
mse_geo_lista = []
mse_learned_lista = []

for k in valori_k:
    x_hat_g_ridotto = np.zeros_like(x_hat_geo)
    x_hat_g_ridotto[:k] = x_hat_geo[:k]
    x_rec_geo = U_geo @ x_hat_g_ridotto
    mse_geo_lista.append(np.mean((x_originale - x_rec_geo) ** 2))
    
    x_hat_l_ridotto = np.zeros_like(x_hat_learned)
    x_hat_l_ridotto[:k] = x_hat_learned[:k]
    x_rec_learned = U_learned @ x_hat_l_ridotto
    mse_learned_lista.append(np.mean((x_originale - x_rec_learned) ** 2))

# Plot MSE vs K
plt.figure(figsize=(10, 5))
plt.plot(valori_k, mse_geo_lista, label="Grafo Geometrico Reale (Distanza)", color="blue", marker='s', linewidth=1.5)
plt.plot(valori_k, mse_learned_lista, label="Grafo Imparato (Graph Learning)", color="red", marker='o', linewidth=1.5)
plt.title("Confronto MSE di Ricostruzione al variare delle Componenti Spettrali (K)")
plt.xlabel("Numero di Autovettori usati (K)")
plt.ylabel("Errore di Ricostruzione (MSE)")
plt.legend()
plt.grid(True, linestyle=":")
plt.show()

# Plot confronto al variare di K = 10, 50, 100
valori_k_scelti = [10, 50, 100]
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

for i, K in enumerate(valori_k_scelti):
    x_rec_geo = U_geo @ np.append(x_hat_geo[:K], np.zeros(N_nodi - K))
    x_rec_learned = U_learned @ np.append(x_hat_learned[:K], np.zeros(N_nodi - K))
    
    mse_geo_locale = np.mean((x_originale - x_rec_geo) ** 2)
    mse_learned_locale = np.mean((x_originale - x_rec_learned) ** 2)
    
    axes[i].plot(x_originale, label="Mappa Incidenti Reale (x)", color="black", linewidth=1.2, marker='o', markersize=2)
    axes[i].plot(x_rec_geo, label=f"Ricostruzione Geo (MSE: {mse_geo_locale:.4f})", color="blue", linestyle=":", linewidth=1.8)
    axes[i].plot(x_rec_learned, label=f"Ricostruzione Learned (MSE: {mse_learned_locale:.4f})", color="red", linestyle="--", linewidth=2.0)
    
    axes[i].set_title(f"Confronto Spettrale con K = {K} Autovettori", fontsize=12, fontweight='bold')
    axes[i].set_ylabel("Numero di Incidenti")
    axes[i].legend(loc="upper right")
    axes[i].grid(True, linestyle=":", alpha=0.6)

axes[-1].set_xlabel("Indice dei Nodi (Quartieri)")
plt.tight_layout()
plt.show()


#Plot heatmap
print("\n--- Generazione delle mappe spaziali su mappa reale di LA (K=50) ---")
K_confronto = 50
x_rec_geo_mappa = U_geo @ np.append(x_hat_geo[:K_confronto], np.zeros(N_nodi - K_confronto))
x_rec_learned_mappa = U_learned @ np.append(x_hat_learned[:K_confronto], np.zeros(N_nodi - K_confronto))

geometrie = [Point(xy) for xy in zip(longitudini, latitudini)]
gdf_base = gpd.GeoDataFrame(df, geometry=geometrie, crs="EPSG:4326").to_crs(epsg=3857)
x_coords = gdf_base.geometry.x
y_coords = gdf_base.geometry.y

fig, axes = plt.subplots(1, 3, figsize=(22, 9), sharex=True, sharey=True)
vmin, vmax = np.min(x_originale), np.max(x_originale)
mappa_colori = plt.cm.YlOrRd 
sfondo_mappa = ctx.providers.CartoDB.Positron

# Originale
scat1 = axes[0].scatter(x_coords, y_coords, c=x_originale, cmap=mappa_colori, vmin=vmin, vmax=vmax, s=55, edgecolors='black', linewidths=0.4, zorder=2)
ctx.add_basemap(axes[0], source=sfondo_mappa, zorder=1)
axes[0].set_title("1. Mappa Originale (Marzo 2023)", fontsize=13, fontweight='bold')
axes[0].set_aspect('equal')
axes[0].axis('off')

# Geometrica
axes[1].scatter(x_coords, y_coords, c=x_rec_geo_mappa, cmap=mappa_colori, vmin=vmin, vmax=vmax, s=55, edgecolors='black', linewidths=0.4, zorder=2)
ctx.add_basemap(axes[1], source=sfondo_mappa, zorder=1)
axes[1].set_title(f"2. Ricostruzione Geometrica (K={K_confronto})", fontsize=13, fontweight='bold')
axes[1].set_aspect('equal')
axes[1].axis('off')

# Learned
axes[2].scatter(x_coords, y_coords, c=x_rec_learned_mappa, cmap=mappa_colori, vmin=vmin, vmax=vmax, s=55, edgecolors='black', linewidths=0.4, zorder=2)
ctx.add_basemap(axes[2], source=sfondo_mappa, zorder=1)
axes[2].set_title(f"3. Ricostruzione Ottima (K={K_confronto})", fontsize=13, fontweight='bold')
axes[2].set_aspect('equal')
axes[2].axis('off')

pad_x = (np.max(x_coords) - np.min(x_coords)) * 0.05
pad_y = (np.max(y_coords) - np.min(y_coords)) * 0.05
plt.xlim(np.min(x_coords) - pad_x, np.max(x_coords) + pad_x)
plt.ylim(np.min(y_coords) - pad_y, np.max(y_coords) + pad_y)

cbar = fig.colorbar(scat1, ax=axes.ravel().tolist(), shrink=0.4, orientation='horizontal', pad=0.05)
cbar.set_label("Numero di Incidenti Registrati / Stimati", fontsize=11)
plt.show()

# Test con segnale non-smoth su grafo A ottimo
print("Generazione di un segnale NON-SMOOTH sul Grafo Appreso")

x_anti_smooth = U_learned[:, -1] * 50  # Aggiungo ampiezza all' ultima componente spettrale per definizione quella del segnale non smooth

# Calcolo GFT per questo segnale
x_hat_anti_geo = U_geo.T @ x_anti_smooth
x_hat_anti_learned = U_learned.T @ x_anti_smooth

valori_k_anti = np.arange(5, 150, 5)
mse_anti_geo = []
mse_anti_learned = []

for k in valori_k_anti:
    # Ricostruzione Geometrica
    x_hat_g_ridotto = np.zeros_like(x_hat_anti_geo)
    x_hat_g_ridotto[:k] = x_hat_anti_geo[:k]
    x_rec_geo = U_geo @ x_hat_g_ridotto
    mse_anti_geo.append(np.mean((x_anti_smooth - x_rec_geo) ** 2))
    
    # Ricostruzione Learned (Modello Appreso)
    x_hat_l_ridotto = np.zeros_like(x_hat_anti_learned)
    x_hat_l_ridotto[:k] = x_hat_anti_learned[:k]
    x_rec_learned = U_learned @ x_hat_l_ridotto
    mse_anti_learned.append(np.mean((x_anti_smooth - x_rec_learned) ** 2))

# Plot del "Fallimento controllato"
plt.figure(figsize=(10, 5))
plt.plot(valori_k_anti, mse_anti_geo, label="Grafo Geometrico", color="blue", marker='s', linewidth=1.5)
plt.plot(valori_k_anti, mse_anti_learned, label="Grafo Imparato (ADMM)", color="red", marker='o', linewidth=1.5)
plt.title("Ricostruzione di un Segnale ad Alta Frequenza (Non-Smooth sul Grafo Appreso)")
plt.xlabel("Numero di Autovettori usati (K)")
plt.ylabel("Errore di Ricostruzione (MSE)")
plt.legend()
plt.grid(True, linestyle=":")
plt.show()