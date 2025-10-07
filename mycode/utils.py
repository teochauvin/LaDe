import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import folium
import numpy as np 

def distribution_date(df):
    # Préfixer l'année pour le parsing
    df['accept_time_full'] = '2000-' + df['accept_time']

    # Convertir en datetime
    df['accept_time_full'] = pd.to_datetime(
        df['accept_time_full'], format='%Y-%m-%d %H:%M:%S', errors='coerce'
    )

    # Supprimer lignes non parsables
    df = df.dropna(subset=['accept_time_full'])

    # Extraire la date
    df['date'] = df['accept_time_full'].dt.date

    # Compter les livraisons par jour
    daily_counts = df.groupby('date').size()

    # Afficher
    print(daily_counts.head())

    # Tracer
    plt.figure(figsize=(12,6))
    plt.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-', color='royalblue')
    plt.title("📦 Distribution des livraisons par jour à Yantai")
    plt.xlabel("Date")
    plt.ylabel("Nombre de livraisons")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def departure_arrival(df): 

    # 3️⃣ Ajouter une année par défaut pour parser correctement
    df['accept_time_full'] = pd.to_datetime('2021-' + df['accept_time'], 
                                            format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['delivery_time_full'] = pd.to_datetime('2021-' + df['delivery_time'], 
                                            format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # 4️⃣ Supprimer les lignes avec timestamps manquants
    df = df.dropna(subset=['accept_time_full', 'delivery_time_full'])

    # 5️⃣ Extraire l'heure du jour (0-23h) pour départ et arrivée
    df['accept_hour'] = df['accept_time_full'].dt.hour + df['accept_time_full'].dt.minute/60
    df['delivery_hour'] = df['delivery_time_full'].dt.hour + df['delivery_time_full'].dt.minute/60

    df['accept_gps_time_full'] = pd.to_datetime('2021-' + df['accept_gps_time'], 
                                            format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['delivery_gps_time_full'] = pd.to_datetime('2021-' + df['delivery_gps_time'], 
                                                format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Corriger passage à minuit
    mask = df['delivery_gps_time_full'] < df['accept_gps_time_full']
    df.loc[mask, 'delivery_gps_time_full'] += timedelta(days=1)

    # Calculer durée en minutes
    df['duration_min'] = (df['delivery_gps_time_full'] - df['accept_gps_time_full']).dt.total_seconds() / 60

    # 7️⃣ Tracer la distribution des heures de départ
    plt.figure(figsize=(12,4))
    plt.hist(df['accept_hour'], bins=100, color='green', alpha=0.7)
    plt.title("Distribution des heures de départ")
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de livraisons")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8️⃣ Tracer la distribution des heures d'arrivée
    plt.figure(figsize=(12,4))
    plt.hist(df['delivery_hour'], bins=100, color='red', alpha=0.7)
    plt.title("Distribution des heures d'arrivée")
    plt.xlabel("Heure de la journée")
    plt.ylabel("Nombre de livraisons")
    plt.grid(True, alpha=0.3)
    plt.show()

def distribution_distance(df_clean, threshold):
    def haversine(lat1, lon1, lat2, lon2):
        """
        Compute Haversine distance (km) between two points.
        """
        R = 6371  # Earth radius in km
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    # Compute distances for all rows
    df_clean['distance_km'] = haversine(
        df_clean['accept_gps_lat'], df_clean['accept_gps_lng'],
        df_clean['delivery_gps_lat'], df_clean['delivery_gps_lng']
    )

    # Show basic statistics
    print(df_clean['distance_km'].describe())

    df_clean = df_clean[df_clean['distance_km'] <= threshold]

    # Plot histogram
    plt.figure(figsize=(10,5))
    plt.hist(df_clean['distance_km'], bins=150, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("📦 Distribution des distances entre pickup et delivery")
    plt.xlabel("Distance (km)")
    plt.ylabel("Nombre de livraisons")
    plt.grid(alpha=0.3)
    plt.show()

def create_map(df, day, month): 

    df['accept_time_full'] = pd.to_datetime('2021-' + df['accept_gps_time'], 
                                        format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Filter by month and day
    filtered_df = df[(df['accept_time_full'].dt.month == month) & (df['accept_time_full'].dt.day == day)]

    # Example: df_yantai contains your data
    gps_cols = ['accept_gps_lat', 'accept_gps_lng', 'delivery_gps_lat', 'delivery_gps_lng']

    df_clean = filtered_df.dropna(subset=gps_cols)

    #  3️⃣ Créer une carte centrée sur Yantai
    m = folium.Map(location=[df_clean['delivery_gps_lat'].mean(),
                             df_clean['delivery_gps_lng'].mean()],
                   zoom_start=12)

    #4️⃣ Ajouter les trajets (ligne départ ➝ arrivée)
    for _, row in df_clean.iterrows():  # échantillonner 500 trajets pour performance
        folium.PolyLine(
            locations=[
                [row['accept_gps_lat'], row['accept_gps_lng']],
                [row['delivery_gps_lat'], row['delivery_gps_lng']]
            ],
            color='blue',
            weight=2,
            opacity=0.5
        ).add_to(m)

    # 5️⃣ Ajouter des points de départ et d’arrivée
    for _, row in df_clean.iterrows():  # moins de points pour lisibilité
        folium.CircleMarker(
            location=[row['accept_gps_lat'], row['accept_gps_lng']],
            radius=3,
            color='green',
            fill=True,
            fill_opacity=0.7,
            popup=f"Accept: {row['accept_time']}"
        ).add_to(m)
        folium.CircleMarker(
            location=[row['delivery_gps_lat'], row['delivery_gps_lng']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"Delivered: {row['delivery_time']}"
        ).add_to(m)

    # 6️⃣ Sauvegarder la carte
    m.save(f"deliveries_{month}-{day}.html")
    print("✅ Carte créée")
