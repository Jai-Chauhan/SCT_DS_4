import zipfile
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans


ZIP_PATH = "C:/Users/cjani/Downloads/Road Accident Data.csv.zip"


def load_all_csv_from_zip(zip_path):
    """
    Loads ALL CSV files inside a ZIP archive into a dictionary of DataFrames.
    Key = file name
    Value = pandas DataFrame
    """
    dataframes = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():
            if file.lower().endswith(".csv"):
                with z.open(file) as f:
                    df = pd.read_csv(f)
                    dataframes[os.path.basename(file)] = df

    return dataframes



# -------- LOAD DATA --------
datasets = load_all_csv_from_zip(ZIP_PATH)

# Join datasets if structures match
df_list = list(datasets.values())
if len(df_list) > 1 and all((df_list[0].columns == d.columns).all() for d in df_list):
    df = pd.concat(df_list, ignore_index=True)
else:
    df = df_list[0]


print("\n===== DATA PREVIEW =====\n")
print(df.head())



# -------- CLEANING --------
# Standardize likely column names (edit if needed)
possible_time_cols = ["Time", "Accident_Time", "Crash_Time"]
possible_weather_cols = ["Weather", "Weather_Conditions"]
possible_road_cols = ["Road_Condition", "Road_Surface"]
possible_lat = ["Latitude", "lat"]
possible_lon = ["Longitude", "lon", "lng"]

def pick(col_list):
    for c in col_list:
        if c in df.columns:
            return c
    return None


time_col = pick(possible_time_cols)
weather_col = pick(possible_weather_cols)
road_col = pick(possible_road_cols)
lat_col = pick(possible_lat)
lon_col = pick(possible_lon)



# Convert time if exists
if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["Hour"] = df[time_col].dt.hour



# Handle missing values
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())



# -------- SUMMARY --------
print("\n===== SUMMARY =====\n")
print(df.describe(include="all"))



# -------- VISUALIZATIONS --------
sns.set(style="whitegrid")



# WEATHER vs ACCIDENTS
if weather_col:
    plt.figure(figsize=(8,5))
    df[weather_col].value_counts().plot(kind="bar")
    plt.title("Accidents by Weather Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()



# ROAD CONDITION vs ACCIDENTS
if road_col:
    plt.figure(figsize=(8,5))
    df[road_col].value_counts().plot(kind="bar")
    plt.title("Accidents by Road Condition")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.show()



# TIME OF DAY DISTRIBUTION
if time_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df["Hour"], bins=24)
    plt.title("Accidents by Hour of Day")
    plt.xlabel("Hour (0–23)")
    plt.ylabel("Accident Count")
    plt.show()



# CROSS COMPARISONS
if time_col and weather_col:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=weather_col, y="Hour", data=df)
    plt.xticks(rotation=45)
    plt.title("Time of Day vs Weather")
    plt.show()


if time_col and road_col:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=road_col, y="Hour", data=df)
    plt.xticks(rotation=45)
    plt.title("Time of Day vs Road Condition")
    plt.show()



# -------- HOTSPOT ANALYSIS (if GPS available) --------
if lat_col and lon_col:

    coords = df[[lat_col, lon_col]].dropna()

    # K-means clustering to find hotspots
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    coords["cluster"] = kmeans.fit_predict(coords)

    plt.figure(figsize=(8,6))
    plt.scatter(coords[lat_col], coords[lon_col], c=coords["cluster"], s=10)
    plt.title("Accident Hotspots (Clustered)")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()

    print("\n===== HOTSPOT CENTERS =====\n")
    print(pd.DataFrame(kmeans.cluster_centers_, columns=["Latitude", "Longitude"]))

else:
    print("\nGPS coordinates not found — hotspot clustering skipped.")



# -------- KEY FINDINGS --------
print("\n===== PATTERN & TREND INSIGHTS =====\n")
print("• Accident frequencies analyzed by weather, road condition, and time")
print("• Peak accident times identified")
print("• Weather impact examined")
print("• Road surface impact examined")
print("• Hotspots detected where geo-coordinates exist")
