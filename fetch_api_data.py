import pandas as pd
import requests
import time

def fix_my_dataset(csv_path="train_data.csv", output_path="new_imd_train_data.csv"):
    print("📊 Loading original Kaggle dataset...")
    df = pd.read_csv(csv_path)
    
    # Create Date string (This worked perfectly in the first run!)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.strftime('%Y-%m-%d')
    
    unique_locations = df['Location'].unique()
    print(f"🔍 Found {len(unique_locations)} unique locations. Starting data fetch...")
    
    all_historical_data = []

    for loc in unique_locations:
        print(f"\n📍 Processing Location: {loc}")
        
        # Better search logic for Open-Meteo
        search_loc = str(loc).replace(',', ' ').strip()
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={search_loc}&count=1&format=json"
        
        try:
            geo_resp = requests.get(geo_url).json()
            if not geo_resp.get('results'):
                # Fallback: Try searching just the first word if the full name fails
                fallback_loc = search_loc.split(' ')[0]
                geo_url_fallback = f"https://geocoding-api.open-meteo.com/v1/search?name={fallback_loc}&count=1&format=json"
                geo_resp = requests.get(geo_url_fallback).json()
                
                if not geo_resp.get('results'):
                    print(f"  ❌ Could not find GPS coordinates for {loc}. Skipping...")
                    continue
            
            lat = geo_resp['results'][0]['latitude']
            lon = geo_resp['results'][0]['longitude']
            print(f"  ✅ GPS Coordinates found: {lat}, {lon}")
            
        except Exception as e:
            print(f"  ❌ Geocoding failed for {loc}: {e}")
            continue
            
        loc_df = df[df['Location'] == loc]
        start_date = loc_df['Date'].min()
        end_date = loc_df['Date'].max()
        print(f"  📅 Fetching hourly data from {start_date} to {end_date}")
        
        aqi_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        )
        
        try:
            aqi_resp = requests.get(aqi_url).json()
            if 'hourly' not in aqi_resp:
                print(f"  ❌ Failed to fetch AQI data.")
                continue
                
            hourly_data = aqi_resp['hourly']
            temp_df = pd.DataFrame(hourly_data)
            
            temp_df['time'] = pd.to_datetime(temp_df['time'])
            temp_df['Year'] = temp_df['time'].dt.year.astype(int)
            temp_df['Month'] = temp_df['time'].dt.month.astype(int)
            temp_df['Day'] = temp_df['time'].dt.day.astype(int)
            temp_df['Hour'] = temp_df['time'].dt.hour.astype(int)
            temp_df['Location'] = loc
            
            all_historical_data.append(temp_df)
            print(f"  ✅ Successfully downloaded {len(temp_df)} hourly sensor readings!")
            
        except Exception as e:
            print(f"  ❌ API Error: {e}")
            
        time.sleep(1)

    print("\n🧬 Merging new high-quality data with your images...")
    
    if len(all_historical_data) == 0:
        print("❌ Critical Error: No data was downloaded at all. Cannot merge.")
        return

    fetched_df = pd.concat(all_historical_data, ignore_index=True)
    
    fetched_df = fetched_df.rename(columns={
        'pm2_5': 'PM2.5',
        'pm10': 'PM10',
        'carbon_monoxide': 'CO',
        'nitrogen_dioxide': 'NO2',
        'sulphur_dioxide': 'SO2',
        'ozone': 'O3'
    })
    
    columns_to_keep = ['Location', 'Year', 'Month', 'Day', 'Hour', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    fetched_df = fetched_df[columns_to_keep]
    
    # 💥 DROP THE CORRUPTED KAGGLE NUMBERS 💥
    kaggle_clean = df.drop(columns=['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2'])
    
    # ======== THE SURGICAL FIX ========
    # Safely force the Kaggle time columns into integers so Pandas can merge them!
    kaggle_clean['Year'] = pd.to_numeric(kaggle_clean['Year'], errors='coerce').fillna(2023).astype(int)
    kaggle_clean['Month'] = pd.to_numeric(kaggle_clean['Month'], errors='coerce').fillna(1).astype(int)
    kaggle_clean['Day'] = pd.to_numeric(kaggle_clean['Day'], errors='coerce').fillna(1).astype(int)
    kaggle_clean['Hour'] = pd.to_numeric(kaggle_clean['Hour'], errors='coerce').fillna(12).astype(int)
    # ==================================
    
    # 🔗 THE MAGIC MERGE
    perfect_dataset = pd.merge(
        kaggle_clean, 
        fetched_df, 
        on=['Location', 'Year', 'Month', 'Day', 'Hour'], 
        how='inner' 
    )
    
    perfect_dataset.to_csv(output_path, index=False)
    print(f"\n🎉 DONE! Saved {len(perfect_dataset)} mathematically perfect rows to '{output_path}'")

if __name__ == "__main__":
    fix_my_dataset(csv_path="val_data.csv", output_path="new_imd_val_data.csv")