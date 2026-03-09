import requests
import pandas as pd
import time
from datetime import datetime
import os


TOMTOM_API_KEY = ""  # the key from TOMTOM
POINT = (45.786685, 4.888233)  # the Latitude and longitude of the position where you want to collect data
OUTPUT_DIR = "traffic_data"
COLLECTION_INTERVAL = 60  # the data will be collected every minute (every 60 seconds)
TOTAL_DURATION = 10800  # the total duration for data collection in seconds (1 hour)

# Assumed road capacity (vehicles per hour)
ROAD_CAPACITY = 2500  # vehicles per hour

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_traffic_data(lat, lon, api_key):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"
    params = {
        "point": f"{lat},{lon}",
        "unit": "KMPH",
        "openLr": "true",
        "key": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[{datetime.now()}] request fails: {e}")
        return None


def extract_info(data, lat, lon):
    segment = data.get("flowSegmentData", {})
    current_speed = segment.get("currentSpeed", 0)
    # Calculate flow
    TTI = segment.get("currentTravelTime", 0) / segment.get("freeFlowTravelTime", 1)


    flow = ROAD_CAPACITY / max(1, TTI)
    flow = int(flow / 60)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": lat,
        "longitude": lon,
        "current_speed": current_speed,
        "flow": flow,  # Add the flow to the data
        "free_flow_speed": segment.get("freeFlowSpeed", 0),
        "current_travel_time": segment.get("currentTravelTime", 0),
        "free_flow_travel_time": segment.get("freeFlowTravelTime", 0),
        "confidence": segment.get("confidence", 0),
        "road_type": segment.get("frc", ""),

    }


def run_monitor():
    print("Starting traffic data collection...")
    records = []
    start_time = time.time()
    end_time = start_time + TOTAL_DURATION

    while time.time() < end_time:
        data = get_traffic_data(POINT[0], POINT[1], TOMTOM_API_KEY)
        if data:
            info = extract_info(data, POINT[0], POINT[1])
            print(
                f"[{info['timestamp']}] current speed: {info['current_speed']} km/h | flow: {info['flow']} vehicles/minute ")
            records.append(info)
        else:
            print("the data of this time is not available ")

        time.sleep(COLLECTION_INTERVAL)

    # Save to CSV
    df = pd.DataFrame(records)
    # filename = f"RD383_Villeurbanne_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    filename = f"RD383_Villeurbanne_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"\n data have been saved in : {filepath}")
    print(df)


if __name__ == "__main__":
    run_monitor()
