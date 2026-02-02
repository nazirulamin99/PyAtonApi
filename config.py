# config.py - Updated with No Data condition
MAP_STYLE = "mapbox://styles/mapbox/satellite-streets-v12"
MAPBOX_TOKEN = "pk.eyJ1IjoiYXp6dWxoaXNoYW0iLCJhIjoiY2s5bjR1NDBqMDJqNDNubjdveXdiOGswYyJ9.SYlfXRzRtpbFoM2PHskvBg"
UTC_OFFSET = 8

# Enhanced color mapping with No Data condition
COLOR_MAP = {
    ("Beacon", "Good"): [0, 200, 180, 200],
    ("Beacon", "Not Good"): [255, 140, 0, 200], 
    ("Beacon", "Low Battery"): [255, 0, 150, 200],
    ("Beacon", "Low Voltage"): [255, 100, 0, 200],
    ("Beacon", "No Data"): [169, 169, 169, 200],  # Gray for no data
    
    ("Buoy", "Good"): [0, 100, 255, 200],
    ("Buoy", "Not Good"): [255, 0, 0, 200],
    ("Buoy", "Low Battery"): [255, 0, 150, 200], 
    ("Buoy", "Low Voltage"): [255, 100, 0, 200],
    ("Buoy", "No Data"): [169, 169, 169, 200],
    
    ("Lighthouse", "Good"): [180, 0, 255, 200],
    ("Lighthouse", "Not Good"): [255, 255, 0, 200],
    ("Lighthouse", "Low Battery"): [255, 0, 150, 200],
    ("Lighthouse", "Low Voltage"): [255, 100, 0, 200], 
    ("Lighthouse", "No Data"): [169, 169, 169, 200],
    
    # Unknown type fallbacks
    ("Unknown", "Good"): [100, 200, 100, 200],
    ("Unknown", "Not Good"): [200, 100, 100, 200],
    ("Unknown", "Low Battery"): [200, 100, 200, 200],
    ("Unknown", "Low Voltage"): [200, 150, 100, 200],
    ("Unknown", "No Data"): [128, 128, 128, 200],
}

# Default map center if no data
DEFAULT_CENTER = (3.1390, 101.6869)  # KL