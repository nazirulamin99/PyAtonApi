# Enhanced data_loader.py with better error handling and caching
# source venv/bin/activate

import clickhouse_connect
import math
import pandas as pd
from datetime import timedelta
from config import COLOR_MAP, UTC_OFFSET
# from openai import OpenAI
import numpy as np
import os
import json
from pathlib import Path
import streamlit as st
import logging
from typing import Optional, Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = Path("aton_index.json")
DATA_CACHE_FILE = Path("aton_data_cache.json")
CACHE_DURATION_HOURS = 1  # Cache data for 1 hour

# Initialize OpenAI client with error handling
# try:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# except Exception as e:
#     logger.error(f"Failed to initialize OpenAI client: {e}")
#     client = None


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert value to float with default"""
    try:
        if val is None or val == '':
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

# @st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_all_aton() -> pd.DataFrame:
    """Load AtoN data with enhanced error handling and caching"""
    
    try:
        client = clickhouse_connect.get_client(
            host='43.216.85.155',
            username='',
            password='',
            connect_timeout=30,
            # query_limit=60
        )

        query = """
        WITH 
            atons AS (
                SELECT * 
                FROM pnav.atonlist
            ),
            aton_static AS (
                WITH rowcountstatic AS (
                    SELECT *, row_number() OVER (PARTITION BY mmsi ORDER BY ts DESC) AS rowcountby_mmsi
                    FROM pnav.ptp_type21
                    WHERE ts >= date_add(DAY, -2, now()) 
                    AND mmsi IN (SELECT mmsi FROM atons)
                )
                SELECT *
                FROM rowcountstatic
                WHERE rowcountby_mmsi = 1
            ),
            aton_meas AS (  
                WITH rowcountdata AS (
                    SELECT *, row_number() OVER (PARTITION BY mmsi ORDER BY ts DESC) AS rowcountby_mmsi
                    FROM pnav.ais_type6_533
                    WHERE ts >= date_add(DAY, -2, now())
                    AND mmsi IN (SELECT mmsi FROM atons)
                )
                SELECT *
                FROM rowcountdata
                WHERE rowcountby_mmsi = 1
            )   
        SELECT
            aa.ts               AS meas_ts,
            CAST(aa.mmsi AS VARCHAR)             AS meas_mmsi,
            aa.volt_int         AS volt_int,
            aa.health           AS health,
            ss.ts               AS static_ts,
            ss.mmsi             AS static_mmsi,
            ss.aidType          AS aidType,
            ss.aidName          AS aidName,
            ss.longitude        AS longitude,
            ss.latitude         AS latitude,
            al.no               AS aton_no,
            initcap(al.name)             AS aton_name,
            al.mmsi             AS al_mmsi,
            al.region           AS region,
            al.type             AS aton_type
        FROM aton_meas aa
        RIGHT JOIN aton_static ss ON aa.mmsi = ss.mmsi
        RIGHT JOIN atons al ON al.mmsi = ss.mmsi
        WHERE aa.mmsi != 0 AND ss.longitude >= 0 AND ss.latitude <= 180
        """

        result = client.query(query)
        
        if not result.result_rows:
            logger.warning("No data returned from database")
            return pd.DataFrame()

        rows = []
        for i in result.result_rows:
            try:
                meas_ts, meas_mmsi, volt_int, health, static_ts, static_mmsi, aidType, aidName, longitude, latitude, \
                aton_no, aton_name, al_mmsi, region, aton_type = i

                # Safe conversions with validation
                voltage = safe_float(volt_int, 0.0)
                lon = safe_float(longitude, 0.0)
                lat = safe_float(latitude, 0.0)
                
                # Skip records with invalid coordinates
                if lat == 0.0 and lon == 0.0:
                    continue

                # Enhanced condition logic
                if voltage < 10:
                    condition = "Low Battery"
                elif health == 1:
                    condition = "Not Good"
                else:
                    condition = "Good"

                row_data = {
                    'ts': (meas_ts + timedelta(hours=UTC_OFFSET)).strftime("%Y-%m-%d %H:%M:%S") if meas_ts else "",
                    'mmsi': meas_mmsi,
                    'name': aton_name,
                    'region': region or "Unknown",
                    'aton_type': aton_type or "Unknown",
                    'minBattAton': voltage,
                    # 'voltage': voltage,
                    'longitude': lon,
                    'latitude': lat,
                    'lastseen': (static_ts + timedelta(hours=UTC_OFFSET)).strftime("%Y-%m-%d %H:%M:%S") if static_ts else "",
                    'health_OKNG': health,
                    'aton_no': aton_no,   
                    'condition': condition,
                    'color': COLOR_MAP.get((aton_type, condition), [128, 128, 128, 200]),
                    'volt_aton': volt_int
                }
                
                rows.append(row_data)
                
            except Exception as e:
                logger.warning(f"Failed to process row: {e}")
                continue

        df = pd.DataFrame(rows)
        
        # Enhanced data validation
        df = df[df['latitude'].notna() & df['longitude'].notna()]
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]  # Remove null island
        df['lastseen_dt'] = pd.to_datetime(df['lastseen'], errors='coerce')
        
        logger.info(f"Successfully loaded {len(df)} AtoN records")
        return df
        
    except Exception as e:
        # logger.error(f"Failed to load AtoN data: {e}")
        # st.error(f"Database connection failed: {e}")
        return None
    
def load_aton_document(df) -> List[str]:
    """Load AtoN data as a document"""

    documents = []
    for index, row in df.iterrows():
        # documents.append(f'''
        #     ATON details No. {index + 1}
        #     name = {row['name']}
        #     location at {row['region']}
        #     longitude = {row['longitude']}
        #     latitude = {row['latitude']}
        #     condition = {row['condition']}
        #     mmsi start with {str(row['mmsi'])[0:5]}
        #     mmsi end with {str(row['mmsi'])[5:]}
        #     mmsi = {row['mmsi']}
        #     voltage = {row['volt_aton']}
        #     aton type = {row['aton_type']}
        # ''')

        documents.append(f'''
            ATON No. {index + 1}, name: {row['name']} with mmsi: {row['mmsi']}, it is a type of {row['aton_type']} located at {row['region']} (longitude = {row['longitude']}, latitude = {row['latitude']}). 
            It condition is {row['condition']} with voltage = {row['volt_aton']}.
            Additional information: it mmsi start with {str(row['mmsi'])[0:5]} and end with {str(row['mmsi'])[5:]}.
        ''')        



    return documents


def getAtonVoltByMMSI(mmsi): # Get Aton volt data for last 7 days 
    client = clickhouse_connect.get_client(
        host='43.216.85.155',
        username='',
        password='',
        connect_timeout=30,
    )

    result = client.query(
    f'''
        select ts, volt_int, volt_ex1, al.name 
        from pnav.ais_type6_533 at
        join pnav.atonlist al on al.mmsi=at.mmsi
        where ts >= date_add(DAY, -7, now()) and mmsi = {mmsi}
        order by ts
    '''
    )

    items = []
    for i in result.result_rows:
        data = {
            'ts': i[0].strftime("%Y-%m-%d %H:%M:%S"),
            'volt_int': i[1],
            'volt_ex1': i[2],
            'aton_name': i[3],
            'mmsi': mmsi
        }

        items.append(data)

    return items

def getAtonHeartbeatByMMSI(mmsi): # Get Aton heartbeat
    client = clickhouse_connect.get_client(
        host='43.216.85.155',
        username='',
        password='',
        connect_timeout=30,
    )

    result = client.query(
    f'''
        select ts, beat, al.name
        from pnav.ais_type6_533 at
        join pnav.atonlist al on al.mmsi=at.mmsi
        where ts >= date_add(DAY, -7, now()) and mmsi = '{mmsi}'
        order by ts
    '''
    )

    lists = []
    for i in result.result_rows:
        data = {
            'ts': i[0].strftime("%Y-%m-%d %H:%M:%S"),
            'beat': i[1],
            'Aton name': i[2],
            'mmsi': mmsi
        }

        lists.append(data)

    return lists

from datetime import timedelta
import clickhouse_connect
import math

UTC_OFFSET = 8  # Malaysia timezone offset

def getATON_summary(mmsi: str = None, date_value: str = None):
    client = clickhouse_connect.get_client(
        host='43.216.85.155',
        user='',
        password=''
    )

    where_clauses = []
    if mmsi:
        where_clauses.append(f"mmsi = '{mmsi}'")
    if date_value:
        where_clauses.append(f"toDate(ts) = toDate('{date_value}')")

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
        SELECT * 
        FROM pnav.aton_summary
        {where_clause}
        ORDER BY ts DESC
    """

    result = client.query(query)
    ret_result = []

    for i in result.result_rows:
        data = {
            # 'ts': (i[0] + timedelta(hours=UTC_OFFSET)).strftime("%Y-%m-%d %H:%M:%S"),
            'ts': (i[0].strftime("%Y-%m-%d %H:%M:%S")),
            'mmsi': i[1],
            'name': i[2],
            'region': i[3],
            'aton_type': i[4],
            'opt6_second': i[5],
            'opt6_percent': i[6],
            'opt21_second': i[7],
            'opt21_percent': i[8],
            'minTemp': i[9] if i[9] is not None and not math.isnan(i[9]) else 0,
            'maxTemp': i[10] if i[10] is not None and not math.isnan(i[10]) else 0,
            'meanTemp': i[11] if i[11] is not None and not math.isnan(i[11]) else 0,
            'last_Temp': i[12] if i[12] is not None and not math.isnan(i[12]) else 0,
            'minBattAton': i[13] if i[13] is not None and not math.isnan(i[13]) else 0,
            'maxBattAton': i[14] if i[14] is not None and not math.isnan(i[14]) else 0,
            'meanBattAton': i[15] if i[15] is not None and not math.isnan(i[15]) else 0,
            'last_BattAton': i[16] if i[16] is not None and not math.isnan(i[16]) else 0,
            'minBattLant': i[17] if i[17] is not None and not math.isnan(i[17]) else 0,
            'maxBattLant': i[18] if i[18] is not None and not math.isnan(i[18]) else 0,
            'meanBattLant': i[19] if i[19] is not None and not math.isnan(i[19]) else 0,
            'last_BattLant': i[20] if i[20] is not None and not math.isnan(i[20]) else 0,
            'off_pos_OKNG': i[21],
            'last_off_pos': i[22],
            'LDR_OKNG': i[23],
            'last_LDR': i[24],
            'light_OKNG': i[25],
            'last_light': i[26],
            'racon_OKNG': i[27],
            'last_racon': i[28],
            'health_OKNG': i[29],
            'last_health': i[30],
            'cnt_msg6': i[31],
            'cnt_msg21': i[32],
            'longitude': i[33],
            'latitude': i[34],
            'lastseen': i[35].strftime("%Y-%m-%d %H:%M:%S") if i[35] else None
        }

        ret_result.append(data)

    return ret_result


def getAtonByMonth(month):

    client = clickhouse_connect.get_client(
        host='43.216.85.155',
        user='',
        password=''
    )

    result = client.query(
    f'''
        SELECT formatDateTime(ts, '%Y-%m') as month, mmsi, name, region, aton_type,
            sum(opt6_second) AS opt6_second,
            avg(opt6_percent) AS opt6_percent,
            sum(opt21_second) AS opt21_second,
            avg(opt21_percent) AS opt21_percent,
            min(minTemp) AS minTemp,
            max(maxTemp) AS maxTemp,
            avg(meanTemp) AS meanTemp,
            last_Temp,
            min(minBattAton) AS minBattAton,
            max(maxBattAton) AS maxBattAton,
            avg(meanBattAton) AS meanBattAton,  
            last_BattAton,
            min(minBattLant) AS minBattLant,
            max(maxBattLant) AS maxBattLant,
            avg(meanBattLant) AS meanBattLant,
            last_BattLant,
            off_pos_OKNG,
            last_off_pos,
            LDR_OKNG,
            last_LDR,
            light_OKNG,
            last_light,
            racon_OKNG,
            last_racon,
            health_OKNG,
            last_health,
            sum(cnt_msg6) AS cnt_msg6,
            sum(cnt_msg21) AS cnt_msg21, 
            longitude,
            latitude,
            lastseen
        FROM pnav.aton_summary
        WHERE month = '{month}'
        GROUP BY
            DATE_FORMAT(ts, '%Y-%m'), mmsi, name, region, aton_type, last_Temp, last_BattAton, last_BattLant, off_pos_OKNG, last_off_pos, LDR_OKNG, last_LDR, light_OKNG, last_light, racon_OKNG, last_racon, health_OKNG, last_health, longitude, latitude, lastseen
        ORDER BY
        month
    '''
    )

    list = []
    for i in result.result_rows:
        data = {
            'month': month,
            'mmsi': i[1],
            'name': i[2],
            'region': i[3],
            'aton_type': i[4],
            'opt6_second': i[5],
            'opt6_percent': i[6],
            'opt21_second': i[7],
            'opt21_percent': i[8],
            'minTemp': i[9],
            'maxTemp': i[10],
            'meanTemp': i[11],
            'last_Temp': i[12],
            'minBattAton': i[13],
            'maxBattAton': i[14],                            
            'meanBattAton': i[15],
            'last_BattAton': i[16],
            'minBattLant': i[17],
            'maxBattLant': i[18],
            'meanBattLant': i[19],
            'last_BattLant': i[20],
            'off_pos_OKNG': i[21],
            'last_off_pos': i[22],
            'LDR_OKNG': i[23],
            'last_LDR': i[24],
            'light_OKNG': i[25],
            'last_light': i[26],
            'racon_OKNG': i[27],
            'last_racon': i[28],
            'health_OKNG': i[29],
            'last_health': i[30],
            'cnt_msg6': i[31],
            'cnt_msg21': i[32],
            'longitude': i[33],
            'latitude': i[34],
            'lastseen': i[35].strftime("%Y-%m-%d %H:%M:%S") if i[35] != None else i[35]
        }

        list.append(data)

    return list