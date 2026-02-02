# Enhanced llm_search.py with better context and error handling
# from openai import OpenAI
import json
import ollama
import numpy as np
import os
import streamlit as st
from typing import List, Dict, Tuple, Optional
import logging
# from langchain_openai import OpenAIEmbeddings
# # from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Initialize client with error handling
# try:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# except Exception as e:
#     logger.error(f"Failed to initialize OpenAI client: {e}")
#     client = None


def nlp_llm_analytics(user_query: str) -> Dict[str, str]:
    """Generate SQL query for get_all_aton() data analytics."""

    system_prompt = '''
        You are a professional in natural language processing (NLP) and SQL query generation.
        You are required to write standard SQL queries for maritime AtoN (Aids to Navigation) data analytics.
    '''

    user_prompt = f'''
        user prompt: "{user_query}"
    '''

    user_prompt += '''

    Understand the context from the user prompt above and then generate a complete SQL statement.

    DATABASE INFORMATION:
     - RECORDS: Real-time ATON data for maritime navigation aids (Lighthouse, Beacon, Buoy).
     - TABLENAME: aton_data (IMPORTANT: Always use "aton_data" as table name)
     - COLUMNS:
        1. ts (text): Timestamp of measurement in format "YYYY-MM-DD HH:MM:SS" (Malaysia Time UTC+8)
        2. mmsi (text): 9-digit AtoN MMSI identifier (VARCHAR type)
        3. name (text): ATON name/label in Title Case
        4. region (text): ATON location in Malaysia. Values: [Wilayah Tengah, Wilayah Selatan, Wilayah Utara, Wilayah Timur, Wilayah Labuan, Unknown]
        5. aton_type (text): ATON type. Values: [Beacon, Lighthouse, Buoy, Unknown]
        6. minBattAton (float): Battery voltage in Volts
        7. longitude (float): Geographic longitude coordinate
        8. latitude (float): Geographic latitude coordinate
        9. lastseen (text): Last seen timestamp in format "YYYY-MM-DD HH:MM:SS"
        10. health_OKNG (integer): Health status. 0 = OK/Good, 1 = Not Good
        11. aton_no (text): ATON reference number
        12. condition (text): Overall condition. Values: [Good, Low Battery, Not Good]
        13. volt_aton (float): Raw voltage reading

    COLUMN DESCRIPTIONS:
        - "low battery" or "low voltage" means minBattAton < 10.0 AND condition = 'Low Battery'
        - "not good" or "bad condition" means health_OKNG = 1 AND condition = 'Not Good'
        - "good" or "healthy" means health_OKNG = 0 AND condition = 'Good'
        - mmsi is VARCHAR type, use CAST(mmsi AS VARCHAR) and LIKE for partial matching
        - For name filtering, use UPPER(name) LIKE UPPER('%search%') for case-insensitive search

    SQL WRITING RULES:
        - SELECT * (all columns) unless specific columns requested
        - Use LIKE with '%' for partial text matching (e.g., mmsi LIKE '%1234%')
        - For date/time filtering on ts column, use: ts >= 'YYYY-MM-DD' AND ts < 'YYYY-MM-DD'
        - ORDER BY ts DESC for time-based queries
        - NO LIMIT clause unless specifically requested

    Return your answer only as a JSON object as below.
    NOT MORE THAN THAT, NO EXPLANATION AND DON'T MAKE UP ANYTHING.

    {
        "sql": "the complete SQL statement goes here",
        "query_type": "the type: SELECT, AGGREGATION, COUNT, etc.",
        "title": "a short descriptive title for the query result"
    }
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            format="json",
            options={"temperature": 0.1}
        )

        content = response['message']['content'].strip()

        # Remove markdown code blocks if present
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        sql = result.get("sql")
        title = result.get("title", "AtoN Analytics Result")

        if not sql:
            raise ValueError("Missing SQL in LLM output")

        return sql, title

    except Exception as e:
        raise Exception(f"❌ Error in nlp_llm_analytics: {e}")

MODEL = "llama3:latest"
from datetime import datetime, timedelta
import re

def fix_clickhouse_sql(sql: str) -> str:
    """Post-process SQL to fix common LLM mistakes for ClickHouse compatibility."""

    # Fix table name: aton_summary -> pnav.aton_summary
    # But don't double-fix if already correct
    sql = re.sub(r'\bFROM\s+aton_summary\b', 'FROM pnav.aton_summary', sql, flags=re.IGNORECASE)

    # Fix datetime syntax: CAST(ts AS TIMESTAMP) -> ts
    sql = re.sub(r'CAST\s*\(\s*ts\s+AS\s+TIMESTAMP\s*\)', 'ts', sql, flags=re.IGNORECASE)

    # Fix datetime syntax: TIMESTAMP 'YYYY-MM-DD HH:MM:SS' -> toDateTime('YYYY-MM-DD HH:MM:SS')
    sql = re.sub(
        r"TIMESTAMP\s+'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'",
        r"toDateTime('\1')",
        sql,
        flags=re.IGNORECASE
    )

    # Also handle TIMESTAMP 'YYYY-MM-DD' without time
    sql = re.sub(
        r"TIMESTAMP\s+'(\d{4}-\d{2}-\d{2})'",
        r"toDateTime('\1 00:00:00')",
        sql,
        flags=re.IGNORECASE
    )

    # Remove unnecessary LIKE '%%' filters
    sql = re.sub(r"\s+AND\s+\(?\s*\w+\s+LIKE\s+'%%'\s*(\s+OR\s+\w+\s+LIKE\s+'%%'\s*)*\)?", '', sql, flags=re.IGNORECASE)

    return sql

def nlp_llm_summary(user_query: str, date_override: Optional[str] = None) -> Dict[str, str]:
    # Calculate UTC date (Malaysia is UTC+8, so subtract 8 hours)
    if date_override:
        utc_date = date_override
    else:
        utc_date = (datetime.today() - timedelta(hours=8)).strftime('%Y-%m-%d')

    current_year = utc_date[:4]
    current_month = int(utc_date[5:7])

    system_prompt = '''
        You are an professional in ClickHouse Database management system. 
        You are required to write a standard SQL queries to produce a report or summary.
    '''

    user_prompt = f'''
        user prompt: "{user_query}"
    '''

    user_prompt += f'''

    You must understand the user prompt and then write a standard SQL queries.

    DATABASE INFORMATION:
     - RECORDS: Daily ATON summary data to generate report.
     - TABLENAME: pnav.aton_summary (IMPORTANT: Always use full table name "pnav.aton_summary" in your SQL, never just "aton_summary")
     - COLUMNS:
        1. ts (timestamp): Timestamp without time zone (daily summary date)
        2. mmsi (integer): 9-digit AtoN MMSI identifier. It is an identity number, not a name or label
        3. name (text): ATON name/label in UPPERCASE. Written in MALAY words. It is NOT mmsi
        4. region (text): ATON location in MALAYSIA. Values: [Wilayah Tengah, Wilayah Selatan, Wilayah Utara, Wilayah Timur, Wilayah Labuan, Labuan]
        5. aton_type (text): ATON type. Values: [Beacon, Lighthouse, Buoy]
        6. opt6_second (integer): Operational time in seconds for message type 6
        7. opt6_percent (double): Operational percentage for message type 6
        8. opt21_second (integer): Operational time in seconds for message type 21
        9. opt21_percent (double): Operational percentage for message type 21
        10. minTemp (double): Minimum temperature recorded (Celsius)
        11. maxTemp (double): Maximum temperature recorded (Celsius)
        12. meanTemp (double): Average temperature (Celsius)
        13. last_Temp (double): Last recorded temperature (Celsius)
        14. minBattAton (double): Minimum ATON battery voltage (Volts)
        15. maxBattAton (double): Maximum ATON battery voltage (Volts)
        16. meanBattAton (double): Average ATON battery voltage (Volts)
        17. last_BattAton (double): Last recorded ATON battery voltage (Volts)
        18. minBattLant (double): Minimum lantern battery voltage (Volts)
        19. maxBattLant (double): Maximum lantern battery voltage (Volts)
        20. meanBattLant (double): Average lantern battery voltage (Volts)
        21. last_BattLant (double): Last recorded lantern battery voltage (Volts)
        22. off_pos_OKNG (integer): Off-position status. 0 = OK, 1 = Not Good (off position detected)
        23. last_off_pos (integer): Last off-position value
        24. LDR_OKNG (integer): Light Dependent Resistor status. 0 = OK, 1 = Not Good
        25. last_LDR (integer): Last LDR sensor value
        26. light_OKNG (integer): Light status. 0 = OK, 1 = Not Good
        27. last_light (integer): Last light sensor value
        28. racon_OKNG (integer): Radar beacon (RACON) status. 0 = OK, 1 = Not Good
        29. last_racon (integer): Last RACON value
        30. health_OKNG (integer): Overall health status. 0 = OK/Good, 1 = Not Good
        31. last_health (integer): Last health value
        32. cnt_msg6 (integer): Count of message type 6 received
        33. cnt_msg21 (integer): Count of message type 21 received
        34. longitude (double): Geographic longitude coordinate
        35. latitude (double): Geographic latitude coordinate
        36. lastseen (timestamp): Last seen timestamp

    COLUMN DESCRIPTIONS:
        - "low battery" means minBattAton < 10.0 or last_BattAton < 10.0
        - "not good" or "bad health" means health_OKNG = 1
        - "good" or "healthy" means health_OKNG = 0
        - "off position" means off_pos_OKNG = 1
        - "light problem" means light_OKNG = 1
        - "racon problem" means racon_OKNG = 1
        - "high temperature" means maxTemp > 50 or last_Temp > 50
        - "low operational" means opt6_percent < 80 or opt21_percent < 80

    WRITTING STANDARD SQL QUERIES ROLES:
        - SELECT * (all data) unless specific columns are requested.
        - DATETIME filtering rules:
            * Today's date is {utc_date} (Year: {current_year}, Month: {current_month})
            * Apply '+ INTERVAL 8 HOUR' to 'ts' column for Malaysia timezone
            * For SINGLE MONTH queries (e.g., "september", "november", "january"):
                - If year is specified, use that year
                - If no year specified: month <= current month ({current_month}) use {current_year}, else use {int(current_year) - 1}
                - IMPORTANT: Range is ALWAYS from 1st of that month to 1st of the IMMEDIATELY NEXT month
                - The end date is ALWAYS the next month, NOT multiple months later
                - Examples:
                    * "january 2025": >= '2025-01-01' AND < '2025-02-01'
                    * "june 2025": >= '2025-06-01' AND < '2025-07-01'
                    * "september 2025": >= '2025-09-01' AND < '2025-10-01'
                    * "november 2025": >= '2025-11-01' AND < '2025-12-01'
                    * "december 2025": >= '2025-12-01' AND < '2026-01-01'
            * For relative dates ('last month', 'days ago'), use {utc_date} as reference
            * SQL syntax: (ts + INTERVAL 8 HOUR) >= toDateTime('YYYY-MM-DD 00:00:00') AND (ts + INTERVAL 8 HOUR) < toDateTime('YYYY-MM-DD 00:00:00')
        - MMSI filtering: Convert 'mmsi' column to text, use CAST(mmsi AS TEXT). Use SQL keyword 'LIKE' for partial matching ['end with', 'start with', 'contain'].
        - NAME filtering: Convert 'name' column to uppercase, use UPPER(name).
        - ORDER the data by ts DESC.
        - NO LIMIT

    DO NOT:
        - Use just "aton_summary" - ALWAYS use full table name "pnav.aton_summary"
        - Use CAST or TIMESTAMP keywords for datetime - use toDateTime() function
        - Add unnecessary filters like LIKE '%%' that match everything
        - Use PostgreSQL syntax - this is ClickHouse database

    EXAMPLE:
        User prompt: "show me lighthouse in september 2025"
        Correct SQL: SELECT * FROM pnav.aton_summary WHERE aton_type = 'Lighthouse' AND (ts + INTERVAL 8 HOUR) >= toDateTime('2025-09-01 00:00:00') AND (ts + INTERVAL 8 HOUR) < toDateTime('2025-10-01 00:00:00') ORDER BY ts DESC

        User prompt: "beacon with low battery in january 2025"
        Correct SQL: SELECT * FROM pnav.aton_summary WHERE aton_type = 'Beacon' AND minBattAton < 10.0 AND (ts + INTERVAL 8 HOUR) >= toDateTime('2025-01-01 00:00:00') AND (ts + INTERVAL 8 HOUR) < toDateTime('2025-02-01 00:00:00') ORDER BY ts DESC

        User prompt: "buoy in wilayah utara"
        Correct SQL: SELECT * FROM pnav.aton_summary WHERE aton_type = 'Buoy' AND region = 'Wilayah Utara' ORDER BY ts DESC
    '''

    user_prompt += '''
        Return your answer only json object as below. 
        NOT MORE THAN THAT, NO EXPLANATION AND DON'T MAKE UP ANYTHING.

        {
            "sql": the SQL statement
            "query_type": the type of the SQL statement for example: 'SELECT', 'AGGREGATION', 'UPDATE', 'DELETE'
            "title": the title for the statement
        }
    '''
    

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            format="json",  # Request JSON format from ollama
            options={
                "temperature": 0.1,  # Lower temperature for more consistent output
            }
        )
        
        content = response['message']['content'].strip()
        
        # Remove markdown code blocks if present
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()
        
        # Parse JSON response
        result = json.loads(content)

        sql = result.get("sql")
        title = result.get("title")

        if not sql:
            raise ValueError("Missing SQL in LLM output")

        # Post-process SQL to fix common LLM mistakes
        sql = fix_clickhouse_sql(sql)

        return sql, title
    
    except Exception as e:
        raise Exception(f"❌ Error in nlp_llm_summary: {e}")



# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def query_llm(user_query: str, aton_doc: List[str],
#               max_matches: int = 10, min_similarity: float = 0.1) -> str:
#     """
#     Run semantic search on AtoN index using FAISS and query LLM with enhanced context.
#     """
#     if not client:
#         return "❌ OpenAI service is not available. Please check your API key configuration."

#     # if not aton_index:
#     #     return "❌ No AtoN data available for search."

#     if not user_query.strip():
#         return "❓ Please provide a search query."

#     system_prompt = f'''
#     You are helpful assistant that only use provided list of data to answer the question.
#     '''

#     user_prompt = f'''
#     Use the following data to answer the question. Only answer based on the data provided.


#     Data:
#     {'----------\n\n\n'.join(aton_doc)}

#     Question:
#         {user_query.strip()}
#     '''


#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]

#     # Gemini approach
#     gemini_api_key = 'AIzaSyD2XCH9qPGwrT17n0egIyiM-Hu0HOK6FME'
#     gemini = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
#     model_name = "gemini-2.0-flash"

#     response = gemini.chat.completions.create(model=model_name, messages=messages)
#     answer = response.choices[0].message.content


#     # Ollama approach
#     # response = ollama.chat(model="llama3:latest", messages=messages)
#     # answer = response['message']['content']

    
#     # OpenAI approach
#     # response = client.chat.completions.create(
#     #     model="gpt-4o-mini",  # Updated model name
#     #     messages=messages,
#     #     temperature=0.7,  # Lower temperature for more consistent responses
#     #     # max_tokens=500
#     # )
#     # answer = response.choices[0].message.content

#     return answer

#     # except Exception as e:
#     #     logger.error(f"Error in LLM query: {e}")
#     #     return f"❌ Error while processing query: {str(e)}"

