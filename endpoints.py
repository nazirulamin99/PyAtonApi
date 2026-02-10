from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import pandas as pd
import logging
import json
import duckdb
import numpy as np
import ollama
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

from llm_search import nlp_llm_summary, nlp_llm_analytics
from data_loader import get_all_aton, getATON_summary, getAtonVoltByMMSI, getAtonHeartbeatByMMSI, getAtonByMonth


# -------------------------------------------
# Response Schemas
# -------------------------------------------

class AtonAllItem(BaseModel):
    ts: Optional[str] = None
    mmsi: Optional[str] = None
    name: Optional[str] = None
    region: Optional[str] = None
    aton_type: Optional[str] = None
    minBattAton: Optional[float] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    lastseen: Optional[str] = None
    health_OKNG: Optional[int] = None
    aton_no: Optional[str] = None
    condition: Optional[str] = None
    color: Optional[List[int]] = None
    volt_aton: Optional[float] = None

class AtonAllResponse(BaseModel):
    title: str
    count: int
    data: List[AtonAllItem]


class AtonSummaryItem(BaseModel):
    ts: Optional[str] = None
    mmsi: Optional[str] = None
    name: Optional[str] = None
    region: Optional[str] = None
    aton_type: Optional[str] = None
    opt6_second: Optional[float] = None
    opt6_percent: Optional[float] = None
    opt21_second: Optional[float] = None
    opt21_percent: Optional[float] = None
    minTemp: Optional[float] = None
    maxTemp: Optional[float] = None
    meanTemp: Optional[float] = None
    last_Temp: Optional[float] = None
    minBattAton: Optional[float] = None
    maxBattAton: Optional[float] = None
    meanBattAton: Optional[float] = None
    last_BattAton: Optional[float] = None
    minBattLant: Optional[float] = None
    maxBattLant: Optional[float] = None
    meanBattLant: Optional[float] = None
    last_BattLant: Optional[float] = None
    off_pos_OKNG: Optional[int] = None
    last_off_pos: Optional[int] = None
    LDR_OKNG: Optional[int] = None
    last_LDR: Optional[int] = None
    light_OKNG: Optional[int] = None
    last_light: Optional[int] = None
    racon_OKNG: Optional[int] = None
    last_racon: Optional[int] = None
    health_OKNG: Optional[int] = None
    last_health: Optional[int] = None
    cnt_msg6: Optional[int] = None
    cnt_msg21: Optional[int] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    lastseen: Optional[str] = None

class AtonSummaryResponse(BaseModel):
    title: str
    count: int
    data: List[AtonSummaryItem]


class AtonVoltItem(BaseModel):
    ts: Optional[str] = None
    volt_int: Optional[float] = None
    volt_ex1: Optional[float] = None
    aton_name: Optional[str] = None
    mmsi: Optional[str] = None

class AtonVoltResponse(BaseModel):
    title: str
    count: int
    data: List[AtonVoltItem]


class AtonHeartbeatItem(BaseModel):
    ts: Optional[str] = None
    beat: Optional[int] = None
    Aton_name: Optional[str] = Field(None, alias="Aton name")
    mmsi: Optional[str] = None

class AtonHeartbeatResponse(BaseModel):
    title: str
    count: int
    data: List[AtonHeartbeatItem]


class AtonByMonthItem(BaseModel):
    month: Optional[str] = None
    mmsi: Optional[str] = None
    name: Optional[str] = None
    region: Optional[str] = None
    aton_type: Optional[str] = None
    opt6_second: Optional[float] = None
    opt6_percent: Optional[float] = None
    opt21_second: Optional[float] = None
    opt21_percent: Optional[float] = None
    minTemp: Optional[float] = None
    maxTemp: Optional[float] = None
    meanTemp: Optional[float] = None
    last_Temp: Optional[float] = None
    minBattAton: Optional[float] = None
    maxBattAton: Optional[float] = None
    meanBattAton: Optional[float] = None
    last_BattAton: Optional[float] = None
    minBattLant: Optional[float] = None
    maxBattLant: Optional[float] = None
    meanBattLant: Optional[float] = None
    last_BattLant: Optional[float] = None
    off_pos_OKNG: Optional[int] = None
    last_off_pos: Optional[int] = None
    LDR_OKNG: Optional[int] = None
    last_LDR: Optional[int] = None
    light_OKNG: Optional[int] = None
    last_light: Optional[int] = None
    racon_OKNG: Optional[int] = None
    last_racon: Optional[int] = None
    health_OKNG: Optional[int] = None
    last_health: Optional[int] = None
    cnt_msg6: Optional[int] = None
    cnt_msg21: Optional[int] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    lastseen: Optional[str] = None

class AtonByMonthResponse(BaseModel):
    title: str
    count: int
    data: List[AtonByMonthItem]


class LLMInteractiveResponse(BaseModel):
    success: bool
    title: Optional[str] = None
    sql: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    analysis: Optional[str] = None
    error: Optional[str] = None

class LLMQueryAnalyticsResponse(BaseModel):
    success: bool
    title: Optional[str] = None
    sql: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    count: Optional[int] = None


def fix_month_range_in_sql(sql: str) -> str:
    """
    Fix incorrect month ranges in SQL where LLM generates end date more than 1 month after start.
    Example: >= '2025-11-01' AND < '2026-01-01' should be >= '2025-11-01' AND < '2025-12-01'
    """
    # Pattern to match date ranges with first-of-month dates
    pattern = r">=\s*(?:TIMESTAMP\s*)?'(\d{4}-\d{2}-01)[^']*'\s*AND\s*[^<]*<\s*(?:TIMESTAMP\s*)?'(\d{4}-\d{2}-01)[^']*'"

    match = re.search(pattern, sql, re.IGNORECASE)
    if match:
        start_date_str = match.group(1)
        end_date_str = match.group(2)

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            # Calculate expected end date (1 month after start)
            expected_end = start_date + relativedelta(months=1)

            # If the end date is more than 1 month after start, fix it
            if end_date > expected_end:
                correct_end_str = expected_end.strftime('%Y-%m-%d')
                sql = sql.replace(end_date_str, correct_end_str)
        except ValueError:
            pass  # If date parsing fails, leave SQL unchanged

    return sql



# -------------------------------------------
# Configuration
# -------------------------------------------
app = FastAPI(title="AtoN Data API", version="1.0")
logger = logging.getLogger(__name__)


# -------------------------------------------
# API Endpoint
# -------------------------------------------
@app.get("/aton/all", summary="Get all AtoN data", tags=["📦 Core Endpoints"], response_model=AtonAllResponse)
async def get_all_aton_api():
    df = get_all_aton()
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No AtoN data available")

    # Convert datetime columns to string
    for col in df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
        df[col] = df[col].astype(str)

    data_json = df.to_dict(orient="records")

    return JSONResponse(
        content={
            "title": "AtoN Summary Data",
            "count": len(data_json),
            "data": data_json
        }
    )


@app.get("/aton/summary", summary="Get AtoN summary by MMSI or Date", tags=["📦 Core Endpoints"], response_model=AtonSummaryResponse)
async def aton_summary(
    mmsi: str = Query(None, description="MMSI number of the AtoN"),
    date: str = Query(None, description="Date (YYYY-MM-DD) to filter data")
):
    data = getATON_summary(mmsi=mmsi, date_value=date)

    if not data:
        raise HTTPException(status_code=404, detail="No data found")

    # Convert to JSON-safe format
    data_json = json.loads(json.dumps(data, default=str))

    return JSONResponse(
        content={
            "title": f"AtoN Summary for {mmsi or date or 'All'}",
            "count": len(data_json),
            "data": data_json
        }
    )

@app.get("/atonvolt_mmsi/{mmsi}", tags=["📦 Core Endpoints"], response_model=AtonVoltResponse)
def atonvolt_mmsi(mmsi: str, request: Request):
    if mmsi == '{mmsi}':
        return HTTPException(status_code=400, detail="Invalid format")
    else:
        result = getAtonVoltByMMSI(mmsi)
        referer = request.headers.get("referer", "")

        if "docs" in referer.lower():
            return JSONResponse(
                content={
                    "title": f"AtoN Voltage Data for MMSI {mmsi}",
                    "count": len(result),
                    "data": result[:10]
                }
            )

        return JSONResponse(
            content={
                "title": f"AtoN Voltage Data for MMSI {mmsi}",
                "count": len(result),
                "data": result
            }
        )


@app.get("/atonheartbeat_mmsi/{mmsi}", tags=["📦 Core Endpoints"], response_model=AtonHeartbeatResponse)
def atonheartbeat_mmsi(mmsi: str, request: Request):
    if mmsi == '{mmsi}':
        return HTTPException(status_code=400, detail="Invalid format")
    else:
        result = getAtonHeartbeatByMMSI(mmsi)
        referer = request.headers.get("referer", "")

        if "docs" in referer.lower():
            return JSONResponse(
                content={
                    "title": f"AtoN Heartbeat Data for MMSI {mmsi}",
                    "count": len(result),
                    "data": result[:10]
                }
            )

        return JSONResponse(
            content={
                "title": f"AtoN Heartbeat Data for MMSI {mmsi}",
                "count": len(result),
                "data": result
            }
        )


@app.get("/aton_by_month/{month}", tags=["📦 Core Endpoints"], response_model=AtonByMonthResponse)
def aton_by_month(month: str, request: Request):
    if month == '{month}':
        raise HTTPException(status_code=400, detail="Invalid format. Use YYYY-MM format (e.g., 2025-10)")

    try:
        result = getAtonByMonth(month)

        if not result:
            raise HTTPException(status_code=404, detail=f"No data found for month: {month}")

        # Clean data for JSON serialization (handle NaN, Infinity, Decimal)
        def clean_value(v):
            if v is None:
                return None
            if isinstance(v, float):
                if np.isnan(v) or np.isinf(v):
                    return None
            return v

        cleaned_result = [
            {k: clean_value(v) for k, v in row.items()}
            for row in result
        ]

        referer = request.headers.get("referer", "")

        # Limit to 10 rows for Swagger docs UI
        if "docs" in referer.lower():
            return JSONResponse(
                content={
                    "title": f"AtoN Data for {month}",
                    "count": len(cleaned_result),
                    "data": cleaned_result[:10]
                }
            )

        return JSONResponse(content={
            "title": f"AtoN Data for {month}",
            "count": len(cleaned_result),
            "data": cleaned_result
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching AtoN data for month {month}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
        


def clean_dataframe_for_json(df):
    """Replace NaN, inf, -inf with None for JSON serialization"""
    return df.replace([np.nan, float('inf'), -float('inf')], None)


def convert_clickhouse_to_duckdb(sql: str) -> str:
    # Replace table with parquet
    sql = sql.replace("pnav.aton_summary", "'aton_summary.parquet'")

    # Replace ClickHouse interval syntax with DuckDB
    sql = re.sub(r"ts \+ INTERVAL (\d+) HOUR", r"ts + INTERVAL '\1 hour'", sql)

    # Replace toDateTime('YYYY-MM-DD') with CAST('YYYY-MM-DD' AS TIMESTAMP)
    sql = re.sub(
        r"toDateTime\('([^']+)'\)",
        r"CAST('\1' AS TIMESTAMP)",
        sql
    )

    # Replace toDate('YYYY-MM-DD') with CAST('YYYY-MM-DD' AS DATE)
    sql = re.sub(
        r"toDate\('([^']+)'\)",
        r"CAST('\1' AS DATE)",
        sql
    )

    # ClickHouse CAST(mmsi AS TEXT) → DuckDB CAST(mmsi AS VARCHAR)
    sql = sql.replace("CAST(mmsi AS TEXT)", "CAST(mmsi AS VARCHAR)")

    # Ensure "ORDER BY ts" is valid
    sql = sql.replace("ORDER BY ts", "ORDER BY ts")

    return sql


@app.get("/llm_interactive", tags=["👨‍💻 LLM"], response_model=LLMInteractiveResponse)
def llm_summary(request: Request, query: str = ""):
    """Execute LLM SQL → ClickHouse → DuckDB → AI analysis → Convert ts to MYT."""

    # 1️⃣ Load data from ClickHouse via data_loader
    data = getATON_summary()
    if not data:
        return {"success": False, "error": "No AtoN summary data available from database"}

    df_parquet = pd.DataFrame(data)

    # Add computed 'condition' column (matches get_all_aton logic)
    def compute_condition(row):
        voltage = row.get('last_BattAton', 0) or 0
        health = row.get('health_OKNG', 0) or 0
        if voltage < 10:
            return "Low Battery"
        elif health == 1:
            return "Not Good"
        else:
            return "Good"

    df_parquet['condition'] = df_parquet.apply(compute_condition, axis=1)

    # -----------------------------------------
    # 2️⃣ DEFAULT: NO QUERY PROVIDED
    # -----------------------------------------
    if not query.strip():

        # JSON safe
        df_clean = df_parquet.astype(object).where(pd.notnull(df_parquet), None)

        rows = df_clean.to_dict(orient="records")

        # Limit to 10 rows ONLY for FastAPI Swagger UI
        if "mozilla" in request.headers.get("user-agent", "").lower():
            rows = rows[:10]

        return {
            "success": True,
            "title": "📋 Default AtoN Summary",
            "rows": rows,
            "sql": None,
            "analysis": None
        }

    # -----------------------------------------
    # 3️⃣ QUERY PROVIDED → USE LLM
    # -----------------------------------------
    try:
        sql, title = nlp_llm_summary(query)

        # ---- SQL Cleanup ----
        # Ensure table name is correct for DuckDB (registered as "aton_summary")
        sql = sql.replace("pnav.aton_summary", "aton_summary")
        sql = sql.replace("pnav.", "")
        # Handle case where LLM uses just "aton_summary" - it's already correct for DuckDB
        # Also ensure no schema prefixes remain
        sql = re.sub(r'\bFROM\s+(?!aton_summary\b)(\w+\.)?aton_summary\b', 'FROM aton_summary', sql, flags=re.IGNORECASE)

        # Fix column name: LLM may use "timestamp" or "datetime" but actual column is "ts"
        sql = re.sub(r'\btimestamp\b', 'ts', sql, flags=re.IGNORECASE)
        sql = re.sub(r'"timestamp"', '"ts"', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bdatetime\b', 'ts', sql, flags=re.IGNORECASE)
        sql = re.sub(r'"datetime"', '"ts"', sql, flags=re.IGNORECASE)

        # Fix to_timestamp() PostgreSQL function
        sql = re.sub(
            r'to_timestamp\s*\([^,]+,\s*["\'][^"\']+["\']\s*\)',
            'CAST(ts AS TIMESTAMP)',
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(r'to_timestamp\s*\(([^)]+)\)', r'CAST(\1 AS TIMESTAMP)', sql, flags=re.IGNORECASE)

        sql = sql.replace("toDate(ts)", "DATE(ts)")
        sql = sql.replace("toDateTime", "TIMESTAMP")
        sql = re.sub(r"TIMESTAMP\s*\('([^']+)'\)", r"TIMESTAMP '\1'", sql)
        sql = re.sub(
            r"\bts\s*\+\s*INTERVAL\s+8\s+HOUR\b",
            "CAST(ts AS TIMESTAMP) + INTERVAL 8 HOUR",
            sql
        )
        sql = sql.replace("≥", ">=").replace("≤", "<=")
        sql = sql.replace("CAST(mmsi AS TEXT)", "CAST(mmsi AS VARCHAR)")

        # Fix: Cast ts to TIMESTAMP when compared with TIMESTAMP expressions
        # Matches: ts >= TIMESTAMP '...' or ts < TIMESTAMP '...' etc.
        sql = re.sub(
            r"\bts\s*(>=|<=|>|<|=)\s*TIMESTAMP\s*'",
            r"CAST(ts AS TIMESTAMP) \1 TIMESTAMP '",
            sql
        )
        # Matches: ts >= (TIMESTAMP '...' + INTERVAL ...) or ts < (TIMESTAMP '...' + INTERVAL ...) etc.
        sql = re.sub(
            r"\bts\s*(>=|<=|>|<|=)\s*\(TIMESTAMP\s*'",
            r"CAST(ts AS TIMESTAMP) \1 (TIMESTAMP '",
            sql
        )

        # Fix incorrect month ranges (e.g., Nov-Jan should be Nov-Dec)
        sql = fix_month_range_in_sql(sql)

        # -----------------------------------------
        # 4️⃣ Run SQL with DuckDB
        # -----------------------------------------
        con = duckdb.connect()
        con.register("aton_summary", df_parquet)

        df_result = con.execute(sql).fetchdf()

        # ---- Convert ALL timestamps to MYT ----
        df_result["ts"] = (
            pd.to_datetime(df_result["ts"]) + pd.Timedelta(hours=8)
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        # ---- JSON clean ----
        df_result = df_result.astype(object).where(pd.notnull(df_result), None)

        rows = df_result.to_dict(orient="records")

        # Limit to 10 rows for FastAPI docs only
        if "mozilla" in request.headers.get("user-agent", "").lower():
            rows = rows[:10]

        # -----------------------------------------
        # 5️⃣ AI ANALYSIS (Ollama)
        # -----------------------------------------
        try:
            analysis_prompt = f"""
            You are an expert maritime analyst. Analyze the following AtoN data.

            User query: "{query}"

            Data sample (first 10 rows):
            {df_result.head(10).to_json(orient="records")}

            Give a short (1-3 sentence) insight.
            Avoid starting with: "The table shows" or "The chart indicates".
            """

            analysis_response = ollama.chat(
                model="llama3:latest",
                messages=[{"role": "user", "content": analysis_prompt}]
            )

            ai_analysis = analysis_response["message"]["content"].strip()

        except Exception as e:
            ai_analysis = f"❌ AI analysis error: {e}"

        # -----------------------------------------
        # 6️⃣ Final Output
        # -----------------------------------------
        return {
            "success": True,
            "title": f"📋 {title}",
            "sql": sql,
            "rows": rows,
            "analysis": ai_analysis
        }

    except Exception as e:
        return {"success": False, "error": str(e), "sql": sql if 'sql' in locals() else None}


@app.get("/llm_query_analytics", tags=["👨‍💻 LLM"], response_model=LLMQueryAnalyticsResponse)
def llm_query_analytics_local(
    request: Request,
    user_query: str = Query(..., description="Natural language query for AtoN Analytics (real-time data)")
):
    sql = None
    try:
        # -------------------------------------------------
        # 1️⃣ Load real-time data from ClickHouse via get_all_aton()
        # -------------------------------------------------
        df = get_all_aton()
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No AtoN data available from database.")

        # -------------------------------------------------
        # 2️⃣ Generate SQL from LLM using nlp_llm_analytics
        # -------------------------------------------------
        sql, title = nlp_llm_analytics(user_query)
        if not sql:
            raise HTTPException(status_code=400, detail="LLM did not return SQL.")

        logger.info(f"LLM Generated SQL: {sql}")

        # -------------------------------------------------
        # 3️⃣ Clean SQL for DuckDB execution
        # -------------------------------------------------
        # Ensure table name is "aton_data" as registered in DuckDB
        sql = re.sub(r'\bFROM\s+\w+\b', 'FROM aton_data', sql, count=1, flags=re.IGNORECASE)

        # Fix column name variations
        sql = re.sub(r'\btimestamp\b', 'ts', sql, flags=re.IGNORECASE)
        sql = re.sub(r'"timestamp"', '"ts"', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bdatetime\b', 'ts', sql, flags=re.IGNORECASE)

        # Fix unicode operators
        sql = sql.replace("≥", ">=").replace("≤", "<=")

        # Fix type casting
        sql = sql.replace("CAST(mmsi AS TEXT)", "CAST(mmsi AS VARCHAR)")

        logger.info(f"Cleaned SQL for DuckDB: {sql}")

        # -------------------------------------------------
        # 4️⃣ Execute SQL via DuckDB
        # -------------------------------------------------
        con = duckdb.connect()
        con.register("aton_data", df)

        result_df = con.execute(sql).fetchdf()

        if result_df.empty:
            raise HTTPException(status_code=404, detail="Query returned no results.")

        # -------------------------------------------------
        # 5️⃣ Timestamps are already in MYT (UTC+8) from get_all_aton()
        # -------------------------------------------------

        # -------------------------------------------------
        # 6️⃣ JSON-safe (remove NaN / Infinity / convert arrays / timestamps)
        # -------------------------------------------------
        # Drop non-serializable columns
        drop_cols = ['color', 'lastseen_dt']
        for col in drop_cols:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])

        # Convert datetime columns to string
        for col in result_df.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
            result_df[col] = result_df[col].astype(str)

        # Convert any remaining numpy arrays to lists
        for col in result_df.columns:
            if result_df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                result_df[col] = result_df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        result_df = result_df.astype(object).where(pd.notnull(result_df), None)

        # Convert to dict and ensure all values are JSON serializable
        records = json.loads(result_df.to_json(orient="records"))

        # Limit to 10 rows for Swagger UI only
        referer = request.headers.get("referer", "")
        is_swagger = "docs" in referer.lower() or "swagger" in referer.lower()
        if is_swagger:
            records = records[:10]

        # -------------------------------------------------
        # 7️⃣ Return final JSON
        # -------------------------------------------------
        return JSONResponse(
            content={
                "success": True,
                "title": f"📋 {title}",
                "sql": sql,
                "rows": records,
                "count": len(result_df)  # Total count before limiting
            }
        )

    except Exception as e:
        logger.exception("Error in /llm_query_analytics")
        raise HTTPException(status_code=500, detail=str(e))

# 1️⃣ Create a request model for POST
class LLMQueryRequest(BaseModel):
    query: str = ""

# 2️⃣ POST endpoint
# @app.post("/llm_interactive", tags=["👨‍💻 LLM"])
# def llm_summary(request: Request, body: LLMQueryRequest):
#     """Execute LLM SQL → ClickHouse → DuckDB → AI analysis → Convert ts to MYT."""

#     # Get query from request body
#     query = body.query

#     # 1️⃣ Load data from ClickHouse via data_loader
#     data = getATON_summary()
#     if not data:
#         return {"success": False, "error": "No AtoN summary data available from database"}

#     df_parquet = pd.DataFrame(data)

#     # Add computed 'condition' column (matches get_all_aton logic)
#     def compute_condition(row):
#         voltage = row.get('last_BattAton', 0) or 0
#         health = row.get('health_OKNG', 0) or 0
#         if voltage < 10:
#             return "Low Battery"
#         elif health == 1:
#             return "Not Good"
#         else:
#             return "Good"

#     df_parquet['condition'] = df_parquet.apply(compute_condition, axis=1)

#     # -----------------------------------------
#     # 2️⃣ DEFAULT: NO QUERY PROVIDED
#     # -----------------------------------------
#     if not query.strip():

#         # JSON safe
#         df_clean = df_parquet.astype(object).where(pd.notnull(df_parquet), None)

#         rows = df_clean.to_dict(orient="records")

#         # Limit to 10 rows ONLY for FastAPI Swagger UI
#         if "mozilla" in request.headers.get("user-agent", "").lower():
#             rows = rows[:10]

#         return {
#             "success": True,
#             "title": "📋 Default AtoN Summary",
#             "rows": rows,
#             "sql": None,
#             "analysis": None
#         }

#     # -----------------------------------------
#     # 3️⃣ QUERY PROVIDED → USE LLM
#     # -----------------------------------------
#     try:
#         sql, title = nlp_llm_summary(query)

#         # ---- SQL Cleanup ----
#         sql = sql.replace("pnav.aton_summary", "aton_summary")
#         sql = sql.replace("pnav.", "")

#         # Fix column name: LLM may use "timestamp" or "datetime" but actual column is "ts"
#         sql = re.sub(r'\btimestamp\b', 'ts', sql, flags=re.IGNORECASE)
#         sql = re.sub(r'"timestamp"', '"ts"', sql, flags=re.IGNORECASE)
#         sql = re.sub(r'\bdatetime\b', 'ts', sql, flags=re.IGNORECASE)
#         sql = re.sub(r'"datetime"', '"ts"', sql, flags=re.IGNORECASE)

#         # Fix to_timestamp() PostgreSQL function
#         sql = re.sub(
#             r'to_timestamp\s*\([^,]+,\s*["\'][^"\']+["\']\s*\)',
#             'CAST(ts AS TIMESTAMP)',
#             sql,
#             flags=re.IGNORECASE
#         )
#         sql = re.sub(r'to_timestamp\s*\(([^)]+)\)', r'CAST(\1 AS TIMESTAMP)', sql, flags=re.IGNORECASE)

#         sql = sql.replace("toDate(ts)", "DATE(ts)")
#         sql = sql.replace("toDateTime", "TIMESTAMP")
#         sql = re.sub(r"TIMESTAMP\s*\('([^']+)'\)", r"TIMESTAMP '\1'", sql)
#         sql = re.sub(
#             r"\bts\s*\+\s*INTERVAL\s+8\s+HOUR\b",
#             "CAST(ts AS TIMESTAMP) + INTERVAL 8 HOUR",
#             sql
#         )
#         sql = sql.replace("≥", ">=").replace("≤", "<=")
#         sql = sql.replace("CAST(mmsi AS TEXT)", "CAST(mmsi AS VARCHAR)")

#         # -----------------------------------------
#         # 4️⃣ Run SQL with DuckDB
#         # -----------------------------------------
#         con = duckdb.connect()
#         con.register("aton_summary", df_parquet)

#         df_result = con.execute(sql).fetchdf()

#         # ---- Convert ALL timestamps to MYT ----
#         df_result["ts"] = (
#             pd.to_datetime(df_result["ts"]) + pd.Timedelta(hours=8)
#         ).dt.strftime("%Y-%m-%d %H:%M:%S")

#         # ---- JSON clean ----
#         df_result = df_result.astype(object).where(pd.notnull(df_result), None)

#         rows = df_result.to_dict(orient="records")

#         # Limit to 10 rows for FastAPI docs only
#         if "mozilla" in request.headers.get("user-agent", "").lower():
#             rows = rows[:10]

#         # -----------------------------------------
#         # 5️⃣ AI ANALYSIS (Ollama)
#         # -----------------------------------------
#         try:
#             analysis_prompt = f"""
#             You are an expert maritime analyst. Analyze the following AtoN data.

#             User query: "{query}"

#             Data sample (first 10 rows):
#             {df_result.head(10).to_json(orient="records")}

#             Give a short (1-3 sentence) insight.
#             Avoid starting with: "The table shows" or "The chart indicates".
#             """

#             analysis_response = ollama.chat(
#                 model="llama3:latest",
#                 messages=[{"role": "user", "content": analysis_prompt}]
#             )

#             ai_analysis = analysis_response["message"]["content"].strip()

#         except Exception as e:
#             ai_analysis = f"❌ AI analysis error: {e}"

#         # -----------------------------------------
#         # 6️⃣ Final Output
#         # -----------------------------------------
#         return {
#             "success": True,
#             "title": f"📋 {title}",
#             "sql": sql,
#             "rows": rows,
#             "analysis": ai_analysis
#         }

#     except Exception as e:
#         return {"success": False, "error": str(e), "sql": sql if 'sql' in locals() else None}
