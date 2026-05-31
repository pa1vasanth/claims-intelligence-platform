"""
Fetches real Medicare claims data from CMS public APIs and stores in SQLite.
Teaches: API ingestion, data normalization, SQLite with SQLAlchemy.
"""
import requests
import sqlite3
import os
import json
from datetime import datetime
from app.config import CMS_APIS, DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS inpatient_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_ccn TEXT,
            provider_name TEXT,
            city TEXT,
            state TEXT,
            drg_code TEXT,
            drg_description TEXT,
            total_discharges INTEGER,
            avg_submitted_charge REAL,
            avg_total_payment REAL,
            avg_medicare_payment REAL,
            charge_to_payment_ratio REAL,
            ingested_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS physician_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            npi TEXT,
            provider_name TEXT,
            provider_type TEXT,
            city TEXT,
            state TEXT,
            hcpcs_code TEXT,
            hcpcs_description TEXT,
            total_beneficiaries INTEGER,
            total_services INTEGER,
            avg_submitted_charge REAL,
            avg_medicare_allowed REAL,
            avg_medicare_payment REAL,
            ingested_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS outpatient_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider_ccn TEXT,
            provider_name TEXT,
            city TEXT,
            state TEXT,
            apc_code TEXT,
            apc_description TEXT,
            beneficiary_count INTEGER,
            total_services INTEGER,
            avg_submitted_charge REAL,
            avg_medicare_allowed REAL,
            avg_medicare_payment REAL,
            outlier_services INTEGER,
            charge_to_allowed_ratio REAL,
            ingested_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS claim_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT,
            claim_type TEXT,
            claim_data TEXT,
            retrieved_policies TEXT,
            analysis TEXT,
            risk_score INTEGER,
            risk_label TEXT,
            analyzed_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def fetch_cms_data(api_url: str, size: int = 500, filters: dict = None) -> list:
    params = {"size": size, "offset": 0}
    if filters:
        params.update(filters)
    try:
        resp = requests.get(api_url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[CMS API Error] {api_url}: {e}")
        return []


def ingest_outpatient(limit: int = 200) -> int:
    records = fetch_cms_data(CMS_APIS["outpatient"], size=limit)
    if not records:
        return 0

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM outpatient_claims")

    count = 0
    now = datetime.utcnow().isoformat()
    for r in records:
        try:
            avg_charge = float(r.get("Avg_Tot_Sbmtd_Chrgs", 0) or 0)
            avg_allowed = float(r.get("Avg_Mdcr_Alowd_Amt", 0) or 0)
            ratio = round(avg_charge / avg_allowed, 2) if avg_allowed > 0 else 0

            c.execute("""
                INSERT INTO outpatient_claims
                (provider_ccn, provider_name, city, state, apc_code, apc_description,
                 beneficiary_count, total_services, avg_submitted_charge,
                 avg_medicare_allowed, avg_medicare_payment, outlier_services,
                 charge_to_allowed_ratio, ingested_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.get("Rndrng_Prvdr_CCN", ""),
                r.get("Rndrng_Prvdr_Org_Name", ""),
                r.get("Rndrng_Prvdr_City", ""),
                r.get("Rndrng_Prvdr_State_Abrvtn", ""),
                r.get("APC_Cd", ""),
                r.get("APC_Desc", ""),
                int(float(r.get("Bene_Cnt", 0) or 0)),
                int(float(r.get("CAPC_Srvcs", 0) or 0)),
                avg_charge,
                avg_allowed,
                float(r.get("Avg_Mdcr_Pymt_Amt", 0) or 0),
                int(float(r.get("Outlier_Srvcs", 0) or 0)),
                ratio,
                now,
            ))
            count += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    return count


def get_outpatient_claims(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM outpatient_claims
        ORDER BY charge_to_allowed_ratio DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def ingest_inpatient(limit: int = 200) -> int:
    records = fetch_cms_data(CMS_APIS["inpatient"], size=limit)
    if not records:
        return 0

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM inpatient_claims")  # refresh on each ingestion

    count = 0
    now = datetime.utcnow().isoformat()
    for r in records:
        try:
            avg_charge = float(r.get("Avg_Submtd_Cvrd_Chrg", 0) or 0)
            avg_payment = float(r.get("Avg_Tot_Pymt_Amt", 0) or 0)
            ratio = round(avg_charge / avg_payment, 2) if avg_payment > 0 else 0

            c.execute("""
                INSERT INTO inpatient_claims
                (provider_ccn, provider_name, city, state, drg_code, drg_description,
                 total_discharges, avg_submitted_charge, avg_total_payment,
                 avg_medicare_payment, charge_to_payment_ratio, ingested_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.get("Rndrng_Prvdr_CCN", ""),
                r.get("Rndrng_Prvdr_Org_Name", ""),
                r.get("Rndrng_Prvdr_City", ""),
                r.get("Rndrng_Prvdr_State_Abrvtn", ""),
                r.get("DRG_Cd", ""),
                r.get("DRG_Desc", ""),
                int(float(r.get("Tot_Dschrgs", 0) or 0)),
                avg_charge,
                avg_payment,
                float(r.get("Avg_Mdcr_Pymt_Amt", 0) or 0),
                ratio,
                now,
            ))
            count += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    return count


def ingest_physician(limit: int = 200) -> int:
    records = fetch_cms_data(CMS_APIS["physician"], size=limit)
    if not records:
        return 0

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM physician_claims")

    count = 0
    now = datetime.utcnow().isoformat()
    for r in records:
        try:
            first = r.get("Rndrng_Prvdr_First_Name", "")
            last = r.get("Rndrng_Prvdr_Last_Org_Name", "")
            name = f"{first} {last}".strip() if first else last

            c.execute("""
                INSERT INTO physician_claims
                (npi, provider_name, provider_type, city, state,
                 hcpcs_code, hcpcs_description, total_beneficiaries,
                 total_services, avg_submitted_charge, avg_medicare_allowed,
                 avg_medicare_payment, ingested_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.get("Rndrng_NPI", ""),
                name,
                r.get("Rndrng_Prvdr_Type", ""),
                r.get("Rndrng_Prvdr_City", ""),
                r.get("Rndrng_Prvdr_State_Abrvtn", ""),
                r.get("HCPCS_Cd", ""),
                r.get("HCPCS_Desc", ""),
                int(float(r.get("Tot_Benes", 0) or 0)),
                int(float(r.get("Tot_Srvcs", 0) or 0)),
                float(r.get("Avg_Sbmtd_Chrg", 0) or 0),
                float(r.get("Avg_Mdcr_Alowd_Amt", 0) or 0),
                float(r.get("Avg_Mdcr_Pymt_Amt", 0) or 0),
                now,
            ))
            count += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    return count


def get_inpatient_claims(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM inpatient_claims
        ORDER BY charge_to_payment_ratio DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_physician_claims(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM physician_claims
        ORDER BY avg_submitted_charge DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    stats = {}

    c.execute("SELECT COUNT(*), AVG(avg_submitted_charge), AVG(avg_medicare_payment) FROM inpatient_claims")
    row = c.fetchone()
    stats["inpatient_count"] = row[0] or 0
    stats["inpatient_avg_charge"] = round(row[1] or 0, 2)
    stats["inpatient_avg_payment"] = round(row[2] or 0, 2)

    c.execute("SELECT COUNT(*), AVG(avg_submitted_charge), AVG(avg_medicare_payment) FROM physician_claims")
    row = c.fetchone()
    stats["physician_count"] = row[0] or 0
    stats["physician_avg_charge"] = round(row[1] or 0, 2)
    stats["physician_avg_payment"] = round(row[2] or 0, 2)

    c.execute("SELECT COUNT(*), AVG(avg_submitted_charge), AVG(avg_medicare_payment) FROM outpatient_claims")
    row = c.fetchone()
    stats["outpatient_count"] = row[0] or 0
    stats["outpatient_avg_charge"] = round(row[1] or 0, 2)
    stats["outpatient_avg_payment"] = round(row[2] or 0, 2)

    c.execute("SELECT COUNT(*) FROM claim_analyses")
    stats["analyses_count"] = c.fetchone()[0] or 0

    conn.close()
    return stats


def save_analysis(claim_id: str, claim_type: str, claim_data: dict,
                  retrieved_policies: list, analysis: str,
                  risk_score: int, risk_label: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO claim_analyses
        (claim_id, claim_type, claim_data, retrieved_policies, analysis,
         risk_score, risk_label, analyzed_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        claim_id,
        claim_type,
        json.dumps(claim_data),
        json.dumps(retrieved_policies),
        analysis,
        risk_score,
        risk_label,
        datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()


def get_recent_analyses(limit: int = 20) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM claim_analyses
        ORDER BY analyzed_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
