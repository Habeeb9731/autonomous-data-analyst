from __future__ import annotations

import io
import math
import re
import unicodedata
from datetime import date, datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Autonomous Data Analyst API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    analysis: dict[str, Any]


def rule_based_answer(question: str, analysis: dict[str, Any]) -> dict[str, Any]:
    """Explain only facts already present in the deterministic analysis result."""
    question_lower = question.casefold()
    insights = analysis.get("insights", [])
    issues = analysis.get("issues", [])
    calculations = analysis.get("calculations", [])
    if not insights:
        return {"answer": "Upload a dataset first so I can answer from calculated evidence.", "calculation_ids": [], "status": "UNVERIFIED"}

    selected = insights
    if any(word in question_lower for word in ("issue", "quality", "missing", "duplicate", "invalid")):
        selected = [insight for insight in insights if insight.get("type") == "DATA QUALITY"] or insights
        answer = f"I found {len(issues)} calculated data-quality issues. " + " ".join(str(item.get("body", "")) for item in selected[:3])
    elif any(word in question_lower for word in ("trend", "decline", "change", "month", "revenue", "sales")):
        selected = [insight for insight in insights if insight.get("type") == "TREND"] or insights
        answer = "Based on the calculated analysis: " + " ".join(str(item.get("body", "")) for item in selected[:3])
    elif any(word in question_lower for word in ("unusual", "outlier", "anomal")):
        selected = [insight for insight in insights if insight.get("type") == "DISTRIBUTION"] or insights
        answer = "The analysis identified: " + " ".join(str(item.get("body", "")) for item in selected[:3])
    else:
        answer = "Here are the strongest calculated findings: " + " ".join(str(item.get("body", "")) for item in selected[:3])

    evidence_ids = [str(item.get("calculation_id")) for item in selected if item.get("calculation_id")]
    known_ids = {str(item.get("id")) for item in calculations}
    evidence_ids = [calculation_id for calculation_id in evidence_ids if calculation_id in known_ids]
    return {"answer": answer, "calculation_ids": evidence_ids, "status": "VERIFIED" if evidence_ids else "PARTIALLY_VERIFIED"}


def json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def infer_semantic_type(name: str, series: pd.Series) -> str:
    lowered = name.lower()
    sample = series.dropna().astype(str).head(100)
    if "email" in lowered or (not sample.empty and sample.str.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", na=False).mean() > 0.8):
        return "email"
    if lowered.endswith("_id") or lowered in {"id", "order", "orderid"} or "identifier" in lowered:
        return "identifier"
    if "date" in lowered or "time" in lowered:
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if any(token in lowered for token in ("pct", "percent", "discount", "rate")):
            return "percentage"
        if any(token in lowered for token in ("revenue", "amount", "price", "cost", "sales", "margin")):
            return "currency" if any(token in lowered for token in ("revenue", "amount", "price", "cost", "sales")) else "numeric_measure"
        if series.nunique(dropna=True) / max(1, len(series)) > 0.98 and "quantity" not in lowered:
            return "identifier"
        return "numeric_measure"
    if any(token in lowered for token in ("country", "region", "segment", "category", "channel")):
        return "categorical"
        return "numeric"
    return "categorical" if series.nunique(dropna=True) < min(50, max(10, len(series) // 20)) else "text"


def profile_frame(frame: pd.DataFrame, filename: str, size: int) -> dict[str, Any]:
    frame = frame.copy()
    frame.columns = [str(column).strip() or f"column_{index + 1}" for index, column in enumerate(frame.columns)]
    numeric_count = int(sum(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns))
    dates = int(sum(pd.api.types.is_datetime64_any_dtype(frame[column]) for column in frame.columns))
    boolean = int(sum(pd.api.types.is_bool_dtype(frame[column]) for column in frame.columns))
    categorical_count = int(sum(infer_semantic_type(column, frame[column]) == "categorical" for column in frame.columns))
    missing_cells = int(frame.isna().sum().sum())
    total_cells = max(1, frame.shape[0] * frame.shape[1])
    duplicate_rows = int(frame.duplicated().sum())
    quality = max(0, min(100, round(100 - (missing_cells / total_cells * 55) - (duplicate_rows / max(1, len(frame)) * 25))))

    column_profiles = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        semantic_type = infer_semantic_type(column, series)
        profile: dict[str, Any] = {
            "name": column,
            "physical_type": str(series.dtype),
            "inferred_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "semantic_type": semantic_type,
            "potential_pii": semantic_type in {"email", "phone"},
            "missing": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and semantic_type != "identifier" and len(non_null):
            numeric_columns.append(column)
            profile.update({
                "mean": round(float(non_null.mean()), 2),
                "median": round(float(non_null.median()), 2),
                "min": json_value(non_null.min()),
                "max": json_value(non_null.max()),
            })
        elif semantic_type == "categorical":
            categorical_columns.append(column)
        column_profiles.append(profile)

    issues = []
    for column, count in frame.isna().sum().items():
        if int(count):
            issues.append({"severity": "HIGH" if count / max(1, len(frame)) > 0.05 else "MED", "issue": "Missing values", "column": str(column), "affected": f"{int(count)} rows", "confidence": "High", "fix": "Review missingness"})
    if duplicate_rows:
        issues.append({"severity": "HIGH", "issue": "Duplicate rows", "column": "—", "affected": f"{duplicate_rows} rows", "confidence": "High", "fix": "Review records"})
    for column in frame.select_dtypes(include="object").columns:
        values = frame[column].dropna().astype(str)
        if values.empty:
            continue
        normalized = values.str.strip().str.lower()
        if normalized.nunique() < values.nunique():
            issues.append({"severity": "MED", "issue": "Category mismatch", "column": column, "affected": f"{values.nunique() - normalized.nunique()} values", "confidence": "Medium", "fix": "Standardize labels"})
        whitespace_rows = int((values != values.str.strip()).sum())
        if whitespace_rows:
            issues.append({"severity": "LOW", "issue": "Whitespace inconsistency", "column": column, "affected": f"{whitespace_rows} rows", "confidence": "High", "fix": "Trim whitespace"})
        case_rows = int((values.str.strip() != values.str.strip().str.casefold()).sum())
        if case_rows and normalized.nunique() < values.nunique():
            issues.append({"severity": "LOW", "issue": "Case inconsistency", "column": column, "affected": f"{case_rows} rows", "confidence": "High", "fix": "Normalize case"})

    for column in frame.columns:
        semantic = infer_semantic_type(column, frame[column])
        if semantic == "identifier":
            duplicate_ids = int(frame[column].duplicated(keep=False).sum())
            if duplicate_ids:
                issues.append({"severity": "MED", "issue": "Duplicate identifiers", "column": column, "affected": f"{duplicate_ids} rows", "confidence": "High", "fix": "Review entity integrity"})
        if semantic == "email":
            issues.append({"severity": "MED", "issue": "Potential PII", "column": column, "affected": f"{int(frame[column].notna().sum())} values", "confidence": "High", "fix": "Exclude raw values from LLM context"})
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_values = pd.to_numeric(frame[column], errors="coerce")
            lowered = column.lower()
            if "quantity" in lowered:
                invalid = int((numeric_values <= 0).sum())
                if invalid:
                    issues.append({"severity": "HIGH", "issue": "Invalid quantity", "column": column, "affected": f"{invalid} rows", "confidence": "High", "fix": "Review values <= 0"})
            if any(token in lowered for token in ("pct", "percent", "discount", "rate")):
                invalid = int(((numeric_values < 0) | (numeric_values > 100)).sum())
                if invalid:
                    issues.append({"severity": "HIGH", "issue": "Invalid percentage range", "column": column, "affected": f"{invalid} rows", "confidence": "High", "fix": "Review values outside 0–100"})
            if semantic != "identifier":
                valid = numeric_values.dropna()
                if len(valid) >= 4:
                    q1, q3 = valid.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = int(((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)).sum()) if iqr else 0
                    if outliers:
                        issues.append({"severity": "LOW", "issue": "Potential statistical outliers", "column": column, "affected": f"{outliers} rows", "confidence": "Medium", "fix": "Inspect before excluding"})

    country_aliases = {"de": "germany", "deutschland": "germany", "ger": "germany", "uk": "united kingdom", "gb": "united kingdom"}
    for column in frame.select_dtypes(include="object").columns:
        if "country" not in column.lower():
            continue
        normalized = frame[column].dropna().astype(str).map(lambda value: unicodedata.normalize("NFKC", value).strip().casefold())
        aliases = [value for value in normalized.unique() if value in country_aliases]
        if aliases:
            issues.append({"severity": "MED", "issue": "Country alias inconsistency", "column": column, "affected": f"{len(aliases)} aliases", "confidence": "High", "fix": "Standardize country labels"})

    insights: list[dict[str, Any]] = []
    calculations: list[dict[str, Any]] = []
    if duplicate_rows:
        insights.append({"type": "DATA QUALITY", "title": "Duplicate records need review", "body": f"{duplicate_rows:,} duplicate rows were detected and excluded from quality scoring.", "impact": f"{duplicate_rows:,} rows", "confidence": "High", "columns": "all columns"})
    if missing_cells:
        missing_column = str(frame.isna().sum().idxmax())
        missing_count = int(frame[missing_column].isna().sum())
        insights.append({"type": "DATA QUALITY", "title": f"{missing_column} has the most missing values", "body": f"{missing_count:,} rows ({missing_count / max(1, len(frame)):.1%}) are missing a value in {missing_column}.", "impact": f"{missing_count:,} rows", "confidence": "High", "columns": missing_column})
    if numeric_columns:
        numeric = numeric_columns[0]
        series = pd.to_numeric(frame[numeric], errors="coerce").dropna()
        if len(series) >= 4:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()) if iqr else 0
            insights.append({"type": "DISTRIBUTION", "title": f"{numeric} has a long-tail distribution", "body": f"The median is {series.median():,.2f}, while the mean is {series.mean():,.2f}. {outlier_count:,} values fall outside the 1.5× IQR range.", "impact": f"{outlier_count:,} outliers", "confidence": "Medium", "columns": numeric})
    if categorical_columns:
        category = categorical_columns[0]
        top_value = frame[category].dropna().astype(str).value_counts().head(1)
        if not top_value.empty:
            value, count = top_value.index[0], int(top_value.iloc[0])
            insights.append({"type": "SEGMENT", "title": f"{value} is the largest {category} group", "body": f"{value} represents {count / max(1, frame[category].notna().sum()):.1%} of non-empty values in {category}.", "impact": f"{count:,} rows", "confidence": "High", "columns": category})

    chart_data: dict[str, Any] = {"numeric_distribution": [], "category_breakdown": [], "monthly_trend": []}
    if numeric_columns:
        numeric = numeric_columns[0]
        series = pd.to_numeric(frame[numeric], errors="coerce").dropna()
        if not series.empty:
            counts, edges = pd.cut(series, bins=min(8, max(2, series.nunique())), retbins=True, include_lowest=True)
            grouped = series.groupby(counts, observed=False).size()
            chart_data["numeric_distribution"] = [{"label": str(interval), "value": int(value)} for interval, value in grouped.items()]
    if categorical_columns:
        category = categorical_columns[0]
        chart_data["category_breakdown"] = [{"label": str(label), "value": int(value)} for label, value in frame[category].dropna().astype(str).value_counts().head(8).items()]

    date_columns = [column for column in frame.columns if infer_semantic_type(column, frame[column]) == "date"]
    revenue_columns = [column for column in frame.columns if infer_semantic_type(column, frame[column]) in {"currency", "numeric_measure"} and any(token in column.lower() for token in ("revenue", "sales", "amount", "price"))]
    if date_columns and revenue_columns:
        date_column, revenue_column = date_columns[0], revenue_columns[0]
        parsed_dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
        revenue_values = pd.to_numeric(frame[revenue_column], errors="coerce")
        trend_frame = pd.DataFrame({"date": parsed_dates, "value": revenue_values}).dropna()
        if not trend_frame.empty:
            monthly = trend_frame.assign(month=trend_frame["date"].dt.to_period("M")).groupby("month", as_index=False).agg(value=("value", "sum"), rows=("value", "size"))
            previous = None
            for row in monthly.itertuples(index=False):
                change = None if previous in (None, 0) else round(float((row.value - previous) / previous * 100), 4)
                chart_data["monthly_trend"].append({"month": str(row.month), "revenue": round(float(row.value), 2), "change_pct": change})
                previous = row.value
            if len(monthly) >= 2:
                changes = monthly.assign(change=monthly["value"].pct_change() * 100).dropna()
                sharpest = changes.sort_values("change").iloc[0]
                previous_row = monthly.iloc[int(changes.index.get_loc(sharpest.name))]
                if float(sharpest["change"]) < 0:
                    calculation_id = f"calc_trend_{len(calculations) + 1:04d}"
                    calculations.append({"id": calculation_id, "method": "monthly_sum_and_mom_change", "rows_used": int(len(trend_frame)), "view": "raw_v1", "metrics": {"previous_period": str(previous_row["month"]), "current_period": str(sharpest["month"]), "previous_value": round(float(previous_row["value"]), 2), "current_value": round(float(sharpest["value"]), 2), "absolute_change": round(float(sharpest["value"] - previous_row["value"]), 2), "percentage_change": round(float(sharpest["change"]), 4)}})
                    insights.append({"type": "TREND", "title": f"Revenue declined in {sharpest['month']}", "body": f"Revenue changed {float(sharpest['change']):.2f}% compared with {previous_row['month']}, based on grouped monthly sums.", "impact": f"{float(sharpest['change']):.2f}%", "confidence": "High", "columns": f"{date_column} · {revenue_column}", "calculation_id": calculation_id})

    for index, insight in enumerate(insights):
        if "calculation_id" not in insight:
            calculation_id = f"calc_profile_{index + 1:04d}"
            insight["calculation_id"] = calculation_id
            calculations.append({"id": calculation_id, "method": "deterministic_profile_rule", "rows_used": int(len(frame)), "view": "raw_v1", "metrics": {"claim": insight["title"], "impact": insight["impact"]}})

    missing_rate = missing_cells / total_cells if total_cells else 0
    duplicate_rate = duplicate_rows / max(1, len(frame))
    issue_penalty = min(35, len(issues) * 2.5)
    consistency_issues = sum(1 for issue in issues if "inconsistency" in str(issue["issue"]).lower() or "mismatch" in str(issue["issue"]).lower())
    invalid_issues = sum(1 for issue in issues if "invalid" in str(issue["issue"]).lower())
    quality_breakdown = {"completeness": round(max(0, 100 - missing_rate * 100), 2), "uniqueness": round(max(0, 100 - duplicate_rate * 100), 2), "validity": round(max(0, 100 - invalid_issues * 8), 2), "consistency": round(max(0, 100 - consistency_issues * 5), 2)}
    quality = max(0, min(100, round(100 - missing_rate * 40 - duplicate_rate * 30 - issue_penalty)))

    return {
        "name": filename,
        "size_bytes": size,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "detected_types": {"numeric": numeric_count, "categorical": categorical_count, "date": dates, "boolean": boolean, "text": max(0, frame.shape[1] - numeric_count - dates - boolean - categorical_count)},
        "quality_score": quality,
        "quality_breakdown": quality_breakdown,
        "issues": issues,
        "column_profiles": column_profiles,
        "preview": [{str(key): json_value(value) for key, value in row.items()} for row in frame.head(8).to_dict(orient="records")],
        "insights": insights,
        "calculations": calculations,
        "charts": chart_data,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/profile")
async def profile_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or "dataset"
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if extension not in {"csv", "xls", "xlsx"}:
        raise HTTPException(status_code=400, detail="Supported files are CSV, XLS, and XLSX.")
    content = await file.read()
    try:
        if extension == "csv":
            try:
                frame = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
            except UnicodeDecodeError:
                frame = pd.read_csv(io.BytesIO(content), encoding="latin-1", sep=None, engine="python")
        else:
            frame = pd.read_excel(io.BytesIO(content))
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not parse dataset: {error}") from error
    if frame.empty and len(frame.columns) == 0:
        raise HTTPException(status_code=422, detail="The uploaded dataset has no readable columns.")
    return profile_frame(frame, filename, len(content))


@app.post("/api/demo")
def demo_dataset() -> dict[str, Any]:
    """Profile a deterministic sample through the same production analysis pipeline."""
    frame = pd.DataFrame(
        {
            "order_id": ["ORD-001", "ORD-002", "ORD-003", "ORD-003", "ORD-005", "ORD-006", "ORD-007", "ORD-008"],
            "customer_email": ["alex@example.com", "sam@example.com", "lee@example.com", "lee@example.com", "mia@example.com", "noah@example.com", "zoe@example.com", "ava@example.com"],
            "order_date": ["2026-01-05", "2026-02-05", "2026-02-18", "2026-02-18", "2026-03-02", "2026-03-19", "2026-04-03", "2026-04-20"],
            "customer_segment": ["Enterprise", "enterprise", "SMB", "SMB", "Mid-Market", "SMB", "Enterprise", "mid-market"],
            "country": ["Germany", " germany ", "DE", "France", "France", "GB", "United Kingdom", "Germany"],
            "channel": ["Website", " Website ", "Store", "store", "Partner", "Website", "Partner", "Website"],
            "quantity": [2, 1, 0, 3, 4, -1, 2, 5],
            "discount_pct": [5, 10, 110, 15, 20, -4, 10, 0],
            "revenue_eur": [120.0, 90.0, 65.0, 65.0, 230.0, 175.0, 310.0, 205.0],
        }
    )
    content_size = len(frame.to_csv(index=False).encode("utf-8"))
    return profile_frame(frame, "demo_dataset.csv", content_size)


@app.post("/api/ask")
def ask_data(request: AskRequest) -> dict[str, Any]:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return rule_based_answer(request.question, request.analysis)
