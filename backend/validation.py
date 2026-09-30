from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from main import profile_frame


def validate(path: str) -> dict:
    source = Path(path)
    content = source.read_bytes()
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, sep=None, engine="python")
    else:
        frame = pd.read_excel(source)
    result = profile_frame(frame, source.name, len(content))
    semantic = {column["name"]: column["semantic_type"] for column in result["column_profiles"]}
    issue_types = {issue["issue"] for issue in result["issues"]}
    failures: list[str] = []
    if not all(insight.get("calculation_id") for insight in result["insights"] if insight.get("impact")):
        failures.append("quantitative insight missing calculation_id")
    for column in semantic:
        if semantic[column] == "identifier" and any(column in insight.get("columns", "") for insight in result["insights"]):
            failures.append(f"identifier used in insight: {column}")
    report = {
        "dataset": result["name"],
        "rows": result["rows"],
        "columns": result["columns"],
        "quality_score": result["quality_score"],
        "quality_breakdown": result["quality_breakdown"],
        "issues_detected": sorted(issue_types),
        "semantic_types": semantic,
        "insights": result["insights"],
        "trend_results": result["charts"]["monthly_trend"],
        "driver_results": [],
        "anomalies": [issue for issue in result["issues"] if "outlier" in issue["issue"].lower()],
        "failed_assertions": failures,
    }
    Path("analysis_validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("AUTONOMOUS ANALYST VALIDATION")
    print(f"{'PASS' if not failures else 'FAIL'} Dataset rows: {result['rows']}")
    print(f"{'PASS' if semantic else 'FAIL'} Semantic types detected: {len(semantic)}")
    print(f"{'PASS' if result['quality_breakdown'] else 'FAIL'} Quality breakdown present")
    print(f"{'PASS' if result['insights'] else 'FAIL'} Computed insights: {len(result['insights'])}")
    print(f"{'PASS' if not failures else 'FAIL'} Provenance validation")
    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: PYTHONPATH=backend .venv/bin/python backend/validation.py path/to/dataset.csv")
    validate(sys.argv[1])
