# merge_kiosc.py
import pandas as pd
import numpy as np
import re
from io import BytesIO
from collections import Counter

# --- Helper functions ---
def make_unique(headers):
    counts = Counter()
    unique = []
    for h in headers:
        if counts[h]:
            unique.append(f"{h} ({counts[h]})")
        else:
            unique.append(h)
        counts[h] += 1
    return unique


def rename_columns(headers):
    option_map = {
        "Delivered - Onsite (face to face at KIOSC)": "Delivered - Onsite",
        "Offsite (face to face at your school by your teachers OR a KIOSC facilitator)": "Delivered - Offsite",
        "Online (delivered zia Zoom, Webex, Teams etc)": "Delivered - Online",
        "Immersion (delivered at an industry site)": "Delivered - Immersion",
        "What day/s did you attend? - Monday": "Attended Monday",
        "Tuesday": "Attended Tuesday",
        "Wednesday": "Attended Wednesday",
        "Thursday": "Attended Thursday",
        "Friday": "Attended Friday",
        "No": "If given the opportunity, would you like to attend another KIOSC program? - no",
        "Unsure": "If given the opportunity, would you like to attend another KIOSC program? - unsure",
        "Male": "Male",
        "Non-binary": "Non-binary",
        "Rather not say": "Rather not say",
        "Year 6": "Level - Year 6",
        "Year 7": "Level - Year 7",
        "Year 8": "Level - Year 8",
        "Year 9": "Level - Year 9",
        "Year 10": "Level - Year 10",
        "Year 11": "Level - Year 11",
        "Year 12": "Level - Year 12",
    }

    new = []
    for col in headers:
        if col in option_map:
            new.append(option_map[col])
            continue
        parts = re.split(r"\s*[–-]\s*", str(col), maxsplit=1)
        if len(parts) == 2:
            q, o = parts
            ql = q.lower()
            if "delivered" in ql:
                new.append(f"Delivered - {o}")
            elif "program did you attend" in ql:
                new.append(o)
            elif "school are you from" in ql:
                new.append(o)
            elif "gender" in ql:
                new.append(o)
            elif "year level are you" in ql:
                new.append(f"Level - {o}")
            elif "attend another kiosk program" in ql:
                new.append(f"{q} - {o.lower()}")
            else:
                new.append(f"{q} - {o}")
        else:
            new.append(col)
    return make_unique(new)


def preprocess_sparkchart(file, source_label):
    raw = pd.read_excel(file, sheet_name=0, header=None)
    q_row = raw.iloc[4]
    o_row = raw.iloc[5]
    combined = [
        f"{q} - {o}" if pd.notna(q) and pd.notna(o)
        else (q if pd.notna(q) else o)
        for q, o in zip(q_row, o_row)
    ]
    renamed = rename_columns(combined)
    df = raw.iloc[6:].reset_index(drop=True)
    df.columns = renamed
    df["Source"] = source_label
    return df


def add_derived_columns(df):
    out = df.copy()
    date_col = next((c for c in ["Survey End", "End Date", "Response Date"] if c in out.columns), None)
    if date_col:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out["response_date"] = dt.dt.date
        out["response_year"] = dt.dt.year.astype("Int64")

        def map_term(date):
            if pd.isna(date):
                return pd.NA
            m, d = date.month, date.day
            if (m == 1 and d >= 30) or (m in (2, 3)) or (m == 4 and d <= 10):
                return "Term 1"
            elif (m == 4 and d >= 20) or (m in (5, 6)) or (m == 7 and d <= 5):
                return "Term 2"
            elif (m == 7 and d >= 20) or (m in (8, 9)) or (m == 9 and d <= 20):
                return "Term 3"
            elif (m == 10 and d >= 5) or (m in (11, 12)):
                return "Term 4"
            return pd.NA

        out["Term"] = dt.apply(map_term)
    return out


def merge_survey_files(disc_file, vce_file):
    """Main callable merger — returns cleaned merged DataFrame."""
    df1 = preprocess_sparkchart(disc_file, "Discovery")
    df2 = preprocess_sparkchart(vce_file, "VCE")

    cols1 = list(df1.columns)
    cols2 = [c for c in df2.columns if c not in cols1]
    final_cols = cols1 + cols2
    df1 = df1.reindex(columns=final_cols)
    df2 = df2.reindex(columns=final_cols)
    merged = pd.concat([df1, df2], ignore_index=True)

    # Drop metadata
    merged.drop(columns=[
        'Last Name', 'Tags', 'Email', 'Organisation','Initial Mail',
        'Participant External ID','Other Comments','Other (1)','Other (2)',
    ], errors='ignore', inplace=True)

    # Delivery
    delivery_cols = [c for c in merged.columns if c.startswith("Delivered - ")]
    if delivery_cols:
        merged["Delivered"] = merged[delivery_cols].idxmax(axis=1).str.replace("Delivered - ", "")
        merged.drop(columns=delivery_cols, inplace=True, errors='ignore')

    # Gender
    gender_cols = ["Female","Male","Non-binary","Rather not say"]
    merged["Gender"] = merged[gender_cols].eq(1).idxmax(axis=1)
    merged.drop(columns=gender_cols, inplace=True, errors='ignore')

    # Grade level
    year_cols = [c for c in merged.columns if c.startswith("Level - Year")]
    merged["Grade Level"] = merged[year_cols].idxmax(axis=1).str.replace("Level - ", "")
    merged.drop(columns=year_cols, inplace=True, errors='ignore')

    # Attend another
    attend_cols = [c for c in merged.columns if c.startswith("If given the opportunity")]
    if attend_cols:
        merged["If given the opportunity, would you like to attend another KIOSC program?"] = (
            merged[attend_cols].eq(1)
            .idxmax(axis=1)
            .str.split(" - ")
            .str.get(1)
            .str.lower()
        )
        merged.drop(columns=attend_cols, inplace=True, errors='ignore')

    # Attendance
    attendance_cols = [c for c in merged.columns if c.startswith("Attended ")]
    if attendance_cols:
        merged["Attendance"] = merged[attendance_cols].eq(1).idxmax(axis=1).str.replace("Attended ", "")
        merged.drop(columns=attendance_cols, inplace=True, errors='ignore')

    # Program
    program_cols = [c for c in merged.columns if c.startswith(("Discovery:", "VCE", "Professional", "Work", "Internship"))]
    merged["Program"] = merged[program_cols].eq(1).idxmax(axis=1)
    merged.drop(columns=program_cols, inplace=True, errors='ignore')

    # School
    school_cols = [c for c in merged.columns if re.search(r"(school|college)", c, re.I)]
    merged["School"] = (
        merged[school_cols].eq(1)
        .idxmax(axis=1)
        .astype(str)
        .str.replace(r"\s*\(\d+\)$", "", regex=True)
    )
    merged.drop(columns=school_cols, inplace=True, errors='ignore')

    # Add derived date/term
    merged = add_derived_columns(merged)
    return merged
