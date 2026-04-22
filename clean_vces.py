import re
from collections import Counter

import pandas as pd


def clean_vces(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file, skiprows=4, dtype=str)

    # Replace pandas "Unnamed: X" placeholders with empty string
    df.columns = ["" if "Unnamed" in str(c) else c for c in df.columns]

    # Row 0 is the sub-option row (e.g. school names, program names, Yes/No).
    # Merge it with the primary header to form combined column names.
    def make_col(h, s):
        h = str(h).strip() if pd.notna(h) and str(h).strip() not in ("", "nan") else ""
        s = str(s).strip() if pd.notna(s) and str(s).strip() not in ("", "nan") else ""
        if h and s:
            return f"{h}-{s}"
        return h or s

    df.columns = [make_col(df.columns[i], df.iloc[0, i]) for i in range(len(df.columns))]
    df = df.iloc[1:].reset_index(drop=True)

    # Drop blank / non-respondent rows
    df = df[df["First Name"].notna() & (df["First Name"].str.strip() != "")].reset_index(drop=True)

    # Deduplicate column names — "Other", "Other Comments", "No", "Yes" each appear
    # multiple times across question blocks, so append the column's positional index.
    raw_cols = df.columns.tolist()
    dup_set = {c for c, n in Counter(raw_cols).items() if n > 1}
    df.columns = [f"{c}_{i}" if c in dup_set else c for i, c in enumerate(raw_cols)]
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()

    # ── Block detection ────────────────────────────────────────────────────────
    # Each multi-select question spans a contiguous range of one-hot columns.
    # Locate each block by its first (anchor) column prefix.

    def block_idx(prefix: str, default: int = len(all_cols)) -> int:
        return next((i for i, c in enumerate(all_cols) if c.startswith(prefix)), default)

    prog_start   = block_idx("What program did you attend?")
    school_start = block_idx("Please Select Your School")
    year_start   = block_idx("What is your year level at school?")
    gender_start = block_idx("What is your gender?")

    # "We want to know…" is a section-header column that follows the gender block.
    gender_end = next(
        (i for i, c in enumerate(all_cols) if "We want to know" in c),
        gender_start + 5,   # fallback: 5 gender options
    )

    prog_cols   = all_cols[prog_start:school_start]
    school_cols = all_cols[school_start:year_start]
    year_cols   = all_cols[year_start:gender_start]
    gender_cols = all_cols[gender_start:gender_end]

    # ── One-hot helpers ────────────────────────────────────────────────────────

    def strip_q_prefix(name: str) -> str:
        for pfx in (
            "What program did you attend?-",
            "Please Select Your School-",
            "What is your year level at school?-",
            "What is your gender?-",
        ):
            if name.startswith(pfx):
                return name[len(pfx):]
        return name

    def is_other(c: str) -> bool:
        return c.strip() == "Other" or bool(re.match(r"^Other_\d+$", c.strip()))

    def is_other_comments(c: str) -> bool:
        return c.strip() == "Other Comments" or bool(re.match(r"^Other Comments_\d+$", c.strip()))

    def get_selected(row, block: list, use_comment: bool = True) -> str:
        """Return the display value of the first '1' column in the block.
        Falls back to the free-text Other Comments field when the Other
        checkbox is ticked and use_comment is True."""
        other_c    = next((c for c in block if is_other(c)),          None)
        comments_c = next((c for c in block if is_other_comments(c)), None)
        for c in block:
            if is_other(c) or is_other_comments(c):
                continue
            if str(row.get(c, "")).strip() == "1":
                return strip_q_prefix(c)
        if other_c and str(row.get(other_c, "")).strip() == "1":
            if use_comment and comments_c:
                comment = str(row.get(comments_c, "")).strip()
                if comment and comment.lower() != "nan":
                    return comment
            return "Other"
        return "Unknown"

    df["Program Name"] = df.apply(lambda r: get_selected(r, prog_cols),                     axis=1)
    df["School"]       = df.apply(lambda r: get_selected(r, school_cols),                   axis=1)
    df["Year Level"]   = df.apply(lambda r: get_selected(r, year_cols),                     axis=1)
    df["Gender"]       = df.apply(lambda r: get_selected(r, gender_cols, use_comment=False), axis=1)

    # ── School name normalisation ──────────────────────────────────────────────
    # Students sometimes type the school name freehand with misspellings.
    SCHOOL_ALIASES = {
        "Nossal high school":        "Nossal High School",
        "Nossal Highschool":         "Nossal High School",
        "Nossal HIGH scgool":        "Nossal High School",
        "Nossal High":               "Nossal High School",
        "Nossal high":               "Nossal High School",
        "Missal High School":        "Nossal High School",
        "Missal HS":                 "Nossal High School",
        "Highvale secondary college":"Highvale Secondary College",
    }
    df["School"] = df["School"].map(lambda s: SCHOOL_ALIASES.get(s, s))

    # ── Binary Yes / No questions ──────────────────────────────────────────────
    def find_yes_col(prefix: str):
        return next((c for c in all_cols if c.strip().startswith(prefix.strip())), None)

    attend_yes    = find_yes_col("Did you attend the activity with students")
    see_life_yes  = find_yes_col("I can see how what I learnt")
    recommend_yes = find_yes_col("I would recommend this activity to another student")

    def yes_no(row, yes_col: str) -> str:
        if yes_col and str(row.get(yes_col, "")).strip() == "1":
            return "Yes"
        return "No"

    df["Did you attend with classmates?"] = df.apply(
        lambda r: yes_no(r, attend_yes), axis=1
    )
    df["I can see how what I learnt in this activity can be used in real life"] = df.apply(
        lambda r: yes_no(r, see_life_yes), axis=1
    )
    df["I would recommend this activity to another student"] = df.apply(
        lambda r: yes_no(r, recommend_yes), axis=1
    )

    # ── Numeric ratings (scale 1–5) ────────────────────────────────────────────
    RATING_QS = [
        "The activity introduced me to new topics and ideas",
        "I enjoy STEM more now because of the activity",
        "This activity made me excited to learn more about STEM by myself",
        "The activity helped me meet other students who like learning about the same things as me",
    ]
    for q in RATING_QS:
        if q in df.columns:
            df[q] = pd.to_numeric(df[q], errors="coerce")

    # ── Identifiers & metadata ─────────────────────────────────────────────────
    df = df.rename(columns={"Survey Start": "Timestamp"})
    df["Record Number"] = df["First Name"].str.extract(r"#(\d+)")
    df["Term"] = ""
    df["ATSI"] = ""

    # ── Open-text comments ─────────────────────────────────────────────────────
    comments_raw = next((c for c in all_cols if "favourite part" in c.lower()), None)
    if comments_raw:
        df = df.rename(columns={comments_raw: "Comments"})

    # ── Final column selection ─────────────────────────────────────────────────
    output_cols = [
        "Record Number",
        "Timestamp",
        "Term",
        "Gender",
        "ATSI",
        "School",
        "Year Level",
        "Program Name",
        "The activity introduced me to new topics and ideas",
        "I enjoy STEM more now because of the activity",
        "This activity made me excited to learn more about STEM by myself",
        "The activity helped me meet other students who like learning about the same things as me",
        "Did you attend with classmates?",
        "I can see how what I learnt in this activity can be used in real life",
        "I would recommend this activity to another student",
        "Comments",
    ]
    return df[[c for c in output_cols if c in df.columns]]
