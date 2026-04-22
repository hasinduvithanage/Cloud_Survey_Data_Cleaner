import re
from collections import Counter

import pandas as pd


def clean_discovery(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file, skiprows=4, dtype=str)

    # Replace pandas "Unnamed: X" placeholders with empty string
    df.columns = ["" if "Unnamed" in str(c) else c for c in df.columns]

    # Row 0 is the sub-option row; row 1 is a partial "Demographic" label row.
    # Merge sub-option row with the primary header.
    def make_col(h, s):
        h = str(h).strip() if pd.notna(h) and str(h).strip() not in ("", "nan") else ""
        s = str(s).strip() if pd.notna(s) and str(s).strip() not in ("", "nan") else ""
        if h and s:
            return f"{h}-{s}"
        return h or s

    df.columns = [make_col(df.columns[i], df.iloc[0, i]) for i in range(len(df.columns))]
    df = df.iloc[1:].reset_index(drop=True)

    # Drop non-respondent rows (partial "Demographic" label row and blanks)
    df = df[df["First Name"].notna() & (df["First Name"].str.strip() != "")].reset_index(drop=True)

    # Deduplicate column names — "Other" and "Other Comments" each appear in
    # the program and school blocks, so append the column's positional index.
    raw_cols = df.columns.tolist()
    dup_set = {c for c, n in Counter(raw_cols).items() if n > 1}
    df.columns = [f"{c}_{i}" if c in dup_set else c for i, c in enumerate(raw_cols)]
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()

    # ── Block detection ────────────────────────────────────────────────────────
    def block_idx(prefix: str, default: int = len(all_cols)) -> int:
        return next((i for i, c in enumerate(all_cols) if c.startswith(prefix)), default)

    delivery_start     = block_idx("How was your KIOSC program delivered?")
    prog_start         = block_idx("What program did you attend?")
    school_start       = block_idx("What school are you from?")
    gender_start       = block_idx("Which of the following most accurately describes your gender?")
    year_start         = block_idx("What year level are you?")
    section_hdr        = next((i for i, c in enumerate(all_cols) if "DURING my KIOSC experience" in c), year_start + 9)
    likelihood_hdr     = next((i for i, c in enumerate(all_cols) if "How likely are you to" in c), section_hdr + 6)
    attend_again_start = block_idx("If given the opportunity, would you like")
    days_start         = block_idx("What day")

    delivery_cols     = all_cols[delivery_start:prog_start]
    school_cols       = all_cols[school_start:gender_start]
    gender_cols       = all_cols[gender_start:year_start]
    year_cols         = all_cols[year_start:section_hdr]
    attend_again_cols = all_cols[attend_again_start:days_start]

    # ── Column name helpers ────────────────────────────────────────────────────
    # Strip the "Question?-" prefix from a combined column name to get the
    # display value (e.g. "What year level are you?-Year 7" → "Year 7").
    QUESTION_PREFIXES = (
        "How was your KIOSC program delivered?-",
        "What program did you attend?-",
        "Which of the following most accurately describes your gender?-",
        "What year level are you?-",
        "If given the opportunity, would you like to attend another KIOSC program?-",
    )

    def strip_q_prefix(name: str) -> str:
        for pfx in QUESTION_PREFIXES:
            if name.startswith(pfx):
                return name[len(pfx):]
        # School question has a long parenthetical; split on the separator dash
        # that follows the closing parenthesis, not on dashes within school names.
        if "What school are you from?" in name:
            return name.split("-", 1)[1].strip()
        return name

    def is_other(c: str) -> bool:
        return c.strip() == "Other" or bool(re.match(r"^Other_\d+$", c.strip()))

    def is_other_comments(c: str) -> bool:
        return c.strip() == "Other Comments" or bool(re.match(r"^Other Comments_\d+$", c.strip()))

    def get_selected(row, block: list, use_comment: bool = True) -> str:
        """Return the display value of the first '1' column in the block."""
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

    # ── Delivery mode ──────────────────────────────────────────────────────────
    DELIVERY_MAP = {
        "Onsite (face to face at KIOSC)":                                                     "Onsite",
        "Offsite (face to face at your school by your teachers OR a KIOSC facilitator)":      "Offsite",
        "Online (delivered zia Zoom, Webex, Teams etc)":                                      "Online",
        "Immersion (delivered at an industry site)":                                          "Immersion",
    }

    def get_delivery(row):
        for c in delivery_cols:
            if str(row.get(c, "")).strip() == "1":
                return DELIVERY_MAP.get(strip_q_prefix(c), strip_q_prefix(c))
        return "Unknown"

    df["Delivery Mode"] = df.apply(get_delivery, axis=1)

    # ── Program name ───────────────────────────────────────────────────────────
    # Use substring-in-column-name matching so we don't rely on block boundaries.
    # Each program name is a substring of exactly one column in all_cols.
    PROGRAMS_2026 = [
        "Discovery: 3D Design and Merge",
        "Discovery: Aspirin Analysis",
        "Discovery: Bioplastics",
        "Discovery: Build Beyond Trades",
        "Discovery: Drones on Mars",
        "Discovery: Emergency Technology",
        "Discovery: Forensic Science: Major Crime",
        "Discovery: Genetics &amp; Microarrays",
        "Discovery: Green Energy Revolution",
        "Discovery: Guardians of the CyNet",
        "Discovery: Hydrogen GRAND PRIX",
        "Discovery: Inquiry In RealLife (IRL)",
        "Discovery: LEGO Robotics",
        "Discovery: Logistic FAILs",
        "Discovery: Ocean Scratch 2",
        "Discovery: OZGRAV Space",
        "Discovery: Peer Support Training",
        "Discovery: Psychology: Brain Tech",
        "Discovery: Scratch AI",
        "Discovery: Sphero Space",
        "Discovery: STEM Communication Conference",
        "Discovery: STEM to the Rescue",
        "Discovery: Sustainable Futures",
        "Discovery: TECHSprint",
        "Discovery: TowerTech Innovators",
        "Discovery: Transformational Design",
        "Discovery: TrashBot Challenge",
        "Discovery: Vitamin C Analysis",
    ]
    # Map each program name → the column that contains it as a substring
    prog_col_map = {p: next((c for c in all_cols if p in c), None) for p in PROGRAMS_2026}
    # "Other" and "Other Comments" for the program block sit between prog_start and school_start
    prog_other_c    = next((c for c in all_cols[prog_start:school_start] if is_other(c)),          None)
    prog_comments_c = next((c for c in all_cols[prog_start:school_start] if is_other_comments(c)), None)

    def get_program(row):
        for prog, col in prog_col_map.items():
            if col and str(row.get(col, "")).strip() == "1":
                return prog
        if prog_other_c and str(row.get(prog_other_c, "")).strip() == "1":
            if prog_comments_c:
                comment = str(row.get(prog_comments_c, "")).strip()
                if comment and comment.lower() != "nan":
                    return comment
            return "Other"
        return "Unknown"

    df["Program Name"] = df.apply(get_program, axis=1)

    # ── School ─────────────────────────────────────────────────────────────────
    # Discovery 2026 only lists 6 schools by name; most students use Other + free text.
    # School name columns (except the anchor) carry no question prefix, so
    # strip_q_prefix returns them as-is.  The anchor column splits on the
    # first dash (the question text contains no dashes before the separator).
    df["School"] = df.apply(lambda r: get_selected(r, school_cols), axis=1)

    # ── Gender ─────────────────────────────────────────────────────────────────
    # "Let me explain" is an open-text field; handle it like Other Comments.
    def get_gender(row):
        let_me_col     = next((c for c in gender_cols if "Let me explain" in c and "Comments" not in c), None)
        let_me_comment = next((c for c in gender_cols if "Let me explain Comments" in c), None)
        for c in gender_cols:
            if "Let me explain" in c:
                continue
            if str(row.get(c, "")).strip() == "1":
                return strip_q_prefix(c)
        if let_me_col and str(row.get(let_me_col, "")).strip() == "1":
            if let_me_comment:
                comment = str(row.get(let_me_comment, "")).strip()
                if comment and comment.lower() != "nan":
                    return comment
            return "Other"
        return "Unknown"

    df["Gender"] = df.apply(get_gender, axis=1)

    # ── Year level ─────────────────────────────────────────────────────────────
    # Uses "Others" (capital S) with a paired "Others Comments" free-text field.
    def get_year_level(row):
        others_c   = next((c for c in year_cols if c.startswith("Others") and "Comments" not in c), None)
        comments_c = next((c for c in year_cols if "Others Comments" in c), None)
        for c in year_cols:
            if c == others_c or c == comments_c:
                continue
            if str(row.get(c, "")).strip() == "1":
                return strip_q_prefix(c)
        if others_c and str(row.get(others_c, "")).strip() == "1":
            if comments_c:
                comment = str(row.get(comments_c, "")).strip()
                if comment and comment.lower() != "nan":
                    return comment
            return "Other"
        return "Unknown"

    df["Year Level"] = df.apply(get_year_level, axis=1)

    # ── "If given the opportunity…" ────────────────────────────────────────────
    # The Yes column has the question prefix; No and Unsure are bare col names.
    # strip_q_prefix handles all three: returns "Yes", "No", or "Unsure".
    def get_attend_again(row):
        for c in attend_again_cols:
            if str(row.get(c, "")).strip() == "1":
                return strip_q_prefix(c)
        return "Unknown"

    df["If given the opportunity, would you like to attend another KIOSC program?"] = df.apply(
        get_attend_again, axis=1
    )

    # ── Metadata ───────────────────────────────────────────────────────────────
    df = df.rename(columns={"Survey Start": "Timestamp"})
    df["Record Number"] = df["First Name"].str.extract(r"#(\d+)")
    df["Term"] = ""
    df["ATSI"] = ""

    # ── Numeric columns ────────────────────────────────────────────────────────
    NUMERIC_COLS = [
        "How much did you enjoy the sessions today?",
        "How much do you think you have learnt today?",
        "I learnt something new today",
        "The program I did motivated me to explore new ideas and concepts",
        "I used technology to help me learn",
        "I had the opportunity to collaborate with other students",
        "I learnt about industries that use science, technology, engineering, or maths (referred to as STEM) in my local area",
        "The learning program I completed at the KIOSC met the Learning Intentions",
    ]
    for col_name in NUMERIC_COLS:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

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
        "Delivery Mode",
        "How much did you enjoy the sessions today?",
        "How much do you think you have learnt today?",
        "I learnt something new today",
        "The program I did motivated me to explore new ideas and concepts",
        "I used technology to help me learn",
        "I had the opportunity to collaborate with other students",
        "I learnt about industries that use science, technology, engineering, or maths (referred to as STEM) in my local area",
        "If given the opportunity, would you like to attend another KIOSC program?",
        "The learning program I completed at the KIOSC met the Learning Intentions",
    ]
    return df[[c for c in output_cols if c in df.columns]]
