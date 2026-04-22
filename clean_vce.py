import re
from collections import Counter

import pandas as pd


def clean_vce(input_file: str) -> pd.DataFrame:
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

    # Deduplicate column names — "Other", "Other Comments", and the six priority
    # schools that appear twice in the school block all collide. Append position.
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

    def strip_dup_suffix(name: str) -> str:
        # Six priority schools appear twice in the school block; after dedup the
        # second copy carries a _<idx> suffix we need to discard for display.
        return re.sub(r"_\d+$", "", name)

    def get_selected(row, block: list, use_comment: bool = True) -> str:
        """Return the display value of the first '1' column in the block."""
        other_c    = next((c for c in block if is_other(c)),          None)
        comments_c = next((c for c in block if is_other_comments(c)), None)
        for c in block:
            if is_other(c) or is_other_comments(c):
                continue
            if str(row.get(c, "")).strip() == "1":
                return strip_dup_suffix(strip_q_prefix(c))
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
    # Substring-in-column-name matching avoids dependence on block boundaries.
    PROGRAMS_2026 = [
        "VCE Masterclass Chem Unit 2: Analytical Techniques Water",
        "VCE Masterclass: Biology Unit 2: Sickle Cell Inheritance",
        "VCE Masterclass: Biology Unit 3: DNA Manipulation and Genetic Technologies",
        "VCE Masterclass: Biology Unit 3: Photosynthesis and Biochemical Pathways",
        "VCE Masterclass: Biology Unit 4: Evolution of Lemurs",
        "VCE Masterclass: Chemistry Unit 2: Analytical Techniques Water",
        "VCE Masterclass: Chemistry Unit 4: Organic Compounds",
        "VCE Masterclass: Environmental Science Unit 2: Water Pollution",
        "VCE Masterclass: Physics Unit 1: Thermodynamics",
        "VCE Masterclass: Physics Unit 2: Mission Gravity with OzGrav",
    ]
    prog_col_map = {p: next((c for c in all_cols if p in c), None) for p in PROGRAMS_2026}
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
    # VCE year block has no "Others" free-text field.
    def get_year_level(row):
        for c in year_cols:
            if str(row.get(c, "")).strip() == "1":
                return strip_q_prefix(c)
        return "Unknown"

    df["Year Level"] = df.apply(get_year_level, axis=1)

    # ── "If given the opportunity…" ────────────────────────────────────────────
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

    # ── Learning Intentions column: header uses inconsistent whitespace ────────
    # Locate it by substring and normalise the name for output selection.
    li_src = next((c for c in df.columns if "Learning Intentions" in c), None)
    LI_OUT = "The learning program I completed at the KIOSC met the Learning Intentions"
    if li_src and li_src != LI_OUT:
        df = df.rename(columns={li_src: LI_OUT})

    # ── Numeric columns ────────────────────────────────────────────────────────
    NUMERIC_COLS = [
        "How much did you enjoy the sessions today?",
        "How much do you think you have learnt today?",
        "I learnt something new today",
        "The program I did motivated me to explore new ideas and concepts",
        "I used technology to help me learn",
        "I had the opportunity to collaborate with other students",
        "I learnt about industries that use science, technology, engineering, or maths (referred to as STEM) in my local area",
        "Study a VCE Science, Maths or Technology subject",
        "Undertake a VET program",
        "Enrol in a STEM-related university or TAFE course after school",
        "Consider a STEM-related career after school",
        LI_OUT,
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
        "Study a VCE Science, Maths or Technology subject",
        "Undertake a VET program",
        "Enrol in a STEM-related university or TAFE course after school",
        "Consider a STEM-related career after school",
        "If given the opportunity, would you like to attend another KIOSC program?",
        LI_OUT,
    ]
    return df[[c for c in output_cols if c in df.columns]]
