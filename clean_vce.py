# Cleaning script for VCE survey data
import pandas as pd


def clean_vce(input_file: str) -> pd.DataFrame:
    """Clean a raw VCE survey CSV and return the processed DataFrame."""

    # Load dataset (skip metadata rows)
    df = pd.read_csv(input_file, skiprows=4)

    # Replace columns named 'Unnamed' with empty strings
    df.columns = ["" if "Unnamed" in str(col) else col for col in df.columns]

    # Helper to combine header and first data row
    def create_column_name(header, row_value):
        if pd.isna(header) or "Unnamed" in str(header) or str(header).strip() == "":
            return str(row_value)
        elif pd.notna(row_value) and str(row_value).strip() != "":
            return f"{header}-{row_value}"
        else:
            return header

    # Build combined column names using the first row
    df.columns = [create_column_name(df.columns[i], df.iloc[0, i]) for i in range(len(df.columns))]

    # # Clean column names
    # df.columns = df.columns.str.replace("â€™", "'", regex=False)
    # df.columns = df.columns.str.replace("\xa0", " ", regex=False)
    # df.columns = df.columns.str.strip()

    # Remove the row used for column renaming
    df = df.iloc[1:].reset_index(drop=True)

    # Drop rows missing a first name
    df = df.dropna(subset=["First Name"])

    # Ensure duplicate column names are unique
    df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col for i, col in enumerate(df.columns)]

    df.columns = df.columns.str.strip()

    # Print column names for debugging
    print("COLUMN NAMES:", df.columns.tolist())

    def get_gender(row):
        if row['Which of the following most accurately describes your gender? -Female'] == '1':
            return 'Female'
        elif row['Male'] == '1':
            return 'Male'
        elif row['Non-binary'] == '1':
            return 'Non-binary'
        elif row['Let me explain'] == '1':
            return row.get('Let me explain Comments', 'Let me explain')
        elif row['Rather not say'] == '1':
            return 'Rather not say'
        else:
            return 'Unknown'

    df['Gender'] = df.apply(get_gender, axis=1)

        # School responses appear as one-hot encoded columns with the school name
        # as the header. The raw data may contain duplicate columns for multiple
        # survey sections which are made unique by an index suffix (e.g. `_31`).
    def get_school_name(row):
        # Combined header: long header that includes Bayswater Secondary College
        for col in row.index:
            if (
                "Bayswater Secondary College" in col
                and str(row[col]) == '1'
            ):
                return "Bayswater Secondary College"

        # General case: one-hot encoded school name columns
        for col in row.index:
            if (
                any(word in col for word in ["School", "College", "House", "Centre"])
                and str(row[col]) == '1'
            ):
                return col.rsplit('_', 1)[0]  # Removes _32 suffix, safer for names with _

        # Handle "Other" option
        if str(row.get('Other_156', '')) == '1':
            return row.get('Other Comments_157', 'Other')

        return 'Unknown'


    df['School'] = df.apply(get_school_name, axis=1)
    if 'Other Comments_157' in df.columns:
        df.loc[df['School'] == 'Unknown', 'School'] = df['Other Comments_157']

    def get_year_level(row):
        if row['What year level are you?-Year 5'] == '1':
            return 'Year 5'
        elif row['Year 6'] == '1':
            return 'Year 6'
        elif row['Year 7'] == '1':
            return 'Year 7'
        elif row['Year 8'] == '1':
            return 'Year 8'
        elif row['Year 9'] == '1':
            return 'Year 9'
        elif row['Year 10'] == '1':
            return 'Year 10'
        elif row['Year 11'] == '1':
            return 'Year 11'
        elif row['Year 12'] == '1':
            return 'Year 12'
        else:
            return 'Unknown'

    df['Year Level'] = df.apply(get_year_level, axis=1)

    def get_program_name(row):
        if row['What\u00a0program did you attend?-VCE Masterclass Chem Unit 2: Analytical Techniques Water'] == '1':
            return 'VCE Masterclass Chem Unit 2: Analytical Techniques Water'
        elif row['VCE Masterclass: Biology Unit 2: Sickle Cell Inheritance'] == '1':
            return 'VCE Masterclass: Biology Unit 2: Sickle Cell Inheritance'
        elif row['VCE Masterclass: Biology Unit 3: DNA Manipulation and Genetic Technologies'] == '1':
            return 'VCE Masterclass: Biology Unit 3: DNA Manipulation and Genetic Technologies'
        elif row['VCE Masterclass: Biology Unit 3: Photosynthesis and Biochemical Pathways'] == '1':
            return 'VCE Masterclass: Biology Unit 3: Photosynthesis and Biochemical Pathways'
        elif row['VCE Masterclass: Biology Unit 4: Evolution of Lemurs'] == '1':
            return 'VCE Masterclass: Biology Unit 4: Evolution of Lemurs'
        elif row['VCE Masterclass: Chemistry Unit 2: Analytical Techniques Water'] == '1':
            return 'VCE Masterclass: Chemistry Unit 2: Analytical Techniques Water'
        elif row['VCE Masterclass: Chemistry Unit 4: Organic Compounds'] == '1':
            return 'VCE Masterclass: Chemistry Unit 4: Organic Compounds'
        elif row['VCE Masterclass: Environmental Science Unit 2: Water Pollution'] == '1':
            return 'VCE Masterclass: Environmental Science Unit 2: Water Pollution'
        elif row['VCE Masterclass: Physics Unit 1: Thermodynamics'] == '1':
            return 'VCE Masterclass: Physics Unit 1: Thermodynamics'
        elif row['VCE Masterclass: Physics Unit 2: Mission Gravity with OzGrav'] == '1':
            return 'VCE Masterclass: Physics Unit 2: Mission Gravity with OzGrav'
        elif row['VCE Masterclass: Unit 4: Evolution of Lemurs'] == '1':
            return 'VCE Masterclass: Unit 4: Evolution of Lemurs'
        elif row['Other_27'] == '1':
            return row.get('Other Comments_28', 'Other')
        else:
            return 'Unknown'

    df['Program Name'] = df.apply(get_program_name, axis=1)

    def get_delivery_mode(row):
        if row['How was your KIOSC program delivered?-Onsite (face to face at KIOSC)'] == '1':
            return 'Onsite'
        elif row['Offsite (face to face at your school by your teachers OR a KIOSC facilitator)'] == '1':
            return 'Offsite'
        elif row['Online (delivered zia Zoom, Webex, Teams etc)'] == '1':
            return 'Online'
        elif row['Immersion (delivered at an industry site)'] == '1':
            return 'Immersion'
        else:
            return 'Unknown'

    df['Delivery Mode'] = df.apply(get_delivery_mode, axis=1)

    df = df.rename(columns={'Survey Start': 'Timestamp'})
    df['Record Number'] = df['First Name'].str.extract(r'#(\d+)')
    df['Term'] = ''
    df['ATSI'] = ''

    selected_columns = [
        'Record Number',
        'Timestamp',
        'Term',
        'Gender',
        'ATSI',
        'School',
        'Year Level',
        'Program Name',
        'Delivery Mode',
        'How much did you enjoy the sessions today?',
        'How much do you think you have learnt today?',
        'I learnt something new today',
        'The program I did motivated me to explore new ideas and concepts',
        'I used technology to help me learn',
        'I had the opportunity to collaborate with other students',
        'I learnt about industries that use science, technology, engineering, or maths (referred to as STEM) in my local area',
        'If given the opportunity, would you like to attend another KIOSC program?-Yes',
        'The learning program I completed at the KIOSC\u00a0 met the Learning Intentions'
    ]

    df_selected = df[selected_columns]
    return df_selected
