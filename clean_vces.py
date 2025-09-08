import pandas as pd


def clean_vces(input_file: str) -> pd.DataFrame:
    """Clean a raw VCES survey CSV and return the processed DataFrame."""

    # Load dataset (skip first 3 rows)
    df = pd.read_csv(input_file, skiprows=4)

    # Replace column names that contain 'Unnamed' with an empty string
    df.columns = ["" if "Unnamed" in str(col) else col for col in df.columns]

    # Function to concatenate header with first-row values
    def create_column_name(header, row_value):
        if pd.isna(header) or "Unnamed" in str(header) or str(header).strip() == "":
            return str(row_value)
        elif pd.notna(row_value) and str(row_value).strip() != "":
            return f"{header}-{row_value}"
        else:
            return header

    # Apply function to generate new column names
    df.columns = [create_column_name(df.columns[i], df.iloc[0, i]) for i in range(len(df.columns))]

    # Drop the first row after using it for renaming
    df = df.iloc[1:].reset_index(drop=True)

    # Drop rows where 'First Name' is empty or NaN
    df = df.dropna(subset=['First Name'])

    # Rename duplicate columns if necessary
    df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col for i, col in enumerate(df.columns)]

    df.columns = df.columns.str.strip()

    def get_gender(row):
        if row['What is your gender?-Female'] == '1':
            return 'Female'
        elif row['Male'] == '1':
            return 'Male'
        elif row['Other_14'] == '1':
            return 'Non-binary'
        elif row['Rather not say_15'] == '1':
            return 'Rather not say'
        else:
            return 'Unknown'

    df['Gender'] = df.apply(get_gender, axis=1)

    def get_school_name(row):
        school_columns = [
            "What School are you from?-Bayswater Secondary College", "Boronia K-12 College", "Fairhills High School",
            "Rowville Secondary College", "Scoresby Secondary College", "Wantirna College",
            "Alamanda College", "Albert Park Primary School", "Aquinas College", "Ashwood College",
            "Auburn High School", "Avila College", "Balwyn High School", "Balwyn Primary School",
            "Beaumaris Secondary College", "Bentleigh West Primary School", "Berwick Primary School",
            "Billanook College", "Blackburn High School", "Box Hill High School", "Brentwood College",
            "Brighton Secondary College", "Brunswick Secondary College", "Cambridge Primary School",
            "Canterbury Primary School", "Carranballac College", "Caulfield Grammar", "Charlton College",
            "CIRE Community School", "Coburg Primary School", "Croydon Community School",
            "Dandenong High School", "Diamond Valley College", "Doncaster Secondary College",
            "Donvale Christian College", "East Doncaster Secondary College", "Edinburgh College",
            "Elliminyt Primary School", "Eltham High School", "Emerald Primary School",
            "Emerald Secondary College", "Emmaus College", "Essendon Keilor College", "Forest Hill College",
            "Glen Waverley Secondary College", "Hazel Glen College", "Healesville High School",
            "Heathmont East Primary School", "Heathmont Secondary College", "Highvale Secondary College",
            "Kananook Primary School", "Keysborough College", "Kew High School", "Killester College",
            "Knox School", "Launching Place Primary School", "Lilydale Heights College", "Lilydale High School",
            "Luther College", "Mansfield Secondary College", "Mary MacKillop Catholic Regional College",
            "Mater Christi College", "Mazenod College", "McClelland College", "McKinnon Secondary College",
            "Melba College", "Mill Park Primary School", "Monbulk College", "Mooroolbark College",
            "Mount Evelyn Christian College", "Mount Lilydale Mercy College", "Mount Waverley Secondary College",
            "Mountain District Christian School", "Mountain District Learning Centre", "Mullauna College",
            "Nazareth College", "Narre Warren South P12 College", "North Ringwood Community House",
            "Northern Bay P-12", "Norwood Secondary College", "Oakwood School", "Our Lady of Sion College",
            "Oxley College", "Oxley Christian College", "Pines Learning Centre", "Ranges TEC",
            "Reservoir West Primary School", "Richmond West primary school", "Ringwood Secondary College",
            "Rosanna Golf Links Primary School", "Sherbrooke Community School", "South Melbourne Park Primary School",
            "St Andrew's Christian College", "St Joseph's College", "St Kilda Park Primary School",
            "Strathmore Secondary College", "Swan Hill College", "Taylors Lakes Secondary College",
            "Tecoma Primary School", "Templestowe College", "Tintern Schools", "Upper Yarra Secondary College",
            "Upwey High School", "Vermont Secondary College", "Victoria Road Primary School",
            "Wantirna South Primary School", "Warrandyte High School", "Waverley Christian College",
            "Wellington College", "Wheelers Hill Secondary College", "Whitefriars College",
            "Whittlesea Secondary College", "Wodonga Middle School", "Woodleigh School",
            "Yarra Hills Secondary College", "Yarra Junction primary", "Yarra Valley Grammar School"
        ]

        if 'What School are you from?-Bayswater Secondary College' in row.index and str(row['What School are you from?-Bayswater Secondary College']) == '1':
            return 'Bayswater Secondary College'

        for school in school_columns:
            if str(row[school]) == '1':
                return school

        if str(row['Other_136']) == '1':
            return row['Other Comments_137']

        return 'Unknown'

    df['School'] = df.apply(get_school_name, axis=1)
    df.loc[df['School'] == 'Unknown', 'School'] = df['Other Comments_137']

    df.columns = df.columns.str.strip()

    def get_grade(row):
        if row['What is your year level at school?-Prep'] == '1':
            return 'Prep'
        elif row['Year 5'] == '1':
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

    df['Year Level'] = df.apply(get_grade, axis=1)

    df.columns = df.columns.str.strip()

    def get_attend_again1(row):
        if row['I would recommend this activity to another student.-Strongly agree'] == '1':
            return 'Strongly agree'
        elif row['Agree_152'] == '1':
            return 'Agree'
        elif row['Neither agree nor disagree _153'] == '1':
            return 'Neither agree nor disagree'
        elif row['Disagree_154'] == '1':
            return 'Disagree'
        elif row['Strongly disagree._155'] == '1':
            return 'Strongly disagree'
        else:
            return 'Unknown'

    df['I would recommend this activity to another student'] = df.apply(get_attend_again1, axis=1)

    df.columns = df.columns.str.strip()

    def get_attend_again2(row):
        if row['The activity introduced me to new topics and ideas.-Strongly agree'] == '1':
            return 'Strongly agree'
        elif row['Agree_157'] == '1':
            return 'Agree'
        elif row['Neither agree nor disagree _158'] == '1':
            return 'Neither agree nor disagree'
        elif row['Disagree_159'] == '1':
            return 'Disagree'
        elif row['Strongly disagree._160'] == '1':
            return 'Strongly disagree'
        else:
            return 'Unknown'

    df['The activity introduced me to new topics and ideas'] = df.apply(get_attend_again2, axis=1)

    df.columns = df.columns.str.strip()

    def get_attend_again3(row):
        if row['The activity made me think hard / carefully.-Strongly agree'] == '1':
            return 'Strongly agree'
        elif row['Agree_162'] == '1':
            return 'Agree'
        elif row['Neither agree nor disagree _163'] == '1':
            return 'Neither agree nor disagree'
        elif row['Disagree_164'] == '1':
            return 'Disagree'
        elif row['Strongly disagree._165'] == '1':
            return 'Strongly disagree'
        else:
            return 'Unknown'

    df['The activity made me think hard / carefully'] = df.apply(get_attend_again3, axis=1)

    df.columns = df.columns.str.strip()

    def get_attend_again4(row):
        if row['The activity was different to regular class at school.-Strongly agree'] == '1':
            return 'Strongly agree'
        elif row['Agree_167'] == '1':
            return 'Agree'
        elif row['Neither agree nor disagree _168'] == '1':
            return 'Neither agree nor disagree'
        elif row['Disagree_169'] == '1':
            return 'Disagree'
        elif row['Strongly disagree._170'] == '1':
            return 'Strongly disagree'
        else:
            return 'Unknown'

    df['The activity was different to regular class at school.'] = df.apply(get_attend_again4, axis=1)

    df.columns = df.columns.str.strip()

    def get_program_name(row):
        if str(row.get('What program did you complete today?-VCES: BioPlastics', '')) == '1':
            return 'VCES: BioPlastics'
        elif str(row.get('VCES: Forensics: Crack the COVID Case', '')) == '1':
            return 'VCES: Forensics: Crack the COVID Case'
        elif str(row.get('VCES: Forensics: Major Crime', '')) == '1':
            return 'VCES: Forensics: Major Crime'
        elif str(row.get('VCES: Genetics and Microarrays', '')) == '1':
            return 'VCES: Genetics and Microarrays'
        elif str(row.get('VCES: Green Energy Revolution', '')) == '1':
            return 'VCES: Green Energy Revolution'
        elif str(row.get('VCES: Hydrogen Car Competition', '')) == '1':
            return 'VCES: Hydrogen Car Competition'
        elif str(row.get('VCES: LEGO', '')) == '1':
            return 'VCES: LEGO'
        elif str(row.get('VCES: Ocean Scratch 1: Food Webs', '')) == '1':
            return 'VCES: Ocean Scratch 1: Food Webs'
        elif str(row.get('VCES: Ocean Scratch 2: The Clean Up', '')) == '1':
            return 'VCES: Ocean Scratch 2: The Clean Up'
        elif str(row.get('VCES: Scratch Ai Part 1', '')) == '1':
            return 'VCES: Scratch Ai Part 1'
        elif str(row.get('VCES: Scratch Ai Part 2', '')) == '1':
            return 'VCES: Scratch Ai Part 2'
        elif str(row.get('VCES: Smart Trains', '')) == '1':
            return 'VCES: Smart Trains'
        elif str(row.get('VCES: Transformational Design', '')) == '1':
            return 'VCES: Transformational Design'
        elif str(row.get('VCES: TrashBot Challenge', '')) == '1':
            return 'VCES: TrashBot Challenge'
        elif str(row.get('Other_185', '')) == '1':
            return row.get('Other Comments_186', 'Other')
        else:
            return 'Unknown'

    df['Program Name'] = df.apply(get_program_name, axis=1)

    df = df.rename(columns={'Survey Start': 'Timestamp'})
    df['Record Number'] = df['First Name'].str.extract(r'#(\d+)')
    df['Term'] = ''
    df['ATSI'] = ''

    df_selected = df[
        [
            'Record Number', 'Timestamp', 'Term', 'Gender', 'ATSI', 'School', 'Year Level', 'Program Name',
            'I would recommend this activity to another student',
            'The activity introduced me to new topics and ideas',
            'The activity made me think hard / carefully',
            'The activity was different to regular class at school.'
        ]
    ]

    return df_selected
