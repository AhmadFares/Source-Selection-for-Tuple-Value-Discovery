import pandas as pd
import sqlite3
import numpy as np


class TestCases:
    """
    This class contains test cases for the Coverage-Guided Row Selection algorithm.
    Each test case is defined as a tuple (T, UR) where:
      - T is the initial table (a pandas DataFrame).
      - UR is the User Request table (a pandas DataFrame) that specifies the required values for each column.
    """

    def __init__(self):
        self.cases = {}  # Dictionary to store test cases
        #self.load_lisa_sheets()
        self.load_fixed_mathe_case()  # Load MATHE case 
        
        #self.load_movielens_case()  # Load MovieLens case
        self.load_fixed_movielens_case()  # Load fixed MovieLens case
        
    def load_lisa_sheets(self):
        """
        Load the Lisa_Sheets table from the SQLite database and store it as T.
        """
        db_path = "data/Lisa_Tabular_Data.db"  # Database path
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM Lisa_Sheets;"
        T = pd.read_sql_query(query, conn)
        conn.close()

        # 🔹 Define User Requests (UR):
        """
        UR1: Answer in the Beggining of the table
        UR2: Answer in the End of Table
        UR3: Answer Distributed in the Table
        UR4: No Answers
        UR5: Partial Answers
        UR6: Multiple Answers Applicable -> To test importance of Penalty_Optimization
        UR7: Intermediate Answers -> To test importance of Optimize_Selection
        """
        user_requests = {
    1: {
        "Keyword1": ["venous approaches", "removal venous", "gestational hypertension", "pre eclampsia", "pregnancy methods"],
        "Keyword2": ["peripheral venous", "pregnancy hypertension", "haemorrhage", "lupus"]
    },
    2: {
        "Keyword1": ["mri lumbar", "sacroiliac tests", "spinal causes"],
        "Keyword2": ["spine mri", "spondylodiscitis pott", "severe undernutrition", "pain spinal"]
    },
    3: {
        "Keyword1": ["venous approaches", "sacroiliac tests", "pre eclampsia", "mri lumbar", "tumour stomach", "splenomegaly enlarged", "preventive cerclage", "rachis cervical"],
        "Keyword2": ["hyperplasia parathyroid", "oedematous syndrome", "schizophrenia following"]
    },
    4: {
        "Keyword1": ["aaaaa", "bbb", "cccc"],
        "Keyword2": ["dddd", "eeee", "ffff"]
    },
    5: {
        "Keyword1": ["venous approaches", "aaaaaa", "removal venous"],
        "Keyword2": ["bbbbbbb", "oedematous syndrome", "hyperplasia parathyroid"]
    },
    6: {
       #ID-FARES-Test||||||||||venous approaches|approach venous||
       "Keyword1": ["venous approaches"],
       "Keyword2": ["approach venous"]
    },
    7: {
       "Keyword1": ["cerebral mri", "limb trauma", "trendelebourg lameness", "complications pregnancy"],
       "Keyword2": ["stroke mri", "saluting trendelebourg", "maternal complications", "complications nerve"]
    }
}

        # 🔹 Convert all User Requests to properly formatted DataFrames
        for case_number, ur_data in user_requests.items():
            self.cases[case_number] = (T, self.create_flexible_dataframe(ur_data))

        # 🔹 Add additional test cases (without Lisa_Sheets)
        self.cases[10] = self.create_penalty_opt_case()
        self.cases[11] = self.create_optimized_selection_case()

    def create_flexible_dataframe(self, data_dict):
        """
        Convert a dictionary to a pandas DataFrame, handling columns with different lengths.
        Uses pd.Series to ensure misaligned columns are handled correctly.
        """
        return pd.DataFrame.from_dict({key: pd.Series(value, dtype=object) for key, value in data_dict.items()})

    def create_penalty_opt_case(self):
        """Returns a predefined penalty optimization test case."""
        T10 = pd.DataFrame({
            "A": ["v1", "v2", "x3", "x4", "v1", "v2"],
            "B": ["x1", "x2", "v3", "v4", "v3", "v4"]
        })
        UR10 = pd.DataFrame({
            "A": ["v1", "v2"],
            "B": ["v3", "v4"]
        })
        return T10, UR10

    def create_optimized_selection_case(self):
        """Returns a predefined optimized selection test case."""
        T11 = pd.DataFrame({
            "A": ["v1", "v2", "v1", "x3"],
            "B": ["x1", "x2", "v3", "v4"]
        })
        UR11 = pd.DataFrame({
            "A": ["v1", "v2"],
            "B": ["v3", "v4"]
        })
        return T11, UR11
    def build_random_ur(self, df, columns, k):
        ur = {}
        for col in columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                selected = np.random.choice(unique_vals, size=min(k, len(unique_vals)), replace=False)
                ur[col] = list(selected)
        return ur
        
    def load_mathe_case(self, csv_path="./data/MATHE/output_table.csv", n_values_per_col=1):
        """
        Load MATHE and generate a random UR from the last 3 columns (for ad-hoc testing).
        """
        mathe_df = pd.read_csv(csv_path, delimiter=";")
        mathe_df.rename(columns={"id_assessment": "Identifiant"}, inplace=True)

        ur_columns = mathe_df.columns[-7:]
        ur_dict = self.build_random_ur(mathe_df, ur_columns, n_values_per_col)
        UR = self.create_flexible_dataframe(ur_dict)

        base_cols = list(ur_columns) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[19] = (T, UR)

    def load_fixed_mathe_case(self, csv_path="/home/slide/faresa/Coverage-Guided-Row-Selection-with-Optimization/data/MATHE/output_table.csv"):
        """
        Load MATHE and use a fixed User Request (UR) across 3 columns.
        """
        mathe_df = pd.read_csv(csv_path, delimiter=";")
        mathe_df.rename(columns={"id_assessment": "Identifiant"}, inplace=True)

        UR_Deep_1 = {
            "keyword_name": ["Two variables", "Orthogonality", "Three points rule", "Mean"],
            "topic_name": ["Linear Algebra", "Probability", "Optimization", "Discrete Mathematics"],
            "subtopic_name": [
                "Linear Transformations",
                "Vector Spaces",
                "Algebraic expressions, Equations, and Inequalities",
                "Triple Integration"
            ]
        }
        UR_Deep_2 = {
            "keyword_name": [
                "Matrix of a linear transformation",
                "Triangles",
                "Event",
                "Roots of a function"
            ],
            "topic_name": [
                "Real Functions of Several Variables",
                "Optimization",
                "Real Functions of a Single Variable",
                "Graph Theory"
            ],
            "subtopic_name": [
                "Double Integration",
                "Triple Integration",
                "Derivatives",
                "Domain, Image and Graphics"
            ]
        }

        UR_Shallow_1 = {
            "keyword_name": ["Cauchy problem"],
            "topic_name": ["Integration", "Discrete Mathematics"],
            "subtopic_name": ["Recursivity"],
            "question_id": [80],
            "id_lect": [2162],
            "answer1": ["The system has no solution."],
            "keyword_id": [139]
        }
        
        UR_Shallow_2 = {
            "newLevel": [2],
            "algorithmLevel": [2],
            "checked": [1.0],
            "keyword_id": [41.0],
            "keyword_name": ["Continuity"],
            "topic_name": ["Discrete Mathematics"],
            "subtopic_name": ["Limits, Continuity, Domain and Image"]
        }

        UR = self.create_flexible_dataframe(UR_Deep_1)
        base_cols = list(UR_Deep_1.keys()) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[20] = (T, UR) # UR_Deep_1
        
        UR = self.create_flexible_dataframe(UR_Deep_2)
        base_cols = list(UR_Deep_2.keys()) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[21] = (T, UR) # UR_Deep_2
        
        UR = self.create_flexible_dataframe(UR_Shallow_1)
        base_cols = list(UR_Shallow_1.keys()) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[22] = (T, UR) # UR_Shallow_1
        
        UR = self.create_flexible_dataframe(UR_Shallow_2)
        base_cols = list(UR_Shallow_2.keys()) + ["Identifiant"]
        T = mathe_df[base_cols].copy()

        self.cases[23] = (T, UR) # UR_Shallow_2






    def load_movielens_case(self, csv_path="/home/slide/faresa/Coverage-Guided-Row-Selection-with-Optimization/movielens-200k.csv", n_values_per_col=4):
        """
        Load MovieLens and generate a random UR from Occupation, Zip-code, and Title.
        'Identifiant' is UserID_MovieID (guaranteed unique per rating).
        """
        df = pd.read_csv(csv_path)
    
        # Create unique identifier
        df["Identifiant"] = df["UserID"].astype(str) + "_" + df["MovieID"].astype(str)
    
        # Use the desired columns
        ur_columns = ["Occupation", "Zip-code", "Title"]
    
        # Build random user request
        ur_dict = self.build_random_ur(df, ur_columns, n_values_per_col)
        UR = self.create_flexible_dataframe(ur_dict)
    
        # Print the UR for inspection
        print("User Request (UR):")
        print(UR)
    
        # Prepare the table T
        base_cols = ur_columns + ["Identifiant"]
        T = df[base_cols].copy()
    
        self.cases[24] = (T, UR)
        
    def load_fixed_movielens_case(self, csv_path="/home/slide/faresa/Coverage-Guided-Row-Selection-with-Optimization/movielens-200k.csv"):
        """
        Load MovieLens and use a fixed User Request (UR) for Occupation, Zip-code, Title.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        df["Identifiant"] = df["UserID"].astype(str) + "_" + df["MovieID"].astype(str)

        UR_Deep_ML = {
            "Occupation": [7, 13, 0, 1],
            "Zip-code": ["11793", "67042", "77459", "97124"],
            "Title": [
                "Swingers (1996)",
                "Very Brady Sequel, A (1996)",
                "Meatballs 4 (1992)",
                "Fiendish Plot of Dr. Fu Manchu, The (1980)"
            ]
        }
        
        UR_Shallow_ML = {
            "Occupation": [13],
            "Zip-code": ["62702"],
            "Title": ["Raise the Red Lantern (1991)"],
            "Genres": ["Drama|Film-Noir"],
            "Rating": [2],
            "Gender": ["M"],
            "Age": [25]
        }


        UR = self.create_flexible_dataframe(UR_Deep_ML)
        base_cols = list(UR_Deep_ML.keys()) + ["Identifiant"]
        T = df[base_cols].copy()

        self.cases[25] = (T, UR)  
        
        UR = self.create_flexible_dataframe(UR_Shallow_ML)
        base_cols = list(UR_Shallow_ML.keys()) + ["Identifiant"]
        T = df[base_cols].copy()

        self.cases[26] = (T, UR) 
        
        
    def get_case(self, case_number):
        """
        Return the tuple (T, UR) for the specified case number.
        Defaults to case 1 if the given case is not found.
        """
        return self.cases.get(case_number, self.cases[case_number])



