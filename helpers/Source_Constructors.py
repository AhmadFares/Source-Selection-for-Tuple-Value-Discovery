# import pandas as pd
# import numpy as np
# import random
# from itertools import product

# def dataframe_to_ur_dict(df):
#         return {
#             col: set(df[col].dropna().unique())
#             for col in df.columns
#         }
# class SourceConstructor:
#     def __init__(self, T: pd.DataFrame, UR: dict, seed: int = 42):
#         self.T = T.copy()
#         self.UR = UR  # UR is a dict {column: set(values)}
#         self.seed = seed
#         np.random.seed(seed)
#         random.seed(seed)
     
#     def random_split(self, df, n_sources=10):
#         """Randomly split a DataFrame into n sources."""
#         df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
#         return np.array_split(df_shuffled, n_sources)
    
#     def low_penalty_sources(self):
#         """Insert cartesian product rows of UR into T."""
#         columns = list(self.UR.keys())
#         value_lists = [
#             [val for val in self.UR[col] if pd.notna(val)] for col in columns
#         ]
#         cartesian_rows = list(product(*value_lists))
#         print(f"Cartesian product rows: {cartesian_rows}")
#         T_augmented = self.T.copy()

#         for row_values in cartesian_rows:
#             new_row = {col: val for col, val in zip(columns, row_values)}
#             for col in T_augmented.columns:
#                 if col not in new_row:
#                     new_row[col] = f"default_{random.Random(self.seed).randint(1000,9999)}"
#             T_augmented = pd.concat([T_augmented, pd.DataFrame([new_row])], ignore_index=True)

#         return self.random_split(T_augmented, n_sources=10)

#     def group_by_sources(self):
#         """Group table T into sources by unique values of the first attribute in UR."""
#         first_attribute = list(self.UR.keys())[0]
#         sources = []
#         for val in self.T[first_attribute].dropna().unique():
#             group = self.T[self.T[first_attribute] == val].copy()
#             if not group.empty:
#                 sources.append(group)
#         return sources

#     def low_coverage_sources(self, remove_fraction=0.3):
#         """Randomly remove a fraction of UR values from T."""
#         T_mutated = self.T.copy()

#         ur_values_to_remove = {}

#         # Decide which values to remove
#         for col, vals in self.UR.items():
#             vals_list = list(vals)
#             n_remove = max(1, int(len(vals_list) * remove_fraction))  # Remove at least 1 if possible
#             vals_to_remove = random.Random(self.seed).sample(vals_list, n_remove)

#             ur_values_to_remove[col] = vals_to_remove

#         # Now mutate T
#         for col, vals in ur_values_to_remove.items():
#             for val in vals:
#                 random_replacement = f"noise_{random.Random(self.seed).randint(1000,9999)}"
#                 T_mutated.loc[T_mutated[col] == val, col] = random_replacement

#         return self.random_split(T_mutated, n_sources=10)

#     def high_penalty_sources(self):
#         """Introduce high-penalty structure by breaking perfect rows into polluted parts."""
#         T_mutated = self.T.copy()

#         #  Handle both dict and DataFrame UR formats
#         if isinstance(self.UR, pd.DataFrame):
#             ur_columns = list(self.UR.columns)
#             ur_dict = {
#                 col: set(self.UR[col].dropna().unique())
#                 for col in self.UR.columns
#             }
#         else:
#             ur_columns = list(self.UR.keys())
#             ur_dict = self.UR

#         print("UR format:", ur_dict)

#         #  Flatten all UR values
#         ur_values_flat = set(val for vals in ur_dict.values() for val in vals)

#         # ✅ Identify perfect rows
#         perfect_rows = []
#         for idx, row in T_mutated.iterrows():
#             if all(row[col] in ur_dict[col] for col in ur_columns):
#                 perfect_rows.append((idx, row.copy()))

#         #  Identify donor rows (0 UR values)
#         donor_rows = []
#         for idx, row in T_mutated.iterrows():
#             if all(row[col] not in ur_values_flat for col in ur_columns):
#                 donor_rows.append((idx, row.copy()))

#         # ✅ Break perfect rows into polluted rows
#         used_donors = set()
#         for perfect_idx, perfect_row in perfect_rows:
#             # Find an unused donor
#             donor = None
#             for donor_idx, donor_row in donor_rows:
#                 if donor_idx not in used_donors:
#                     donor = donor_row
#                     used_donors.add(donor_idx)
#                     break
#             if donor is None:
#                 break  # Not enough donors

#             # Create two polluted rows
#             row1 = perfect_row.copy()
#             row2 = perfect_row.copy()
#             row1[ur_columns[1]] = donor[ur_columns[1]]
#             row2[ur_columns[0]] = donor[ur_columns[0]]
            

#             # Replace perfect row with polluted ones
#             T_mutated = T_mutated.drop(index=perfect_idx)
#             T_mutated = pd.concat([T_mutated, pd.DataFrame([row1, row2])], ignore_index=True)

#         return self.random_split(T_mutated, n_sources=10)


import pandas as pd
import numpy as np
import random
from itertools import product

def dataframe_to_ur_dict(df):
    """Convert a DataFrame to a user request dictionary."""
    return {col: set(df[col].dropna().unique()) for col in df.columns}

class SourceConstructor:
    def __init__(self, T: pd.DataFrame, UR: dict, seed: int = 42):
        """
        T: base table (DataFrame)
        UR: user request dict {column: set(values)}
        seed: seed for reproducibility
        """
        self.T = T.copy()
        self.UR = UR  # dict {column: set(values)}
        self.seed = seed
        self.rng = random.Random(seed)  # Instance-level RNG
        self.np_rng = np.random.RandomState(seed)  # Instance-level NumPy RNG

    def random_split(self, df, n_sources=10):
        """Randomly split a DataFrame into n sources."""
        df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return np.array_split(df_shuffled, n_sources)

    def low_penalty_sources(self):
        """Insert cartesian product rows of UR into T and split into sources."""
        columns = list(self.UR.keys())
        value_lists = [[val for val in self.UR[col] if pd.notna(val)] for col in columns]
        cartesian_rows = list(product(*value_lists))
        print(f"Cartesian product rows: {cartesian_rows}")
        T_augmented = self.T.copy()

        for row_values in cartesian_rows:
            new_row = {col: val for col, val in zip(columns, row_values)}
            for col in T_augmented.columns:
                if col not in new_row:
                    new_row[col] = f"default_{self.rng.randint(1000,9999)}"  # Use self.rng
            T_augmented = pd.concat([T_augmented, pd.DataFrame([new_row])], ignore_index=True)

        return self.random_split(T_augmented, n_sources=10)

    def group_by_sources(self):
        """Group table T into sources by unique values of the first attribute in UR."""
        first_attribute = list(self.UR.keys())[0]
        sources = []
        for val in self.T[first_attribute].dropna().unique():
            group = self.T[self.T[first_attribute] == val].copy()
            if not group.empty:
                sources.append(group)
        return sources

    def low_coverage_sources(self, remove_fraction=0.3):
        """Randomly remove a fraction of UR values from T, mutate, and split."""
        T_mutated = self.T.copy()
        ur_values_to_remove = {}

        for col, vals in self.UR.items():
            vals_list = list(vals)
            n_remove = max(1, int(len(vals_list) * remove_fraction))  # Remove at least 1 if possible
            if len(vals_list) >= n_remove:
                vals_to_remove = self.rng.sample(vals_list, n_remove)  # Use self.rng
            else:
                vals_to_remove = vals_list
            ur_values_to_remove[col] = vals_to_remove

        for col, vals in ur_values_to_remove.items():
            for val in vals:
                random_replacement = f"noise_{self.rng.randint(1000,9999)}"  # Use self.rng
                T_mutated.loc[T_mutated[col] == val, col] = random_replacement

        return self.random_split(T_mutated, n_sources=10)

    def high_penalty_sources(self):
        """Break perfect rows into polluted parts using donor rows, then split."""
        T_mutated = self.T.copy()

        #  Handle both dict and DataFrame UR formats
        if isinstance(self.UR, pd.DataFrame):
            ur_columns = list(self.UR.columns)
            ur_dict = {col: set(self.UR[col].dropna().unique()) for col in self.UR.columns}
        else:
            ur_columns = list(self.UR.keys())
            ur_dict = self.UR

        print("UR format:", ur_dict)

        # Flatten all UR values
        ur_values_flat = set(val for vals in ur_dict.values() for val in vals)

        # Identify perfect rows (all values match UR)
        perfect_rows = []
        for idx, row in T_mutated.iterrows():
            if all(row[col] in ur_dict[col] for col in ur_columns):
                perfect_rows.append((idx, row.copy()))

        # Identify donor rows (no values from UR)
        donor_rows = []
        for idx, row in T_mutated.iterrows():
            if all(row[col] not in ur_values_flat for col in ur_columns):
                donor_rows.append((idx, row.copy()))

        # Break perfect rows into polluted rows using donors
        used_donors = set()
        for perfect_idx, perfect_row in perfect_rows:
            # Find an unused donor
            donor = None
            for donor_idx, donor_row in donor_rows:
                if donor_idx not in used_donors:
                    donor = donor_row
                    used_donors.add(donor_idx)
                    break
            if donor is None:
                break  # Not enough donors

            # Create two polluted rows
            row1 = perfect_row.copy()
            row2 = perfect_row.copy()
            if len(ur_columns) > 1:
                row1[ur_columns[1]] = donor[ur_columns[1]]
                row2[ur_columns[0]] = donor[ur_columns[0]]
            else:
                # If only one column in UR, just pollute it
                row1[ur_columns[0]] = donor[ur_columns[0]]

            # Replace perfect row with polluted ones
            T_mutated = T_mutated.drop(index=perfect_idx)
            T_mutated = pd.concat([T_mutated, pd.DataFrame([row1, row2])], ignore_index=True)

        return self.random_split(T_mutated, n_sources=10)
