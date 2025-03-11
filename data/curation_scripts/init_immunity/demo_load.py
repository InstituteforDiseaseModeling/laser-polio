import pandas as pd

# Define the file path (adjust as needed)
h5_file_path = "data/init_immunity_0.5coverage_january.h5"  # Change for other coverages

df = pd.read_hdf(h5_file_path, key="immunity")
df.head()


# # Open the HDF5 file
# with h5py.File(h5_file_path, "r") as h5f:
#     # Check available datasets
#     print("Datasets in HDF5 file:", list(h5f.keys()))

#     # Load numeric data
#     if "numeric_data" in h5f:
#         numeric_data = np.array(h5f["numeric_data"])  # Convert to NumPy array
#         print("Numeric data shape:", numeric_data.shape)
#     else:
#         numeric_data = None

#     # Load character data if present
#     if "char_data" in h5f:
#         char_data = np.array(h5f["char_data"], dtype=str)  # Ensure string format
#         print("Character data shape:", char_data.shape)
#     else:
#         char_data = None

# # Convert to pandas DataFrame (if both datasets exist)
# if numeric_data is not None and char_data is not None:
#     # Combine numeric and character data
#     df = pd.DataFrame(numeric_data)
#     char_df = pd.DataFrame(char_data)

#     # If column names are stored, assign them manually (if known)
#     # Example (assuming character data contains column names):
#     if char_data.shape[1] == numeric_data.shape[1]:
#         df.columns = char_df.iloc[0]  # Use first row as column names
#         char_df = char_df.iloc[1:]  # Remove the first row

#     print(df.head())  # Preview data
# else:
#     df = None  # Handle missing data scenario

# print("Data successfully loaded!")
