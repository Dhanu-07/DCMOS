import scipy.io
import pandas as pd

# Load the .mat file
mat_file = '/home/arunp24/python_lsgs/examples/ckt10k.mat'  # Replace with your .mat file path
mat_data = scipy.io.loadmat(mat_file)

# Extract data from the .mat file
# Assuming the data is stored in a variable like 'data' inside the .mat file
# The structure of the .mat file can vary, so you need to check the keys in mat_data to identify the correct one
print("Keys in .mat file:", mat_data.keys())

# Example: If the data you want to convert is under the key 'data'
a = mat_data['data']  # Replace 'data' with the correct key if necessary
dmin = mat_data['a']
F = mat_data['F']
g = mat_data['g']
# Convert the data to a pandas DataFrame (if it's 2D or can be converted into a DataFrame)
df = pd.DataFrame(data)

# Save the DataFrame to a .csv file
csv_file = '/home/arunp24/python_lsgs/csv/ckt10k.csv'  # Replace with your desired output file name
df.to_csv(csv_file, index=False)

print(f"File converted and saved as {csv_file}")

