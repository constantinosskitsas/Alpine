import pandas as pd

# Read the file
filename = "raw_data/douban_ground_True.txt"  # Change this to your filename
df = pd.read_csv(filename, delim_whitespace=True, header=None, names=["Col1", "Col2"])

# Sort by the second column
df_sorted = df.sort_values(by="Col2")

# Check if the second column is a sequence from 1 to N
N = len(df_sorted)
if list(df_sorted["Col2"]) == list(range(0, N )):
    print(df_sorted)
    print("The second column is a sequence from 1 to N.")
else:
    print("The second column is NOT a sequence from 1 to N.")

# Save only the first column
df_sorted["Col1"].to_csv("data3_/douban/10/nodes.txt", index=False, header=False)
print("Processed file saved as output.txt")