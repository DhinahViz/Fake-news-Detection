import pandas as pd

# Step 1: Load both CSVs
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Step 2: Add labels
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# Step 3: Combine them
df = pd.concat([fake_df, real_df], axis=0)

# Step 4: Keep only necessary columns
df = df[["text", "label"]]

# Step 5: Save to a new CSV
df.to_csv("data/news.csv", index=False)

print("✅ Combined dataset saved as 'data/news.csv'")
print(df.head())
