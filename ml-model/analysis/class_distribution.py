import os
import pandas as pd

DATA_DIR = "data/raw_filtered"

records = []

for label in os.listdir(DATA_DIR):

    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    count = len(os.listdir(label_path))

    records.append({
        "class": label,
        "samples": count
    })

df = pd.DataFrame(records)

df = df.sort_values("samples", ascending=False)

print("\nTop classes by sample count:\n")
print(df.head(20))

print("\nDataset statistics:\n")
print("Total classes:", len(df))
print("Total samples:", df["samples"].sum())
print("Average samples per class:", df["samples"].mean())

df.to_csv("class_distribution.csv", index=False)