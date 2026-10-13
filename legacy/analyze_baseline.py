import pandas as pd

print("Loading baseline results...")

df = pd.read_csv("baseline_results.csv")

print("Results loaded!")

print("\nTotal examples:", len(df))
print("Correct:", df["correct"].sum())
print("Incorrect:", (~df["correct"]).sum())

# Select incorrect predictions
wrong = df[df["correct"] == False]

print("\nNumber of incorrect predictions:", len(wrong))

print("\n========== FIRST 20 INCORRECT PREDICTIONS ==========\n")

for _, row in wrong.head(20).iterrows():

    print("----------------------------------------")
    print("Example:", row["example"])
    print("Question:", row["question"])
    print("Ground truth:", row["ground_truth"])
    print("Prediction:", row["prediction"])

print("\nAnalysis complete!")