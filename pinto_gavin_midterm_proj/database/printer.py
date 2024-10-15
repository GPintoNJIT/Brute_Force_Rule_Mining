import pandas as pd

files = {"barnes_and_noble" : "Barnes and Noble", "costco": "Costco", "gorilla_mind": "Gorilla Mind", "rogue_fitness" : "Rogue Fitness", "vilros" : "Vilros"}

for file in files:
    df = pd.read_csv("pinto_gavin_midterm_proj/database/transaction_" + file + ".csv")

    # Print DataFrame in a nicely formatted table
    print("---", files[file], "---")
    print(df.to_string(index=False))
    print()
    print()