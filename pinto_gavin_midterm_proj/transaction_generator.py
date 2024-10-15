import random
import pandas as pd

''' This file generates the transactional database for the datasets. This was ran ONCE to generate the files. '''

# Generates a random transaction of a given length from a given array with no repeats
def generate_random_transaction (array, length):
    # Keep track of usable elements from array
    available_elements = array.copy()
    transaction = []
    # Append to transaction length times
    for i in range(length):
        # Choose random element from available elements and add it to transaction
        random_index = random.randint(0, len(available_elements) - 1)
        transaction.append(available_elements[random_index])
        # Prevent repeats
        available_elements.pop(random_index)
    # Sort final transaction alphabetically
    return sorted(transaction)

# Appends random transactions generated from dataset_file to transaction_file
def append_transactions_csv (num_transactions, dataset_file, transaction_file):
    # Store elements from dataset_file in a list
    df = pd.read_csv(dataset_file)
    elements = df['Element'].tolist()
    transactions = []
    # Append num_transactions transactions
    for i in range(num_transactions):
        # I have decided to make transactions of size 3, 4, or 5
        random_transaction = generate_random_transaction(elements, random.randint(3,5))
        transaction_string = ",".join(random_transaction)
        # Append transaction to transaction_file with i and transaction in ID and Transaction columns
        transactions.append({"ID": i + 1, "Transaction": transaction_string})
    # Convert transactions to df
    transactions_df = pd.DataFrame(transactions)
    # Create file if it doesn't exist and append transactions
    transactions_df.to_csv(transaction_file, mode='a', index=False, header=not pd.io.common.file_exists(transaction_file))

# Generate transaction databases for each company
companies = ["barnes_and_noble", "costco", "gorilla_mind", "rogue_fitness", "vilros"]
for company in companies:
    transaction_filename = "pinto_gavin_midterm_proj/database/transaction_" + company + ".csv"
    dataset_filename = "pinto_gavin_midterm_proj/database/dataset_" + company + ".csv"
    append_transactions_csv(20, dataset_filename, transaction_filename)

