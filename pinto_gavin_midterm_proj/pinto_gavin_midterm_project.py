#!/usr/bin/env python
# coding: utf-8

# **Part 1:**
# The datasets were manually created, and the transactions were created using transaction_generator.py, and stored in database. Below are necessary imports and helper functions to extract information from a file in the form of a list.

# In[12]:


# pip install mlxtend pandas


# In[13]:


import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth
import time

def get_transactions_as_array (filename):
    df = pd.read_csv(filename)
    transactions = df['Transaction'].tolist()
    return transactions

def get_dataset_as_array (filename):
    df = pd.read_csv(filename)
    elements = df['Element'].tolist()
    return elements


# **Part 2:** Implement the brute force method to generate the frequent itemsets and their association rules.

# Below are helper functions to aid with the brute force algorithm.

# In[14]:


# Returns number of occurrences of itemset in transactions
def get_frequency (itemset, transactions):
    frequency = 0
    for transaction in transactions:
        transaction_items = transaction.split(",")
        if set(itemset) <= set(transaction_items):
            frequency += 1
    return frequency

# Returns confidence of rule according to transactions
def get_confidence (rule, transactions):
    # formula: freq(combined) / freq(left)
    combined = rule[0] + rule[1]
    left = rule[0]
    return (1.0 * get_frequency(combined, transactions)) / (1.0 * get_frequency(left, transactions))
    
# Returns all itemsets of size n from dataset in the form 
# list[list[], list[]] with list[0] as the itemsets and list[1] initialized to 0
def get_k_itemsets (dataset, k):
    k_itemsets = []
    frequencies = []
    for i in range(len(dataset) - k + 1):
        element_itemset = [dataset[i]]
        if (k > 1):
            for second in range(i + 1, len(dataset) - k + 2):
                element_itemset = [dataset[i]]
                for j in range(second, second + k - 1):
                    element_itemset.append(dataset[j])
                k_itemsets.append(element_itemset)
                frequencies.append(0)
        else:
            k_itemsets.append(element_itemset)
            frequencies.append(0)
    return [k_itemsets, frequencies] 

# Returns all possible rules from itemset in the form [[[],[]], ... ]
def generate_rules(itemset):
    n = len(itemset)
    rules = []
    for r in range(1, n):
        for left in combinations(itemset, r):
            left = list(left)
            right = [item for item in itemset if item not in left]
            rules.append([left, right])
    return rules


# Below are the functions that generate the frequent itemsets and their association rules.

# In[15]:


# Returns all frequent itemsets from transactions according to min_support
def brute_force_frequent_itemsets (dataset, transactions, min_support):
    # Find all frequent k-itemsets
    frequent_itemsets = []
    frequent_itemsets_frequencies = []
    k = 1
    while True:
        # Generate k-itemsets
        k_itemset_info = get_k_itemsets(dataset, k)
        k_itemsets = k_itemset_info[0]
        k_itemsets_frequencies = k_itemset_info[1]
        frequent_itemsets_found = 0
        # Update frequencies of each itemset in transactions
        for i in range(len(k_itemsets)):
            k_itemsets_frequencies[i] = get_frequency(k_itemsets[i], transactions)
        # Add frequent itemsets
        for i in range(len(k_itemsets)):
            supp = k_itemsets_frequencies[i] / 20
            if supp >= min_support:
                frequent_itemsets.append(k_itemsets[i])
                frequent_itemsets_frequencies.append(k_itemsets_frequencies[i])
                frequent_itemsets_found += 1
        # Terminate algorithm if no frequent k-itemsets found
        if frequent_itemsets_found == 0:
            break
        k += 1
    return [frequent_itemsets, frequent_itemsets_frequencies]  

# Returns association rules based on min_confidence
def brute_force_association_rules (dataset, transactions, min_support, min_confidence):
    association_rules = []
    # Get frequent itemsets
    frequent_itemsets = brute_force_frequent_itemsets(dataset, transactions, min_support)[0]
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            # Get all possible rules of itemset
            rules = generate_rules(itemset)
            # Append all rules above min_confidence to result
            for rule in rules:
                conf = get_confidence(rule, transactions)
                if conf >= min_confidence:
                    association_rules.append([rule, conf])
    return association_rules


# **Part 3:**
# Use an existing Apriori implementation from Python libraries/packages to
# verify the results from your brute force algorithm implementation.
# Use Python existing package for fpgrowth (as known as fp-tree algorithm)
# to generate the items and rules.
# Compare the results from your brute-force, Apriori, and FP-Tree/Growth.

# In[16]:


# Prompt the user for their company choice
print("Welcome! Please select the company you'd like to analyze.")
print()
print("1. Barnes & Noble")
print("2. Costco")
print("3. Gorilla Mind")
print("4. Rogue Fitness")
print("5. Vilros")
user_choice = int(input("Please enter the number next to your choice: "))

# Prompt the user for support and confidence
print()
support = float(input("Please enter the minimum support (as a percentage): ")) / 100.0
confidence = float(input("Please enter the minimum confidence (as a percentage): ")) / 100.0
print()

# Process which files to analyze
companies = ["barnes_and_noble", "costco", "gorilla_mind", "rogue_fitness", "vilros"]
company = companies[user_choice - 1]
transaction_file = "pinto_gavin_midterm_proj/database/transaction_" + company + ".csv"
dataset_file = "pinto_gavin_midterm_proj/database/dataset_" + company + ".csv"

dataset = get_dataset_as_array(dataset_file)
transactions = get_transactions_as_array(transaction_file)

# Preprocess data for library implementations
transactions_proper = []
for transaction in transactions:
    transactions_proper.append(transaction.split(","))
df = pd.DataFrame(pd.Series(transactions_proper).apply(lambda x: pd.Series(1, index=x)).fillna(0))

# Start timing for Apriori
start_time = time.time()

# Use the existing Apriori implementation
frequent_itemsets_apriori = apriori(df.astype('bool'), min_support=support, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=confidence)

# End timing for Apriori
end_time = time.time()

# Print Apriori results
print("---APRIORI ALGORITHM---")
print()
print("* Frequent Itemsets (Apriori Algorithm):")
print(frequent_itemsets_apriori)
print("\n* Association Rules (Apriori Algorithm):")
print(rules_apriori[['antecedents', 'consequents', 'confidence']])
elapsed_time = end_time - start_time
print(f"\n* Time taken for Apriori Algorithm: {elapsed_time:.6f} seconds")
print()

# Start timing for FP-Growth
start_time = time.time()

# Use the existing package for FP-Growth
frequent_itemsets_fp = fpgrowth(df.astype('bool'), min_support=support, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=confidence)

# End timing for FP-Growth
end_time = time.time()

# Print FP-Growth results
print("---FP-GROWTH ALGORITHM---")
print()
print("* Frequent Itemsets (FP-Growth):")
print(frequent_itemsets_fp)
print("\n* Association Rules (FP-Growth):")
print(rules_fp[['antecedents', 'consequents', 'confidence']])
elapsed_time = end_time - start_time
print(f"\n* Time taken for FP-Growth: {elapsed_time:.6f} seconds")
print()

# Start timing for brute
start_time = time.time()

# Compare the results with the brute force algorithm
frequent_itemsets_brute = brute_force_frequent_itemsets(dataset, transactions, support)[0]
rules_brute = brute_force_association_rules(dataset, transactions, support, confidence)

# End timing for brute
end_time = time.time()

# Print Brute force results
print("---BRUTE FORCE ALGORITHM---")
print()
print("* Frequent Itemsets (Brute Force):")
for item in frequent_itemsets_brute:
    print(item)
print("\n* Association Rules (Brute Force):")
for item in rules_brute:
    rule = item[0]
    conf = item[1]
    print(rule[0], "->", rule[1], f"confidence: {conf:.6f}")
elapsed_time = end_time - start_time
print(f"\n* Time taken for Brute Force Algorithm: {elapsed_time:.6f} seconds")


# In[ ]:




