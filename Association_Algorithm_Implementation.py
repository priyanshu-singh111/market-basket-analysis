import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load dataset
df = pd.read_csv("study_habits_dataset.csv")

# Convert dataset rows to list of transactions
transactions = df.values.tolist()

# One-hot encoding using TransactionEncoder
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.15, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Sort rules by Lift (interestingness measure)
rules_sorted = rules.sort_values(by="lift", ascending=False)

# Show top rules
print("Top Association Rules:\n")
print(rules_sorted[['antecedents','consequents','support','confidence','lift']].head(15))
