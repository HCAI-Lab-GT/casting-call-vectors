import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("scratch.csv")

# Calculate percentage of True values for each column
base_pct = (df['base'].sum() / len(df)) * 100
sarcastic_pct = (df['sarcastic'].sum() / len(df)) * 100

# Create bar graph
plt.figure(figsize=(8, 6))
plt.bar(['Base', 'Sarcastic'], [base_pct, sarcastic_pct])
plt.ylabel('Percentage (%)')
plt.title('Percentage of True Values')
plt.ylim(0, 100)

# Add percentage labels on bars
for i, v in enumerate([base_pct, sarcastic_pct]):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center')

plt.show()
