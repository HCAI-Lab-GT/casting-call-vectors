import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the boardgameqa data
df = pd.read_csv('boardgameqa.csv')

# Calculate accuracy for control and sarcastic
control_accuracy = df['control'].mean() * 100  # Convert to percentage
sarcastic_accuracy = df['sarcastic'].mean() * 100

print(f"Dataset size: {len(df)} examples")
print(f"Control Accuracy: {control_accuracy:.2f}%")
print(f"Sarcastic Accuracy: {sarcastic_accuracy:.2f}%")
print(f"Difference: {sarcastic_accuracy - control_accuracy:.2f}%")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart comparing accuracies
ax1 = axes[0]
conditions = ['Control', 'Sarcastic']
accuracies = [control_accuracy, sarcastic_accuracy]
colors = ['#3498db', '#e74c3c']

bars = ax1.bar(conditions, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Comparison: Control vs Sarcastic Persona', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Confusion/agreement matrix
ax2 = axes[1]
control_correct = df['control'].sum()
control_incorrect = len(df) - control_correct
sarcastic_correct = df['sarcastic'].sum()
sarcastic_incorrect = len(df) - sarcastic_correct

# Calculate agreement
both_correct = ((df['control'] == True) & (df['sarcastic'] == True)).sum()
both_incorrect = ((df['control'] == False) & (df['sarcastic'] == False)).sum()
control_only = ((df['control'] == True) & (df['sarcastic'] == False)).sum()
sarcastic_only = ((df['control'] == False) & (df['sarcastic'] == True)).sum()

agreement_data = [[both_correct, control_only], 
                  [sarcastic_only, both_incorrect]]
agreement_labels = [['Both Correct\n' + str(both_correct), 'Control Only\n' + str(control_only)],
                    ['Sarcastic Only\n' + str(sarcastic_only), 'Both Wrong\n' + str(both_incorrect)]]

# Create heatmap
sns.heatmap(agreement_data, annot=agreement_labels, fmt='', cmap='YlOrRd', 
            cbar_kws={'label': 'Count'}, ax=ax2, linewidths=2, linecolor='black')
ax2.set_xlabel('Control', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sarcastic', fontsize=12, fontweight='bold')
ax2.set_title('Agreement Matrix', fontsize=14, fontweight='bold')
ax2.set_xticklabels(['Correct', 'Incorrect'])
ax2.set_yticklabels(['Correct', 'Incorrect'])

agreement_rate = (both_correct + both_incorrect) / len(df) * 100
print(f"\nAgreement rate: {agreement_rate:.2f}%")
print(f"Both correct: {both_correct}")
print(f"Both incorrect: {both_incorrect}")
print(f"Control correct, Sarcastic wrong: {control_only}")
print(f"Sarcastic correct, Control wrong: {sarcastic_only}")

# Print examples where they disagreed or both were wrong
print("\n" + "="*80)
print("DETAILED ERROR ANALYSIS")
print("="*80)

# Both got it wrong
both_wrong_indices = df[(df['control'] == False) & (df['sarcastic'] == False)].index
if len(both_wrong_indices) > 0:
    print(f"\n\n{'='*80}")
    print(f"BOTH CONTROL AND SARCASTIC GOT WRONG ({len(both_wrong_indices)} cases):")
    print(f"{'='*80}")
    for idx in both_wrong_indices:
        print(f"\n--- Example {idx} ---")
        example_text = df.loc[idx, 'example']
        print(f"Question: {example_text}")
        print(f"Control: WRONG | Sarcastic: WRONG")

# Control correct, Sarcastic wrong
control_only_indices = df[(df['control'] == True) & (df['sarcastic'] == False)].index
if len(control_only_indices) > 0:
    print(f"\n\n{'='*80}")
    print(f"CONTROL CORRECT, SARCASTIC WRONG ({len(control_only_indices)} cases):")
    print(f"{'='*80}")
    for idx in control_only_indices:
        print(f"\n--- Example {idx} ---")
        example_text = df.loc[idx, 'example']
        print(f"Question: {example_text}")
        print(f"Control: CORRECT | Sarcastic: WRONG")

# Sarcastic correct, Control wrong
sarcastic_only_indices = df[(df['control'] == False) & (df['sarcastic'] == True)].index
if len(sarcastic_only_indices) > 0:
    print(f"\n\n{'='*80}")
    print(f"SARCASTIC CORRECT, CONTROL WRONG ({len(sarcastic_only_indices)} cases):")
    print(f"{'='*80}")
    for idx in sarcastic_only_indices:
        print(f"\n--- Example {idx} ---")
        example_text = df.loc[idx, 'example']
        print(f"Question: {example_text}")
        print(f"Control: WRONG | Sarcastic: CORRECT")

print(f"\n\n{'='*80}")
print("END OF ERROR ANALYSIS")
print(f"{'='*80}\n")

# Test hypothesis: authority-related concepts affect sarcastic performance
print("\n" + "="*80)
print("AUTHORITY-RELATED KEYWORD ANALYSIS")
print("="*80)

# Define authority-related keywords
authority_keywords = [
    'respect', 'respects', 'respected',
    'authority', 'authorities',
    'obey', 'obeys', 'obeyed',
    'command', 'commands', 'commanded',
    'order', 'orders', 'ordered',
    'control', 'controls', 'controlled',
    'dominance', 'dominant', 'dominate',
    'submit', 'submits', 'submission'
]

# Check which examples contain authority keywords
df['has_authority_keyword'] = df['example'].str.lower().apply(
    lambda x: any(keyword in x for keyword in authority_keywords)
)

# Count examples with/without authority keywords
with_authority = df[df['has_authority_keyword'] == True]
without_authority = df[df['has_authority_keyword'] == False]

print(f"\nTotal examples: {len(df)}")
print(f"Examples WITH authority keywords: {len(with_authority)}")
print(f"Examples WITHOUT authority keywords: {len(without_authority)}")

# Calculate accuracy for each group
if len(with_authority) > 0:
    control_auth_acc = with_authority['control'].mean() * 100
    sarcastic_auth_acc = with_authority['sarcastic'].mean() * 100
    print(f"\n--- WITH Authority Keywords ({len(with_authority)} examples) ---")
    print(f"Control accuracy: {control_auth_acc:.2f}%")
    print(f"Sarcastic accuracy: {sarcastic_auth_acc:.2f}%")
    print(f"Difference (Sarcastic - Control): {sarcastic_auth_acc - control_auth_acc:.2f}%")

if len(without_authority) > 0:
    control_no_auth_acc = without_authority['control'].mean() * 100
    sarcastic_no_auth_acc = without_authority['sarcastic'].mean() * 100
    print(f"\n--- WITHOUT Authority Keywords ({len(without_authority)} examples) ---")
    print(f"Control accuracy: {control_no_auth_acc:.2f}%")
    print(f"Sarcastic accuracy: {sarcastic_no_auth_acc:.2f}%")
    print(f"Difference (Sarcastic - Control): {sarcastic_no_auth_acc - control_no_auth_acc:.2f}%")

# Analyze where sarcastic got wrong but control got right - check for authority keywords
sarcastic_worse = df[(df['control'] == True) & (df['sarcastic'] == False)]
if len(sarcastic_worse) > 0:
    sarcastic_worse_with_auth = sarcastic_worse[sarcastic_worse['has_authority_keyword'] == True]
    print(f"\n--- Cases where Sarcastic WRONG, Control CORRECT ---")
    print(f"Total: {len(sarcastic_worse)}")
    print(f"With authority keywords: {len(sarcastic_worse_with_auth)} ({len(sarcastic_worse_with_auth)/len(sarcastic_worse)*100:.1f}%)")
    print(f"Without authority keywords: {len(sarcastic_worse) - len(sarcastic_worse_with_auth)} ({(len(sarcastic_worse) - len(sarcastic_worse_with_auth))/len(sarcastic_worse)*100:.1f}%)")
    
    if len(sarcastic_worse_with_auth) > 0:
        print(f"\nExamples where Sarcastic got wrong on authority-related questions:")
        for idx in sarcastic_worse_with_auth.index:
            example = df.loc[idx, 'example']
            # Find which keywords appear
            found_keywords = [kw for kw in authority_keywords if kw in example.lower()]
            print(f"\n  Example {idx}:")
            print(f"    Keywords found: {', '.join(found_keywords)}")
            print(f"    Question preview: {example[:300]}...")

# Create additional visualization comparing performance on authority vs non-authority questions
if len(with_authority) > 0 and len(without_authority) > 0:
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = range(4)
    accuracies_by_category = [
        control_auth_acc, sarcastic_auth_acc,
        control_no_auth_acc, sarcastic_no_auth_acc
    ]
    labels = ['Control\n(Authority)', 'Sarcastic\n(Authority)', 
              'Control\n(No Authority)', 'Sarcastic\n(No Authority)']
    colors_cat = ['#3498db', '#e74c3c', '#3498db', '#e74c3c']
    alphas = [0.9, 0.9, 0.5, 0.5]
    
    # Create bars individually with different alphas
    bars = []
    for i, (pos, acc, color, alpha) in enumerate(zip(x, accuracies_by_category, colors_cat, alphas)):
        bar = ax.bar(pos, acc, color=color, alpha=alpha, edgecolor='black', linewidth=1.5)
        bars.append(bar)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance on Authority vs Non-Authority Questions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (pos, acc) in enumerate(zip(x, accuracies_by_category)):
        ax.text(pos, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('authority_keyword_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n\nAuthority analysis plot saved as 'authority_keyword_analysis.png'")

print("\n" + "="*80 + "\n")

plt.tight_layout()
plt.savefig('control_vs_sarcastic_accuracy.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'control_vs_sarcastic_accuracy.png'")
plt.show()
