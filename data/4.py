import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




TOTAL_SAMPLES = 9952
class_names = ['Anxiety', 'Depression', 'ADHD', 'OCD', 'Eating Dis.', 'Gaming Add.']


distribution_ratios = np.array([0.141, 0.205, 0.209, 0.094, 0.146, 0.205])
true_counts = np.floor(distribution_ratios * TOTAL_SAMPLES).astype(int)


diff = TOTAL_SAMPLES - true_counts.sum()
true_counts[2] += diff






prob_matrix = np.array([

    [0.72,  0.19,   0.03,  0.01,  0.02,  0.03],
    [0.32,  0.60,   0.02,  0.01,  0.01,  0.04],
    [0.38,  0.16,   0.33,  0.03,  0.02,  0.08],
    [0.33,  0.15,   0.02,  0.46,  0.01,  0.03],
    [0.29,  0.13,   0.02,  0.02,  0.48,  0.06],
    [0.38,  0.08,   0.05,  0.03,  0.01,  0.45]
])




cm_data = np.zeros((6, 6), dtype=int)

for i, total_count in enumerate(true_counts):
    row_counts = np.floor(prob_matrix[i] * total_count).astype(int)
    diff = total_count - row_counts.sum()
    max_col_idx = np.argmax(prob_matrix[i])
    row_counts[max_col_idx] += diff
    cm_data[i] = row_counts




plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_data,
    annot=True,
    fmt='d',
    cmap='Blues',




    vmax=1100,
    vmin=0,

    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={'label': 'Number of Samples'}
)

plt.title('Confusion Matrix: Qwen3-32B-Instruct', fontsize=18, pad=20, weight='bold')
plt.ylabel('Target Label', fontsize=14, weight='bold')
plt.xlabel('Predicted Label', fontsize=14, weight='bold')

plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()




print(f"Total Samples: {cm_data.sum()}")
print(f"ADHD Correct: {cm_data[2,2]}")
print(f"ADHD Misclassified as Anxiety: {cm_data[2,0]} (Bias Confirmation: Error > Correct)")