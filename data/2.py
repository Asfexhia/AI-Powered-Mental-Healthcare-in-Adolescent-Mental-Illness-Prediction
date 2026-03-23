import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




TOTAL_SAMPLES = 9758
class_names = ['Anxiety', 'Depression', 'ADHD', 'OCD', 'Eating Dis.', 'Gaming Add.']



raw_distribution = np.array([1332, 1935, 1994, 854, 1313, 2273])
distribution_ratios = raw_distribution / raw_distribution.sum()


true_counts = np.floor(distribution_ratios * TOTAL_SAMPLES).astype(int)

true_counts[-1] += TOTAL_SAMPLES - true_counts.sum()





prob_matrix = np.array([

    [0.60,  0.34,   0.01,  0.03,  0.01,  0.01],
    [0.12,  0.82,   0.01,  0.01,  0.01,  0.03],
    [0.22,  0.32,   0.34,  0.08,  0.02,  0.02],
    [0.22,  0.24,   0.01,  0.50,  0.01,  0.02],
    [0.10,  0.24,   0.01,  0.05,  0.58,  0.02],
    [0.14,  0.30,   0.06,  0.10,  0.01,  0.39]
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


plt.title('Confusion Matrix: Gemma3-27B', fontsize=18, pad=20, weight='bold')
plt.ylabel('Target Label', fontsize=14, weight='bold')
plt.xlabel('Predicted Label', fontsize=14, weight='bold')

plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()




print(f"Total Samples Generated: {cm_data.sum()}")
print(f"Depression Column Sum (The 'Sink'): {cm_data[:, 1].sum()}")
print(f"ADHD Misclassified as Depression: {cm_data[2, 1]}")
print(f"Gaming Misclassified as Depression: {cm_data[5, 1]}")