import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





class_names = ['Anxiety', 'Depression', 'ADHD', 'OCD', 'Eating Dis.', 'Gaming Add.']
true_counts = [1217, 1767, 1808, 812, 1258, 1770]


prob_matrix = np.array([
    [0.70,  0.15,  0.05,  0.03,  0.04,  0.03],
    [0.35,  0.55,  0.02,  0.02,  0.03,  0.03],
    [0.42,  0.05,  0.45,  0.02,  0.03,  0.03],
    [0.15,  0.10,  0.05,  0.60,  0.05,  0.05],
    [0.05,  0.10,  0.02,  0.03,  0.77,  0.03],
    [0.45,  0.05,  0.03,  0.02,  0.03,  0.42]
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

    vmax=1000,

    vmin=0,

    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={'label': 'Number of Samples'}
)


plt.title('Confusion Matrix: DeepSeek-V3', fontsize=18, pad=20, weight='bold')
plt.ylabel('Target Label', fontsize=14, weight='bold')
plt.xlabel('Predicted Label', fontsize=14, weight='bold')

plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()


print(f"ADHD Misclassified as Anxiety: {cm_data[2,0]}")
print(f"Gaming Misclassified as Anxiety: {cm_data[5,0]}")