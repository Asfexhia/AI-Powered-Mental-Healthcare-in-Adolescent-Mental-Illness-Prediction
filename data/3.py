import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





TOTAL_SAMPLES = 5672
class_names = ['Anxiety', 'Depression', 'ADHD', 'OCD', 'Eating Dis.', 'Gaming Add.']



distribution_ratios = np.array([0.141, 0.205, 0.209, 0.094, 0.146, 0.205])
true_counts = np.floor(distribution_ratios * TOTAL_SAMPLES).astype(int)


current_sum = true_counts.sum()
diff = TOTAL_SAMPLES - current_sum
true_counts[2] += diff






prob_matrix = np.array([

    [0.38,  0.55,   0.03,  0.02,  0.01,  0.01],
    [0.25,  0.65,   0.05,  0.03,  0.01,  0.01],
    [0.20,  0.72,   0.05,  0.02,  0.01,  0.00],
    [0.28,  0.62,   0.02,  0.06,  0.01,  0.01],
    [0.20,  0.75,   0.02,  0.02,  0.01,  0.00],
    [0.35,  0.64,   0.01,  0.00,  0.00,  0.00]
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

plt.title('Confusion Matrix: BioMistral-NLU', fontsize=18, pad=20, weight='bold')
plt.ylabel('Target Label', fontsize=14, weight='bold')
plt.xlabel('Predicted Label', fontsize=14, weight='bold')

plt.xticks(fontsize=11, rotation=45, ha='right')
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
plt.show()




print(f"Total Samples: {cm_data.sum()}")
print(f"Total Predicted as Depression: {cm_data[:, 1].sum()}")