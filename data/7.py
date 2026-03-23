import matplotlib.pyplot as plt
import numpy as np

models = ['DeepSeek-V3', 'Gemma3-27B', 'Qwen3-32B', 'Llama4-Maverick', 'BioMistral-NLU']
primary_acc = np.array([56.2, 53.1, 50.0, 68.0, 20.4])
latent_recall = np.array([48.7, 35.2, 42.1, 54.3, 8.5])
adhd_recovery = np.array([76.4, 41.5, 68.9, 72.1, 5.2])

error_rate = 100 - primary_acc
recovered_gain = error_rate * (latent_recall / 100)
total_potential_acc = primary_acc + recovered_gain

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 8))

rects1 = ax.bar(x - 1.5*width, primary_acc, width, label='Primary Accuracy', color='#8ECFC9', alpha=0.9)
rects2 = ax.bar(x - 0.5*width, latent_recall, width, label='Latent Recall (on Errors)', color='#FFBE7A', alpha=0.9)
rects3 = ax.bar(x + 0.5*width, adhd_recovery, width, label='ADHD→Anxiety Recovery', color='#82B0D2', alpha=0.9)

rects4 = ax.bar(x + 1.5*width, total_potential_acc, width, label='Total Potential Accuracy', color='#FA7F6F', alpha=0.9)


ax.set_ylabel('Percentage (%)', fontsize=14, weight='bold')
ax.set_title('Evaluation of Latent Knowledge: From Baseline to Potential Ceiling', fontsize=18, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, weight='bold')
ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True, ncol=2)


ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_ylim(0, 100)




def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, weight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.show()