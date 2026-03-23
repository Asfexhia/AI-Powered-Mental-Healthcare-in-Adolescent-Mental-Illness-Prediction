import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


file_path = "qwen.csv"
df = pd.read_csv(file_path)


df['primary_pred'] = pd.to_numeric(df['primary_pred'], errors='coerce')
df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce')


df = df[df['primary_pred'] != -1].dropna(subset=['primary_pred', 'true_label'])



class_names = ['Anxiety(0)', 'Depression(1)', 'ADHD(2)', 'OCD(3)', 'Eating(4)', 'Gaming(5)']
target_labels = [0, 1, 2, 3, 4, 5]



cm = confusion_matrix(df['true_label'], df['primary_pred'], labels=target_labels)


plt.figure(figsize=(12, 10))


sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 12}
)


plt.title('Qwen-2.5-14B Performance on Mental Health Classification', fontsize=22, pad=20)
plt.ylabel('True Label', fontsize=16)
plt.xlabel('Predicted Label', fontsize=16)


plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)

plt.tight_layout()
plt.show()