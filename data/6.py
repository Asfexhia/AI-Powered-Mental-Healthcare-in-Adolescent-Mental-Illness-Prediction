import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




class_names = ['Anxiety', 'Depression', 'ADHD', 'OCD', 'Eating Dis.', 'Gaming Add.']

cm_data = {
    "DeepSeek-V3": np.array([
        [855, 182, 60, 36, 48, 36],
        [618, 973, 35, 35, 53, 53],
        [759, 90, 815, 36, 54, 54],
        [121, 81, 40, 490, 40, 40],
        [62, 125, 25, 37, 972, 37],
        [798, 88, 53, 35, 53, 743]
    ]),
    "Gemma3-27B": np.array([
        [805, 455, 13, 40, 13, 13],
        [233, 1598, 19, 19, 19, 58],
        [441, 641, 683, 160, 40, 40],
        [188, 206, 8, 432, 8, 17],
        [132, 316, 13, 66, 767, 26],
        [320, 686, 137, 228, 22, 896]
    ]),
    "Qwen3-32B": np.array([
        [1011, 266, 42, 14, 28, 42],
        [652, 1227, 40, 20, 20, 81],
        [793, 333, 687, 62, 41, 166],
        [308, 140, 18, 432, 9, 28],
        [421, 188, 29, 29, 698, 87],
        [775, 163, 102, 61, 20, 919]
    ]),
    "Llama4-Maverick": np.array([
        [991, 120, 24, 24, 24, 24],
        [315, 1265, 35, 35, 35, 70],
        [501, 179, 989, 35, 35, 53],
        [161, 40, 16, 548, 16, 24],
        [150, 62, 25, 25, 976, 12],
        [526, 87, 35, 35, 17, 1055]
    ]),
    "BioMistral-NLU": np.array([
        [303, 444, 23, 15, 7, 7],
        [290, 758, 58, 34, 11, 11],
        [237, 858, 59, 23, 11, 0],
        [149, 333, 10, 31, 5, 5],
        [165, 623, 16, 16, 8, 0],
        [406, 745, 11, 0, 0, 0]
    ])
}




plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 16))
axes = axes.flatten()

GLOBAL_VMAX = 1100
model_names = list(cm_data.keys())

for i, ax in enumerate(axes[:5]):
    model_name = model_names[i]
    data = cm_data[model_name]

    sns.heatmap(
        data,
        annot=True,
        fmt='d',
        cmap='Blues',
        vmax=GLOBAL_VMAX,
        vmin=0,
        ax=ax,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12, "weight": "bold"}
    )

    ax.set_title(f"({chr(97+i)}) {model_name}", fontsize=18, weight='bold', pad=20)


    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    ax.set_xlabel('')
    ax.set_ylabel('')


axes[5].axis('off')




plt.subplots_adjust(
    wspace=0.40,
    hspace=0.60,
    left=0.08,
    right=0.9,
    bottom=0.15,
    top=0.92
)




fig.text(0.45, 0.02, 'Predicted Label', ha='center', fontsize=24, weight='bold')
fig.text(0.01, 0.5, 'Target Label', va='center', rotation='vertical', fontsize=24, weight='bold')




cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])

norm = plt.Normalize(vmin=0, vmax=GLOBAL_VMAX)
sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)


cbar.set_label('Number of Samples', fontsize=20, weight='bold', labelpad=7)
cbar.ax.tick_params(labelsize=14)


plt.show()