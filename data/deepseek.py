import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score






cm_deepseek = np.array([
    [855, 182, 60, 36, 48, 36],
    [618, 973, 35, 35, 53, 53],
    [759, 90, 815, 36, 54, 54],
    [121, 81, 40, 490, 40, 40],
    [62, 125, 25, 37, 972, 37],
    [798, 88, 53, 35, 53, 743]
])


cm_gemma = np.array([
    [805, 455, 13, 40, 13, 13],
    [233, 1598, 19, 19, 19, 58],
    [441, 641, 683, 160, 40, 40],
    [188, 206, 8, 432, 8, 17],
    [132, 316, 13, 66, 767, 26],
    [320, 686, 137, 228, 22, 896]
])


cm_qwen = np.array([
    [1011, 266, 42, 14, 28, 42],
    [652, 1227, 40, 20, 20, 81],
    [793, 333, 687, 62, 41, 166],
    [308, 140, 18, 432, 9, 28],
    [421, 188, 29, 29, 698, 87],
    [775, 163, 102, 61, 20, 919]
])


cm_llama = np.array([
    [991, 120, 24, 24, 24, 24],
    [315, 1265, 35, 35, 35, 70],
    [501, 179, 989, 35, 35, 53],
    [161, 40, 16, 548, 16, 24],
    [150, 62, 25, 25, 976, 12],
    [526, 87, 35, 35, 17, 1055]
])


cm_biomistral = np.array([
    [303, 444, 23, 15, 7, 7],
    [290, 758, 58, 34, 11, 11],
    [237, 858, 59, 23, 11, 0],
    [149, 333, 10, 31, 5, 5],
    [165, 623, 16, 16, 8, 0],
    [406, 745, 11, 0, 0, 0]
])

models = {
    "DeepSeek-V3": cm_deepseek,
    "Gemma3-27B": cm_gemma,
    "Qwen3-32B": cm_qwen,
    "Llama4-Maverick": cm_llama,
    "BioMistral-NLU": cm_biomistral
}




def calculate_metrics(cm):



    TP = np.diag(cm)

    FP = np.sum(cm, axis=0) - TP

    FN = np.sum(cm, axis=1) - TP


    accuracy = np.sum(TP) / np.sum(cm)



    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.nan_to_num(TP / (TP + FP))
        recall = np.nan_to_num(TP / (TP + FN))
        f1 = np.nan_to_num(2 * (precision * recall) / (precision + recall))


    macro_prec = np.mean(precision)
    macro_rec = np.mean(recall)
    macro_f1 = np.mean(f1)

    return accuracy, macro_f1, macro_prec, macro_rec





print(f"{'Model':<20} | {'Acc':<8} | {'Macro-F1':<8} | {'Macro-P':<8} | {'Macro-R':<8}")
print("-" * 65)

for name, cm in models.items():
    acc, f1, prec, rec = calculate_metrics(cm)

    print(f"{name:<20} | {acc:.3f}    | {f1:.3f}     | {prec:.3f}    | {rec:.3f}")

print("-" * 65)


print("\n=== LaTeX Table Rows (Copy Paste) ===")
for name, cm in models.items():
    acc, f1, prec, rec = calculate_metrics(cm)
    print(f"{name} & {acc:.3f} & {f1:.3f} & {prec:.3f} & {rec:.3f} \\\\")