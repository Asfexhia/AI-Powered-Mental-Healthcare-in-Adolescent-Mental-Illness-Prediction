import pandas as pd

file_name = 'biomistral.csv'  
output_name = 'cleaned_biomistral.csv'

print(f"正在读取文件: {file_name}...")
try:
    df = pd.read_csv(file_name, sep=None, engine='python', encoding='utf-8-sig', quoting=3, on_bad_lines='skip')
except Exception as e:
    print(f"读取失败: {e}")
    exit()

df.columns = df.columns.str.strip()

if 'true_label' not in df.columns or 'primary_pred' not in df.columns:
    print(f"错误：找不到指定的列。当前列名为: {df.columns.tolist()}")
    exit()

df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce')
df['primary_pred'] = pd.to_numeric(df['primary_pred'], errors='coerce')

df['Accuracy'] = (df['true_label'] == df['primary_pred'])
pre_cleaning_count = len(df)
pre_cleaning_acc = df['Accuracy'].mean()

print(f"\n[1. 清洗前统计 (包含预测失败的 -1)]")
print(f"原始总行数: {pre_cleaning_count}")
print(f"原始整体准确率: {pre_cleaning_acc:.2%}")

df_cleaned = df[df['primary_pred'] != -1].copy()
df_cleaned = df_cleaned.dropna(subset=['true_label', 'primary_pred'])

post_cleaning_count = len(df_cleaned)
failed_analysis_count = pre_cleaning_count - post_cleaning_count

print(f"\n[2. 清洗后统计 (剔除无效预测)]")
print(f"被剔除的行数 (primary_pred 为 -1 或数据损坏): {failed_analysis_count}")
print(f"模型分析成功行数: {post_cleaning_count}")

if post_cleaning_count > 0:
    df_cleaned['Accuracy'] = (df_cleaned['true_label'] == df_cleaned['primary_pred'])
    post_cleaning_acc = df_cleaned['Accuracy'].mean()
    print(f"清洗后有效预测准确率: {post_cleaning_acc:.2%}")

    print(f"\n[3. 各类别 (True Label) 详细表现统计]")
    category_stats = df_cleaned.groupby('true_label')['Accuracy'].agg(['count', 'mean']).reset_index()
    category_stats.columns = ['真实类别 (Label)', '样本量', '准确率 (Accuracy)']

    category_stats['准确率 (Accuracy)'] = category_stats['准确率 (Accuracy)'].apply(lambda x: f"{x:.2%}")
    
    print(category_stats.to_string(index=False))

    df_cleaned.to_csv(output_name, index=False, encoding='utf-8-sig')
    print(f"\n--- 处理完成！清洗后的结果已保存至: {output_name} ---")

else:
    print("\n警告：清洗后没有剩余数据，请检查是否所有 primary_pred 都是 -1。")