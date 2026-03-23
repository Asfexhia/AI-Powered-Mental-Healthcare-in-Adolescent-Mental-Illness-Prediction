import os
import re

def remove_comments(content):
    """删除Python代码中的注释"""
    lines = content.split('\n')
    result = []
    in_multiline = False
    multiline_char = None

    for line in lines:
        # 处理多行注释
        if in_multiline:
            if multiline_char in line:
                in_multiline = False
                # 删除多行注释后的内容
                line = line.split(multiline_char, 1)[1]
            else:
                continue

        # 检查是否开始多行注释
        if '"""' in line:
            if line.count('"""') == 2:
                # 同一行内的多行注释
                line = re.sub(r'""".*?"""', '', line)
            else:
                in_multiline = True
                multiline_char = '"""'
                line = line.split('"""', 1)[0]
        elif "'''" in line:
            if line.count("'''") == 2:
                # 同一行内的多行注释
                line = re.sub(r"'''.*?'''", '', line)
            else:
                in_multiline = True
                multiline_char = "'''"
                line = line.split("'''", 1)[0]

        # 删除单行注释
        if '#' in line:
            # 检查#是否在字符串内
            in_string = False
            string_char = None
            for i, char in enumerate(line):
                if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                elif char == '#' and not in_string:
                    line = line[:i]
                    break

        # 去掉行尾空白，但保留非空行
        line = line.rstrip()
        if line or result:  # 保留代码行或保留空行（如果之前有代码）
            result.append(line)

    # 删除末尾空行
    while result and not result[-1]:
        result.pop()

    return '\n'.join(result)

# 处理所有Python文件
py_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'remove_comments.py']

for py_file in py_files:
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()

        cleaned_content = remove_comments(content)

        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"[OK] {py_file}")
    except Exception as e:
        print(f"[ERROR] {py_file}: {e}")

print("\n完成！所有Python文件的注释已删除。")
