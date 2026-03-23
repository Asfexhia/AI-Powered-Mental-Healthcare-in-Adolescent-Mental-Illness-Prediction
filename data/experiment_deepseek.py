import torch
import pandas as pd
import json
import os
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM




MODEL_PATH = "/root/models/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5"
INPUT_CSV = "network_cleaned1.csv"
OUTPUT_CSV = f"biomistral_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


LABEL_MAP = {
    "anxiety": 0, "depression": 1, "adhd": 2, "ocd": 3,
    "eating": 4, "eating disorder": 4,
    "gaming": 5, "video game": 5, "video game addiction": 5
}





DIAGNOSIS_PROMPT =





tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)




def robust_parse(text):




    text = text.replace("“", '"').replace("”", '"')

    text = text.replace("'", '"')

    text = text.replace("\\_", "_")

    parsed_data = None


    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1)

            if "[" in json_str and "]" not in json_str:
                json_str += "]}"
            elif "{" in json_str and "}" not in json_str:
                json_str += "}"

            parsed_data = json.loads(json_str)
    except:
        pass



    preds = []
    rationale = ""

    if parsed_data:
        raw_preds = parsed_data.get("predicted_labels", [])
        rationale = parsed_data.get("rationale", "")
    else:

        num_match = re.findall(r'labels"?:?\s*\[([\d,\s]+)', text)
        if num_match:

            preds = [int(n) for n in re.findall(r'\d+', num_match[0])]
        else:

            text_match = re.findall(r'labels"?:?\s*\[(.*?)\]', text)
            if text_match:

                words = re.findall(r'"(.*?)"', text_match[0])
                for w in words:
                    w_lower = w.lower().strip()
                    if w_lower in LABEL_MAP:
                        preds.append(LABEL_MAP[w_lower])

    if parsed_data and isinstance(raw_preds, list):

        final_preds = []
        for p in raw_preds:
            if isinstance(p, int) or (isinstance(p, str) and p.isdigit()):
                final_preds.append(int(p))
            elif isinstance(p, str):
                p_lower = p.lower().strip()
                if p_lower in LABEL_MAP:
                    final_preds.append(LABEL_MAP[p_lower])
        preds = final_preds

    return preds, rationale




df_full = pd.read_csv(INPUT_CSV)
df = df_full.sample(n=1000, random_state=42) if len(df_full) > 1000 else df_full

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write("id,true_label,primary_pred,all_preds,rationale,raw_snippet\n")

print("▶️ 开始清洗式分析...")

for index, row in df.iterrows():
    try:
        c_title = str(row.get('title', ""))
        c_text = str(row.get('text', ""))[:800]
        true_label = row.get('target_label', row.get('target', 'N/A'))

        prompt = DIAGNOSIS_PROMPT.format(title=c_title, text=c_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.4,
                do_sample=True,
                repetition_penalty=1.2
            )

        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()


        preds, rationale = robust_parse(response)

        p_pred = -1
        a_preds = "[]"

        if preds:
            p_pred = preds[0]
            a_preds = ";".join(map(str, preds))


        rationale = str(rationale).replace("\n", " ").replace(",", "，")

        print(f"[{index}] True: {true_label} | Pred: {p_pred}")

        with open(OUTPUT_CSV, 'a', encoding='utf-8') as f:
            clean_raw = response[:50].replace("\n", " ").replace('"', "'")
            f.write(f"{index},{true_label},{p_pred},{a_preds},\"{rationale}\",\"{clean_raw}\"\n")

    except Exception as e:
        print(f"Error {index}: {e}")
        continue

print("✅ 完成")