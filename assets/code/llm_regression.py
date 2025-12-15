
!git clone https://github.com/hiyouga/LLaMA-Factory.git "/content/drive/MyDrive/LLaMA-Factory"
# %cd "/content/drive/MyDrive/LLaMA-Factory"
# %ls
!pip install .

import torch
try:
  assert torch.cuda.is_available() is True
except AssertionError:
  print("Please set up a GPU before using LLaMA Factory: https://medium.com/mlearning-ai/training-yolov4-on-google-colab-316f8fff99c6")

!huggingface-cli login

import csv
import json
from pathlib import Path
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

stop_words.discard('no')
stop_words.discard('not')
def process_text(text):
    return [word for word in word_tokenize(text.lower()) if ((word not in string.punctuation) and (word not in stop_words))]

PATH_TO_TRAIN_DATA = 'train.csv'
df = pd.read_csv(PATH_TO_TRAIN_DATA)
df.head()

df['negative'] = df['negative'].apply(process_text)
df['positive'] = df['positive'].apply(process_text)
df["all_features"] = df["positive"].apply(" ".join) + " " + df["negative"].apply(" ".join)


json_train_path = Path("data/product_sft_train.json")
json_val_path = Path("data/product_sft_validation.json")

train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)

print(f"Full dataset: {len(df)}")
print(f"Training set: {len(train_df)} (Saved to {json_train_path})")
print(f"Validation set: {len(val_df)} (Saved to {json_val_path})")

# format to Alpaca json for LLaMA factory model
def to_alpaca(dataframe):
    entries = []
    for _, row in dataframe.iterrows():
        # Ensure score is a string for generation
        entries.append({
            "instruction": "Predict the score for the review.",
            "input": row["all_features"],
            "output": str(row["score"]),
            "system": "You are a regression assistant.",
            "history": []
        })
    return entries

with json_train_path.open("w", encoding="utf-8") as f:
    json.dump(to_alpaca(train_df), f, ensure_ascii=False, indent=2)

with json_val_path.open("w", encoding="utf-8") as f:
    json.dump(to_alpaca(val_df), f, ensure_ascii=False, indent=2)

print("Files created successfully.")

# register the dataset
dataset_info_path = Path("/content/drive/MyDrive/LLaMA-Factory/data/dataset_info.json")

with dataset_info_path.open("r", encoding="utf-8") as f:
    dataset_info = json.load(f)

dataset_info["product_sft_train"] = {
    "file_name": "product_sft_train.json",
    "formatting": "alpaca",
    "columns": {
        "prompt":   "instruction",
        "query":    "input",
        "response": "output",
        "system":   "system",
        "history":  "history"
    }
}

with dataset_info_path.open("w", encoding="utf-8") as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print("Updated data/dataset_info.json with 'product_category_sft'.")


# %cd LLaMA-Factory
# %ls
!pip install -e .[torch,bitsandbytes]

args = dict(
    stage="sft",
    do_train=True,
    model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit",

    dataset="product_sft_train",
    dataset_dir="data",

    # Monitor health using a small slice of the training data
    val_size=0.05,
    eval_strategy="steps",
    eval_steps=100,           # Check health every 100 steps
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,

    template="llama3",
    finetuning_type="lora",
    lora_target="all",
    output_dir="saves/llama3_lora_product_category_large", # New output folder

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
    logging_steps=1,
    warmup_ratio=0.1,
    learning_rate=5e-5,
    num_train_epochs = 3.0,     # Large data usually needs fewer epochs
    max_samples = 10000,        # Increase this to fit your larger dataset
    max_grad_norm=1.0,
    quantization_bit=4,
    loraplus_lr_ratio=16.0,
    fp16=True,
    report_to="none"
)

with open("train_llama3_product_category.json", "w", encoding="utf-8") as f:
    json.dump(args, f, ensure_ascii=False, indent=2)

print("Wrote training config to train_llama3_product_category.json")


# %cd /content/drive/MyDrive/LLaMA-Factory/
!ls train_llama3*
!llamafactory-cli train train_llama3_product_category.json



# --- CONFIGURATION ---
BASE_MODEL_ID = "unsloth/llama-3-8b-Instruct-bnb-4bit" # The base model you used
ADAPTER_PATH = "saves/llama3_lora_product_category_large"    # Where your training saved the weights
DATA_PATH = "data/product_sft_validation.json"           # Your data file
VAL_SIZE = 0.1                                         # Must match your training config
SEED = 42                                              # Default seed often used by HF/Sklearn

# 1. Load Data and Recreate Split
print("Loading data...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Recreate the shuffle and split to isolate the validation set
# Note: This assumes standard random shuffling. If the rows don't look familiar,
# you might be testing on training data, but for a quick check this is standard.
_, val_data = train_test_split(full_data, test_size=VAL_SIZE, random_state=SEED)

print(f"Validation set size: {len(val_data)} examples")

# 2. Load Model & Tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True, # Make sure bitsandbytes is installed
)

# Load the fine-tuned adapter on top
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval() # Set to evaluation mode

# 3. Helper function to parse numbers from text
def extract_score(text):
    # Looks for the last number in the text (e.g. "Score: 4.5" -> 4.5)
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        return float(matches[-1])
    return None

# 4. Inference Loop
true_scores = []
predicted_scores = []

print("Running inference...")
for entry in tqdm(val_data):
    # Prepare the prompt exactly as LLaMA-3 expects it
    messages = [
        {"role": "system", "content": entry["system"]},
        {"role": "user", "content": entry["instruction"] + "\n" + entry["input"]}
    ]

    # Format input using the tokenizer's chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,        # We only expect a short number
            eos_token_id=terminators,
            do_sample=False,          # Greedy decoding (deterministic) for regression
            temperature=0.0
        )

    # Decode only the new tokens (the response)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # Parse values
    pred_val = extract_score(response)
    true_val = float(entry["output"])

    if pred_val is not None:
        predicted_scores.append(pred_val)
        true_scores.append(true_val)
        # Optional: Print first few to debug
        if len(predicted_scores) <= 3:
            print(f"Sample - True: {true_val}, Pred: {pred_val}, Raw Text: {response}")
    else:
        print(f"Could not parse score from: {response}")

# 5. Calculate Metrics
mae = mean_absolute_error(true_scores, predicted_scores)
print(f"\nResults on {len(predicted_scores)} items:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Optional: Show a few discrepancies
diffs = [abs(t - p) for t, p in zip(true_scores, predicted_scores)]
worst_idx = np.argmax(diffs)
print(f"Worst Prediction: True {true_scores[worst_idx]} vs Pred {predicted_scores[worst_idx]}")