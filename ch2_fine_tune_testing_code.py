from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd


# Load dataset
df = pd.read_csv('GoldStandard2024.csv')  # change path

# Basic cleaning
df = df[['Text', 'Biased']].dropna()
df['Biased'] = df['Biased'].astype(int)

# Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Biased'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Model choice
model_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(example):
    encoding = tokenizer(example["Text"], padding="max_length", truncation=True, max_length=128)
    encoding["labels"] = example["Biased"] # Add the labels
    return encoding

train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# Training Arguments and Hyperparameters for fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", # Corrected parameter name
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
tokenizer.save_pretrained("./fine_tuned_model")
metrics = trainer.evaluate()
print(metrics)
# Save the model

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# --- 1. Compute predictions on the evaluation (test) set ---
pred_output = trainer.predict(test_dataset)  # returns predictions, label_ids, metrics
logits = pred_output.predictions  # shape (N, num_labels)
true_labels = pred_output.label_ids  # shape (N,)
pred_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
pred_ids = np.argmax(logits, axis=1)

# --- 2. Classification report: precision, recall, F1 ---
report = classification_report(true_labels, pred_ids, target_names=["not_offensive", "offensive"], output_dict=True)
print("=== Classification Report ===")
print(classification_report(true_labels, pred_ids, target_names=["not_offensive", "offensive"]))

# --- 3. Confusion matrix ---
cm = confusion_matrix(true_labels, pred_ids)
print("=== Confusion Matrix ===")
print(cm)
# Optionally pretty-print with labels:
cm_df = pd.DataFrame(cm, index=["true_not", "true_off"], columns=["pred_not", "pred_off"])
print(cm_df)

# --- 4. Error analysis: false positives (predicted offensive but actually not) ---
# Need original texts. Assume your test_dataset has the "Text" field preserved.
# If you used a HuggingFace Dataset, you can get it as pandas:
test_df = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in ["Text", "labels"]]).to_pandas()
# But depending on your naming: if label column became "labels" or something else adjust accordingly.

# Reconstruct a DataFrame with predictions
analysis_df = pd.DataFrame({
    "text": test_dataset["Text"],
    "true_label": true_labels,
    "pred_label": pred_ids,
    "prob_not_off": pred_probs[:, 0],
    "prob_off": pred_probs[:, 1],
})

# False positives: predicted offensive (1) but true is not (0)
false_positives = analysis_df[(analysis_df["true_label"] == 0) & (analysis_df["pred_label"] == 1)]
print(f"\n=== Number of False Positives: {len(false_positives)} ===")
print("=== Sample False Positives ===")
print(false_positives[["text", "true_label", "pred_label", "prob_off"]].head(10).to_string(index=False))

# Similarly you can look at false negatives if you want:
false_negatives = analysis_df[(analysis_df["true_label"] == 1) & (analysis_df["pred_label"] == 0)]
print(f"\n=== Number of False Negatives: {len(false_negatives)} ===")
print("=== Sample False Negatives ===")
print(false_negatives[["text", "true_label", "pred_label", "prob_off"]].head(10).to_string(index=False))