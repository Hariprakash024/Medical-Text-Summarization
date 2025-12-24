import torch
import numpy as np
import os
import json
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from peft import LoraConfig, get_peft_model, TaskType
from rouge_score import rouge_scorer
from datetime import datetime
import traceback
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        print("NLTK punkt download failed, using simple tokenization")

def setup_logging():
    """Create log directory and return filename"""
    log_dir = "training_results"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_dir}/Summarised_results{timestamp}.txt"
    
    return filename

def save_results_to_file(filename, content, mode='a'):
    """Save results to text file"""
    with open(filename, mode, encoding='utf-8') as f:
        f.write(content + '\n')

def simple_tokenize(text):
    """Simple tokenization fallback if NLTK fails"""
    return text.lower().split()

def plot_confusion_matrix_rouge(predictions, targets, save_path, threshold=0.5):
    """Create confusion matrix for ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    rouge_scores = []
    for pred, target in zip(predictions, targets):
        score = scorer.score(target, pred)['rouge1'].fmeasure
        rouge_scores.append(score)
    
    # Convert to binary classification (above/below threshold)
    y_true = [1 if score >= threshold else 0 for score in rouge_scores]
    y_pred = [1] * len(y_true)  # All predictions are considered positive for this analysis
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low ROUGE', 'High ROUGE'],
                yticklabels=['Low ROUGE', 'High ROUGE'])
    plt.title(f'ROUGE Score Distribution (Threshold: {threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rouge_scores, cm

def plot_confusion_matrix_bleu(predictions, targets, save_path, threshold=0.3):
    """Create confusion matrix for BLEU scores using simple tokenization"""
    
    # Calculate BLEU scores for each sample using simple tokenization
    bleu_scores = []
    for pred, target in zip(predictions, targets):
        try:
            # Try NLTK tokenization first
            pred_tokens = nltk.word_tokenize(pred.lower())
            target_tokens = nltk.word_tokenize(target.lower())
        except:
            # Fallback to simple tokenization
            pred_tokens = simple_tokenize(pred)
            target_tokens = simple_tokenize(target)
        
        # Simple BLEU calculation
        if len(target_tokens) == 0:
            bleu = 0.0
        else:
            # Calculate unigram precision as simple BLEU approximation
            pred_set = set(pred_tokens)
            target_set = set(target_tokens)
            if len(pred_set) == 0:
                bleu = 0.0
            else:
                overlap = len(pred_set.intersection(target_set))
                bleu = overlap / len(pred_set)
        
        bleu_scores.append(bleu)
    
    # Convert to binary classification (above/below threshold)
    y_true = [1 if score >= threshold else 0 for score in bleu_scores]
    y_pred = [1] * len(y_true)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Low BLEU', 'High BLEU'],
                yticklabels=['Low BLEU', 'High BLEU'])
    plt.title(f'BLEU Score Distribution (Threshold: {threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bleu_scores, cm

def plot_score_distributions(rouge_scores, bleu_scores, save_path):
    """Plot distribution of ROUGE and BLEU scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROUGE distribution
    ax1.hist(rouge_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('ROUGE-1 Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of ROUGE-1 Scores')
    ax1.axvline(np.mean(rouge_scores), color='red', linestyle='--', label=f'Mean: {np.mean(rouge_scores):.3f}')
    ax1.legend()
    
    # BLEU distribution
    ax2.hist(bleu_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('BLEU Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of BLEU Scores')
    ax2.axvline(np.mean(bleu_scores), color='red', linestyle='--', label=f'Mean: {np.mean(bleu_scores):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def preprocess_dataset(df):
    df_clean = df.copy()
    
    def clean_medical_notes(text, is_target=False):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        
        # Basic Cleaning ONLY
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)?', '', text)
        text = re.sub(r'Clip #.*?(?=\s*[A-Z]|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'TECHNIQUE:.*?(\.|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'REASON FOR THIS EXAMINATION:.*?(\.|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # MINIMAL abbreviation expansion
        minimal_replacements = {
            r'\bs/p\b': 'status post',
            r'\bPOD(\d+)\b': r'postoperative day \1',
            r'\bNPO\b': 'nothing by mouth',
            r'\bVSS\b': 'vital signs stable',
        }
        
        for pattern, replacement in minimal_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if not is_target:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            unique_sentences = []
            seen = set()
            for sent in sentences:
                sent_lower = sent.lower().strip()
                if sent_lower and sent_lower not in seen:
                    seen.add(sent_lower)
                    unique_sentences.append(sent)
            text = ' '.join(unique_sentences)
        
        return text.strip()
    
    df_clean["input"] = df_clean["input"].apply(clean_medical_notes)
    df_clean["target"] = df_clean["target"].apply(lambda x: clean_medical_notes(x, is_target=True))
    
    def word_overlap(input_text, target_text):
        input_words = set(input_text.lower().split())
        target_words = set(target_text.lower().split())
        if not target_words:
            return 0
        return len(input_words.intersection(target_words)) / len(target_words)
    
    df_clean['overlap'] = df_clean.apply(lambda row: word_overlap(row['input'], row['target']), axis=1)
    initial_len = len(df_clean)
    
    df_clean = df_clean[df_clean['overlap'] >= 0.08]
    df_clean = df_clean[(df_clean["input"].str.len() > 30) & (df_clean["target"].str.len() > 10)]
    
    print(f"Medical preprocessing complete: {len(df_clean)} samples (filtered from {initial_len}, avg overlap: {df_clean['overlap'].mean():.2%})")
    return df_clean

class MedicalDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_input_len=1024, max_output_len=256):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        clinical_notes = str(item["input"])
        summary = str(item["target"])
        
        prompt = "Generate a concise clinical discharge summary: "
        full_input = prompt + clinical_notes
        
        inputs = self.tokenizer(
            full_input,
            max_length=self.max_input_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            text_target=summary,
            max_length=self.max_output_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        labels_tensor = labels["input_ids"].squeeze(0)
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels_tensor,
            "original_target": summary
        }

def collate_fn(batch):
    batch_dict = {key: [d[key] for d in batch] for key in batch[0].keys()}
    stacked_batch = {}
    for key in batch_dict:
        if key != "original_target":
            stacked_batch[key] = torch.stack(batch_dict[key])
        else:
            stacked_batch[key] = batch_dict[key]
    return stacked_batch

def BartModel(model_name, lora_rank=16, lora_alpha=32, lora_dropout=0.1):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer

def calculate_bleu_score(predictions, targets):
    """Calculate simplified BLEU score between predictions and targets"""
    bleu_scores = []
    
    for pred, target in zip(predictions, targets):
        try:
            # Try NLTK tokenization first
            pred_tokens = nltk.word_tokenize(pred.lower())
            target_tokens = nltk.word_tokenize(target.lower())
        except:
            # Fallback to simple tokenization
            pred_tokens = simple_tokenize(pred)
            target_tokens = simple_tokenize(target)
        
        # Simple BLEU calculation (unigram precision)
        if len(target_tokens) == 0 or len(pred_tokens) == 0:
            bleu = 0.0
        else:
            pred_set = set(pred_tokens)
            target_set = set(target_tokens)
            overlap = len(pred_set.intersection(target_set))
            bleu = overlap / len(pred_set) if len(pred_set) > 0 else 0.0
        
        bleu_scores.append(bleu)
    
    return np.mean(bleu_scores) if bleu_scores else 0

def calculate_rouge_scores(predictions, targets):
    """Calculate ROUGE scores (ROUGE-1 and ROUGE-L only)"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []
    
    for pred, target in zip(predictions, targets):
        scores = scorer.score(target, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0,
        'rouge_avg': (np.mean(rouge1_scores) + np.mean(rougeL_scores)) / 2  # Only ROUGE-1 and ROUGE-L
    }

def calculate_comprehensive_metrics(predictions, targets, split_name="validation"):
    """Calculate all metrics: ROUGE and BLEU"""
    # ROUGE scores
    rouge_metrics = calculate_rouge_scores(predictions, targets)
    
    # BLEU score
    bleu_score = calculate_bleu_score(predictions, targets)
    
    # Combine all metrics
    all_metrics = {
        'split': split_name,
        'num_samples': len(predictions)
    }
    all_metrics.update(rouge_metrics)
    all_metrics.update({
        'bleu': bleu_score
    })
    
    return all_metrics

def calculate_accuracy_metrics(model, dataloader, tokenizer, device, max_batches=None, split_name="validation", comprehensive=False):
    """Enhanced metrics calculation with all scores"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    batch_count = 0
    
    # Use all batches for test, limited batches for train/val
    if max_batches is None:
        max_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            if batch_count >= max_batches:
                break
                
            original_targets = batch["original_target"]
            model_batch = {k: v.to(device) for k, v in batch.items() if k != "original_target"}
            
            outputs = model(**model_batch)
            total_loss += outputs.loss.item()
            
            # Faster generation for evaluation
            generated_ids = model.generate(
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                max_length=256,  # Reduced for faster generation
                num_beams=4,    # Reduced for faster generation
                repetition_penalty=1.5,
                early_stopping=True,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
            
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(predictions)
            all_targets.extend(original_targets)
            batch_count += 1
    
    if batch_count == 0:
        return {'loss': float('inf'), 'rouge_accuracy': 0, 'num_samples': 0}, [], []
    
    avg_loss = total_loss / batch_count
    
    # Calculate comprehensive metrics
    if comprehensive or split_name == "test":
        metrics = calculate_comprehensive_metrics(all_predictions, all_targets, split_name)
    else:
        # For training/validation, just use ROUGE for speed
        metrics = calculate_rouge_scores(all_predictions, all_targets)
        metrics['split'] = split_name
        metrics['num_samples'] = len(all_predictions)
        metrics['bleu'] = 0
    
    metrics['loss'] = avg_loss
    
    return metrics, all_predictions, all_targets

def train_model(model, train_loader, val_loader, tokenizer, learning_rate, epochs, device, log_filename):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=300,  # Reduced for faster training
        num_training_steps=len(train_loader) * epochs
    )
    
    model.to(device)
    best_val_rouge = 0
    train_history = []
    val_history = []
    
    initial_log = f"Training Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    initial_log += f"Device: {device}\n"
    initial_log += f"Train batches: {len(train_loader)}\n"
    initial_log += f"Validation batches: {len(val_loader)}\n"
    initial_log += f"Epochs: {epochs}\n"
    initial_log += f"Learning rate: {learning_rate}\n"
    initial_log += "="*80 + "\n"
    save_results_to_file(log_filename, initial_log, 'w')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            model_batch = {k: v.to(device) for k, v in batch.items() if k != "original_target"}
            
            optimizer.zero_grad()
            outputs = model(**model_batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 200 == 0:
                log_msg = f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                print(log_msg)
                save_results_to_file(log_filename, log_msg)
        
        avg_train_loss = total_train_loss / train_batches
        
        # LIMITED evaluation during training (for speed)
        train_metrics, _, _ = calculate_accuracy_metrics(model, train_loader, tokenizer, device, max_batches=3, split_name="train")
        val_metrics, _, _ = calculate_accuracy_metrics(model, val_loader, tokenizer, device, max_batches=5, split_name="validation")
        
        epoch_log = f"\nEpoch {epoch + 1}/{epochs}\n"
        epoch_log += f"TRAINING (sampled):\n"
        epoch_log += f"  Loss: {avg_train_loss:.4f}\n"
        epoch_log += f"  ROUGE-1: {train_metrics['rouge1']:.4f}, ROUGE-L: {train_metrics['rougeL']:.4f}\n"
        epoch_log += f"  ROUGE Avg: {train_metrics['rouge_avg']:.4f}\n"
        epoch_log += f"VALIDATION (sampled):\n"
        epoch_log += f"  Loss: {val_metrics['loss']:.4f}\n"
        epoch_log += f"  ROUGE-1: {val_metrics['rouge1']:.4f}, ROUGE-L: {val_metrics['rougeL']:.4f}\n"
        epoch_log += f"  ROUGE Avg: {val_metrics['rouge_avg']:.4f}\n"
        
        print(epoch_log)
        save_results_to_file(log_filename, epoch_log)
        
        # Save best model based on validation ROUGE
        if val_metrics['rouge_avg'] > best_val_rouge:
            best_val_rouge = val_metrics['rouge_avg']
            torch.save(model.state_dict(), 'best_model.pth')
            save_msg = f"Saved best model with Validation ROUGE: {best_val_rouge:.4f}"
            print(save_msg)
            save_results_to_file(log_filename, save_msg)
        
        train_history.append({
            'epoch': epoch + 1,
            'loss': avg_train_loss,
            'rouge_avg': train_metrics['rouge_avg'],
            'rouge1': train_metrics['rouge1'],
            'rougeL': train_metrics['rougeL']
        })
        
        val_history.append({
            'epoch': epoch + 1,
            'loss': val_metrics['loss'],
            'rouge_avg': val_metrics['rouge_avg'],
            'rouge1': val_metrics['rouge1'],
            'rougeL': val_metrics['rougeL']
        })
    
    return model, train_history, val_history

def evaluate_test_set(model, test_loader, tokenizer, device, log_filename):
    """COMPREHENSIVE Test set evaluation with ALL metrics and confusion matrices"""
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION (COMPREHENSIVE - WITH CONFUSION MATRICES)")
    print("="*80)
    
    # Use limited batches for test evaluation to save time
    test_metrics, test_predictions, test_targets = calculate_accuracy_metrics(
        model, test_loader, tokenizer, device, 
        max_batches=20,  # Limited batches for faster evaluation
        split_name="test",
        comprehensive=True
    )
    
    # Create confusion matrices and save them
    log_dir = os.path.dirname(log_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ROUGE confusion matrix
    rouge_scores, rouge_cm = plot_confusion_matrix_rouge(
        test_predictions, test_targets,
        os.path.join(log_dir, f"rouge_confusion_matrix_{timestamp}.png"),
        threshold=0.5
    )
    
    # BLEU confusion matrix
    bleu_scores, bleu_cm = plot_confusion_matrix_bleu(
        test_predictions, test_targets,
        os.path.join(log_dir, f"bleu_confusion_matrix_{timestamp}.png"),
        threshold=0.3
    )
    
    # Score distributions
    plot_score_distributions(
        rouge_scores, bleu_scores,
        os.path.join(log_dir, f"score_distributions_{timestamp}.png")
    )
    
    test_log = f"\nFINAL TEST SET RESULTS (Comprehensive Metrics):\n"
    test_log += f"Loss: {test_metrics['loss']:.4f}\n"
    test_log += f"\nROUGE Scores:\n"
    test_log += f"  ROUGE-1: {test_metrics['rouge1']:.4f}\n"
    test_log += f"  ROUGE-L: {test_metrics['rougeL']:.4f}\n"
    test_log += f"  ROUGE Average: {test_metrics['rouge_avg']:.4f}\n"
    test_log += f"\nOther Metrics:\n"
    test_log += f"  BLEU Score: {test_metrics['bleu']:.4f}\n"
    test_log += f"\nConfusion Matrix Analysis:\n"
    test_log += f"  ROUGE Confusion Matrix (Threshold 0.5):\n{rouge_cm}\n"
    test_log += f"  BLEU Confusion Matrix (Threshold 0.3):\n{bleu_cm}\n"
    test_log += f"  High ROUGE samples: {np.sum(np.array(rouge_scores) >= 0.5)}/{len(rouge_scores)}\n"
    test_log += f"  High BLEU samples: {np.sum(np.array(bleu_scores) >= 0.3)}/{len(bleu_scores)}\n"
    test_log += f"\nTotal samples evaluated: {test_metrics['num_samples']}\n"
    test_log += f"Visualizations saved to: {log_dir}\n"
    
    print(test_log)
    save_results_to_file(log_filename, test_log)
    
    return test_metrics, test_predictions, test_targets

def generate_improved_summary(model, tokenizer, text, max_input_len=768, max_output_len=256, device='cuda'):
    model.eval()  
    
    prompt = "Generate a clinical discharge summary: "
    full_input = prompt + text
    
    inputs = tokenizer(
        full_input,
        max_length=max_input_len,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=4,  # Reduced for faster generation
            repetition_penalty=1.5,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def generate_first_n_summaries(model, tokenizer, test_df, num_samples=3, device='cuda', log_filename=None):
    model.eval()
    
    analysis_header = f"\n{'='*80}\n"
    analysis_header += f"QUALITATIVE ANALYSIS - FIRST {num_samples} TEST SAMPLES\n"
    analysis_header += f"{'='*80}\n"
    
    print(analysis_header)
    if log_filename:
        save_results_to_file(log_filename, analysis_header)
    
    num_samples = min(num_samples, len(test_df))
    
    for i in range(num_samples):
        sample_text = test_df.iloc[i]["input"]
        true_summary = test_df.iloc[i]["target"]
        generated_summary = generate_improved_summary(model, tokenizer, sample_text, device=device)
        
        case_header = f"\n{'─'*40} TEST CASE {i+1} {'─'*40}\n"
        print(case_header)
        if log_filename:
            save_results_to_file(log_filename, case_header)
        
        print(f"INPUT KEY POINTS:")
        input_log = "INPUT KEY POINTS:\n"
        input_lines = sample_text.split('.')
        for line in input_lines[:3]:
            if len(line.strip()) > 20:
                point = f"  • {line.strip()}\n"
                print(point, end='')
                input_log += point
        if log_filename:
            save_results_to_file(log_filename, input_log)
        
        print(f"\nEXPECTED SUMMARY:")
        expected_log = f"\nEXPECTED SUMMARY:\n{true_summary}\n"
        print(expected_log)
        if log_filename:
            save_results_to_file(log_filename, expected_log)
        
        print(f"GENERATED SUMMARY:")
        generated_log = f"GENERATED SUMMARY:\n{generated_summary}\n"
        print(generated_log)
        if log_filename:
            save_results_to_file(log_filename, generated_log)
        
        # Calculate metrics for this sample
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(true_summary, generated_summary)
        
        bleu_score = calculate_bleu_score([generated_summary], [true_summary])
        
        print(f"SAMPLE-LEVEL COMPREHENSIVE METRICS:")
        quality_log = f"SAMPLE-LEVEL COMPREHENSIVE METRICS:\n"
        quality_log += f"  ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}\n"
        quality_log += f"  ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}\n"
        quality_log += f"  BLEU: {bleu_score:.4f}\n"
        print(quality_log)
        if log_filename:
            save_results_to_file(log_filename, quality_log)
        
        separator = f"\n{'─'*80}\n"
        print(separator)
        if log_filename:
            save_results_to_file(log_filename, separator)

if __name__ == "__main__":
    log_filename = setup_logging()
    print(f"Results will be saved to: {log_filename}")
    
    model_name = "facebook/bart-base"
    max_input_len = 768
    max_output_len = 256
    
    lora_rank = 32    
    lora_alpha = 32
    lora_dropout = 0.05
    
    batch_size = 4
    learning_rate = 2e-4
    epochs = 10  # Reduced epochs for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load and split data properly
        train_df = pd.read_parquet(r"train-00001-of-00004-23a9cfd849f3e2a9.parquet")
        train_df["input"] = train_df["extractive_notes_summ"]
        train_df["target"] = train_df["target_text"]
        
        test_df = pd.read_parquet(r"test-00000-of-00001-e79c5993b9fe60d6.parquet")
        test_df["input"] = test_df["extractive_notes_summ"]
        test_df["target"] = test_df["target_text"]
        
        # Log dataset info
        dataset_info = f"Dataset Information:\n"
        dataset_info += f"Original train samples: {len(train_df)}\n"
        dataset_info += f"Original test samples: {len(test_df)}\n"
        dataset_info += f"Evaluation Metrics: ROUGE-1, ROUGE-L, BLEU with Confusion Matrices\n"
        print(dataset_info)
        save_results_to_file(log_filename, dataset_info)
        
        # Preprocess
        train_df_clean = preprocess_dataset(train_df)
        test_df_clean = preprocess_dataset(test_df)
        
        # Further split train into train/validation
        train_split, val_split = train_test_split(train_df_clean, test_size=0.2, random_state=42)
        
        preprocess_info = f"After preprocessing and splitting:\n"
        preprocess_info += f"Train samples: {len(train_split)}\n"
        preprocess_info += f"Validation samples: {len(val_split)}\n"
        preprocess_info += f"Test samples: {len(test_df_clean)}\n"
        print(preprocess_info)
        save_results_to_file(log_filename, preprocess_info)
        
        model, tokenizer = BartModel(model_name, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        # Create datasets for all splits
        train_dataset = MedicalDataset(tokenizer, train_split, max_input_len, max_output_len)
        val_dataset = MedicalDataset(tokenizer, val_split, max_input_len, max_output_len)
        test_dataset = MedicalDataset(tokenizer, test_df_clean, max_input_len, max_output_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # TRAIN with train/validation
        trained_model, train_history, val_history = train_model(
            model, train_loader, val_loader, tokenizer, learning_rate, epochs, device, log_filename
        )
        
        # Load best model for FINAL TEST evaluation
        trained_model.load_state_dict(torch.load('best_model.pth'))
        
        # FINAL TEST SET EVALUATION with ALL metrics and confusion matrices
        test_metrics, test_predictions, test_targets = evaluate_test_set(trained_model, test_loader, tokenizer, device, log_filename)
        
        # Qualitative analysis on TEST set with comprehensive metrics
        generate_first_n_summaries(trained_model, tokenizer, test_df_clean, num_samples=3, device=device, log_filename=log_filename)
        
        # Save everything
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'tokenizer': tokenizer,
            'train_history': train_history,
            'val_history': val_history,
            'test_metrics': test_metrics
        }, 'improved_medical_summarization_model.pth')
        
        # Final comprehensive comparison
        best_val_rouge = max([v['rouge_avg'] for v in val_history])
        
        completion_msg = f"\n{'='*60}\n"
        completion_msg += f"FINAL COMPREHENSIVE PERFORMANCE\n"
        completion_msg += f"{'='*60}\n"
        completion_msg += f"Best Validation ROUGE: {best_val_rouge:.4f}\n"
        completion_msg += f"Final Test ROUGE: {test_metrics['rouge_avg']:.4f}\n"
        completion_msg += f"Test BLEU: {test_metrics['bleu']:.4f}\n"
        completion_msg += f"\nConfusion Matrix Insights:\n"
        completion_msg += f"- ROUGE confusion matrix shows score distribution\n"
        completion_msg += f"- BLEU confusion matrix shows n-gram precision distribution\n"
        completion_msg += f"- Score distribution plots show overall performance patterns\n"
        completion_msg += f"\nResults and visualizations saved to training_results directory\n"
        completion_msg += f"Model saved to: improved_medical_summarization_model.pth\n"
        
        print(completion_msg)
        save_results_to_file(log_filename, completion_msg)
            
    except Exception as e:
        error_msg = f"Error: {e}\n"
        error_msg += f"Traceback: {traceback.format_exc()}\n"
        print(error_msg)
        save_results_to_file(log_filename, error_msg)