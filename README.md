## 🔬 Model & Methodology

- Developed a **BART-based sequence-to-sequence summarization model**, fine-tuned using **LoRA (Low-Rank Adaptation)** to generate clinical discharge summaries from detailed physician notes.
- Trained and evaluated the model on the **MIMIC-III clinical dataset**, using standard NLP evaluation metrics such as **ROUGE-1, ROUGE-L, and BLEU**.
- Applied **LoRA fine-tuning** to significantly reduce the number of trainable parameters, resulting in **lower memory usage and faster training** compared to full model fine-tuning.