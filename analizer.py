import os
import sys
import io
import json
import torch
import pdfplumber
from transformers import BertTokenizer, BertModel, GPTNeoForCausalLM, GPT2Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')


def read_pdf_files(pdf_dir, cache_file="pdf_texts_cache.json"):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            pdf_texts = json.load(f)
    else:
        pdf_texts = []
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                with pdfplumber.open(pdf_path) as pdf:
                    pdf_text = ' '.join(page.extract_text() for page in pdf.pages)
                    pdf_texts.append(pdf_text)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(pdf_texts, f)
    return pdf_texts


def generate_answer(model, tokenizer, input_text, max_length):
    input_ids = tokenizer.encode(input_text, padding=True, max_length=max_length, truncation=True, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).float()
    model.config.pad_token_id = model.config.eos_token_id
    generated_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return generated_text

def main(pdf_texts):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    embeddings = []
    for pdf_text in pdf_texts:
        inputs = bert_tokenizer(pdf_text, return_tensors='pt', max_length=512, truncation=True)
        outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach())

    gpt_tokenizer = GPT2Tokenizer.from_pretrained("finetuned_tokenizer")
    gpt_model = GPTNeoForCausalLM.from_pretrained("finetuned_model")
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    while True:
        input_text = input("質問を入力してください（終了するには'q'を入力）: ")
        if input_text.lower() == 'q':
            break

        generated_answer = generate_answer(gpt_model, gpt_tokenizer, input_text, max_length=100)
        print(f"Generated answer: {generated_answer}")

if __name__ == "__main__":
    pdf_dir = './'
    pdf_texts = read_pdf_files(pdf_dir)
    main(pdf_texts)

