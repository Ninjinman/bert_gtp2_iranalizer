import os
import sys
import io
import torch
import pdfplumber
from transformers import BertTokenizer, BertModel, GPTNeoForCausalLM, GPT2Tokenizer
from googletrans import Translator

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def translate_text(text, target_language="en"):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

def read_pdf_files(pdf_dir):
    pdf_texts = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            with pdfplumber.open(pdf_path) as pdf:
                pdf_text = ' '.join(page.extract_text() for page in pdf.pages)
                pdf_texts.append(pdf_text)
    return pdf_texts

def generate_answer(model, tokenizer, input_text, max_length):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    generated_output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return generated_text

def main():
    pdf_dir = './'
    input_text = "コロナウイルスの影響が２年続いた場合、株価はどう変動しますか？"
    # Translate the input text to English
    translated_input_text = translate_text(input_text)

    pdf_texts = read_pdf_files(pdf_dir)

    pdf_texts = read_pdf_files(pdf_dir)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
    embeddings = []
    for pdf_text in pdf_texts:
        inputs = bert_tokenizer(pdf_text, return_tensors='pt', max_length=512, truncation=True)
        outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach())


    gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    generated_answer = generate_answer(gpt_model, gpt_tokenizer, translated_input_text, max_length=150)
    print(f"Generated answer: {generated_answer}")

if __name__ == "__main__":
    main()
