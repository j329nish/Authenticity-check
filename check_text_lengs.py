from transformers import AutoTokenizer

def check_text_lengs(text: str, max_length: int = 512) -> bool:
    model_name = "ku-nlp/roberta-base-japanese-char-wwm"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(len(tokenizer.tokenize(text)) )
    return len(tokenizer.tokenize(text)) <= max_length

