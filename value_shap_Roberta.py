import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import torch.nn.functional as F

# RoBERTaの定義
class RoBERTaClassifier(pl.LightningModule):
    def __init__(self, model):
        super(RoBERTaClassifier, self).__init__()
        self.model = model
    def forward(self, input_ids, output_attentions=False):  
        outputs = self.model(input_ids, output_attentions=output_attentions)
        return outputs

model_name = "ku-nlp/roberta-base-japanese-char-wwm" #事前学習済みモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
checkpoint_path = "appri/model\epoch=9-step=520-v1.ckpt" #ファインチューニングモデル
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_loaded = RoBERTaClassifier.load_from_checkpoint(checkpoint_path, model=model)
model_loaded.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_loaded.to(device)
MAX_LENGTH = 512

#予測関数予測ラベルと確信度を返す。
def pred(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH  )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model_loaded(inputs['input_ids'])
        logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    probabilities = F.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_label].item()
    return predicted_label, confidence

#SHAP値計算のための関数
def f(sentences):
    input_ids = torch.tensor([tokenizer.encode(text, padding='max_length', max_length=MAX_LENGTH , truncation=True) for text in sentences]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits.detach().cpu().numpy()  
explainer = shap.Explainer(f, masker=tokenizer)

#SHAP値の計算
def calculate_shap_values(text):
    shap_values = explainer([text])
    return shap_values

#閾値を超えるトークンの計算
def filter_valid_indices(shap_class_values, threshold=0.001):
    valid_indices = [i for i, shap_value in enumerate(shap_class_values) if abs(shap_value) > threshold]
    return valid_indices

#同じSHAP値かつ連続するトークンをつなげる
#つなげたトークン、SHAP値、開始位置、終了位置を返す
def decode_tokens_and_positions(input_ids, shap_class_values, valid_indices):
    top_tokens = []
    top_shap_values = []
    top_start_positions = []
    top_end_positions = []

    prev_value = None
    prev_index = None
    combined_token = ""
    combined_start_pos = None
    combined_end_pos = None

    for i in valid_indices:
        if i < len(input_ids):
            decoded_token = tokenizer.decode([input_ids[i]])

            if decoded_token not in tokenizer.all_special_tokens:
                token_start_pos = tokenizer.decode(input_ids[:i]).count(" ")
                token_end_pos = token_start_pos + len(decoded_token)

                if shap_class_values[i] == prev_value and i == prev_index + 1:
                    combined_token += decoded_token
                    combined_end_pos = token_end_pos
                else:
                    if prev_value is not None:
                        top_tokens.append(combined_token)
                        top_shap_values.append(prev_value)
                        top_start_positions.append(combined_start_pos)
                        top_end_positions.append(combined_end_pos)
                    combined_token = decoded_token
                    combined_start_pos = token_start_pos
                    combined_end_pos = token_end_pos

                prev_value = shap_class_values[i]
                prev_index = i

    if prev_value is not None:
        top_tokens.append(combined_token)
        top_shap_values.append(prev_value)
        top_start_positions.append(combined_start_pos)
        top_end_positions.append(combined_end_pos)

    return top_tokens, top_shap_values, top_start_positions, top_end_positions

#SHAP値、開始位置、終了位置をタプルに
def create_highlight_ranges(top_tokens, top_shap_values, top_start_positions, top_end_positions):
    shape_list = [
        {'token': token, 'shape': round(value.item(), 3), 'start_pos': start_pos, 'end_pos': end_pos}
        for token, value, start_pos, end_pos in zip(top_tokens, top_shap_values, top_start_positions, top_end_positions)
    ]

    highlight_ranges_and_score = []
    for attention in shape_list:
        start_pos = attention['start_pos']
        end_pos = attention['end_pos']
        shape_value = attention['shape']
        highlight_ranges_and_score.append((start_pos, end_pos, shape_value))

    return highlight_ranges_and_score

#appから呼び出される関数
#予測ラベル、確信度、タプル(開始位置、終了位置、SHAP値)を返す
def process_text(text):
    pred_label, total_score = pred(text) #予測ラベルと確信度の取得

    shap_values = calculate_shap_values(text) # SHAP値の計算
    shap_class_values = shap_values.values[0, :, pred_label] #予測ラベルのSHAP値を取得

    input_ids = tokenizer.encode(text, padding='max_length', max_length=MAX_LENGTH , truncation=True) #テキストのトークン化

    valid_indices = filter_valid_indices(shap_class_values) #SHAP値が閾値以上のトークンを取得

    # トークンの連結、位置情報の取得
    top_tokens, top_shap_values, top_start_positions, top_end_positions = decode_tokens_and_positions(
        input_ids, shap_class_values, valid_indices
    )

    # タプルの作成
    highlight_ranges_and_score = create_highlight_ranges(
        top_tokens, top_shap_values, top_start_positions, top_end_positions
    )

    return pred_label, total_score, highlight_ranges_and_score

