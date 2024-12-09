import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap
import torch.nn.functional as F
import numpy as np

# LUKE分類器の定義
class RoBERTaClassifier(pl.LightningModule):

    def __init__(self, model):
        super(RoBERTaClassifier, self).__init__()
        self.model = model
        self.validation_losses = []  # 検証損失を保存するリスト

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        val_loss = output.loss
        self.validation_losses.append(val_loss)  # 検証損失をリストに追加
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self):
        # 各エポック終了時に検証損失を平均化してログ
        avg_val_loss = torch.stack(self.validation_losses).mean()
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.validation_losses.clear()  # リストをリセット

    def test_step(self, batch, batch_idx):
        output = self.model(**batch)
        labels_predicted = output.logits.argmax(-1)
        labels = batch.pop("labels")
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def forward(self, **inputs):
        return self.model(**inputs)

model_name = "ku-nlp/roberta-base-japanese-char-wwm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
checkpoint_path = "model/houdou_Roberta_ver2.0.ckpt"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_loaded = RoBERTaClassifier.load_from_checkpoint(checkpoint_path, model=model)
model_loaded.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_loaded.to(device)

def pred(sentence):
    # 文をエンコード（トークンID、attention maskを取得）
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=512 )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model_loaded(**inputs)
        logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    probabilities = F.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_label].item()
    return predicted_label, confidence

from typing import List
max_len = 512
class_names = ['human', 'AI']


def find_str_positions(text, substring):
    positions = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        end = start + len(substring)
        positions.append((start, end)) 
        start += len(substring) 

    return positions

def predictor(texts: List, max_length: int = max_len) -> np:
    # 文章をID化する
    encoding = tokenizer.batch_encode_plus(
                texts, 
                add_special_tokens=True,
                padding="max_length", 
                max_length=max_length,
                truncation=True)
    
    input_ids = torch.tensor(encoding['input_ids']).to(device)
    
    # 学習済みモデルによる推論
    with torch.no_grad():
        output = model(input_ids, output_attentions=True)  # Correct argument
    
    # output.logitsにsoftmaxを適用して確率を計算
    probas = F.softmax(output.logits, dim=1).cpu().detach().numpy()

    return probas

explainer = LimeTextExplainer(class_names=class_names)


def process_text(text):
    pred_label, total_score = pred(text)
    lime_list = []
    
    # LIMEを使って説明を得る
    exp = explainer.explain_instance(text, predictor, num_features=20, num_samples=70)
    
    # 返されたラベルの中で最も高いスコアを持つラベルを取得
    top_label = exp.available_labels()[0]  # 最も高いスコアを持つラベル
    
    for i in exp.as_list(label=top_label):
        lime_list.append({'token': i[0], 'lime': round(i[1], 3)})#-が人間ぽくって

    highlight_ranges_and_score = []
    cnt = 0
    for attention in lime_list:
        positions = find_str_positions(text, attention['token'])
        for range in positions:
            t = range + (attention['lime'],)
            highlight_ranges_and_score.append(t)
        if cnt >= 9:
            break
        cnt += 1
    
    return pred_label, total_score, highlight_ranges_and_score