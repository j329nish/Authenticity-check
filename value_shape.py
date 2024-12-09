import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import shap
import torch.nn.functional as F

# LUKE分類器の定義
class LUKEClassifier(pl.LightningModule):

    def __init__(self, model):
        super(LUKEClassifier, self).__init__()
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

model_name = "studio-ousia/luke-japanese-base-lite"
tokenizer = AutoTokenizer.from_pretrained(model_name)
checkpoint_path = "model/tekagemi_luke_vr1.ckpt"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model_loaded = LUKEClassifier.load_from_checkpoint(checkpoint_path, model=model)
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
def f(sentences):
    input_ids = torch.tensor([tokenizer.encode(text, padding='max_length', max_length=68, truncation=True) for text in sentences]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.logits.detach().cpu().numpy()  
explainer = shap.Explainer(f, masker=tokenizer)

def find_str_positions(text, substring):
    positions = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        end = start + len(substring)
        positions.append((start + 1, end)) 
        start += len(substring) 

    return positions



def process_text(text):
    pred_label, total_score = pred(text)
    shap_values = explainer([text])
    shap_class_values = shap_values.values[0, :, pred_label]
    input_ids = tokenizer.encode(text, padding='max_length', max_length=68, truncation=True)
    top_indices = torch.topk(torch.tensor(shap_class_values), 12).indices.numpy()
    top_tokens = [tokenizer.decode([input_ids[i]]) for i in top_indices]
    top_shap_values = [shap_class_values[i] for i in top_indices]


    shape_list = []
    special_tokens = tokenizer.all_special_tokens
    for token, value in zip(top_tokens, top_shap_values):
        if token not in special_tokens:  # 特殊トークンを除外
            shape_list.append({'token': token, 'shape': round(value.item(), 3)})
    highlight_ranges_and_score = []
    for attention in shape_list:
        positions = find_str_positions(text, attention['token'])
        for range in positions:
            t = range + (attention['shape'],)
            highlight_ranges_and_score.append(t)
    
    return pred_label, total_score, highlight_ranges_and_score

