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

def get_attention(inputs, pred_label, attention_weight):
    seq_len = attention_weight.size()[2]
    all_attens = torch.zeros(seq_len).to(device)

    for i in range(12):
        all_attens += attention_weight[0, i, 0, :]
    return all_attens

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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512 )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions 
    special_tokens = tokenizer.all_special_tokens
    attention_weight = get_attention(inputs, pred_label, attentions[-1])
    non_zero_count = torch.sum(attention_weight != 0).item()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention_list = []
    for attention, token in zip(attention_weight, tokens):
        if token not in special_tokens:  # 特殊トークンを除外
            attention_list.append({'token': token, 'attention': round(attention.item(), 3)})
    attention_list = sorted(attention_list, key=lambda x: abs(x['attention']), reverse=True)



    highlight_ranges_and_score = []
    cnt = 0
    for attention in attention_list:
        positions = find_str_positions(text, attention['token'])
        for range in positions:
            t = range + (attention['attention'],)
            highlight_ranges_and_score.append(t)
        if cnt >= 9:
            break
        cnt += 1

    
    return pred_label, total_score, highlight_ranges_and_score
text ="ChatGPTやGeminiといった生成系AIが普及する中、AIが生成した文章には誤った情報が含まれる可能性があります。また、その文章がAIによって書かれたものか、人間によるものかを判別することがますます難しくなっています。誤情報は信頼性を損なう原因となるため、文章がAIによって生成されたものかどうかを識別することは非常に重要です。そこで本発表では、生成AIが書いた文章と人間が書いた文章を収集してデータセットを構築し、自然言語処理モデルを活用した真贋判定システムを提案します。"
print(process_text(text))