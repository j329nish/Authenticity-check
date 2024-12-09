import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

    def forward(self, input_ids, output_attentions=False):  # output_attentions 引数を追加
        # output_attentions 引数をモデルに渡す
        outputs = self.model(input_ids, output_attentions=output_attentions)
        return outputs

model_name = "ku-nlp/roberta-base-japanese-char-wwm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
checkpoint_path = "model/houdou_Roberta_ver2.0.ckpt"
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
        outputs = model_loaded(inputs['input_ids'])
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
        positions.append((start, end)) 
        start += len(substring) 

    return positions

def process_text(text):
    pred_label, total_score = pred(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    special_tokens = tokenizer.all_special_tokens
    attention_weight = get_attention(inputs, pred_label, attentions[-1])

    # トークン化されたトークンを取得
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attention_list = []
    for i, (attention, token) in enumerate(zip(attention_weight, tokens)):
        if token not in special_tokens:  # 特殊トークンを除外
            attention_list.append({'token': token, 'attention': round(attention.item(), 3), 'index': i})

    # アテンションスコアでソート
    attention_list = sorted(attention_list, key=lambda x: abs(x['attention']), reverse=True)

    # トークン位置を元のテキストにマッピング
    highlight_ranges_and_score = []
    text_pointer = 0
    for i, attention in enumerate(attention_list[:20]):  # 上位10個のみ処理
        token_text = attention['token'].replace("##", "")  # サブワードを修正
        token_index = attention['index']
        
        # 元のテキスト内の一致する位置を計算
        start = text.find(token_text, text_pointer)
        if start != -1:
            end = start + len(token_text)
            highlight_ranges_and_score.append((start, end, attention['attention']))
            text_pointer = end  # 次の検索開始位置を更新

    return pred_label, total_score, highlight_ranges_and_score
