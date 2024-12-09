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
        positions.append((start, end)) 
        start += len(substring) 

    return positions

def process_text(text): 
    pred_label, total_score = pred(text)

    # SHAP値の計算
    shap_values = explainer([text])
    shap_class_values = shap_values.values[0, :, pred_label]

    # トークンのエンコード
    input_ids = tokenizer.encode(text, padding='max_length', max_length=512, truncation=True)

    # SHAP値に基づいて上位トークンを取得
    top_indices = torch.topk(torch.tensor(shap_class_values), k=min(40, len(input_ids))).indices.numpy()
    
    # トークンとSHAP値を格納するリスト
    top_tokens = []
    top_shap_values = []
    top_start_positions = []  # 開始位置を格納
    top_end_positions = []    # 終了位置を格納
    
    prev_value = None
    prev_index = None
    combined_token = ""
    combined_start_pos = None
    combined_end_pos = None
    
    for i in top_indices:
        if i < len(input_ids):  # インデックス範囲を確認
            decoded_token = tokenizer.decode([input_ids[i]])

            if decoded_token not in tokenizer.all_special_tokens and shap_class_values[i] > 0.001:  # 特殊トークンとSHAP値が0.001以下のものを除外
                token_start_pos = tokenizer.decode(input_ids[:i]).count(" ")  # 現トークンの開始位置
                token_end_pos = token_start_pos + len(decoded_token)  # 現トークンの終了位置
                
                # SHAP値が同じかつインデックスが連続している場合はトークンを連結
                if shap_class_values[i] == prev_value and i == prev_index + 1:
                    combined_token += decoded_token  # トークンを連結
                    combined_end_pos = token_end_pos  # 終了位置を更新
                else:
                    if prev_value is not None:
                        top_tokens.append(combined_token)
                        top_shap_values.append(prev_value)
                        top_start_positions.append(combined_start_pos)
                        top_end_positions.append(combined_end_pos)
                    combined_token = decoded_token
                    combined_start_pos = token_start_pos  # 新しいトークンの開始位置
                    combined_end_pos = token_end_pos  # 新しいトークンの終了位置
                prev_value = shap_class_values[i]
                prev_index = i
    
    # 最後に追加
    if prev_value is not None:
        top_tokens.append(combined_token)
        top_shap_values.append(prev_value)
        top_start_positions.append(combined_start_pos)
        top_end_positions.append(combined_end_pos)

    # トークンとSHAP値をまとめる
    shape_list = [{'token': token, 'shape': round(value.item(), 3), 'start_pos': start_pos+1, 'end_pos': end_pos+1} 
                  for token, value, start_pos, end_pos in zip(top_tokens, top_shap_values, top_start_positions, top_end_positions)]


    # トークン位置の特定
    highlight_ranges_and_score = []
    for attention in shape_list:
        token = attention['token']
        shape_value = attention['shape']
        start_pos = attention['start_pos']
        end_pos = attention['end_pos']

        # トークンの位置を特定（開始位置と終了位置を使う）
        positions = [(start_pos, end_pos)]  # ここで位置を直接指定

        # トークン位置とSHAP値をまとめる
        for range_ in positions:
            t = range_ + (shape_value,)
            highlight_ranges_and_score.append(t)

    return pred_label, total_score, highlight_ranges_and_score


#text ="愛媛労働局は、新型コロナウイルス感染症の影響により県内の企業が申請した雇用調整助成金の支給状況を発表した。9月末時点で、申請件数は385件、支給額は約2.1億円に上る。同局は雇用維持を目的に、特例措置を講じながら支給の迅速化を図っている。取材によると、一部の中小企業では依然として経営の厳しさが続いており、助成金の支援を受けることで何とか従業員の雇用を守っているという。松山市にある製造業の「佐藤製作所」では、需要減少に伴い生産調整を余儀なくされているが、雇用調整助成金により従業員の解雇を回避できたと佐藤社長が語る。一方、手続きの煩雑さや情報提供の不足により、一部の企業は助成金の申請に難儀している。県内の商工会議所も、これを受けて申請サポートを充実させる動きを見せている。さらなる改善が求められるが、愛媛労働局の藤田課長は「企業の皆様が安心して活用できるよう、今後も支援体制を強化していく」と述べた。新型コロナの影響が長引く中、迅速で適切な支援が求められている。"
#print(process_text(text))
