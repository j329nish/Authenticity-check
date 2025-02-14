import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning.loggers import TensorBoardLogger

def make_dataset(tokenizer, max_length, texts, labels):
    dataset_for_loader = list()
    for text, label in zip(texts, labels):
        encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True)
        # tokenizerメソッドは辞書を返す。その辞書にラベルのIDも持たせる。
        encoding["labels"] = label

        # テンソルに変換
        encoding = {key: torch.tensor(value) for key, value in encoding.items()}

        # 前処理済みのデータを保存して次の文へ
        dataset_for_loader.append(encoding)
    return dataset_for_loader

#報道記事
data_path_train1="/home/hamamoto/pbl/dataset/houdou_train.csv"
data_path_train2="/home/hamamoto/pbl/dataset/GPT_houdou_train.csv"
data_path_val1="/home/hamamoto/pbl/dataset/houdou_valid.csv"
data_path_val2="/home/hamamoto/pbl/dataset/GPT_houdou_valid.csv"

#報道記事＋エッセイ＋取材
# data_path_train1="/home/hamamoto/pbl/dataset/article_train.csv"
# data_path_train2="/home/hamamoto/pbl/dataset/GPT_article_train.csv"
# data_path_val1="/home/hamamoto/pbl/dataset/article_valid.csv"
# data_path_val2="/home/hamamoto/pbl/dataset/GPT_article_valid.csv"


train_df1 = pd.read_csv(data_path_train1)
text_list1 = train_df1['body'].tolist()
label_list1 = train_df1['label'].tolist()
train_df2 = pd.read_csv(data_path_train2)
text_list2 = train_df2['body'].tolist()
label_list2 = train_df2['label'].tolist()

train_text_list = text_list1 + text_list2
train_label_list = label_list1 + label_list2

val_df1 = pd.read_csv(data_path_val1)
text_list3 = val_df1['body'].tolist()
label_list3 = val_df1['label'].tolist()
val_df2 = pd.read_csv(data_path_val2)
text_list4 = val_df2['body'].tolist()
label_list4 = val_df2['label'].tolist()

val_text_list = text_list3 + text_list4
val_label_list = label_list3 + label_list4

train_data = pd.DataFrame({
    'text': train_text_list,
    'label': train_label_list,
})
val_data = pd.DataFrame({
    'text': val_text_list,
    'label': val_label_list,
})


# モデルとトークナイザーの設定
model_name = "ku-nlp/roberta-base-japanese-char-wwm"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 512
dataset_train = make_dataset(tokenizer, max_length, [train_data.iloc[i]['text'] for i in range(len(train_data))], [train_data.iloc[i]["label"] for i in range(len(train_data))])
dataset_val = make_dataset(tokenizer, max_length, [val_data.iloc[i]['text'] for i in range(len(val_data))], [val_data.iloc[i]["label"] for i in range(len(val_data))])

# データローダを作成。訓練用データはシャッフルしながら使う。
# 検証用と評価用は損失の勾配を計算する必要がないため、バッチサイズを大きめにとれる。
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=False)


# RoBERTa分類器の定義
class RoBERTClassifier(pl.LightningModule):
    def __init__(self, model):
        super(RoBERTClassifier, self).__init__()
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
hparams = {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "max_epochs": 1000,
    "model_name": model_name,
    "max_length": max_length
}
logger = TensorBoardLogger(save_dir="logs/", name="Roberta_training")
logger.log_hyperparams(hparams) 
# モデルとコールバックの設定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=1,
    save_weights_only=True, dirpath="model/"
)

early_stopping = pl.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, mode="min", verbose=True
)
# 訓練
trainer = pl.Trainer(
    devices=[1], logger=logger,max_epochs=10000, callbacks=[checkpoint, early_stopping]
)
trainer.fit(RoBERTClassifier(model), dataloader_train, dataloader_val)

# ベストモデルの確認
print("ベストモデル: ", checkpoint.best_model_path)
print("ベストモデルの検証用データにおける損失: ", checkpoint.best_model_score)