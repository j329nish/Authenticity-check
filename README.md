## 真贋判定アプリ（Authenticity-check）

新聞記事を見て、人間が書いた記事かAIが書いた記事かを自動で判定するWebアプリです。

## 概要

<img src="image/attempt.png" width="70%">

近年、大規模言語モデルの進化に伴って自動生成文章は流暢性を増していますが、誤情報を含む自動生成文章の拡散は社会に悪影響を及ぼすリスクがあります。私たちはこの問題に取り組むため、新聞記事の見出しから大規模言語モデルを用いて本文相当の文章を自動生成しました。その後、これらの記事と実際の新聞記事を言語モデルで学習させ、AIが生成した記事を識別するWebアプリを作成しました。

## 機能一覧

(作成予定)

## 使用技術

|Category|Tecknology Stack|
|:-|:-|
|Frontend|Python, Streamlit, Flask|
|Backend|Python, RoBRTa, SHAP, LIME|

## インストール方法

以下の手順でプロジェクトをローカル環境にインストールしてください。

```
ディレクトリを作成し移動
mkdir repository
cd repository

リポジトリをクローン
git clone https://github.com/j329nish/Authenticity-check.git

ディレクトリに移動
cd repository

依存関係をインストール
pip install -r requirements.txt
```

## 使い方

実行方法の例

```
python main.py --option value
```

## （作成者向け）GitHubの使い方

GitHubとは、ソースコードの管理や共同作業を行うためのウェブプラットフォームです。成果物の管理を共有でき、チーム開発がしやすいため、多くのITエンジニアに重宝されています。
GitHubは以下のような形で構成されてます。

<img src="image/GitHubComponents.png" width="70%">

### リポジトリとmainブランチの作成

### 2つ目以降のブランチの作成

### 外部アカウントでのリポジトリへの参加

### ファイルの更新と削除（ローカルからリモート）

### ファイルの更新と削除（リモートからローカル）

### vimの使い方

### ブランチでの統合作業

### コンフリクトへの対応

## ライセンス

このプロジェクトは...ライセンスのもとで公開されています。

## クレジット

作成者：西田+他2名..


