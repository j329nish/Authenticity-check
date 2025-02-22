# 真贋判定アプリ（Authenticity-check Apps）

新聞記事を見て、人間が書いた記事かAIが書いた記事かを自動で判定するWebアプリです。

## 1.概要

<img src="image/attempt.png" width="70%">

近年、大規模言語モデルの進化に伴って自動生成文章は流暢性を増していますが、誤情報を含む自動生成文章の拡散は社会に悪影響を及ぼすリスクがあります。私たちはこの問題に取り組むため、新聞記事の見出しから大規模言語モデルを用いて本文相当の文章を自動生成しました。その後、これらの記事と実際の新聞記事を言語モデルで学習させ、AIが生成した記事を識別するWebアプリを作成しました。

## 2.使用方法

１.前準備

Google Colab用のファイルを開き、画面上部のバナーで"ランタイム"をクリックし、続いて"ランタイムのタイプを変更"をクリックすることにより使用するハードウェアを変更できます。GPUの部分にチェックが入っていることを確認して保存してください。<br>
次に、画面左側のバナーにあるフォルダアイコンをクリックし、app.py、value_shap_roberta.py、check_text_lengs.py、roberta.ckptの4つをドラッグ&ドロップしてください。<br>
また、本アプリケーションの利用にはngrokのアカウントが必要となります。[こちら](https://ngrok.com/)でngrokのアカウントを作成していただき、ngrokのメニューバーから"Your Authtoken"を選択後トークンの文字列を本アプリケーションにおけるコード上の"トークンを入力してください"の部分に入力してください。

２.アプリケーションの使い方

ファイルがアップロードされ、トークンが入力されている状態でGoogle Colabを実行していただくとリンクが生成されます。リンクの先でアプリケーションを操作することができます。<br>
txtファイルを上の欄にドラッグ&ドロップしていただくか下の欄にテキストを打ち込むことにより判定するテキストを入力できます。<ins>※txtファイルは1ファイルしか読み込めません。</ins><br>
テキストがある状態で"判定"ボタンをクリックすると、判定が開始されます。判定が終了すると判定結果が画面下部に表示されます。<ins>※文字数が多い場合は判定出来ません。</ins>

## 3.機能説明

|分類|機能名|機能説明|
|:-|:-|:-|
|入力|記事の入力機能|textファイルに書かれている記事、記事を入力できます。|
|ボタン|使い方の説明|Webアプリの使い方についての説明を表示します。|
||判定機能|入力した記事を見て人間が書いた記事かAIが書いた記事かを判定します。|
|出力|判定結果の表示|判断結果とその確信度を表示します。|
||判断根拠の表示|判断根拠となる単語がハイライト表示されます。赤色だと影響度が高く、青色だと負の影響度が高いです。また、濃い色ほど影響度が高く、薄い色ほど影響度が少ないです。|

## 4.使用技術

|Category|Tecknology Stack|
|:-|:-|
|Frontend|Python, Google Colab, Streamlit, ngrok|
|Backend|Python, RoBERTa, Shap|

## 5.ファイル構成

main/<br>
┣app.ipynb<br>
┣app.py<br>
┣check_text_lengs.py<br>
┣value_shap_roberta.py<br>
└train.py<br>
※roberta.ckptはありません。<br>

## 6.Webアプリのデモ

Webアプリを立ち上げた時の画面<br>
<img src="image/before-streamlit.png" width="70%">

真贋判定を行った結果の画面<br>
<img src="image/after-streamlit.png" width="70%">

## クレジット

作成者<br>
西田：https://github.com/j329nish<br>
濱本：https://github.com/j348hama<br>
松浦：https://github.com/j396mats<br>

