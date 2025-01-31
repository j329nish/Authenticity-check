## 真贋判定アプリ（Authenticity-check）

新聞記事を見て、人間が書いた記事かAIが書いた記事かを自動で判定するWebアプリです。

## 概要

<img src="image/attempt.png" width="70%">

近年、大規模言語モデルの進化に伴って自動生成文章は流暢性を増していますが、誤情報を含む自動生成文章の拡散は社会に悪影響を及ぼすリスクがあります。私たちはこの問題に取り組むため、新聞記事の見出しから大規模言語モデルを用いて本文相当の文章を自動生成しました。その後、これらの記事と実際の新聞記事を言語モデルで学習させ、AIが生成した記事を識別するWebアプリを作成しました。

## 機能説明

・ txtファイルのアップロード、テキストの記入が可能<br>
・ 「判定」ボタンを押すと判定を開始（CPUを使用するため約5分かかります）<br>
・ 判定が終了すると、判定結果、確信度が表示されます。<br>
・ また、判定を行った文章が表示され、判断結果の根拠となる部分がハイライト表示されます。

## Webアプリのデモ

Webアプリを立ち上げた時の画面<br>
<img src="image/before-streamlit.png" width="70%">

真贋判定を行った結果の画面<br>
<img src="image/after-streamlit.png" width="70%">

## 使用技術

|Category|Tecknology Stack|
|:-|:-|
|Frontend|Python, Google Colab, Streamlit, ngrok|
|Backend|Python, RoBRTa, Shap|

## 使用方法

１．プロジェクトのインストール<br>
２．仮想環境の構築<br>
３．依存関係のインストール<br>


## （作成者向け）GitHubの使い方

GitHubとは、ソースコードの管理や共同作業を行うためのウェブプラットフォームです。成果物の管理を共有でき、チーム開発がしやすいため、多くのITエンジニアに重宝されています。
GitHubは以下のような形で構成されてます。

<img src="image/GitHubComponents.png" width="70%">

### GitHubアカウントの作成
1. gitアカウントの作成
    
    [GitHub](https://github.com/)に入りアカウントの作成をする
    
2. アクセストークンの取得
    
    GitHubのホーム画面へ行く
    
    右のアバター画像をクリック→「Settings」へ
    
    左メニューの「Developer settings」を押す
    
    左メニューの「Personal access tokens」→「Tokens（classic）」を押す
    
    「Generate new token」を押す
    
    Noteに「git client token」など、トークンだと分かるような名前を付ける
    
    「repo」のすべてのチェックマークをクリックし、「Update token」を押す
    
    期間を設定し（30日とかで大丈夫）ボタンを押すとアクセストークンが出てくる
    
    アクセストークンを忘れない場所へ保存する
    
    ※忘れた場合、期限が切れた場合は3.へ
    
3. アクセストークンの更新
    
    2.を参考にTokens（classic）を押す
    
    「Regenerate token」を押す
    
    出てきたアクセストークンを保存する

### リポジトリとmainブランチの作成
1. リモートリポジトリの作成
    
    GitHubのホームで、「New」ボタンを押す
    
    リポジトリの名前を決定し、「Create repository」を押す
    
    ※リポジトリの名前は、プロジェクトの名前が良い
    
2. ローカルリポジトリの作成
    
    使用するディレクトリをPC上に作成
    
    GitBushを使用し、使用するディレクトリまで移動
    
```
ローカルリポジトリの作成
git init
    
ステージングに追加（-Aで全ファイルを選択可能）
git add
    
コミット  ※変更箇所が分かるようにすること
git commit -m “{説明}”
    
リモートリポジトリとローカルリポジトリの紐付け
git remote add {リモートリポジトリ名} {リモートリポジトリURL}
    
プッシュ
git push -u {リモートリポジトリ名} {ブランチ名}
```

※ここでのリモートリポジトリ名は、originにする

※リモートリポジトリURLは、GitHubのリポジトリに入り、「Code」ボタンを押すと取得可能

以下のように変更して記述する
```
https://github.com/xx/xx.git→https://{ユーザ名}:{アクセストークン}@github.com/xx/xx.git
```

### 2つ目以降のブランチの作成

```
1. mainブランチに移動
git checkout main

2. リモートリポジトリのmainブランチを最新状態にする
git pull origin main

3. ローカルでsecondブランチを作成
git checkout -b {新しいブランチ}

4. プッシュ
git push -u origin {新しいブランチ}
```

※2つ目以降のブランチを作成する際は、mainを反映する

### 外部アカウントでのリポジトリへの参加
1. 使用するディレクトリをPC上に作成

2. GitBushを使用し、使用するディレクトリまで移動

3. ローカルリポジトリの作成
```
git init 
```

4. 古いとmasterとつくので、これを{ブランチ名}に変える
```
git branch -m {ブランチ名}
```
※ブランチ名は作業する人によって変わる。西田の場合、j329nishとなる。
    
5. ターミナルに入力して、ユーザー情報を設定
```
git config --global [user.name](http://user.name/) "Your Name"
git config --global user.email "[you@example.com](mailto:you@example.com)"
```

6. ディレクトリにファイルを取り込む
```
git clone {リポジトリURL}
```

### ファイルの更新と削除（ローカルからリモート）
#### ファイルの更新

VSCodeを使用する場合

1. 作業用ディレクトリでVSCodeに入る
2. 左メニューの3つの○が結合したマークを押す
3. 変更を加えたファイルが表示されるため、そのファイルの「+」を押す
4. メッセージ欄に変更箇所が分かるように説明を書く
5. コミットを押す
6. GitHubでログイン認証がされるため、それを許可する

GitBushを使用する場合

1. 自分のPC内の作業用ディレクトリに入る
2. GitBushを使用し、作業用ディレクトリまで移動
3. 変更を加えるブランチに移動（$ git checkout {ブランチ名}）
4. ステージングに追加（$ git add）-Aで全ファイルを選択可能
5. コミット（$ git commit -m “{説明}”）※変更箇所が分かるようにすること
6. プッシュ（$ git push -u {リモートリポジトリ名} {ブランチ名}）

#### ファイルの削除

1. 自分のPC内の作業用ディレクトリに入る
2. ファイルを削除
3. GitBushを使用し、作業用ディレクトリまで移動
4. 変更を加えるブランチに移動（$ git checkout {ブランチ名}）
5. ステージングに追加（$ git add .）
6. コミット（$ git commit -m “{説明}”）※変更箇所が分かるようにすること
7. プッシュ（$ git push {リモートリポジトリ名} {ブランチ名}）

### ファイルの更新と削除（リモートからローカル）
#### ファイルの更新

1. GitHubでファイルの内容を更新
2. GitBushを使用し、使用するディレクトリまで移動
3. 作業しているブランチに移動（$ git checkout {ブランチ名}）
4. リモートリポジトリの情報を取得（$ git pull origin {ブランチ名}）

#### ファイルの削除

1. GitHubで削除するファイルを押す
2. 「Delete file」を押し、コミット&プッシュ
3. GitBushを使用し、使用するディレクトリまで移動
4. 作業しているブランチに移動（$ git checkout {ブランチ名}）
5. リモートリポジトリの情報を取得（$ git pull origin {ブランチ名}）

### vimの使い方
vimを立ち上げる（$ vim {ファイル名}）

挿入（Esc + i ）

保存（Esc + :wq）

保存せず終了（Esc + :q!）

### ブランチでの統合作業
下級ブランチ→mainブランチへの統合

1. GitHubのホームで、「New pull request」を押す　
    
    ※プルリクエストができない場合「コンフリクトへの対応」ページへ
    
2. baseをmain、compareを結合する元のブランチ名とする
3. プルリクエストを作成するため、「Create pull request」を押す
4. プルリクエストのタイトルを適当なものに変更し、コメントがあれば追記する
5. チームメンバーと変更点を確認し、レビューを行う※重要
6. 問題がなければ再び「Create pull request」を押す
7. mainブランチに結合するため、「Merge pull request」を押す
8. 「Confirm merge」を押す
9. （元のブランチが必要なくなったら、「Delete branch」を押して削除する）

### コンフリクトへの対応
ローカルリポジトリ上でmainブランチ→結合元ブランチへの統合を行う

1. GitBushで作業用ディレクトリへ移動する
2. 作業するディレクトリをmainに切り替える（$ git checkout main）
3. ローカルのmainブランチを最新にする（$ git pull）
4. 結合元ブランチへ移動（$ git checkout {結合元ブランチ}）
5. mainブランチを結合元ブランチへマージ（$ git marge main）
6. コンフリクトの解消（IDEの機能を利用or手作業）
7. 変更したファイルの更新を行う ※「変更したファイルの更新」ページへ
8. 結合作業へ移行　※「ブランチでの結合作業」ページへ

## ライセンス

このプロジェクトは...ライセンスのもとで公開されています。

## クレジット

作成者：<br>
西田：https://github.com/j329nish<br>
濱本：https://github.com/j348hama<br>
松浦：https://github.com/j396mats<br>

