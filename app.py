import streamlit as st
import plotly.graph_objects as go
from value_shap_Roberta import process_text  # バックから関数をインポート
from check_text_lengs import check_text_lengs

def highlight_text(text, ranges):
    highlighted = ""
    last_idx = 0
    ranges = sorted(ranges, key=lambda x: x[0])
    for start_idx, end_idx,score in ranges:
        # start_idx, end_idxがテキストの範囲内であるかを確認
        if start_idx < 0 or end_idx >= len(text) + 1 or start_idx >= end_idx:
            continue  # 範囲外の場合はスキップ
 
        color = get_highlight_color(score)
        # ハイライト処理
        highlighted += text[last_idx:start_idx]
        highlighted += f'<span style="background-color: {color};">{text[start_idx:end_idx]}</span>'
        last_idx = end_idx
   
    # 残りのテキストを追加
    highlighted += text[last_idx:]
   
    return highlighted

def get_highlight_color(score):
    """
    スコアに基づいてハイライト色を決定する関数
    """
    if score >= 0:
        return f'rgb({255},{255 - min(score * 8 * 255,255)},{255 - min(score * 8 * 255,255)})' #AI度を赤く描写
    else:
        score *= -1
        return f'rgb({255 - min(score * 8 * 255,255)},{255 - min(score * 8 * 255,255)},{255})' #人間度を青く描写
    

st.title("真贋判定アプリ")

@st.dialog("使い方")
def using():
    st.write("ファイル読み込み欄にtxtファイルをアップロードするか直接文字を打ち込むことにより判定対象の文字列を入力することができ、対象の文字列が入力されている状態で判定ボタンを押下すると判定が開始されます。")
    st.write("判定結果において、結果と確信度が表示され、その下に結果に影響を及ぼした部分がハイライト表示されて文章が表示されます。赤くなっている部分が影響が大きい部分、青くなっている部分が影響が小さい部分に対応します。")

if st.button("使い方"):
    using()
uploaded_file = st.file_uploader("テキストファイルをアップロード", type='txt')
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
else:
    text = st.text_area("テキストを入力")

if st.button("判定"):
    if text:
        if(check_text_lengs(text)):
            # テキストをvalues.pyに渡して処理し、複数の範囲とスコアを受け取る
            pred_label, total_score, highlight_ranges = process_text(text)
            # スコアを一度だけ表示（パーセント表示）
            if pred_label!=1:
                st.write("""<h1 align="center">結果：人間</h1>""",unsafe_allow_html=True)
            else:
                st.write("""<h1 align="center">結果：AI</h1>""",unsafe_allow_html=True)
            score = round(100 * total_score,2)
            st.write(f"""<h3 align="center">確信度:{score:.2f}%</h3>""",unsafe_allow_html=True)
            st.progress(total_score)

            # 複数のハイライト処理を行う
            highlighted_text = highlight_text(text, highlight_ranges)
            st.markdown(f"<pre>{highlighted_text}</pre>", unsafe_allow_html=True)
        else:
            st.warning("テキストが長すぎます。")
        
        
    else:
        st.warning("テキストを入力してください。")
        