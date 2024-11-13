import streamlit as st
from value import process_text  # value.pyから関数をインポート

def highlight_text(text, ranges, color):
    highlighted = ""
    last_idx = 0
    
    for start_idx, end_idx in ranges:
        # start_idx, end_idxがテキストの範囲内であるかを確認
        if start_idx < 0 or end_idx >= len(text) or start_idx >= end_idx:
            continue  # 範囲外の場合はスキップ

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
    if score >= 80:
        return "red"
    elif score >= 50:
        return "yellow"
    else:
        return "green"

st.title("真贋判定アプリ")

uploaded_file = st.file_uploader("テキストファイルをアップロード", type='txt')
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
else:
    text = st.text_area("テキストを入力")

if st.button("判定"):
    if text:
        # テキストをvalues.pyに渡して処理し、複数の範囲とスコアを受け取る
        highlight_ranges, score = process_text(text)

        # スコアに基づいてハイライト色を決定
        highlight_color = get_highlight_color(score)

        # 複数のハイライト処理を行う
        highlighted_text = highlight_text(text, highlight_ranges, highlight_color)
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # スコアを一度だけ表示（パーセント表示）
        st.write(f"AI度: {score:.2f}%")
    else:
        st.warning("テキストを入力してください。")
