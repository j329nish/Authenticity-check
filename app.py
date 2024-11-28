import streamlit as st
from value import process_text  # value.pyから関数をインポート

def highlight_text(text, ranges):
    highlighted = ""
    last_idx = 0
    
    for start_idx, end_idx,score in ranges:
        # start_idx, end_idxがテキストの範囲内であるかを確認
        if start_idx < 0 or end_idx >= len(text) or start_idx >= end_idx:
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

uploaded_file = st.file_uploader("テキストファイルをアップロード", type='txt')
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
else:
    text = st.text_area("テキストを入力")

if st.button("判定"):
    if text:
        # テキストをvalues.pyに渡して処理し、複数の範囲とスコアを受け取る
        pred_label, total_score, highlight_ranges = process_text(text)

        # 複数のハイライト処理を行
        highlighted_text = highlight_text(text, highlight_ranges)
        st.markdown(f"<pre>{highlighted_text}</pre>", unsafe_allow_html=True)
        
        # スコアを一度だけ表示（パーセント表示）
        if pred_label!=1:
            st.write(f"結果：人間")
            st.write(f"人間度: {1 - total_score:.2f}%")
        else:
            st.write(f"結果：AI")
            st.write(f"AI度: {total_score:.2f}%")
        
    else:
        st.warning("テキストを入力してください。")
        