import streamlit as st
import plotly.graph_objects as go
from value import process_text  # value.pyから関数をインポート

def highlight_text(text, ranges):
    highlighted = ""
    last_idx = 0
    
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
    
# 完全な円メーターを描画する関数
def create_meter(percentage):
    # 進捗部分の割合と残り部分の割合を設定
    progress = percentage
    remaining = 100 - percentage
    
    # 円メーターを作成
    fig = go.Figure(data=[go.Pie(
        values=[progress, remaining],
        hole=0.5,  # ドーナツ型にするための穴の大きさ
        marker=dict(colors=['skyblue', 'lightgray']),  # 進捗部分と残り部分の色
        showlegend=False,  # 凡例を非表示にする
        textinfo="none",
        hoverinfo="percent"
    )])

    # グラフのレイアウト設定
    fig.update_layout(
        annotations=[dict(
            text=f"確信度:{percentage}%",
            x = 0.5,
            y = 0.5,
            font_size = 28,
            font_color = "black",
            showarrow = False
        )
        ],
        margin=dict(t=0, b=0, l=0, r=0)  # 上下左右の余白を小さく設定
    )
    
    return fig

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
        # スコアを一度だけ表示（パーセント表示）
        if pred_label!=1:
            st.write("""<h1 align="center">結果：人間</h1>""",unsafe_allow_html=True)
            #st.write(f"人間度: {total_score:.2f}%")
            fig = create_meter(total_score*100)
            st.plotly_chart(fig)
        else:
            st.write("""<h1 align="center">結果：AI</h1>""",unsafe_allow_html=True)
            #st.write(f"AI度: {total_score:.2f}%")
            fig = create_meter(total_score*100)
            st.plotly_chart(fig)

        # 複数のハイライト処理を行う
        highlighted_text = highlight_text(text, highlight_ranges)
        st.markdown(f"<pre>{highlighted_text}</pre>", unsafe_allow_html=True)
        
        
        
    else:
        st.warning("テキストを入力してください。")
        