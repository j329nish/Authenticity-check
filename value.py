def process_text(text):
    #仮作成
    highlight_ranges = [(9, 19,0.03), (30, 39,0.113)]  # ハイライト範囲,スコア
    total_score = 0.60 #仮のAI度
    pred_label = 1 #仮ラベル 

    return pred_label,total_score,highlight_ranges
