def process_text(text):
    #仮作成
    highlight_ranges = [(10, 20,0.03), (30, 40,0.113)]  # ハイライト範囲,スコア
    total_score = 60.0 #仮のAI度
    pred_label = 1 #仮ラベル 

    return pred_label,total_score,highlight_ranges
