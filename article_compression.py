import pandas as pd
from bs4 import BeautifulSoup
import re
import csv

def remove_rows_containing_strings(input_file_path, output_file_path, strings_to_remove):
    """指定された文字列のリストが 'midasi' カラムに含まれている行を削除し、結果を新しいCSVファイルに保存します。"""
    try:
        df = pd.read_csv(input_file_path, dtype={0: str, 13: str, 23: str}, low_memory=False)
        
        filter_condition = df['midasi'].apply(lambda x: any(s in str(x) for s in strings_to_remove))
        filtered_df = df[~filter_condition]

        filtered_df.to_csv(output_file_path, index=False)
        print(f"フィルタリングされたデータを {output_file_path} に保存しました。")
    except FileNotFoundError:
        print("指定されたファイルが見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def replace_symbols_with_space(input_file_path, output_file_path, symbols):
    """
    指定したCSVファイルの 'honbun' カラムから特定の記号を空白に置き換えて、新しいCSVファイルとして保存する関数。
    """
    try:
        df = pd.read_csv(input_file_path, dtype={'honbun': str}, low_memory=False)

        pattern = '[' + re.escape(''.join(symbols)) + ']'
        df['honbun'] = df['honbun'].str.replace(pattern, ' ', regex=True)

        df.to_csv(output_file_path, index=False)
        print(f"処理結果を {output_file_path} に保存しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

def convert_ruby_to_kanji(input_file_path, output_file_path):
    """指定されたCSVファイルの 'honbun' カラムに含まれるルビを漢字に変換し、残りのテキストと結合します。"""
    try:
        df = pd.read_csv(input_file_path, dtype={0: str, 13: str, 23: str}, low_memory=False)

        def replace_ruby_with_kanji(ruby_text):
            """<ruby> タグを漢字に変換し、元のテキストを保持する関数。"""
            soup = BeautifulSoup(ruby_text, 'html.parser')
            for ruby in soup.find_all('ruby'):
                kanji = ruby.get_text()
                ruby.replace_with(kanji)
            return str(soup)

        df['honbun'] = df['honbun'].apply(replace_ruby_with_kanji)

        df.to_csv(output_file_path, index=False)
        print(f"変換されたデータを {output_file_path} に保存しました。")
    except FileNotFoundError:
        print("指定されたファイルが見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def convert_fullwidth_to_halfwidth(input_file, output_file):
    def fullwidth_to_halfwidth(text):
        if isinstance(text, str):
            return text.translate(str.maketrans({
                '０':'0', '１':'1', '２':'2', '３':'3', '４':'4',
                '５':'5', '６':'6', '７':'7', '８':'8', '９':'9',
                'Ａ':'A', 'Ｂ':'B', 'Ｃ':'C', 'Ｄ':'D', 'Ｅ':'E',
                'Ｆ':'F', 'Ｇ':'G', 'Ｈ':'H', 'Ｉ':'I', 'Ｊ':'J',
                'Ｋ':'K', 'Ｌ':'L', 'Ｍ':'M', 'Ｎ':'N', 'Ｏ':'O',
                'Ｐ':'P', 'Ｑ':'Q', 'Ｒ':'R', 'Ｓ':'S', 'Ｔ':'T',
                'Ｕ':'U', 'Ｖ':'V', 'Ｗ':'W', 'Ｘ':'X', 'Ｙ':'Y',
                'Ｚ':'Z', 'ａ':'a', 'ｂ':'b', 'ｃ':'c', 'ｄ':'d',
                'ｅ':'e', 'ｆ':'f', 'ｇ':'g', 'ｈ':'h', 'ｉ':'i',
                'ｊ':'j', 'ｋ':'k', 'ｌ':'l', 'ｍ':'m', 'ｎ':'n',
                'ｏ':'o', 'ｐ':'p', 'ｑ':'q', 'ｒ':'r', 'ｓ':'s',
                'ｔ':'t', 'ｕ':'u', 'ｖ':'v', 'ｗ':'w', 'ｘ':'x',
                'ｙ':'y', 'ｚ':'z',
                '　':' ',  
                '！':'!', '“':'"', '”':'"', '＃':'#', '＄':'$',
                '％':'%', '＆':'&', '’':"'", '（':'(', '）':')',
                '＊':'*', '＋':'+', '，':',', '－':'-', '．':'.',
                '／':'/', '：':':', '；':';', '＜':'<', '＝':'=',
                '＞':'>', '？':'?', '＠':'@', '［':'[', '＼':'\\',
                '］':']', '＾':'^', '＿':'_', '｀':'`', '｛':'{',
                '｜':'|', '｝':'}', '〜':'~'
            }))
        return text

    try:
        df = pd.read_csv(input_file, dtype={0: str, 13: str, 23: str}, low_memory=False)
        
        if 'honbun' in df.columns:
            df['honbun'] = df['honbun'].apply(fullwidth_to_halfwidth)
        else:
            print("Error: 'honbun' column not found in the CSV file.")
            return

        df.to_csv(output_file, index=False)
        print(f"Converted data saved to {output_file}")
    except FileNotFoundError:
        print("指定されたファイルが見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def filter_text_length(input_file, output_file):
    df = pd.read_csv(input_file)
    
    filtered_df = df[(df['honbun'].str.len() > 299) & (df['honbun'].str.len() < 501)]
    
    filtered_df.to_csv(output_file, index=False)


def split_csv(input_file, output_prefix, lines_per_file=150):
    try:
        with open(input_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader) 
            
            file_count = 1
            lines = []
            
            for i, row in enumerate(reader, start=1):
                lines.append(row)
                if i % lines_per_file == 0:
                    output_file = f"{output_prefix}_{file_count:03}.csv"
                    with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
                        writer = csv.writer(out_file)
                        writer.writerow(headers) 
                        writer.writerows(lines)
                    lines = []  
                    file_count += 1
            
            if lines:
                output_file = f"{output_prefix}_{file_count:03}.csv"
                with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(headers)  
                    writer.writerows(lines)
                    
            print(f"{file_count} 個のファイルに分割しました。")
    except FileNotFoundError:
        print("指定されたファイルが見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def clean_honbun(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    
    
    def remove_parentheses(text):
        if isinstance(text, str):
            paren_index = text.rfind('(')
            if paren_index != -1:
                return text[:paren_index].rstrip() 
        return text

    df['honbun'] = df['honbun'].apply(remove_parentheses)

    df.to_csv(output_file_path, index=False)
    print(f"変換後のデータが {output_file_path} に保存されました。")


strings_to_remove = [] 

symbols_to_remove = ['■','□','◇','◆','〇','◎','▷','▽','▼','▲','▶','△','・','●'] 

for i in range(1, 2):
    if i != 34:
        input_file = f'kijidata_202306\ehime_kiji_{i:03}.csv'  
        mid_file = f'editing\data_{i:03}.csv'
        output_file = f'edited\data{i:03}.csv' 
        input = f'edited\data{i:03}.csv'  
        output = f'split_datas\data_{i:03}\data'  

        remove_rows_containing_strings(input_file, mid_file, strings_to_remove)  #見出しで削除
        filter_text_length(mid_file, mid_file) #本文の長さで削除
        replace_symbols_with_space(mid_file, mid_file, symbols_to_remove) #記号変換
        convert_fullwidth_to_halfwidth(mid_file, mid_file) #全角to半角
        convert_ruby_to_kanji(mid_file, mid_file) #ルビ訂正
        clean_honbun(mid_file, output_file) #文末の人？を削除
        split_csv(input, output)

