import os


os.makedirs("editing", exist_ok=True)
os.makedirs("edited", exist_ok=True)

split_dir = "split_datas"
os.makedirs(split_dir, exist_ok=True)

for i in range(1, 41):
    sub_dir = os.path.join(split_dir, f"data_{i:03}")
    os.makedirs(sub_dir, exist_ok=True)

print("40個のサブディレクトリが作成されました。")
