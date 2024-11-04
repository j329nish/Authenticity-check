import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
import os

class CSVEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Editor")
        self.csv_file = None
        self.df = None
        self.checkbuttons = []
        self.check_vars = []
        self.current_page = 0
        self.rows_per_page = 200
        self.finished_data_dir = "finished_data"
        os.makedirs(self.finished_data_dir, exist_ok=True)
        
        self.create_widgets()

    def create_widgets(self):
        
        self.open_button = tk.Button(self.root, text="Open CSV", command=self.open_csv)
        self.open_button.pack(pady=10)

        self.canvas = tk.Canvas(self.root)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.root.bind("<Up>", self.scroll_up)
        self.root.bind("<Down>", self.scroll_down)

        self.save_button = tk.Button(self.root, text="Save CSV", command=self.save_csv)
        self.save_button.pack(pady=10)

        self.delete_button = tk.Button(self.root, text="Delete Selected", command=self.delete_selected)
        self.delete_button.pack(pady=10)

        self.bulk_delete_label = tk.Label(self.root, text="Search for Category")
        self.bulk_delete_label.pack()
        self.search_entry = tk.Entry(self.root)
        self.search_entry.pack()
        self.bulk_delete_button = tk.Button(self.root, text="Search and Select", command=self.bulk_select)
        self.bulk_delete_button.pack(pady=10)

    def scroll_up(self, event):
        self.canvas.yview_scroll(-1, "units")

    def scroll_down(self, event):
        self.canvas.yview_scroll(1, "units")

    def open_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.csv_file = file_path
            thread = threading.Thread(target=self.load_csv, args=(file_path,))
            thread.start()

    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path)
        self.rows_per_page = len(self.df)
        self.root.after(0, self.display_table)

    def display_table(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        start_index = self.current_page * self.rows_per_page
        end_index = start_index + self.rows_per_page
        page_data = self.df.iloc[start_index:end_index]

        self.checkbuttons = []
        self.check_vars = []

        for i, (index, row) in enumerate(page_data.iterrows()):
            var = tk.BooleanVar()
            checkbutton = tk.Checkbutton(
                self.scrollable_frame, variable=var, height=2, width=3, command=lambda idx=i: self.highlight_row(idx)
            )
            checkbutton.grid(row=i, column=0, sticky='w')
            self.checkbuttons.append(checkbutton)
            self.check_vars.append(var)

            midasi_text = f"{start_index + i + 1}. {row['midasi']}"
            midasi_label = tk.Label(self.scrollable_frame, text=midasi_text + "\n\n", wraplength=450, anchor="w", justify="left")
            midasi_label.grid(row=i, column=1, sticky='w')

            honbun_label = tk.Label(self.scrollable_frame, text=row['honbun'] + "\n\n", wraplength=900, anchor="w", justify="left")
            honbun_label.grid(row=i, column=2, sticky='w')

            honbun_length = len(row['honbun']) 
            length_label = tk.Label(self.scrollable_frame, text=f"文字数: {honbun_length}", anchor="w", justify="left")
            length_label.grid(row=i, column=4, sticky='w')

            if var.get():
                for widget in [midasi_label, honbun_label]:
                    widget.configure(bg='lightblue')

            edit_button = tk.Button(self.scrollable_frame, text="Edit", command=lambda idx=index: self.edit_content(idx))
            edit_button.grid(row=i, column=3)

    def highlight_row(self, idx):
        row_widgets = self.scrollable_frame.grid_slaves(row=idx)
        for widget in row_widgets:
            if self.check_vars[idx].get():
                widget.configure(bg='lightblue')
            else:
                widget.configure(bg='white')

    def delete_selected(self):
        indices_to_delete = [i for i, var in enumerate(self.check_vars) if var.get()]
        if indices_to_delete:
            global_indices_to_delete = [self.current_page * self.rows_per_page + i for i in indices_to_delete]
            self.df.drop(self.df.index[global_indices_to_delete], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.display_table()

    def save_csv(self):
        confirm = messagebox.askyesno("確認", "この内容をCSVファイルに保存しますか？")
        if not confirm:
            return  

        base_filename = os.path.join(self.finished_data_dir, "data_001.csv")

        if os.path.exists(base_filename):
            existing_df = pd.read_csv(base_filename)
            duplicates = self.df[self.df['midasi'].isin(existing_df['title'])]
            if not duplicates.empty:
                messagebox.showwarning("重複警告", "以下のタイトルは既存のデータと重複しています:\n" + "\n".join(duplicates['midasi'].tolist()))
                confirm_duplicate = messagebox.askyesno("確認", "それでも保存しますか？")
                if not confirm_duplicate:
                    return  

            new_data = pd.DataFrame({
                'id': range(len(existing_df), len(existing_df) + len(self.df)),
                'title': self.df['midasi'],
                'body': self.df['honbun'],
                'label': 0
            })
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            combined_df = pd.DataFrame({
                'id': range(len(self.df)),
                'title': self.df['midasi'],
                'body': self.df['honbun'],
                'label': 0
            })

        combined_df.to_csv(base_filename, index=False)
        messagebox.showinfo("Save CSV", f"CSVファイルは保存されました: {base_filename}!")


    def bulk_select(self):
        search_term = self.search_entry.get()
        for i, row in self.df.iterrows():
            if search_term in row['midasi']:
                if (i // self.rows_per_page) == self.current_page:
                    self.check_vars[i % self.rows_per_page].set(True)

    def edit_content(self, row_index):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Content")
        edit_window.geometry("1200x800")

        row_data = self.df.iloc[row_index]

        honbun_label = tk.Label(edit_window, text="Edit 'honbun':")
        honbun_label.pack(pady=5)

        honbun_entry = tk.Text(edit_window, height=25, width=100)
        honbun_entry.insert(tk.END, row_data['honbun'])
        honbun_entry.pack(pady=5)

        save_edit_button = tk.Button(edit_window, text="Save", command=lambda: self.save_edit(row_index, honbun_entry.get("1.0", tk.END).strip(), edit_window))
        save_edit_button.pack(pady=10)

    def save_edit(self, row_index, new_honbun, edit_window):
        self.df.at[row_index, 'honbun'] = new_honbun
        edit_window.destroy()
        self.display_table()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVEditorApp(root)
    root.mainloop()
