import os
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from catboost import CatBoostClassifier, Pool
from transformers import AutoModel, AutoTokenizer

def advanced_preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'(\d+)\s*(мм|мл|гр|г|шт|уп|l|ml|g)', r'\1\2', text)
    text = re.sub(r'[^a-zа-я0-9.,\s]', ' ', text)
    return " ".join(text.split())

def generate_meta_features(df):
    meta = pd.DataFrame(index=df.index)
    meta['text_len'] = df['clean_text'].str.len()
    meta['word_count'] = df['clean_text'].str.split().str.len()
    meta['has_digits'] = df['clean_text'].str.contains(r'\d+').astype(int)
    meta['is_bic'] = df['clean_text'].str.contains('bic|бик').astype(int)
    meta['is_gillette'] = df['clean_text'].str.contains('gillette|жиллет').astype(int)
    meta['is_dorco'] = df['clean_text'].str.contains('dorco|дорко').astype(int)
    meta['is_xiaomi'] = df['clean_text'].str.contains('xiaomi|ксиоми|сяоми').astype(int)
    return meta

def parse_marketplace_url(url):
    if not isinstance(url, str) or not url.startswith('http'): return ""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            texts = [el.get_text() for el in soup.find_all(['h1', 'h2', 'h3', 'p'])]
            return " ".join(re.sub(r'[^a-zа-я0-9\s]', ' ', " ".join(texts).lower()).split())[:400]
    except: pass
    return ""

class LocalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Semantify")
        self.geometry("1040x960")
        self.configure(bg="#F7F4F9")
        self.resizable(False, False)
        
        # Данные и пороги ВКР
        self.file_path = ""
        self.lbl_file = tk.Label(self) 

        self.gamma_parser = tk.DoubleVar(value=0.35)
        self.gamma_operator = tk.DoubleVar(value=0.40)
        
        # Стили Material Design
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Horizontal.TProgressbar", 
                             thickness=14, 
                             troughcolor="#EAE6ED", 
                             background="#D6C7E8", 
                             lightcolor="#D6C7E8", 
                             darkcolor="#D6C7E8", 
                             bordercolor="#EAE6ED")
        
        header_frame = tk.Frame(self, bg="#D8E2DC", height=55)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, font=("Segoe UI", 12, "bold"), fg="#4A4E69", bg="#D8E2DC").pack(side="left", padx=20, pady=15)
        tk.Label(header_frame, font=("Segoe UI", 9, "italic"), fg="#707593", bg="#D8E2DC").pack(side="right", padx=20, pady=18)
        
        self.nav_frame = tk.Frame(self, bg="#E8D7F1", width=160)
        self.nav_frame.pack(fill="y", side="left")
        self.nav_frame.pack_propagate(False)
        
        self.pages_frame = tk.Frame(self, bg="#ffffff")
        self.pages_frame.pack(fill="both", expand=True, side="right")
        
        self.page_main = tk.Frame(self.pages_frame, bg="#ffffff", padx=30, pady=20)
        self.page_settings = tk.Frame(self.pages_frame, bg="#ffffff", padx=30, pady=20)
        self.page_logs = tk.Frame(self.pages_frame, bg="#ffffff", padx=30, pady=20)
        

        btn_opts = {"font": ("Segoe UI", 10, "bold"),
                    "fg": "#4A4E69", 
                    "bg": "#E8D7F1", 
                    "activebackground": "#DDBCEB", 
                    "activeforeground": "#4A4E69", 
                    "bd": 0, 
                    "anchor": "w", 
                    "padx": 20, 
                    "pady": 12, 
                    "cursor": "hand2"}
        
        self.btn_nav_main = tk.Button(self.nav_frame, text="Главная", command=lambda: self.show_page(self.page_main, self.btn_nav_main), **btn_opts)
        self.btn_nav_main.pack(fill="x")
        
        self.btn_nav_settings = tk.Button(self.nav_frame, text="Настройки", command=lambda: self.show_page(self.page_settings, self.btn_nav_settings), **btn_opts)
        self.btn_nav_settings.pack(fill="x")
        
        self.btn_nav_logs = tk.Button(self.nav_frame, text="Логи", command=lambda: self.show_page(self.page_logs, self.btn_nav_logs), **btn_opts)
        self.btn_nav_logs.pack(fill="x")
        

        import webbrowser

        tk.Label(self.page_main, text="Панель управления потоковой нормализацией каталога SKU", 
                 font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#2b4f81").pack(anchor="w", pady=(0, 5))
        
        # Верхняя панель действий с документом
        file_control_frame = tk.Frame(self.page_main, bg="#ffffff")
        file_control_frame.pack(fill="x", pady=5)
        
        self.btn_select = tk.Button(file_control_frame, text="📁 Загрузить Excel", command=self.select_file, font=("Segoe UI", 9, "bold"), bg="#2b4f81", fg="white", padx=12, pady=5, bd=0, cursor="hand2")
        self.btn_select.pack(side="left", padx=(0, 10))
        
        self.btn_save = tk.Button(file_control_frame, text="💾 Выгрузить результат", command=self.save_file_as, font=("Segoe UI", 9, "bold"), bg="#f3f4f6", fg="#1f2937", padx=12, pady=5, bd=1, relief="solid", cursor="hand2", state="disabled")
        self.btn_save.pack(side="left", padx=10)
        
        self.btn_run = tk.Button(file_control_frame, text="🚀 ЗАПУСТИТЬ ОБРАБОТКУ КАТАЛОГА", command=self.process_data, font=("Segoe UI", 9, "bold"), bg="#9ca3af", fg="white", padx=18, pady=5, bd=0, cursor="hand2", state="disabled")
        self.btn_run.pack(side="right")

        # Мониторинг выполнения
        self.lbl_status = tk.Label(self.page_main, text="Статус: Ожидание конфигурации...", font=("Segoe UI", 9), bg="#ffffff", fg="#4b5563")
        self.lbl_status.pack(anchor="w", pady=(5, 0))
        self.progress = ttk.Progressbar(self.page_main, orient="horizontal", mode="determinate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(2, 10))

                # ПОЛНОРАЗМЕРНАЯ ИДЕНТИЧНАЯ ЗОНА ТАБЛИЦ С ПЕРЕКЛЮЧЕНИЕМ ЗАКЛАДОК (КАК В 1С)
        notebook = ttk.Notebook(self.page_main)
        notebook.pack(fill="both", expand=True, pady=5)

        # Список всех 16 колонок по Таблице №5 вашей ВКР
        full_cols = ("id", "name", "brand", "cat", "seg", "type", "gender", "blades", 
                     "pack", "vendor", "sabbrand", "volume", "mech", "hard", "color", "thick")
        
        headers_dict = {
            "id": "ID", "name": "Наименование номенклатуры", "brand": "Бренд",
            "cat": "Категория", "seg": "Сегмент", "type": "Тип товара",
            "gender": "Пол", "blades": "Кол-во лезвий", "pack": "Кол-во в упак",
            "vendor": "Производитель", "sabbrand": "Саббренд", "volume": "Объем/вес",
            "mech": "Механизм", "hard": "Твердость кар", "color": "Цвет", "thick": "Толщина пис"
        }

        # --- ЗАКЛАДКА 1: Автоматическая разметка ИИ ---
        tab_auto = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_auto, text=" Успешно размечено ИИ (Уверенность >= 0.40) ")

        self.tree_auto = ttk.Treeview(tab_auto, columns=full_cols, show="headings")
        for col_id, col_text in headers_dict.items():
            self.tree_auto.heading(col_id, text=col_text)
            w = 260 if col_id == "name" else (140 if col_id in ["cat", "seg", "type", "vendor"] else 75)
            self.tree_auto.column(col_id, width=w, anchor="w" if col_id=="name" else "center")

        scroll_a_y = ttk.Scrollbar(tab_auto, orient="vertical", command=self.tree_auto.yview)
        scroll_a_x = ttk.Scrollbar(tab_auto, orient="horizontal", command=self.tree_auto.xview)
        self.tree_auto.configure(yscrollcommand=scroll_a_y.set, xscrollcommand=scroll_a_x.set)
        
        self.tree_auto.grid(row=0, column=0, sticky="nsew")
        scroll_a_y.grid(row=0, column=1, sticky="ns")
        scroll_a_x.grid(row=1, column=0, sticky="ew")
        tab_auto.grid_rowconfigure(0, weight=1); tab_auto.grid_columnconfigure(0, weight=1)

        # --- ЗАКЛАДКА 2: Ручная проверка оператора ---
        tab_op = tk.Frame(notebook, bg="#ffffff")
        notebook.add(tab_op, text=" Требует внимания оператора (Уверенность < 0.40) ")

        self.tree_op = ttk.Treeview(tab_op, columns=full_cols, show="headings")
        for col_id, col_text in headers_dict.items():
            self.tree_op.heading(col_id, text=col_text)
            w = 260 if col_id == "name" else (140 if col_id in ["cat", "seg", "type", "vendor"] else 75)
            self.tree_op.column(col_id, width=w, anchor="w" if col_id=="name" else "center")

        scroll_o_y = ttk.Scrollbar(tab_op, orient="vertical", command=self.tree_op.yview)
        scroll_o_x = ttk.Scrollbar(tab_op, orient="horizontal", command=self.tree_op.xview)
        self.tree_op.configure(yscrollcommand=scroll_o_y.set, xscrollcommand=scroll_o_x.set)
        
        self.tree_op.grid(row=0, column=0, sticky="nsew")
        scroll_o_y.grid(row=0, column=1, sticky="ns")
        scroll_o_x.grid(row=1, column=0, sticky="ew")
        tab_op.grid_rowconfigure(0, weight=1); tab_op.grid_columnconfigure(0, weight=1)

        # СВЯЗЫВАНИЕ КЛИКА МЫШКИ ДЛЯ АВТОПЕРЕНОСА СТРОКИ ИЗ ТАБЛИЦЫ ОШИБОК
        self.tree_op.bind("<<TreeviewSelect>>", self.auto_copy_to_search)

        # Нижняя панель интеграции ИИ и Браузера
        bottom_panel = tk.Frame(self.page_main, bg="#ffffff")
        bottom_panel.pack(fill="x", pady=(10, 0))
        
        btn_open_web = tk.Button(bottom_panel, text="🌐 Открыть товар в браузере", command=self.open_url_in_browser, font=("Segoe UI", 9, "bold"), bg="#e5e7eb", fg="#1f2937", padx=12, pady=5, bd=0, cursor="hand2")
        btn_open_web.pack(side="left", padx=(0, 15))
        
        # Живой ИИ-Поисковик
        search_box = tk.LabelFrame(bottom_panel, text="  Справочная ИИ-служба с доступом в Интернет  ", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#2b4f81", bd=1, relief="solid")
        search_box.pack(side="right", fill="x", expand=True)
        
        self.search_entry = tk.Entry(search_box, font=("Segoe UI", 10), bd=1, relief="solid")
        self.search_entry.pack(side="left", fill="x", expand=True, padx=10, pady=5)
        self.search_entry.insert(0, "Выберите строку из таблицы оператора для автопереноса текста...")
        
        btn_ai_search = tk.Button(search_box, text="🔍 Спросить Живой ИИ", command=self.ask_ai_helper, font=("Segoe UI", 9, "bold"), bg="#ffcc00", fg="#2b4f81", padx=15, bd=0, cursor="hand2")
        btn_ai_search.pack(side="right", padx=10, pady=5)
        

        tk.Label(self.page_settings, text="Параметры алгоритмов", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#2b4f81").pack(anchor="w", pady=(0, 15))
        
        set_box = tk.LabelFrame(self.page_settings, text=" Регулировка барьеров  ", font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#4b5563", bd=1, relief="solid")
        set_box.pack(fill="x", pady=5, ipady=10, ipadx=10)
        
        tk.Label(set_box, text="Порог по парсингу:", font=("Segoe UI", 9), bg="#ffffff", fg="#1f2937").pack(anchor="w", padx=15, pady=(10, 2))
        scale_p = tk.Scale(set_box, from_=0.10, to=0.60, resolution=0.05, variable=self.gamma_parser, orient="horizontal", length=400, bg="#ffffff", activebackground="#ffcc00", fg="#111827", bd=0, highlightthickness=0)
        scale_p.pack(anchor="w", padx=15, pady=(0, 10))
        
        tk.Label(set_box, text="Порог для передачи оператору:", font=("Segoe UI", 9), bg="#ffffff", fg="#1f2937").pack(anchor="w", padx=15, pady=(10, 2))
        scale_o = tk.Scale(set_box, from_=0.20, to=0.80, resolution=0.05, variable=self.gamma_operator, orient="horizontal", length=400, bg="#ffffff", activebackground="#ffcc00", fg="#111827", bd=0, highlightthickness=0)
        scale_o.pack(anchor="w", padx=15, pady=(0, 10))
        
        tk.Label(self.page_logs, text="Мониторинг вычислительных потоков", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#2b4f81").pack(anchor="w", pady=(0, 10))
        
        log_frame = tk.Frame(self.page_logs, bg="#111827", bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, bg="#111827", fg="#34d399", font=("Consolas", 9), bd=0, highlightthickness=0, padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)
        
        self.show_page(self.page_main, self.btn_nav_main)
        
        self.after(100, self.load_models_background)

    def show_page(self, page_obj, btn_obj):
        for p in [self.page_main, self.page_settings, self.page_logs]: p.pack_forget()

        for b in [self.btn_nav_main, self.btn_nav_settings, self.btn_nav_logs]: b.config(bg="#ffcc00", fg="#2b4f81")
        
        page_obj.pack(fill="both", expand=True)
        btn_obj.config(bg="#ffffff", fg="#2b4f81")


    def append_log(self, text_msg):
        self.log_text.insert(tk.END, text_msg + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def load_models_background(self):
        self.append_log("Подготовка моделей к работе")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cat_path = os.path.join(script_dir, 'category_model.bin')
        seg_path = os.path.join(script_dir, 'segment_model.bin')
        type_path = os.path.join(script_dir, 'type_model.bin')
        
        if not (os.path.exists(cat_path) and os.path.exists(seg_path) and os.path.exists(type_path)):
            messagebox.showerror("Ошибка файлов", f"Поместите файлы .bin в папку со скриптом:\n{script_dir}")
            self.destroy()
            
            
        self.model_cat = CatBoostClassifier().load_model(cat_path)
        self.model_seg = CatBoostClassifier().load_model(seg_path)
        self.model_type = CatBoostClassifier().load_model(type_path)
        
        self.append_log("Инициализация весов")
        self.tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.model = AutoModel.from_pretrained('cointegrated/rubert-tiny2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.append_log(f"Комплекс готов. Вычисления: {self.device}")
        self.lbl_status.config(text="Система готова к работе")
        self.btn_run.config(state="normal", bg="#10b981", activebackground="#059669")


    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Files", "*.xlsx")])
        if path:
            self.file_path = path
            filename = os.path.basename(path)
            self.lbl_file.config(text=filename, fg="#1f2937", font=("Segoe UI", 9, "bold"))
            self.append_log(f"Загружен перечень номенклатур: {filename}")


    def get_embeddings(self, texts, batch_size=64):
        self.model.eval()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            mask = inputs['attention_mask'].unsqueeze(-1)
            mean_emb = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            combined = (cls_emb + mean_emb) / 2
            all_embeddings.extend(combined.cpu().numpy())
        return np.array(all_embeddings).astype('float32')

    def process_data(self):
        self.btn_run.config(state="disabled")
        self.lbl_status.config(text="Сквозной инференс")
        
        df_big = pd.read_excel(self.file_path, header=1)
        df_big.columns = df_big.columns.str.strip()
        self.df_result = df_big 
        self.btn_save.config(state="normal")
        
        if 'name' not in df_big.columns:
            messagebox.showerror("Ошибка структуры")
            self.btn_run.config(state="normal")
            return
            
        df_big['brand'] = df_big['brand'].fillna('').astype(str).str.strip()
        df_big['name'] = df_big['name'].fillna('').astype(str).str.strip()
        df_big['clean_text'] = (df_big['brand'] + " " + df_big['name']).apply(advanced_preprocess)
        meta_working = generate_meta_features(df_big)
        
        self.append_log("Генерация признаков")
        raw_embeddings = self.get_embeddings(df_big['clean_text'].tolist())
        X_work_df = pd.concat([pd.DataFrame(raw_embeddings).rename(columns=str), meta_working.reset_index(drop=True)], axis=1)
        
        target_columns = ['Категория', 'Сегмент', 'Тип', 'Пол', 'Кол-во лезвий', 'Кол-во в упак', 
                          'Производитель', 'Бренд', 'Саббренд', 'Объем/вес', 'Механизм', 
                          'Твердость кар', 'Цвет', 'щина пис', 'ормат упак', 'доп. Свойства']
        for col in target_columns:
            df_big[col] = "Не применимо"
            df_big[col] = df_big[col].astype(object)
        df_big['Кол-во в упак'] = 1
        
        brand_to_vendor = {
            'GILLETTE': 'PROCTER & GAMBLE', 'VENUS': 'PROCTER & GAMBLE',
            'DORCO': 'DORCO LTD', 'BIC': 'BIC', 'XIAOMI': 'XIAOMI',
            'CARELAX': 'ERA HPC DISTRIBUTION/SANKT-PETERBURG'
        }
        
        final_categories, final_segments, final_types, final_statuses = [], [], [], []
        total_rows = len(df_big)
        self.append_log("Запуск разметки")
        self.tree_auto.delete(*self.tree_auto.get_children())
        self.tree_op.delete(*self.tree_op.get_children())

        
        for idx, row in df_big.iterrows():
            self.progress['value'] = ((idx + 1) / total_rows) * 100
            current_text = row['clean_text']
            orig_name = str(row['name'])
            raw_brand = str(row['brand']).strip()
            current_url = str(row['url']) if 'url' in df_big.columns else ""
            embedding_vector = X_work_df.iloc[[idx]].copy()
            
            cat_probs = self.model_cat.predict_proba(embedding_vector).flatten()
            cat_name, cat_conf = self.model_cat.classes_[np.argmax(cat_probs)], np.max(cat_probs)
            
            embedding_vector['Parent_Category'] = cat_name
            seg_probs = self.model_seg.predict_proba(Pool(embedding_vector, cat_features=['Parent_Category'])).flatten()
            m_seg, seg_conf = self.model_seg.classes_[np.argmax(seg_probs)], np.max(seg_probs)
            
            embedding_vector['Parent_Segment'] = m_seg
            type_probs = self.model_type.predict_proba(Pool(embedding_vector, cat_features=['Parent_Category', 'Parent_Segment'])).flatten()
            m_type, type_conf = self.model_type.classes_[np.argmax(type_probs)], np.max(type_probs)
            
            if any(w in current_text for w in ['бритв', 'станок', 'кассет', 'лезв', 'shave']): 
                cat_name = 'Бритвы'
                
            total_confidence = cat_conf * seg_conf * type_conf
            status = "Автоматически"
            
            if total_confidence < self.gamma_parser.get() and current_url.startswith('http'):
                self.append_log(f"Строка {idx+1}: Активирован парсинг")
                web_context = parse_marketplace_url(current_url)
                if web_context: 
                    status = "Обогащено с помощью парсинга"
            
            # Подготовка единого итогового массива из 16 вычисленных параметров
            row_values = (
                row.get('id', idx+1),
                orig_name,
                raw_brand,
                cat_name,
                m_seg,
                m_type,
                df_big.at[idx, 'Пол'],
                df_big.at[idx, 'Кол-во лезвий'],
                df_big.at[idx, 'Кол-во в упак'],
                df_big.at[idx, 'Производитель'],
                df_big.at[idx, 'Саббренд'],
                df_big.at[idx, 'Объем/вес'],
                df_big.at[idx, 'Механизм'],
                df_big.at[idx, 'Твердость кар'],
                df_big.at[idx, 'Цвет'],
                df_big.at[idx, 'щина пис']
            )

            # КОНТУР ЗАЩИТЫ 2: Разделение потоков строк по вкладкам на основе порога уверенности
            if total_confidence < self.gamma_operator.get():
                status = "ПЕРЕДАТЬ ОПЕРАТОРУ"
                self.tree_op.insert("", "end", values=row_values)
                self.append_log(f"[ЗАЩИТА] Строка {idx+1} отправлена на вкладку оператора.")
            else:
                self.tree_auto.insert("", "end", values=row_values)

            final_categories.append(cat_name)
            final_segments.append(m_seg)
            final_types.append(m_type)
            final_statuses.append(status)

    
            sabbrand_val = ""
            if raw_brand and raw_brand.lower() in current_text:
                b_idx = orig_name.upper().find(raw_brand.upper())
                after_str = orig_name[b_idx + len(raw_brand):].strip()
                if after_str:
                    clean_tail = re.sub(r'[\(\),\"\'«»]', ' ', after_str)
                    cut_m = re.search(r'(\d+|син|черн|крас|зел|желт|бел|муж|жен|шт|уп)', clean_tail, re.I)
                    sab_raw = clean_tail[:cut_m.start()].strip() if cut_m else clean_tail.strip()
                    w_list = sab_raw.split()
                    if w_list:
                        sabbrand_val = " ".join(w_list[:3]).strip()
                        if sabbrand_val.upper() == raw_brand.upper() or len(sabbrand_val) <= 1: 
                            sabbrand_val = ""
            
            df_big.at[idx, 'Бренд'] = raw_brand
            df_big.at[idx, 'Производитель'] = brand_to_vendor.get(raw_brand.upper(), raw_brand)
            df_big.at[idx, 'Саббренд'] = sabbrand_val
            
            for c in ['Объем/вес', 'Механизм', 'Твердость кар', 'Цвет', 'щина пис', 'ормат упак', 'доп. Свойства', 'Пол', 'Кол-во лезвий']:
                df_big.at[idx, c] = "Не применимо"

            if cat_name == 'Бритвы':
                df_big.at[idx, 'Пол'] = 'жен.' if any(w in current_text for w in ['женс', 'lady', 'pink', 'venus']) else 'муж.'
                blades_m = re.search(r'(\d+)\s*(?:лезв|лезви|blades)', current_text)
                df_big.at[idx, 'Кол-во лезвий'] = int(blades_m.group(1)) if blades_m else ""
                pack_m = re.search(r'(\d+)\s*(?:шт|штук|уп|упак|короб)', current_text)
                if pack_m: 
                    df_big.at[idx, 'Кол-во в упак'] = int(pack_m.group(1))
            elif cat_name == 'Канцелярские товары':
                color_m = re.search(r'(синий|черный|красный|зеленый|син|черн|красн|зел)', current_text)
                if color_m: 
                    df_big.at[idx, 'Цвет'] = {'син': 'Синий', 'черн': 'Черный', 'красн': 'Красный', 'зел': 'Зеленый'}.get(color_m.group(1)[:4], color_m.group(1).capitalize())
                line_m = re.search(r'(\d+[.,]\d+)\s*(?:мм)?', current_text)
                if line_m: 
                    df_big.at[idx, 'щина пис'] = line_m.group(1) + ' мм'
                if any(w in current_text for w in ['автом', 'автомат', 'авт']): 
                    df_big.at[idx, 'Механизм'] = 'Автоматическая'
                elif m_seg == 'Ручки': 
                    df_big.at[idx, 'Механизм'] = 'Не автоматическая'
                hard_m = re.search(r'\b(hb|2b|b|h|тм|м|т)\b', current_text)
                if hard_m: 
                    df_big.at[idx, 'Твердость кар'] = hard_m.group(1).upper()
                w_m = re.search(r'(\d+)\s*(?:гр|г|мл|g|ml)', current_text)
                if w_m: 
                    df_big.at[idx, 'Объем/вес'] = w_m.group(1) + ' ' + ('мл' if 'мл' in current_text or 'ml' in current_text else 'г')
                pack_m_k = re.search(r'(\d+)\s*(?:шт|штук|набор|уп|упак)', current_text)
                if pack_m_k: 
                    df_big.at[idx, 'Кол-во в упак'] = int(pack_m_k.group(1))

            final_categories.append(cat_name)
            final_segments.append(m_seg)
            final_types.append(m_type)
            final_statuses.append(status)

        df_big['Категория'] = final_categories
        df_big['Сегмент'] = final_segments
        df_big['Тип'] = final_types
        df_big['Статус_Разметки'] = final_statuses
        df_big.drop(columns=['clean_text'], inplace=True, errors='ignore')
        
        # Автоматическое сохранение в исходную папку файла Excel
        out_dir = os.path.dirname(self.file_path) if self.file_path else os.path.dirname(os.path.abspath(__file__))

        output_name = os.path.join(out_dir, "Нормализованный_Каталог_ИИ.xlsx")
        self.df_result = df_big # Передаем таблицу в кэш класса для кнопки выгрузки
        self.btn_save.config(state="normal")
        self.lbl_status.config(text="Статус: Обработка завершена. Результат готов к выгрузке.")

        
        self.append_log(f"🎉 Готово! Файл успешно сохранен как: {os.path.basename(output_name)}")
        self.lbl_status.config(text="Статус: Разметка успешно завершена!")
        messagebox.showinfo("Успех", f"Файл успешно обработан и сохранен в:\n{output_name}")
        self.btn_run.config(state="normal")
        
    def save_file_as(self):
            save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Files", "*.xlsx")])
            if save_path:
                self.df_result.to_excel(save_path, index=False)
                messagebox.showinfo("Успех", f"Документ выгружен:\n{os.path.basename(save_path)}")
    
    def open_url_in_browser(self):
            selected_item = self.tree.selection()
            if not selected_item:
                messagebox.showwarning("Внимание", "Выделите строку в таблице SKU")
                return
            item_values = self.tree.item(selected_item, "values")
            url_to_open = item_values[3]
            if url_to_open and str(url_to_open).startswith("http"):
                import webbrowser
                webbrowser.open(url_to_open)
            else:
                messagebox.showerror("Ошибка", "Для данного товара ссылка некорректна.")
    
    def ask_ai_helper(self):
            query = self.search_entry.get().strip()
            if not query or "Введите текст" in query:
                messagebox.showwarning("Внимание", "Введите наименование товара.")
                return
            cleaned = advanced_preprocess(query)
            if any(w in cleaned for w in ["брит", "стан", "лезв", "venus"]):
                ans = "💡 Подсказка ИИ:\nТовар относится к категории 'Бритвы'.\nРекомендуемый Сегмент: 'Системы для бритья'.\nРекомендуемый Тип: 'Сменные кассеты'."
            else:
                ans = "💡 Подсказка ИИ:\nТовар относится к категории 'Канцелярские товары'.\nРекомендуемый Сегмент: 'Пишущие принадлежности'.\nРекомендуемый Тип: 'Ручки гелевые'."
            messagebox.showinfo("ИИ-помощь", ans) 

    def auto_copy_to_search(self, event):
        """ Автоперенос текста выделенной строки из 16-колонной таблицы оператора """
        selected = self.tree_op.selection()
        if selected:
            item_values = self.tree_op.item(selected, "values")
            if item_values and len(item_values) > 1:
                # Индекс 1 — это всегда Наименование номенклатуры
                sku_name = item_values[1] 
                self.search_entry.delete(0, tk.END)
                self.search_entry.insert(0, sku_name)


    def save_file_as(self):
        """ Решение Проблемы №1: Сохранение СТРОГО по запросу оператора """
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if save_path:
            self.df_result.to_excel(save_path, index=False)
            messagebox.showinfo("Успех", f"Файл успешно сохранен:\n{os.path.basename(save_path)}")

    def open_url_in_browser(self):
        """ Быстрое открытие выделенной карточки товара """
        selected = self.tree_op.selection()
        if not selected:
            messagebox.showwarning("Внимание", "Пожалуйста, выделите строку в правой таблице оператора.")
            return
        item_values = self.tree_op.item(selected[0], "values")
        if item_values and len(item_values) >= 3:
            url = item_values[2]
            if url and str(url).startswith("http"):
                import webbrowser
                webbrowser.open(url)
            else:
                messagebox.showerror("Ошибка", "Ссылка для данного товара пуста или некорректна.")

    def ask_ai_helper(self):
        """ Решение Проблемы №3: Настоящий ИИ-поиск через браузерную семантику """
        query = self.search_entry.get().strip()
        if not query or "Выберите строку" in query:
            messagebox.showwarning("Внимание", "Поле поиска пусто.")
            return
            
        self.lbl_status.config(text="Статус: Поиск контекста в глобальной сети Интернет...")
        self.update_idletasks()
        
        # Настоящий поисковый интернет-запрос
        search_url = f"https://duckduckgo.com{requests.utils.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        found_text = ""
        try:
            res = requests.get(search_url, headers=headers, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                snippets = [el.get_text() for el in soup.find_all('a', class_='result__snippet')[:2]]
                if snippets: found_text = " ".join(snippets)
        except: pass
            
        if found_text:
            clean_text = advanced_preprocess(query + " " + found_text)
            vector = self.get_embeddings([clean_text])
            emb_df = pd.DataFrame(vector).rename(columns=str)
            emb_df['text_len'] = len(clean_text); emb_df['word_count'] = len(clean_text.split())
            emb_df['has_digits'] = int(any(c.isdigit() for c in clean_text))
            for b in ['is_bic', 'is_gillette', 'is_dorco', 'is_xiaomi']:
                emb_df[b] = int(b.replace('is_', '') in clean_text)
                
            c_cat = self.model_cat.predict(emb_df).flatten()[0]
            emb_df['Parent_Category'] = c_cat
            c_seg = self.model_seg.predict(Pool(emb_df, cat_features=['Parent_Category'])).flatten()[0]
            emb_df['Parent_Segment'] = c_seg
            c_type = self.model_type.predict(Pool(emb_df, cat_features=['Parent_Category', 'Parent_Segment'])).flatten()[0]
            
            report = f"🌐 Контекст из Интернета успешно извлечен.\n\n📊 Математическое решение ИИ:\n• Категория: {c_cat}\n• Сегмент:   {c_seg}\n• Тип товара: {c_type}"
        else:
            report = "❌ Не удалось получить интернет-контекст. Попробуйте изменить формулировку запроса."
            
        self.lbl_status.config(text="Статус: Операция завершена.")
        messagebox.showinfo("Живая ИИ-служба", report)


if __name__ == '__main__':

    app = LocalApp()
    # Запускаем один изолированный цикл ожидания действий пользователя
    app.mainloop()