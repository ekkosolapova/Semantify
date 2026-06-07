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
import webbrowser
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
        self.geometry("1100x850")  
        self.minsize(1040, 750)
        self.resizable(True, True)
        self.configure(bg="#F7F4F9")
        
        # Данные и пороги ВКР
        self.file_path = ""
        self.lbl_file = tk.Label(self) 
        self.df_result = None

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
        
        tk.Label(header_frame, text="🧠 Semantify — Интеллектуальная нормализация каталогов", font=("Segoe UI", 12, "bold"), fg="#4A4E69", bg="#D8E2DC").pack(side="left", padx=20, pady=15)
        tk.Label(header_frame, text="Бакалаврская работа ВКР v1.0", font=("Segoe UI", 9, "italic"), fg="#707593", bg="#D8E2DC").pack(side="right", padx=20, pady=18)
        
        self.nav_frame = tk.Frame(self, bg="#E8D7F1", width=160)
        self.nav_frame.pack(fill="y", side="left")
        self.nav_frame.pack_propagate(False)
        
        self.pages_frame = tk.Frame(self, bg="#ffffff")
        self.pages_frame.pack(fill="both", expand=True, side="right")
        
        self.page_main = tk.Frame(self.pages_frame, bg="#ffffff", padx=30, pady=20)
        self.page_settings = tk.Frame(self.pages_frame, bg="#ffffff", padx=30, pady=20)
        self.page_logs = tk.Frame(self.pages_frame, bg="#ffffff", padx=30, pady=20)
        
        tk.Label(self.page_logs, text="Мониторинг вычислительных потоков", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#2b4f81").pack(anchor="w", pady=(0, 10))
        
        log_frame = tk.Frame(self.page_logs, bg="#111827", bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True)
        
        # Строго self.log_text, чтобы метод self.append_log видел его из любой точки программы
        self.log_text = tk.Text(log_frame, bg="#111827", fg="#34d399", font=("Consolas", 9), bd=0, highlightthickness=0, padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)
        
        tk.Label(self.page_logs, text="Мониторинг вычислительных потоков", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#2b4f81").pack(anchor="w", pady=(0, 10))
        log_frame = tk.Frame(self.page_logs, bg="#111827", bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, bg="#111827", fg="#34d399", font=("Consolas", 9), bd=0, highlightthickness=0, padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)

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
        

        tk.Label(self.page_main, text="Панель управления потоковой нормализацией каталога SKU", 
                 font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#2b4f81").pack(anchor="w", pady=(0, 5))
        
        file_control_frame = tk.Frame(self.page_main, bg="#ffffff")
        file_control_frame.pack(fill="x", pady=5)
        
        self.btn_select = tk.Button(file_control_frame, text="📁 Загрузить Excel", command=self.select_file, font=("Segoe UI", 9, "bold"), bg="#2b4f81", fg="white", padx=12, pady=5, bd=0, cursor="hand2")
        self.btn_select.pack(side="left", padx=(0, 10))
        
        self.lbl_file = tk.Label(file_control_frame, text="Файл не выбран", font=("Segoe UI", 9, "italic"), bg="#ffffff", fg="#6b7280")
        self.lbl_file.pack(side="left", padx=5)

        self.btn_save = tk.Button(file_control_frame, text="💾 Выгрузить результат", command=self.save_file_as, font=("Segoe UI", 9, "bold"), bg="#f3f4f6", fg="#1f2937", padx=12, pady=5, bd=1, relief="solid", cursor="hand2", state="disabled")
        self.btn_save.pack(side="left", padx=10)
        
        self.btn_browse = tk.Button(file_control_frame, text="🌐 Открыть в браузере", command=self.open_url_in_browser, font=("Segoe UI", 9, "bold"), bg="#e5e7eb", fg="#1f2937", padx=12, pady=5, bd=1, relief="solid", cursor="hand2")
        self.btn_browse.pack(side="left", padx=10)

        self.btn_run = tk.Button(file_control_frame, text="🚀 ЗАПУСТИТЬ ОБРАБОТКУ КАТАЛОГА", command=self.process_data, font=("Segoe UI", 9, "bold"), bg="#9ca3af", fg="white", padx=18, pady=5, bd=0, cursor="hand2", state="disabled")
        self.btn_run.pack(side="right")

        # Мониторинг выполнения
        # Текстовая метка статуса с увеличенным нижним отступом
        self.lbl_status = tk.Label(self.page_main, text="Статус: Ожидание конфигурации...", font=("Segoe UI", 10), bg="#ffffff", fg="#4b5563")
        self.lbl_status.pack(anchor="w", pady=(8, 6)) # Увеличили расстояние от верхнего и нижнего краев
        
        # Индикатор выполнения с корректными вертикальными границами
        self.progress = ttk.Progressbar(self.page_main, orient="horizontal", mode="determinate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(0, 12)) # Отодвинули прогрессбар ниже, убрав наложение


        self.notebook = ttk.Notebook(self.page_main)
        self.notebook.pack(fill="both", expand=True, pady=5)

        # Ровно 18 колонок в строгом порядке по вашему требованию
        full_cols = ("id", "name", "cat", "seg", "type", "gender", "blades", "pack", 
                     "volume", "vendor", "brand", "sabbrand", "mech", "hard", "color", 
                     "thick", "format", "props")
        
        # Точный маппинг заголовков на экране
        headers_dict = {
            "id": "ID", "name": "Наименование номенклатуры", "cat": "Категория", 
            "seg": "Сегмент", "type": "Тип товара", "gender": "Пол", 
            "blades": "Кол-во лезвий", "pack": "Кол-во в уп", "volume": "Объем/вес", 
            "vendor": "Производитель", "brand": "Бренд", "sabbrand": "Саббренд", 
            "mech": "Механизм", "hard": "Твердость каран-а", "color": "Цвет", 
            "thick": "Толщина письма", "format": "Формат упак", "props": "Доп. Свойства"
        }



        # --- ЗАКЛАДКА 1: Автоматическая разметка ИИ ---
        tab_auto = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(tab_auto, text=" Успешно размечено ИИ (0 шт.) ")
        
        # Разметка tree_auto занимает 100% пространства вкладки 1
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
        tab_op = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(tab_op, text=" Требует внимания оператора (0 шт.) ")

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
            return
            
        self.model_cat = CatBoostClassifier().load_model(cat_path)
        self.model_seg = CatBoostClassifier().load_model(seg_path)
        self.model_type = CatBoostClassifier().load_model(type_path)
        
        self.append_log("ИИ-Инициализация весов")
        self.tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.model = AutoModel.from_pretrained('cointegrated/rubert-tiny2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # === БРОНЕБОЙНЫЙ ПАРСЕР МЕТОДИЧКИ НСИ ПО КООРДИНАТАМ СТОЛБЦОВ ===
        self.append_log("Загрузка нормативно-справочной информации (НСИ)...")
        try:
            ref_path = os.path.join(script_dir, 'Структура.xlsx')
            
            if not os.path.exists(ref_path):
                messagebox.showwarning("Внимание", f"Файл справочника не найден:\n{ref_path}\nБудут использованы резервные контуры.")
                self.dict_subbrands = {}
                self.list_colors = []
            else:
                excel_file = pd.ExcelFile(ref_path)
                compiled_subbrands = {}
                compiled_colors = set()
                total_loaded_series = 0
                
                for sheet in excel_file.sheet_names:
                    # Читаем лист с самой первой строки (header=0)
                    df_sheet = pd.read_excel(ref_path, sheet_name=sheet, header=0)
                    df_sheet.dropna(how='all', inplace=True)
                    
                    sheet_key = sheet.lower().strip()
                    if sheet_key not in compiled_subbrands:
                        compiled_subbrands[sheet_key] = []
                        
                    # Проверяем, что в листе физически достаточно колонок (минимум 16 столбцов)
                    if df_sheet.shape[1] >= 16:
                        for _, row_ref in df_sheet.iterrows():
                            # Извлекаем данные по физическому номеру столбца (индексы 13, 14, 15)
                            b_name = str(row_ref.iloc[13]).strip() # Колонка N (Бренд)
                            s_name = str(row_ref.iloc[14]).strip() # Колонка O (Саббренд)
                            v_name = str(row_ref.iloc[15]).strip() # Колонка P (Производитель)
                            
                            if v_name.lower() == 'nan' or not v_name:
                                v_name = b_name.upper()
                                
                            # Если ячейки не пустые — сохраняем их в ОЗУ
                            if b_name and s_name and b_name.lower() != 'nan' and s_name.lower() != 'nan':
                                compiled_subbrands[sheet_key].append((s_name, v_name))
                                total_loaded_series += 1
                                
                                # Генерируем дубликат с пробелом (Mach3 -> Mach 3) для Ozon карточек
                                m_digits = re.search(r'([a-zA-Zа-яА-Я]+)(\d+)', s_name)
                                if m_digits:
                                    spaced_name = f"{m_digits.group(1)} {m_digits.group(2)}"
                                    compiled_subbrands[sheet_key].append((spaced_name, v_name))
                                    total_loaded_series += 1
                                    
                    # Вытаскиваем Цвета по физическому индексу колонки K (11-я колонка, индекс 10)
                    if df_sheet.shape[1] >= 11:
                        for c_val in df_sheet.iloc[:, 10].dropna():
                            c_str = str(c_val).strip().lower()
                            if c_str and c_str != 'nan' and 'цвет' not in c_str:
                                compiled_colors.add(c_str)
                                
                self.dict_subbrands = compiled_subbrands
                self.list_colors = sorted(list(compiled_colors), key=len, reverse=True)
                self.append_log(f"Успешно загружено НСИ: {len(self.list_colors)} цветов, {total_loaded_series} саббрендов.")
                
        except Exception as e:
            self.append_log(f"Ошибка при чтении файла методички НСИ: {str(e)}")
            self.dict_subbrands = {}
            self.list_colors = []
            
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
        
        self.append_log("Генерация признаков и векторизация текста")
        raw_embeddings = self.get_embeddings(df_big['clean_text'].tolist())
        X_work_df = pd.concat([pd.DataFrame(raw_embeddings).rename(columns=str), meta_working.reset_index(drop=True)], axis=1)
        
        # Целевые колонки для записи в Excel (15 свойств + ID и Name идут изначально)
        target_columns = [
            'Категория', 'Сегмент', 'Тип', 'Пол', 'Кол-во лезвий', 'Кол-во в упак', 
            'Объем/вес', 'Производитель', 'Бренд', 'Саббренд', 'Механизм', 
            'Твердость кар', 'Цвет', 'щина пис', 'ормат упак', 'доп. Свойства'
        ]
        
        existing_targets = [c for c in target_columns if c in df_big.columns]
        if existing_targets:
            df_big.drop(columns=existing_targets, inplace=True)
            
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

        count_auto = 0
        count_op = 0
        
        for idx, row in df_big.iterrows():
            # ЖЕЛЕЗОБЕТОННОЕ ОБНУЛЕНИЕ КОНТЕКСТА ПЕРЕД КАЖДОЙ СТРОКОЙ
            # Это полностью сотрет остаточную память от предыдущих товаров в цикле!
            web_context = "" 
            
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
            
                        # ВЫЧИСЛЯЕМ КУМУЛЯТИВНУЮ УВЕРЕННОСТЬ СЕТИ
            total_confidence = cat_conf * seg_conf * type_conf
            status = "Автоматически"
            
            # =========================================================================
            # ТОТАЛЬНЫЙ ЗАЩИТНЫЙ БАРЬЕР "ПРОЧЕЕ" (БЛОКИРОВКА ЗАПИСИ МУСОРА В ОЗУ)
            # =========================================================================
            if cat_name == 'Прочее' or m_seg == 'Прочее' or m_type == 'Прочее':
                # Жестко зануляем абсолютно все 16 целевых полей в DataFrame для итогового Excel
                for c in target_columns:
                    df_big.at[idx, c] = ""
                
                # Принудительно возвращаем Сегмент или Категорию "Прочее", если они были определены ИИ
                if cat_name == 'Прочее':
                    df_big.at[idx, 'Категория'] = 'Прочее'
                elif m_seg == 'Прочее':
                    df_big.at[idx, 'Категория'] = cat_name
                    df_big.at[idx, 'Сегмент'] = 'Прочее'
                
                # Сразу собираем кристально пустой кортеж для вывода на экран интерфейса
                row_values = (
                    int(row.get('id', idx+1)),
                    orig_name,
                    df_big.at[idx, 'Категория'],
                    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" # Все остальные 13 полей гасятся в ноль
                )
                
                # Отправляем пустую строчку на вкладку оператора или ИИ без дублирования
                if total_confidence < self.gamma_operator.get():
                    self.tree_op.insert("", "end", values=row_values)
                    count_op += 1
                    final_statuses.append("ПЕРЕДАТЬ ОПЕРАТОРУ")
                else:
                    self.tree_auto.insert("", "end", values=row_values)
                    count_auto += 1
                    final_statuses.append("Автоматически")
                
                # КРИТИЧЕСКИ ВАЖНАЯ КОМАНДА: полностью пропускаем все идущие ниже регулярные выражения,
                # извлечения саббрендов и брендов. Переходим строго к следующей строчке файла Excel!
                continue
                
            # =========================================================================
            # ДЛЯ ВСЕХ ОСТАЛЬНЫХ ТИПОВЫХ ТОВАРОВ (Ниже идет ваш стандартный код)
            # =========================================================================
            if total_confidence < self.gamma_parser.get() and current_url.startswith('http'):
                # (Парсинг по ссылке, если уверенность низкая...)

            
                if any(w in current_text for w in ['бритв', 'станок', 'кассет', 'лезв', 'shave']): 
                    cat_name = 'Бритвы'
                
            total_confidence = cat_conf * seg_conf * type_conf
            status = "Автоматически"
            
            if total_confidence < self.gamma_parser.get() and current_url.startswith('http'):
                self.append_log(f"Строка {idx+1}: Активирован парсинг")
                web_context = parse_marketplace_url(current_url)
                if web_context: 
                    status = "Обогащено с помощью парсинга"
            
                       
            sabbrand_val = ""
            vendor_val = brand_to_vendor.get(raw_brand.upper(), raw_brand)

            # === РАСШИРЕННЫЙ СИНХРОННЫЙ КОНТУР ПОИСКА С УЧЕТОМ ВЕБ-ПАРСИНГА ===
            # Склеиваем оригинальное наименование и скачанный из интернета текст карточки (если он есть)
            # Это гарантирует, что ИИ найдет серию товара, даже если в самом Excel-файле имя обрезано!
            combined_search_text = orig_name.lower().strip()
            if 'web_context' in locals() and web_context:
                combined_search_text += " " + str(web_context).lower().strip()
                
            # Очищаем объединенный текст от мусорных слэшей и кавычек Ozon
            name_clean_search = re.sub(r'[\(\),\"\'«»\-\/\s+]', ' ', combined_search_text)
            name_clean_search = " ".join(name_clean_search.split())

            
            if cat_name != 'Прочее':
                # ШАГ 1: АБСОЛЮТНЫЙ СКВОЗНОЙ ПОИСК ПО ВСЕМ ЛИСТАМ ЭКСЕЛЬ-МЕТОДИЧКИ БЕЗ ЗАВИСИМОСТИ ОТ КЛЮЧЕЙ
                if hasattr(self, 'dict_subbrands') and self.dict_subbrands:
                    flat_reference_list = []
                    
                    # Собираем все пары (Саббренд, Производитель) со всех листов методички в один плоский список
                    for sheet_key, series_pairs in self.dict_subbrands.items():
                        flat_reference_list.extend(series_pairs)
                        
                    # Сортируем эталонные серии от длинных к коротким (защита от ложных частичных срезов)
                    flat_reference_list.sort(key=lambda x: len(x[0]), reverse=True)
                    
                    for s_name, v_name in flat_reference_list:
                        # Очищаем эталонное имя саббренда из методички для идеального совпадения фраз
                        s_name_clean = str(s_name).lower().strip()
                        s_name_clean = re.sub(r'[\(\),\"\'«»\-\/\s+]', ' ', s_name_clean)
                        s_name_clean = " ".join(s_name_clean.split())
                        
                        # Если очищенный саббренд из методички найден внутри очищенного названия товара
                        if s_name_clean and s_name_clean in name_clean_search:
                            sabbrand_val = str(s_name).strip()
                            vendor_val = str(v_name).strip()
                            break
                            
                if not sabbrand_val and cat_name != 'Прочее' and raw_brand:
                    sabbrand_val = vendor_val
                
                    # Записываем выверенные, объединенные данные в кэш DataFrame
                    df_big.at[idx, 'Саббренд'] = sabbrand_val
                    df_big.at[idx, 'Производитель'] = vendor_val
                      
                  
            # Если Категория, Сегмент или Тип признаны нетиповыми ("Прочее") — гасим все 16 полей в пустоту
            if  m_seg == 'Прочее' or m_type == 'Прочее':
                row_values = (
                    int(row.get('id', idx+1)), # ID остается
                    orig_name,                 # Наименование остается
                    cat_name if cat_name == 'Прочее' else "", # Показываем Категорию, только если она "Прочее"
                    "",
                    "",
                    "",
                    "",                        # Тип товара
                    "",                        # Пол
                    "",                        # Кол-во лезвий
                    "",                        # Кол-во в уп
                    "",                        # Объем/вес
                    "",                        # Производитель
                    "",                        # Бренд
                    "",                        # Саббренд
                    "",                        # Механизм
                    "",                        # Твердость каран-а
                    "",                        # Цвет
                    "",                        # Толщина письма
                    "",                        # Формат упак
                    ""                         # Доп. Свойства
                )
                
                # Мгновенно зануляем ячейки в кэше DataFrame для итогового Excel-файла
                for c in target_columns:
                    df_big.at[idx, c] = ""
            else:
            # Глубокое извлечение физических параметров в зависимости от категории товара
                if cat_name == 'Бритвы':
                    # Создаем текстовый маркер в нижнем регистре для гарантированного поиска совпадений
                    search_text = current_text.lower()
                    
                    # ШАГ 1: Жесткая привязка пола на основе известных коммерческих саббрендов (по методичке)
                    if any(sub_brand in search_text for sub_brand in ['fusion', 'mach', 'proglide', 'skinguard', 'turbo', 's200', 'электрическая']):
                        detected_gender = 'муж.'
                    elif any(sub_brand in search_text for sub_brand in ['venus', 'lady', 'pink', 'swirl', 'breeze', 'embrace', 'simply']):
                        detected_gender = 'жен.'
                    
                    # ШАГ 2: Если по серии определить не удалось, ищем явные текстовые упоминания корней
                    else:
                        if any(w in search_text for w in ['мужск', 'мужч', 'муж', 'men', 'for men', 'man']):
                            detected_gender = 'муж.'
                        elif any(w in search_text for w in ['женск', 'женщ', 'жен', 'women', 'lady', 'for women']):
                            detected_gender = 'жен.'
                        else:
                            # Защитный откат по умолчанию: бритвы без маркеров чаще всего мужские номенклатуры
                            detected_gender = 'муж.'
                            
                    # Фиксируем правильное значение в DataFrame
                    df_big.at[idx, 'Пол'] = detected_gender
                    
                    # Дальше идет ваш стандартный поиск количества лезвий
                    blades_m = re.search(r'(\d+)\s*(?:лезв|лезви|blades)', search_text)

                    df_big.at[idx, 'Кол-во лезвий'] = int(blades_m.group(1)) if blades_m else ""
                    pack_m = re.search(r'(\d+)\s*(?:шт|штук|уп|упак|короб)', current_text)
                    if pack_m: 
                        df_big.at[idx, 'Кол-во в упак'] = int(pack_m.group(1))
                        
                elif cat_name == 'Канцелярские товары':
                    # Переводим текст в нижний регистр для точной работы всех регулярных выражений
                    search_text = current_text.lower()
                
                # УМНЫЙ ПОИСК ЦВЕТА НА ОСНОВЕ КОЛОНКИ 'К' ВАШЕЙ МЕТОДИЧКЕ
                detected_color = "Не применимо"
                if hasattr(self, 'list_colors') and self.list_colors:
                    for color_ref in self.list_colors:
                        if color_ref in search_text:
                            detected_color = color_ref.capitalize()
                            break
                df_big.at[idx, 'Цвет'] = detected_color
                
                # ЖЕЛЕЗОБЕТОННОЕ ИСПРАВЛЕНИЕ: объявляем поиск толщины письма (цифры с точкой или запятой)
                line_m = re.search(r'(\d+[.,]\d+)\s*(?:мм)?', search_text)
                if line_m: 
                    df_big.at[idx, 'щина пис'] = line_m.group(1).replace(',', '.') + ' мм'
                    
                # Поиск типа механизма
                if any(w in search_text for w in ['автом', 'автомат', 'авт']): 
                    df_big.at[idx, 'Механизм'] = 'Автоматическая'
                elif m_seg == 'Ручки': 
                    df_big.at[idx, 'Mechanizm'] = 'Не автоматическая'
                    
                # Поиск твердости карандашей (строгие границы \b защищают от ложных срабатываний внутри слов)
                hard_m = re.search(r'\b(hb|2b|b|h|тм|м|т)\b', search_text)
                if hard_m: 
                    df_big.at[idx, 'Твердость кар'] = hard_m.group(1).upper()
                    
                # Поиск объема/веса клея, корректоров или чернил
                w_m = re.search(r'(\d+)\s*(?:гр|г|мл|g|ml)', search_text)
                if w_m: 
                    df_big.at[idx, 'Объем/вес'] = w_m.group(1) + ' ' + ('мл' if 'мл' in search_text or 'ml' in search_text else 'г')
                    
                # Поиск количества штук в наборе/упаковке канцелярии
                pack_m_k = re.search(r'(\d+)\s*(?:шт|штук|набор|уп|упак)', search_text)
                if pack_m_k: 
                    df_big.at[idx, 'Кол-во в упак'] = int(pack_m_k.group(1))


            val_thick = df_big.at[idx, 'щина пис'] if cat_name == 'Канцелярские товары' else "Не применимо"
            val_hard = df_big.at[idx, 'Твердость кар'] if cat_name == 'Канцелярские товары' else "Не применимо"
            val_mech = df_big.at[idx, 'Механизм'] if cat_name == 'Канцелярские товары' else "Не применимо"
            val_color = df_big.at[idx, 'Цвет'] if cat_name == 'Канцелярские товары' else "Не применимо"
            val_volume = df_big.at[idx, 'Объем/вес'] if cat_name == 'Канцелярские товары' else "Не применимо"
            
            val_gender = df_big.at[idx, 'Пол'] if cat_name == 'Бритвы' else "Не применимо"
            val_blades = df_big.at[idx, 'Кол-во лезвий'] if cat_name == 'Бритвы' else "Не применимо"

            # Сборка кортежа из 18 элементов строго по вашему новому списку колонок
            row_values = (
                int(row.get('id', idx+1)),
                orig_name,
                cat_name,
                m_seg,
                m_type,
                val_gender,
                val_blades,
                df_big.at[idx, 'Кол-во в упак'],
                val_volume,
                df_big.at[idx, 'Производитель'],
                raw_brand,
                df_big.at[idx, 'Саббренд'], # Саббренд теперь на 12-м месте, строго под своей колонкой!
                val_mech,
                val_hard,
                val_color,
                val_thick,
                df_big.at[idx, 'ормат упак'],
                df_big.at[idx, 'доп. Свойства']
            )

            # Жесткое разделение потоков по вкладкам (Контур защиты №2)
            if total_confidence < self.gamma_operator.get():
                status = "ПЕРЕДАТЬ ОПЕРАТОРУ"
                self.tree_op.insert("", "end", values=row_values)
                count_op += 1
            else:
                status = "Автоматически"
                self.tree_auto.insert("", "end", values=row_values)
                count_auto += 1

            final_statuses.append(status)

        # Выход из цикла iterrows — завершение разметки
        df_big['Статус_Разметки'] = final_statuses
        df_big.drop(columns=['clean_text'], inplace=True, errors='ignore')
        
        # Обновляем счетчики на вкладках
        self.notebook.tab(0, text=f" Успешно размечено ИИ ({count_auto} шт.) ")
        self.notebook.tab(1, text=f" Требует внимания оператора ({count_op} шт.) ")
        
        # === ЖЕЛЕЗОБЕТОННОЕ ОБНОВЛЕНИЕ БИНДИНГОВ ПОСЛЕ ЗАПОЛНЕНИЯ ТАБЛИЦ ===
        # Принудительно связываем двойной клик мыши на обеих вкладках с формой редактирования
        self.tree_auto.bind("<Double-1>", lambda event: self.open_edit_dialog(event, self.tree_auto))
        self.tree_op.bind("<Double-1>", lambda event: self.open_edit_dialog(event, self.tree_op))
        
        self.df_result = df_big
        self.btn_save.config(state="normal")
        self.btn_run.config(state="normal")
        self.lbl_status.config(text="Статус: Разметка успешно завершена!")
        messagebox.showinfo("Успех", "Каталог успешно нормализован и готов к выгрузке.")

        self.btn_run.config(state="normal")
        
    def save_file_as(self):
            save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Files", "*.xlsx")])
            if save_path:
                self.df_result.to_excel(save_path, index=False)
                messagebox.showinfo("Успех", f"Документ выгружен:\n{os.path.basename(save_path)}")
    
    def open_edit_dialog(self, event, target_tree):
        """ Сквозное модальное окно для исправления ошибок ИИ или оператора """
        selected = target_tree.selection()
        if not selected: 
            return
            
        item_id = selected[0]
        current_values = list(target_tree.item(item_id, "values"))
        
        # Создаем всплывающее диалоговое окно
        edit_win = tk.Toplevel(self)
        edit_win.title("Корректировка параметров номенклатуры")
        edit_win.geometry("500x650")
        edit_win.grab_set()  # Делаем окно модальным
        
        # Контейнер со скроллом, чтобы 16 полей ввода поместились на любом экране
        canvas = tk.Canvas(edit_win, bg="#ffffff")
        scrollbar = ttk.Scrollbar(edit_win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#ffffff", padx=15, pady=10)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Маппинг всех полей для циклической генерации GUI формы
        fields_labels = [
            "ID", "Наименование номенклатуры", "Бренд", "Категория", "Сегмент", "Тип товара", 
            "Пол", "Кол-во лезвий", "Кол-во в упак", "Объем/вес", "Производитель", "Бренд", 
            "Саббренд", "Механизм", "Твердость кар", "Цвет", "Толщина письма", "Формат упак", "Доп. Свойства"
        ]

        
        entries_dict = {}
        
        for idx, label_text in enumerate(fields_labels):
            tk.Label(scroll_frame, text=label_text, font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#4b5563").pack(anchor="w", pady=(5, 2))
            
            val = current_values[idx] if idx < len(current_values) else ""
            
            # Поле ID и Наименование защищаем от случайного изменения, если нужно
            if idx == 0:
                entry = tk.Entry(scroll_frame, font=("Segoe UI", 10), bd=1, relief="solid", bg="#f3f4f6", fg="#9ca3af")
                entry.insert(0, val)
                entry.config(state="disabled")
            else:
                entry = tk.Entry(scroll_frame, font=("Segoe UI", 10), bd=1, relief="solid")
                entry.insert(0, val)
            
            entry.pack(fill="x", pady=(0, 5))
            entries_dict[label_text] = entry
            
        def save_and_close():
            # Собираем измененные данные обратно в кортеж
            updated_values = []
            for label_text in fields_labels:
                # Особая обработка для отключенного поля ID
                if label_text == "ID":
                    updated_values.append(current_values[0])
                else:
                    updated_values.append(entries_dict[label_text].get().strip())
            
            # 1. Обновляем визуальную строчку в текущей активной таблице
            target_tree.item(item_id, values=updated_values)
            
            # 2. Синхронизируем изменения с кэшем DataFrame для итогового Excel
            try:
                sku_id = int(current_values[0])
                # Находим соответствие в таблице результатов по оригинальному ID
                if hasattr(self, 'df_result') and self.df_result is not None:
                    # Динамически сопоставляем столбцы
                    # Синхронизируем выгрузку правок с сокращенными столбцами методички
                    col_map = {
                        "Наименование номенклатуры": "name", "Бренд (заполн.)": "Бренд", "Категория": "Категория", 
                        "Сегмент": "Сегмент", "Тип товара": "Тип", "Пол": "Пол", "Кол-во лезвий": "Кол-во лезвий", 
                        "Кол-во в упак": "Кол-во в упак", "Производитель": "Производитель", "Саббренд": "Саббренд", 
                        "Объем/вес": "Объем/вес", "Механизм": "Механизм", "Твердость кар": "Твердость кар", 
                        "Цвет": "Цвет", "Толщина письма": "щина пис", "ормат упак": "ормат упак", "доп. Свойства": "доп. Свойства"
                    }


                    
                    for lbl, df_col in col_map.items():
                        if df_col in self.df_result.columns:
                            self.df_result.loc[self.df_result['id'] == sku_id, df_col] = entries_dict[lbl].get().strip()
            except Exception as e:
                self.append_log(f"Предупреждение при синхронизации ОЗУ: {str(e)}")
            
            self.append_log(f"Данные SKU ID {current_values[0]} успешно изменены оператором.")
            edit_win.destroy()
            
        tk.Button(scroll_frame, text="💾 Сохранить изменения", command=save_and_close, font=("Segoe UI", 10, "bold"), bg="#10b981", fg="white", bd=0, padx=20, pady=8, cursor="hand2").pack(pady=15)

    def save_file_as(self):

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
        if save_path:
            self.df_result.to_excel(save_path, index=False)
            messagebox.showinfo("Успех", f"Файл успешно сохранен:\n{os.path.basename(save_path)}")


    def open_url_in_browser(self):
        """ Универсальное сквозное открытие интернет-ссылки товара из ОЗУ для ЛЮБЫХ брендов """
        # 1. Жестко определяем, какую вкладку сейчас видит оператор
        try:
            active_tab = self.notebook.index(self.notebook.select())
        except:
            active_tab = 0 # Защитный откат на первую вкладку, если блокнот еще не сфокусирован
            
        # 2. Берем выделение строго из той таблицы, на которую смотрит пользователь
        if active_tab == 0:
            selection = self.tree_auto.selection()
            target_tree = self.tree_auto
        else:
            selection = self.tree_op.selection()
            target_tree = self.tree_op
            
        if not selection:
            messagebox.showwarning("Внимание", "Пожалуйста, выделите строку в текущей таблице SKU.")
            return
            
        # Извлекаем первый выделенный элемент (избегаем остаточных кликов с других вкладок)
        selected_item = selection[0]
        item_values = target_tree.item(selected_item, "values")
        
        if item_values and len(item_values) > 0:
            try:
                # 3. Вытягиваем ID товара из первой ячейки (индекс 0) и приводим к чистому строковому виду
                target_id = str(item_values[0]).strip()
                if target_id.endswith('.0'):
                    target_id = target_id[:-2]
                
                # 4. Проверяем кэш данных в оперативной памяти
                if hasattr(self, 'df_result') and self.df_result is not None:
                    # Переводим колонку ID в строки только на время поиска совпадения
                    df_ids = self.df_result['id'].fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                    row_data = self.df_result[df_ids == target_id]
                    
                    if not row_data.empty:
                        # Ищем оригинальный URL-адрес из файла Excel
                        url = str(row_data['url'].values[0]).strip() if 'url' in row_data.columns else ""
                        
                        # ЗАЩИТНЫЙ КОНТУР НА ВСЕ СЛУЧАИ ЖИЗНИ: 
                        # Если ссылка пустая, битая или Ozon выдает ошибку, мы генерируем 
                        # идеальную универсальную поисковую ссылку по артикулу товара
                        if not url or url == "nan" or not url.startswith("http"):
                            url = f"https://ozon.ru{target_id}&from_global=true"
                            self.append_log(f"Контур защиты: Сформирован универсальный поисковый URL Ozon по артикулу: {target_id}")
                        
                        # Запускаем браузер
                        import webbrowser
                        webbrowser.open(url)
                        return
                
                messagebox.showerror("Ошибка", "Не удалось локализовать данную номенклатурную единицу в кэше приложения.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось выполнить переход в браузер: {str(e)}")

if __name__ == '__main__':

    app = LocalApp()
    # Запускаем один изолированный цикл ожидания действий пользователя
    app.mainloop()