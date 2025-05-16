import pandas as pd
import re
from typing import List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import string


# 配置
INPUT_FILE = "../data/all.tsv"

# ---------- Step 1: 加载 TSV 数据 ----------
def load_dialogs(file_path: str) -> List[List[Tuple[str, str, str]]]:
    dialogs = []
    current_dialog = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_dialog:
                    dialogs.append(current_dialog)
                    current_dialog = []
                continue
            try:
                label, text, role = line.split("\t")
                current_dialog.append((label, text.replace("__eou__", "").strip(), role))
            except:
                continue  # 跳过错误行

    if current_dialog:
        dialogs.append(current_dialog)
    return dialogs

# ---------- Step 2: 解析对话结构 ----------
def flatten_dialogs(dialogs: List[List[Tuple[str, str, str]]]) -> pd.DataFrame:
    rows = []
    for dialog_id, dialog in enumerate(dialogs):
        for turn_id, (labels_str, text, role) in enumerate(dialog):
            labels = labels_str.split("_")
            is_starter = 1 if turn_id == 0 and role == "User" else 0
            rows.append({
                "dialog_id": dialog_id,
                "turn_id": turn_id,
                "text": text,
                "labels": labels,
                "role": role,
                "is_starter": is_starter
            })
    return pd.DataFrame(rows)

# ---------- Step 3: 标签二值化 ----------
def binarize_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(df["labels"])
    label_df = pd.DataFrame(label_matrix, columns=mlb.classes_)
    return pd.concat([df.reset_index(drop=True), label_df], axis=1), mlb

# 提取结构化特征
def extract_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_pos"] = df["turn_id"]
    df["norm_pos"] = df["turn_id"] / df.groupby("dialog_id")["turn_id"].transform("max")
    
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))
    df["text_unique_len"] = df["text"].apply(lambda x: len(set(x.split())))
    
    return df

# 提取内容特征
def extract_content_keywords(df: pd.DataFrame) -> pd.DataFrame:
    df["has_question_mark"] = df["text"].str.contains(r"\?", regex=True).astype(int)
    df["has_duplicate_words"] = df["text"].str.contains(r"\bsame\b|\bsimilar\b", regex=True).astype(int)

    def has_5w1h(text):
        flags = [int(word in text.lower()) for word in ["what", "where", "when", "why", "who", "how"]]
        return sum(flags)

    df["has_5w1h"] = df["text"].apply(has_5w1h)
    return df


# 计算初始发言相似度
def extract_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    first_utterances = df[df["turn_id"] == 0][["dialog_id", "text"]].rename(columns={"text": "first_text"})
    df = df.merge(first_utterances, on="dialog_id", how="left")

    tfidf = TfidfVectorizer()
    all_texts = df["text"].tolist() + df["first_text"].tolist()
    tfidf_matrix = tfidf.fit_transform(all_texts)
    tfidf_text = tfidf_matrix[:len(df)]
    tfidf_first = tfidf_matrix[len(df):]

    similarities = cosine_similarity(tfidf_text, tfidf_first).diagonal()
    df["sim_with_first"] = similarities

    return df


# ---------- Step 4: 构建特征矩阵(包含所有特征) ----------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_structural_features(df)
    df = extract_content_keywords(df)
    df = extract_similarity_features(df)  # 你应确保它内部调用了 extract_tfidf_similarity_features

    # 选择最终用于模型的特征列（✅ 添加 sim_with_thread）
    features = [
        "abs_pos", "norm_pos", "is_starter",
        "text_len", "text_unique_len",
        "has_question_mark", "has_duplicate_words", "has_5w1h",
        "sim_with_first", "sim_with_thread"  # ✅ 新增这一行
    ]
    return df[features]







