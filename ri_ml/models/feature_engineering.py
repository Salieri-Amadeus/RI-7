from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def extract_tfidf_similarity_features(df):
    df = df.copy()
    
    # 1. 获取对话首句（第一条发言）
    first_utterances = df[df['turn_id'] == 0][['dialog_id', 'text']].rename(columns={"text": "first_text"})
    df = df.merge(first_utterances, on="dialog_id", how="left")
    
    # 2. 获取整段对话文本
    thread_texts = df.groupby("dialog_id")["text"].apply(lambda texts: " ".join(texts)).reset_index()
    thread_texts.columns = ["dialog_id", "thread_text"]
    df = df.merge(thread_texts, on="dialog_id", how="left")
    
    # 3. 计算 TF-IDF 相似度
    vectorizer = TfidfVectorizer(stop_words="english")
    combined_text = df["text"].tolist() + df["first_text"].tolist() + df["thread_text"].tolist()
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    # 拆分向量
    n = len(df)
    tfidf_text = tfidf_matrix[:n]
    tfidf_first = tfidf_matrix[n:2*n]
    tfidf_thread = tfidf_matrix[2*n:]
    
    # 4. 计算余弦相似度
    df["sim_with_first"] = cosine_similarity(tfidf_text, tfidf_first).diagonal()
    df["sim_with_thread"] = cosine_similarity(tfidf_text, tfidf_thread).diagonal()

    return df
