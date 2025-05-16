#!/usr/bin/env python3
# train_msdialog_word2vec.py

import os
import argparse
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import json

def load_complete_df(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        dialogs = json.load(f)
    texts = []
    for dialog in dialogs.values():
        for utt in dialog.get("utterances", []):
            # depending on your JSON, the field may be 'text' or 'utterance'; adjust if needed
            text = utt.get("text") or utt.get("utterance")
            if text and text.strip():
                texts.append(text.strip())
    return texts

def main():
    parser = argparse.ArgumentParser(
        description="Train CBOW Word2Vec on MSDialog-Complete via pandas"
    )
    parser.add_argument("--input_json",
                        default="../datasets/MSDialog-Complete.json",
                        help="Path to MSDialog-Complete.json")
    parser.add_argument("--output_model",
                        default="msdialog_w2v.model",
                        help="Path to save the Gensim .model")
    parser.add_argument("--output_txt",
                        default="msdialog_w2v.txt",
                        help="Path to save text-format embeddings")
    parser.add_argument("--vector_size", type=int, default=100,
                        help="Dimensionality of the embeddings")
    parser.add_argument("--window", type=int, default=5,
                        help="Word2Vec context window size")
    parser.add_argument("--min_count", type=int, default=5,
                        help="Minimum word frequency to include")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads to train the model")
    args = parser.parse_args()

    print("⏳ Loading MSDialog-Complete.json into pandas…")
    texts = load_complete_df(args.input_json)
    print(f"👉 {len(texts):,} utterances found.")

    print("⏳ Tokenizing utterances…")
    sentences = [simple_preprocess(t, deacc=True, min_len=1) for t in texts]

    print("⏳ Training Word2Vec CBOW model…")
    model = Word2Vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=0,       # CBOW
        epochs=5
    )

    print("✅ Saving binary model to", args.output_model)
    model.save(args.output_model)

    print("✅ Saving text embeddings to", args.output_txt)
    model.wv.save_word2vec_format(args.output_txt, binary=False)

    print("🎉 All done.")

if __name__ == "__main__":
    main()
