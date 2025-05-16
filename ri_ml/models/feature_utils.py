import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
sid = SentimentIntensityAnalyzer()

def tokenizer(doc):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    return token_pattern.findall(doc.lower())

def question_mark(utterances):
    return [int('?' in utt) for utt in utterances]

def duplicate(utterances):
    return [int('same' in utt or 'similar' in utt) for utt in utterances]

def W5H1(utterances):
    keys = ['how', 'what', 'why', 'who', 'where', 'when']
    res = []
    for utt in utterances:
        vec = [int(k in utt.lower()) for k in keys]
        res.append(vec)
    return res

def thank(utterances):
    return [int('thank' in utt.lower()) for utt in utterances]

def exclamation_mark(utterances):
    return [int('!' in utt) for utt in utterances]

def ve_feedback(utterances):
    return [int('did not' in utt.lower() or 'does not' in utt.lower()) for utt in utterances]

def sentiment_scores(utterances):
    return [[str(sid.polarity_scores(utt)[k]) for k in ['neg', 'neu', 'pos']] for utt in utterances]

def lexicon(utterances, pos_dict, neg_dict):
    res = []
    for utt in utterances:
        tokens = tokenizer(utt)
        pos_count = sum(1 for t in tokens if t in pos_dict)
        neg_count = sum(1 for t in tokens if t in neg_dict)
        res.append([str(pos_count), str(neg_count)])
    return res

def post_length(utterances):
    lengths = []
    unique_lengths = []
    stemmed_lengths = []

    for utt in utterances:
        tokens = [t for t in tokenizer(utt) if t not in stop_words]
        unique_tokens = set(tokens)
        stemmed_tokens = set(stemmer.stem(t) for t in unique_tokens)

        lengths.append(len(tokens))
        unique_lengths.append(len(unique_tokens))
        stemmed_lengths.append(len(stemmed_tokens))

    return lengths, unique_lengths, stemmed_lengths


def load_sentiment_lexicon(pos_file, neg_file):
    pos_dict, neg_dict = {}, {}

    with open(pos_file, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            if word and not word.startswith(";"):
                pos_dict[word] = 1

    with open(neg_file, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            if word and not word.startswith(";"):
                neg_dict[word] = 1

    return pos_dict, neg_dict



def extract_all_features(df, term_to_idf_dict, pos_dict, neg_dict):
    new_rows = []

    for dialog_id, group in df.groupby("dialog_id"):
        utterances = group["text"].tolist()
        roles = group["role"].tolist()
        turn_ids = group["turn_id"].tolist()
        is_starter = group["is_starter"].tolist()

        # 特征提取
        _, init_sim, thread_sim = cosine_similarity("", utterances, term_to_idf_dict)
        qm = question_mark(utterances)
        dup = duplicate(utterances)
        wh = W5H1(utterances)

        abs_pos = [i + 1 for i in range(len(utterances))]
        norm_pos = [i / len(utterances) for i in abs_pos]
        length, unique_length, stemmed_length = post_length(utterances)

        thx = thank(utterances)
        exclam_mark = exclamation_mark(utterances)
        vf = ve_feedback(utterances)
        ss = sentiment_scores(utterances)
        lexicon_counts = lexicon(utterances, pos_dict, neg_dict)

        # 构建新的特征 DataFrame
        for i in range(len(utterances)):
            row = {
                "dialog_id": dialog_id,
                "turn_id": turn_ids[i],
                "text": utterances[i],
                "role": roles[i],
                "is_starter": is_starter[i],

                # structural
                "abs_pos": abs_pos[i],
                "norm_pos": norm_pos[i],
                "text_len": length[i],
                "text_unique_len": unique_length[i],
                "text_stem_len": stemmed_length[i],

                # content
                "has_question_mark": qm[i],
                "has_duplicate_words": dup[i],
                "has_5w1h": int(any(int(x) for x in wh[i])),
                "sim_with_first": init_sim[i],
                "sim_with_thread": thread_sim[i],

                # sentiment
                "thank": thx[i],
                "exclam": exclam_mark[i],
                "ve_feedback": vf[i],
                "sent_pos": float(ss[i][2]),
                "sent_neu": float(ss[i][1]),
                "sent_neg": float(ss[i][0]),
                "lexicon_pos": int(lexicon_counts[i][0]),
                "lexicon_neg": int(lexicon_counts[i][1]),
            }
            new_rows.append(row)

    return pd.DataFrame(new_rows)
