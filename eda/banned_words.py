import pandas as pd
import re
from pathlib import Path

def add_flagged_column(df, abstract_col='abstract'):
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()

    with open((base / '../data/banned_words_PEN.txt').resolve(), encoding='utf-8') as f:
        pen_words = sorted(set(line.strip().casefold() for line in f if line.strip() and not line.startswith('#')))

    with open((base / '../data/banned_words.txt').resolve(), encoding='utf-8') as f:
        nyt_words = sorted(set(line.strip().casefold() for line in f if line.strip() and not line.startswith('#')))

    def words_and_phrases(words):
        single_words = []
        phrase_words = []
        for word in words:
            if ' ' in word:
                pattern = r'\s'.join(re.escape(p) for p in word.split())
                phrase_words.append((word, re.compile(r'(?i)' + pattern)))
            else:
                pattern = r'\b' + re.escape(word) + r'\b'
                single_words.append((word, re.compile(r'(?i)' + pattern)))
        return single_words, phrase_words

    pen_single_words, pen_phrases = words_and_phrases(pen_words)
    nyt_single_words, nyt_phrases = words_and_phrases(nyt_words)

    def flag_in_abstract(abstract, words, phrases):
        if pd.isna(abstract):
            return set()
        text = str(abstract)
        found_words = set()
        for word, pattern in words:
            if pattern.search(text):
                found_words.add(word)
        for word, pattern in phrases:
            if pattern.search(text):
                found_words.add(word)
        return found_words

    # for nih specifically
    def parse_nih_flagged_words(cell):
        if pd.isna(cell):
            return set()
        parts = re.split(r'[,\n;|]+', str(cell))
        cleaned = {p.strip().casefold() for p in parts if p.strip()}
        return cleaned

    pen_flagged = []
    nyt_flagged = []
    all_flagged = []

    for i, row in df.iterrows():
        abstract = row[abstract_col]
        pen_found = flag_in_abstract(abstract, pen_single_words, pen_phrases)
        nyt_found = flag_in_abstract(abstract, nyt_single_words, nyt_phrases)
        existing_found = ()
        if 'flagged_words' in df.columns:
            existing_found = parse_nih_flagged_words(row['flagged_words'])
            existing_found = set(existing_found)
        else:
            existing_found = set()
        combined_found = pen_found | nyt_found | existing_found
        pen_flagged.append(pen_found)
        nyt_flagged.append(nyt_found)
        all_flagged.append(combined_found)

    df = df.copy()
    df['flagged_words_pen'] = ['; '.join(sorted(words)) if words else '' for words in pen_flagged]
    df['flagged_words_nyt'] = ['; '.join(sorted(words)) if words else '' for words in nyt_flagged]
    df['flagged_words_all'] = ['; '.join(sorted(words)) if words else '' for words in all_flagged]
    df['has_flagged_word'] = [1 if words else 0 for words in all_flagged]
    df['num_flagged_words'] = [len(words) for words in all_flagged]
    df['num_pen_words'] = [len(words) for words in pen_flagged]
    df['num_nyt_words'] = [len(words) for words in nyt_flagged]

    return df
