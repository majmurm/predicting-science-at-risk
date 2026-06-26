import pandas as pd
import re
from pathlib import Path


def add_flagged_column_vectorized(df, abstract_col='abstract', title_col='project_title'):
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()

    # load words
    def load_words(path):
        with open(path, encoding='utf-8') as f:
            return sorted(set(
                line.strip().casefold()
                for line in f
                if line.strip() and not line.startswith('#')
            ))

    pen_words = load_words((base / '../data/banned_words_PEN_2604.txt').resolve())
    nyt_words = load_words((base / '../data/banned_words.txt').resolve())

    # build regex
    def build_regex(words):
        single = [r'\b' + re.escape(w) + r'\b' for w in words if ' ' not in w]
        phrases = [r'\s+'.join(re.escape(p) for p in w.split()) for w in words if ' ' in w]
        return re.compile('|'.join(single + phrases), flags=re.IGNORECASE)

    pen_regex = build_regex(pen_words)
    nyt_regex = build_regex(nyt_words)
    combined_regex = re.compile(pen_regex.pattern + "|" + nyt_regex.pattern, re.IGNORECASE)

    df = df.copy()

    # clean text
    df[abstract_col] = df[abstract_col].fillna('').astype(str)
    df[title_col] = df[title_col].fillna('').astype(str)

    # total counts
    df['total_pen_terms_abstract'] = df[abstract_col].str.count(pen_regex)
    df['total_nyt_terms_abstract'] = df[abstract_col].str.count(nyt_regex)

    df['total_pen_terms_title'] = df[title_col].str.count(pen_regex)
    df['total_nyt_terms_title'] = df[title_col].str.count(nyt_regex)

    df['total_flagged_terms_abstract'] = (
        df['total_pen_terms_abstract'] + df['total_nyt_terms_abstract']
    )

    df['total_flagged_terms_title'] = (
        df['total_pen_terms_title'] + df['total_nyt_terms_title']
    )

    # extract unique words
    pen_set = frozenset(pen_words)
    nyt_set = frozenset(nyt_words)

    def extract_unique_fast(text_series, regex, word_set):
        _ws = re.compile(r'\s+')
        def find_matches(text):
            return {_ws.sub(' ', m.lower()) for m in regex.findall(text)} & word_set
        return text_series.apply(find_matches)

    pen_abs_sets   = extract_unique_fast(df[abstract_col], pen_regex, pen_set)
    nyt_abs_sets   = extract_unique_fast(df[abstract_col], nyt_regex, nyt_set)

    pen_title_sets = extract_unique_fast(df[title_col], pen_regex, pen_set)
    nyt_title_sets = extract_unique_fast(df[title_col], nyt_regex, nyt_set)

    # combine
    all_abs_sets = pen_abs_sets.combine(nyt_abs_sets, lambda x, y: x | y)
    all_title_sets = pen_title_sets.combine(nyt_title_sets, lambda x, y: x | y)

    #store unique
    df['unique_flagged_words_abstract'] = all_abs_sets.apply(len)
    df['unique_flagged_words_title'] = all_title_sets.apply(len)

    df['unique_pen_words_abstract'] = pen_abs_sets.apply(len)
    df['unique_nyt_words_abstract'] = nyt_abs_sets.apply(len)

    df['unique_pen_words_title'] = pen_title_sets.apply(len)
    df['unique_nyt_words_title'] = nyt_title_sets.apply(len)

    # strings
    df['flagged_words_all_abstract'] = all_abs_sets.apply(lambda x: '; '.join(sorted(x)))
    df['flagged_words_all_title'] = all_title_sets.apply(lambda x: '; '.join(sorted(x)))

    # total aggregates
    df['total_flagged_terms_total'] = (
        df['total_flagged_terms_abstract'] + df['total_flagged_terms_title']
    )

    df['unique_flagged_words_total'] = (
        df['unique_flagged_words_abstract'] + df['unique_flagged_words_title']
    )

    # flags
    df['has_flagged_word_total'] = (df['unique_flagged_words_total'] > 0).astype(int)
    df['has_flagged_terms_total'] = (df['total_flagged_terms_total'] > 0).astype(int)

    # repetition
    df['avg_repetition'] = (
        df['total_flagged_terms_total'] / df['unique_flagged_words_total']
    ).replace([float('inf')], 0).fillna(0)

    return df