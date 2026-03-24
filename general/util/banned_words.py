import pandas as pd
import re
from pathlib import Path

def add_flagged_column(df, abstract_col='abstract', title_col='project_title'):
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
                pattern = r'\s+'.join(re.escape(p) for p in word.split())
                phrase_words.append((word, re.compile(r'(?i)' + pattern)))
            else:
                pattern = r'\b' + re.escape(word) + r'\b'
                single_words.append((word, re.compile(r'(?i)' + pattern)))
        return single_words, phrase_words

    pen_single_words, pen_phrases = words_and_phrases(pen_words)
    nyt_single_words, nyt_phrases = words_and_phrases(nyt_words)

    def flag_in_text(text_value, words, phrases):
        """Return unique matched banned terms."""
        if pd.isna(text_value):
            return set()
        text = str(text_value)
        found_words = set()
        for word, pattern in words:
            if pattern.search(text):
                found_words.add(word)
        for word, pattern in phrases:
            if pattern.search(text):
                found_words.add(word)
        return found_words

    def count_matches_in_text(text_value, words, phrases):
        """Return total number of banned-term occurrences, including repeats."""
        if pd.isna(text_value):
            return 0
        text = str(text_value)
        total = 0
        for _, pattern in words:
            total += len(pattern.findall(text))
        for _, pattern in phrases:
            total += len(pattern.findall(text))
        return total

    # for nih specifically
    def parse_nih_flagged_words(cell):
        if pd.isna(cell):
            return set()
        parts = re.split(r'[,\n;|]+', str(cell))
        cleaned = {p.strip().casefold() for p in parts if p.strip()}
        return cleaned

    pen_flagged_abstract = []
    nyt_flagged_abstract = []
    all_flagged_abstract = []

    pen_flagged_title = []
    nyt_flagged_title = []
    all_flagged_title = []

    n_banned_terms_abstract = []
    n_banned_terms_title = []

    for _, row in df.iterrows():
        abstract = row[abstract_col] if abstract_col in df.columns else None
        title = row[title_col] if title_col in df.columns else None

        pen_found_abstract = flag_in_text(abstract, pen_single_words, pen_phrases)
        nyt_found_abstract = flag_in_text(abstract, nyt_single_words, nyt_phrases)

        pen_found_title = flag_in_text(title, pen_single_words, pen_phrases)
        nyt_found_title = flag_in_text(title, nyt_single_words, nyt_phrases)

        if 'flagged_words' in df.columns:
            existing_found = parse_nih_flagged_words(row['flagged_words'])
        else:
            existing_found = set()

        combined_found_abstract = pen_found_abstract | nyt_found_abstract | existing_found
        combined_found_title = pen_found_title | nyt_found_title

        # total occurrences (not unique)
        total_matches_abstract = (
            count_matches_in_text(abstract, pen_single_words, pen_phrases)
            + count_matches_in_text(abstract, nyt_single_words, nyt_phrases)
        )
        total_matches_title = (
            count_matches_in_text(title, pen_single_words, pen_phrases)
            + count_matches_in_text(title, nyt_single_words, nyt_phrases)
        )

        # optional: add pre-existing NIH flagged words count if you want
        # currently this counts only terms found in text via regex, not pre-parsed NIH list

        pen_flagged_abstract.append(pen_found_abstract)
        nyt_flagged_abstract.append(nyt_found_abstract)
        all_flagged_abstract.append(combined_found_abstract)

        pen_flagged_title.append(pen_found_title)
        nyt_flagged_title.append(nyt_found_title)
        all_flagged_title.append(combined_found_title)

        n_banned_terms_abstract.append(total_matches_abstract)
        n_banned_terms_title.append(total_matches_title)

    df = df.copy()

    if 'flagged_words' in df.columns:
        df['flagged_words'] = df['flagged_words'].fillna('')

    # abstract columns
    df['flagged_words_pen'] = ['; '.join(sorted(words)) if words else '' for words in pen_flagged_abstract]
    df['flagged_words_nyt'] = ['; '.join(sorted(words)) if words else '' for words in nyt_flagged_abstract]
    df['flagged_words_all'] = ['; '.join(sorted(words)) if words else '' for words in all_flagged_abstract]
    df['has_flagged_word'] = [1 if words else 0 for words in all_flagged_abstract]
    df['num_flagged_words'] = [len(words) for words in all_flagged_abstract]   # unique
    df['num_pen_words'] = [len(words) for words in pen_flagged_abstract]       # unique
    df['num_nyt_words'] = [len(words) for words in nyt_flagged_abstract]       # unique
    df['n_banned_terms'] = n_banned_terms_abstract                             # total occurrences

    # title columns
    df['flagged_words_pen_title'] = ['; '.join(sorted(words)) if words else '' for words in pen_flagged_title]
    df['flagged_words_nyt_title'] = ['; '.join(sorted(words)) if words else '' for words in nyt_flagged_title]
    df['flagged_words_all_title'] = ['; '.join(sorted(words)) if words else '' for words in all_flagged_title]
    df['has_flagged_word_title'] = [1 if words else 0 for words in all_flagged_title]
    df['num_flagged_words_title'] = [len(words) for words in all_flagged_title]   # unique
    df['num_pen_words_title'] = [len(words) for words in pen_flagged_title]       # unique
    df['num_nyt_words_title'] = [len(words) for words in nyt_flagged_title]       # unique
    df['n_banned_terms_title'] = n_banned_terms_title                             # total occurrences

    # totals across abstract + title

    df['num_flagged_words_total'] = (
        df['num_flagged_words'] + df['num_flagged_words_title']
    )

    df['num_pen_words_total'] = (
        df['num_pen_words'] + df['num_pen_words_title']
    )

    df['num_nyt_words_total'] = (
        df['num_nyt_words'] + df['num_nyt_words_title']
    )

    df['n_banned_terms_total'] = (
        df['n_banned_terms'] + df['n_banned_terms_title']
    )

    df['has_flagged_word_total'] = (
        (df['has_flagged_word'] + df['has_flagged_word_title']) > 0
    ).astype(int)

    return df