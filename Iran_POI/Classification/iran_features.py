"""
Iran-specificity features  —  Step-8 feature enrichment.
============================================================================
Goal (per the user's key insight): distinguish specifically IRANIAN users, not
merely Persian speakers — Dari (Afghanistan), Tajik, and diaspora also use
Persian. So instead of a single "Persian-script ratio", we score membership in
Iran-SPECIFIC lexicons (cities, politics/regime, protest campaigns, media,
sport/economy, flag) and subtract ANTI-signals (Afghanistan / Tajikistan terms).

Two text surfaces per user:
  * profile   — description + display_name + location   (available for everyone)
  * tweets    — Text_en + original text from posts.csv  (only ~27% of users)

Every category becomes its own numeric feature (profile_* and tw_*), so the
downstream classifier learns the weights (a #WomanLifeFreedom hit should count
for more than a generic "persian" hit). Missing tweets -> zeros + has_tweets=0.
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data'
POSTS_FILE = DATA_DIR / 'posts.csv'

# ----------------------------------------------------------------------------
# Iran-SPECIFIC lexicons (high precision — ambiguous terms deliberately omitted:
# e.g. "kurdistan"/"azerbaijan" also denote Iraqi/other regions).
# English terms are matched on word boundaries; Persian/Arabic on raw substring.
# ----------------------------------------------------------------------------
GEO_IRAN = [
    'iran', 'iranian', 'tehran', 'shiraz', 'isfahan', 'esfahan', 'mashhad',
    'tabriz', 'kerman', 'ahvaz', 'ahwaz', 'karaj', 'rasht', 'urmia', 'kermanshah',
    'hamedan', 'yazd', 'qazvin', 'zahedan', 'khuzestan',
    'ایران', 'ایرانی', 'تهران', 'شیراز', 'اصفهان', 'مشهد', 'تبریز', 'اهواز', 'کرمان',
]
POLITICS_IRAN = [
    'khamenei', 'khomeini', 'raisi', 'rouhani', 'ahmadinejad', 'pahlavi',
    'reza shah', 'mossadegh', 'irgc', 'sepah', 'basij', 'evin', 'ayatollah',
    'islamic republic', 'morality police', 'gasht ershad', 'velayat',
    'خامنه‌ای', 'خامنه ای', 'سپاه', 'بسیج', 'ولایت فقیه', 'جمهوری اسلامی', 'گشت ارشاد',
]
CAMPAIGN_IRAN = [
    'mahsa amini', 'mahsa', 'jina amini', 'woman life freedom',
    'womanlifefreedom', 'mahsaamini', 'iranprotests', 'iranrevolution',
    'freeiran', 'irgcterrorists', 'green movement',
    'مهسا امینی', 'مهسا', 'زن زندگی آزادی', 'زن_زندگی_آزادی', 'اعتراضات',
]
MEDIA_IRAN = [
    'iran international', 'iranintl', 'manoto', 'irib', 'press tv', 'presstv',
    'radio farda', 'bbc persian', 'ایران اینترنشنال', 'من و تو', 'رادیو فردا',
]
SPORTECON_IRAN = [
    'team melli', 'persepolis', 'esteghlal', 'toman', 'rial',
    'تیم ملی', 'پرسپولیس', 'استقلال', 'تومان', 'ریال',
]
# Generic Persian — spoken by non-Iranians too, so a SEPARATE, weak feature.
GENERIC_PERSIAN = ['persian', 'farsi', 'persia', 'فارسی', 'پارسی']
# ANTI-signals: strong markers of Persian speakers who are NOT Iranian.
ANTI_NON_IRAN = [
    'afghanistan', 'afghan', 'kabul', 'herat', 'kandahar', 'mazar', 'taliban',
    'ashraf ghani', 'dari', 'pashto', 'tajikistan', 'dushanbe', 'tajik',
    'افغانستان', 'افغان', 'کابل', 'هرات', 'طالبان', 'تاجیکستان',
]

FLAG_IRAN = '\U0001F1EE\U0001F1F7'      # 🇮🇷
FLAG_AFG = '\U0001F1E6\U0001F1EB'       # 🇦🇫
FLAG_TJK = '\U0001F1F9\U0001F1EF'       # 🇹🇯

CATEGORIES = {
    'geo': GEO_IRAN, 'politics': POLITICS_IRAN, 'campaign': CAMPAIGN_IRAN,
    'media': MEDIA_IRAN, 'sportecon': SPORTECON_IRAN, 'persian_generic': GENERIC_PERSIAN,
    'anti': ANTI_NON_IRAN,
}
# Feature columns this module contributes (profile_* for all users, tw_* for tweeters).
FEATURE_COLS = (
    [f'prof_iran_{k}' for k in CATEGORIES]
    + ['prof_iran_flag', 'prof_persian_char_ratio']
    + [f'tw_iran_{k}' for k in CATEGORIES]
    + ['tw_iran_flag', 'tw_persian_char_ratio', 'has_tweets', 'n_tweets']
)

# Precompiled word-boundary regexes for ASCII terms (Persian handled by substring).
_ASCII = re.compile(r'^[\x00-\x7f]+$')


@lru_cache(maxsize=None)
def _matcher(term: str):
    if _ASCII.match(term):
        return re.compile(r'\b' + re.escape(term) + r'\b')
    return None                      # Persian/Arabic -> substring test


def _count_hits(text: str, terms: list) -> int:
    if not text:
        return 0
    low = text.lower()
    n = 0
    for t in terms:
        rx = _matcher(t)
        if rx is not None:
            n += len(rx.findall(low))
        else:
            n += low.count(t.lower())
    return n


_PERSIAN_RANGE = re.compile(r'[؀-ۿ]')      # Arabic/Persian block


def _persian_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return len(_PERSIAN_RANGE.findall(text)) / len(letters)


def _score_text(text: str) -> dict:
    text = '' if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    out = {k: _count_hits(text, terms) for k, terms in CATEGORIES.items()}
    out['flag'] = int(FLAG_IRAN in text) - int(FLAG_AFG in text or FLAG_TJK in text)
    out['persian_ratio'] = round(_persian_char_ratio(text), 4)
    return out


@lru_cache(maxsize=1)
def _tweets_by_user() -> dict:
    """username(lower) -> (concatenated tweet text, n_tweets). Prefers Text_en,
    falls back to raw text; concatenates both so lexicon hits fire in any language."""
    if not POSTS_FILE.exists():
        return {}
    cols = pd.read_csv(POSTS_FILE, nrows=0).columns
    use = [c for c in ('username', 'text', 'Text_en') if c in cols]
    posts = pd.read_csv(POSTS_FILE, usecols=use)
    posts['username'] = posts['username'].astype(str).str.lower().str.strip()
    parts = []
    if 'Text_en' in posts:
        parts.append(posts['Text_en'].fillna(''))
    if 'text' in posts:
        parts.append(posts['text'].fillna(''))
    posts['_blob'] = parts[0]
    for p in parts[1:]:
        posts['_blob'] = posts['_blob'] + ' ' + p
    grp = posts.groupby('username')
    blob = grp['_blob'].apply(lambda s: ' '.join(s.astype(str))[:200_000])
    cnt = grp.size()
    return {u: (blob[u], int(cnt[u])) for u in blob.index}


def add_iran_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the profile_* and tw_* Iran-specificity numeric features (keyed by username)."""
    df = df.copy()
    prof_text = (df.get('description', '').fillna('').astype(str) + ' '
                 + df.get('display_name', '').fillna('').astype(str) + ' '
                 + df.get('location', '').fillna('').astype(str))
    tw = _tweets_by_user()
    uname = df['username'].astype(str).str.lower().str.strip()

    rows = []
    for u, ptxt in zip(uname, prof_text):
        p = _score_text(ptxt)
        blob, n = tw.get(u, ('', 0))
        t = _score_text(blob)
        row = {f'prof_iran_{k}': p[k] for k in CATEGORIES}
        row['prof_iran_flag'] = p['flag']
        row['prof_persian_char_ratio'] = p['persian_ratio']
        row.update({f'tw_iran_{k}': t[k] for k in CATEGORIES})
        row['tw_iran_flag'] = t['flag']
        row['tw_persian_char_ratio'] = t['persian_ratio']
        row['has_tweets'] = int(n > 0)
        row['n_tweets'] = n
        rows.append(row)
    feats = pd.DataFrame(rows, index=df.index)
    for c in FEATURE_COLS:
        df[c] = feats[c]
    return df


def tweet_blob_series(df: pd.DataFrame) -> pd.Series:
    """Per-row concatenated tweet text (English-preferred) for a TF-IDF feature set."""
    tw = _tweets_by_user()
    uname = df['username'].astype(str).str.lower().str.strip()
    return pd.Series([tw.get(u, ('', 0))[0] for u in uname], index=df.index)
