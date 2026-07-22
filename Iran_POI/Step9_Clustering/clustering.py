#!/usr/bin/env python3
"""
Step 9 — Clustering (חלוקה לאשכולות) for the Iran_POI project.

Goal (from the project brief): cluster the candidate X/Twitter users, find the
*optimal* number of clusters, and characterise each cluster — who are these users
and what do they have in common?

Design choices (documented so the report can justify them):
  * Unit of analysis  : one row per candidate user (Candidates_user_data_MERGED.csv, 946).
  * Clustering features: BEHAVIOURAL / numeric only (followers, following, statuses,
    ratio, bio length, account age, verified, has-description/location, Iran mentions,
    network degree). These are defined for *every* user, so the clustering is not
    driven by a missing-data artefact.
  * Sentiment / emotion : only 236/946 users have collected posts, so these are used
    to *characterise* the clusters (overlay), NOT as clustering inputs — otherwise a
    huge "no posts" cluster would dominate.
  * Text (description)   : used for per-cluster top-terms (characterisation) and for a
    text-augmented robustness check, but the headline solution is behavioural.
  * Feature scaling      : heavy-tailed counts are log1p-transformed, then everything
    is standardised (z-score) so no single feature dominates the Euclidean distance.

Reproducible: RANDOM_SEED = 42 everywhere.

Outputs (all in Iran_POI/Clustering/):
  cluster_assignments.csv          user -> cluster label + key features
  cluster_profiles.csv             one row per cluster: size + median/%% profile
  cluster_top_terms.csv            top TF-IDF terms per cluster
  cluster_sentiment_emotion.csv    mean sentiment/emotion per cluster (posts subset)
  plot_elbow.png                   inertia vs k (elbow)
  plot_silhouette.png              silhouette vs k, KMeans vs Agglomerative
  plot_pca_clusters.png            PCA 2D scatter coloured by cluster
  plot_tsne_clusters.png           t-SNE 2D scatter coloured by cluster
  plot_cluster_profile_heatmap.png standardised numeric profile per cluster
  plot_cluster_sizes.png           cluster sizes
  plot_emotion_by_cluster.png      dominant-emotion mix per cluster
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

HERE = Path(__file__).resolve().parent                 # Iran_POI/Clustering/
DATA = HERE.parent / "data"
CLS = HERE.parent / "Classification"
POOL_FILE = DATA / "Candidates_user_data_MERGED.csv"
SENT_FILE = DATA / "Posts_sentiment_emotion_cleaned.csv"
CONN_FILE = DATA / "POIs_candidate_connections_UNIQUE.csv"

K_RANGE = list(range(2, 11))                            # search k = 2..10

# Iran keyword lists — identical to active_learning.py so features stay consistent.
IRAN_KEYWORDS = [
    "iran", "iranian", "persian", "persia", "tehran", "shiraz", "esfahan",
    "isfahan", "mashhad", "tabriz", "kerman", "qom", "farsi",
    "ایران", "ایرانی", "تهران", "شیراز", "اصفهان", "مشهد", "تبریز", "فارسی",
    "إيران", "ايران", "إيراني", "ايراني", "طهران", "شيراز",
]

EMOTION_COLS = ["emotion_joy", "emotion_anger", "emotion_sadness",
                "emotion_fear", "emotion_surprise", "emotion_disgust"]
SENT_COLS = ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def _has_iran(text) -> int:
    if pd.isna(text):
        return 0
    s = str(text).lower()
    return int(any(kw in s for kw in IRAN_KEYWORDS))


# ---------------------------------------------------------------------------
# 1. Load users + build behavioural features
# ---------------------------------------------------------------------------
def load_users() -> pd.DataFrame:
    df = pd.read_csv(POOL_FILE, encoding="utf-8-sig")
    df["username"] = _norm(df["username"])
    df = df.drop_duplicates(subset="username").reset_index(drop=True)

    for c in ["followers_count", "following_count", "statuses_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["bio_length"] = df["description"].fillna("").astype(str).str.len()
    df["followers_following_ratio"] = df["followers_count"] / (df["following_count"] + 1)
    dt = pd.to_datetime(df["created_at"], format="%B %Y", errors="coerce")
    df["account_age_years"] = (pd.Timestamp.today() - dt).dt.days / 365.25
    df["account_age_years"] = df["account_age_years"].fillna(df["account_age_years"].median())
    df["has_description"] = df["description"].notna().astype(int)
    df["has_location"] = df["location"].notna().astype(int)
    df["verified_flag"] = (
        df.get("verified", pd.Series(False, index=df.index))
        .astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    )
    df["bio_mentions_iran"] = df["description"].apply(_has_iran)
    df["name_mentions_iran"] = df["display_name"].apply(_has_iran)
    df["location_mentions_iran"] = df["location"].apply(_has_iran)

    # network degree from the connections graph (how "central" the user is)
    deg = _network_degree()
    df["network_degree"] = df["username"].map(deg).fillna(0).astype(int)
    return df


def _network_degree() -> dict:
    if not CONN_FILE.exists():
        return {}
    c = pd.read_csv(CONN_FILE, encoding="utf-8-sig")
    a = _norm(c["target_username"])
    b = _norm(c["other_username"])
    return pd.concat([a, b]).value_counts().to_dict()


# ---------------------------------------------------------------------------
# 2. Description translation cache (for text characterisation)
# ---------------------------------------------------------------------------
def load_translation_map() -> dict:
    """username(lower) -> description_en, from every translated cache on disk."""
    out = {}
    caches = sorted(glob.glob(str(CLS / "Iteration_*/iteration_*_combined_translated.csv")))
    consensus = CLS / "Iteration_1" / "Step5_Analysis" / "iteration_1_consensus_translated.csv"
    if consensus.exists():
        caches = [str(consensus)] + caches
    for f in caches:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "username" not in d.columns or "description_en" not in d.columns:
            continue
        for u, t in zip(_norm(d["username"]), d["description_en"].fillna("")):
            if t and u not in out:
                out[u] = t
    return out


def description_en(df: pd.DataFrame) -> pd.Series:
    """Translated bio where available (cache), else the raw bio (offline fallback)."""
    tmap = load_translation_map()
    cached = df["username"].map(tmap)
    return cached.fillna(df["description"].fillna("")).astype(str)


# ---------------------------------------------------------------------------
# 3. Sentiment / emotion aggregates per user (characterisation overlay)
# ---------------------------------------------------------------------------
def sentiment_by_user() -> pd.DataFrame:
    if not SENT_FILE.exists():
        return pd.DataFrame()
    s = pd.read_csv(SENT_FILE, encoding="utf-8-sig")
    s["username"] = _norm(s["username"])
    for c in SENT_COLS + EMOTION_COLS:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    agg = {c: "mean" for c in SENT_COLS + EMOTION_COLS if c in s.columns}
    g = s.groupby("username").agg(agg)
    g["post_count"] = s.groupby("username").size()
    dom = s.groupby("username")["emotion_label"].agg(
        lambda x: x.value_counts().idxmax() if len(x.dropna()) else "")
    g["dominant_emotion"] = dom
    return g.reset_index()


# ---------------------------------------------------------------------------
# 4. Build the clustering feature matrix
# ---------------------------------------------------------------------------
BEHAV_LOG = ["followers_count", "following_count", "statuses_count",
             "followers_following_ratio", "network_degree"]
BEHAV_LIN = ["bio_length", "account_age_years", "has_description", "has_location",
             "verified_flag", "bio_mentions_iran", "name_mentions_iran",
             "location_mentions_iran"]


def behavioural_matrix(df: pd.DataFrame):
    feats = pd.DataFrame(index=df.index)
    for c in BEHAV_LOG:
        feats[c + "_log"] = np.log1p(df[c].clip(lower=0))
    for c in BEHAV_LIN:
        feats[c] = df[c].astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)
    return X, list(feats.columns), scaler


def text_svd(df: pd.DataFrame, n_components=30):
    desc = description_en(df)
    vec = TfidfVectorizer(max_features=400, lowercase=True, stop_words="english",
                          ngram_range=(1, 2), min_df=3)
    tfidf = vec.fit_transform(desc)
    n_comp = min(n_components, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
    Z = svd.fit_transform(tfidf)
    Z = StandardScaler().fit_transform(Z)
    return Z, vec, tfidf


# ---------------------------------------------------------------------------
# 5. Pick optimal k
# ---------------------------------------------------------------------------
def choose_k(X):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lab_km = km.fit_predict(X)
        sil_km = silhouette_score(X, lab_km)
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab_ag = agg.fit_predict(X)
        sil_ag = silhouette_score(X, lab_ag)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette_kmeans": sil_km, "silhouette_agglo": sil_ag})
        print(f"  k={k:2d}  inertia={km.inertia_:9.1f}  "
              f"sil(KMeans)={sil_km:.3f}  sil(Agglo)={sil_ag:.3f}")
    res = pd.DataFrame(rows)
    best_k = int(res.loc[res["silhouette_kmeans"].idxmax(), "k"])
    return res, best_k


# ---------------------------------------------------------------------------
# 6. Characterise clusters
# ---------------------------------------------------------------------------
def profile_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels
    rows = []
    for c in sorted(df["cluster"].unique()):
        g = df[df["cluster"] == c]
        rows.append({
            "cluster": c,
            "size": len(g),
            "pct": round(100 * len(g) / len(df), 1),
            "median_followers": int(g["followers_count"].median()),
            "median_following": int(g["following_count"].median()),
            "median_statuses": int(g["statuses_count"].median()),
            "median_ff_ratio": round(g["followers_following_ratio"].median(), 2),
            "median_bio_len": int(g["bio_length"].median()),
            "median_age_years": round(g["account_age_years"].median(), 1),
            "median_network_degree": int(g["network_degree"].median()),
            "pct_verified": round(100 * g["verified_flag"].mean(), 1),
            "pct_has_location": round(100 * g["has_location"].mean(), 1),
            "pct_has_desc": round(100 * g["has_description"].mean(), 1),
            "pct_iran_in_bio": round(100 * g["bio_mentions_iran"].mean(), 1),
        })
    return pd.DataFrame(rows)


def top_terms_per_cluster(df, labels, tfidf, vec, topn=12):
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for c in sorted(np.unique(labels)):
        mask = labels == c
        mean_tfidf = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top = terms[mean_tfidf.argsort()[::-1][:topn]]
        rows.append({"cluster": int(c), "top_terms": ", ".join(top)})
    return pd.DataFrame(rows)


def label_overlay(df, labels):
    """Distribution of the manual labels (where known) inside each cluster."""
    lab_path = sorted(glob.glob(str(CLS / "Iteration_*/iteration_*_combined_labeled.csv")))
    if not lab_path:
        return pd.DataFrame()
    lab = pd.read_csv(lab_path[-1])          # largest / latest combined labeled set
    lab["username"] = _norm(lab["username"])
    keep = ["username", "target_population", "person_vs_organization"]
    keep = [c for c in keep if c in lab.columns]
    lab = lab[keep].drop_duplicates("username")
    m = df.assign(cluster=labels).merge(lab, on="username", how="left")
    rows = []
    for c in sorted(m["cluster"].unique()):
        g = m[m["cluster"] == c]
        known = g["target_population"].notna().sum()
        rows.append({
            "cluster": c,
            "n_labeled": int(known),
            "pct_target(=1)": _pct(g["target_population"], 1),
            "pct_nontarget(=0)": _pct(g["target_population"], 0),
            "pct_person(=1)": _pct(g.get("person_vs_organization"), 1),
            "pct_org(=0)": _pct(g.get("person_vs_organization"), 0),
        })
    return pd.DataFrame(rows)


def _pct(series, val):
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return round(100 * (s == val).mean(), 1)


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
def plot_selection(res, best_k):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["k"], res["inertia"], "o-", color="#2b6cb0")
    ax.axvline(best_k, ls="--", color="red", alpha=.6, label=f"chosen k={best_k}")
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("inertia (within-cluster SSE)")
    ax.set_title("Elbow — inertia vs k (KMeans)")
    ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "plot_elbow.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["k"], res["silhouette_kmeans"], "o-", label="KMeans")
    ax.plot(res["k"], res["silhouette_agglo"], "s--", label="Agglomerative (ward)")
    ax.axvline(best_k, ls="--", color="red", alpha=.6, label=f"chosen k={best_k}")
    ax.set_xlabel("number of clusters (k)")
    ax.set_ylabel("mean silhouette score")
    ax.set_title("Silhouette vs k — higher is better")
    ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "plot_silhouette.png", dpi=140); plt.close(fig)


def plot_2d(X, labels, method, fname, title):
    if method == "pca":
        emb = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
    else:
        emb = TSNE(n_components=2, random_state=RANDOM_SEED,
                   perplexity=30, init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=18, alpha=.75)
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1"); ax.set_ylabel(f"{method.upper()} 2")
    legend = ax.legend(*sc.legend_elements(), title="cluster",
                       loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout(); fig.savefig(HERE / fname, dpi=140); plt.close(fig)


def plot_profile_heatmap(X, feat_names, labels):
    dfX = pd.DataFrame(X, columns=feat_names)
    dfX["cluster"] = labels
    means = dfX.groupby("cluster").mean()
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(feat_names) + 1))
    im = ax.imshow(means.T.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(means.index)))
    ax.set_xticklabels([f"C{c}\n(n={ (labels==c).sum() })" for c in means.index])
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_title("Standardised feature profile per cluster (z-scores)")
    fig.colorbar(im, ax=ax, shrink=.7, label="mean z-score")
    fig.tight_layout(); fig.savefig(HERE / "plot_cluster_profile_heatmap.png", dpi=140)
    plt.close(fig)


def plot_sizes(labels):
    vals, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"C{v}" for v in vals], counts, color="#4c9f70")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set_ylabel("users"); ax.set_title("Cluster sizes")
    fig.tight_layout(); fig.savefig(HERE / "plot_cluster_sizes.png", dpi=140); plt.close(fig)


def plot_emotion(df, labels, sent):
    m = df.assign(cluster=labels).merge(sent[["username", "dominant_emotion"]],
                                        on="username", how="left")
    m = m.dropna(subset=["dominant_emotion"])
    m = m[m["dominant_emotion"] != ""]
    if len(m) == 0:
        return
    ct = pd.crosstab(m["cluster"], m["dominant_emotion"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("share of users w/ posts")
    ax.set_title("Dominant emotion mix per cluster (users with posts)")
    ax.legend(title="emotion", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "plot_emotion_by_cluster.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=== Step 9 — Clustering ===")
    df = load_users()
    print(f"users: {len(df)}")

    X, feat_names, _ = behavioural_matrix(df)
    print(f"behavioural feature matrix: {X.shape}")

    print("searching optimal k (behavioural features):")
    res, best_k = choose_k(X)
    res.to_csv(HERE / "k_search.csv", index=False)
    print(f"chosen k = {best_k} (max KMeans silhouette)")

    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)
    final_sil = silhouette_score(X, labels)
    print(f"final KMeans silhouette (k={best_k}): {final_sil:.3f}")

    # robustness: text-augmented clustering, same k
    Zt, vec, tfidf = text_svd(df)
    Xc = np.hstack([X, Zt])
    lab_txt = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10).fit_predict(Xc)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels, lab_txt)
    print(f"text-augmented vs behavioural agreement (ARI): {ari:.3f}")

    # characterise
    profiles = profile_clusters(df, labels)
    terms = top_terms_per_cluster(df, labels, tfidf, vec)
    overlay = label_overlay(df, labels)
    sent = sentiment_by_user()

    # per-cluster sentiment/emotion means
    if not sent.empty:
        se = df.assign(cluster=labels).merge(sent, on="username", how="left")
        se_cols = [c for c in SENT_COLS + EMOTION_COLS + ["post_count"] if c in se.columns]
        se_prof = se.groupby("cluster")[se_cols].mean().round(3).reset_index()
        se_prof["n_users_with_posts"] = se.dropna(subset=["post_count"]).groupby("cluster").size().reindex(se_prof["cluster"]).fillna(0).astype(int).values
        se_prof.to_csv(HERE / "cluster_sentiment_emotion.csv", index=False)

    profiles.to_csv(HERE / "cluster_profiles.csv", index=False)
    terms.to_csv(HERE / "cluster_top_terms.csv", index=False)
    if not overlay.empty:
        overlay.to_csv(HERE / "cluster_label_overlay.csv", index=False)

    out = df[["username", "display_name", "followers_count", "following_count",
              "statuses_count", "followers_following_ratio", "bio_length",
              "account_age_years", "verified_flag", "network_degree",
              "bio_mentions_iran"]].copy()
    out["cluster"] = labels
    out.to_csv(HERE / "cluster_assignments.csv", index=False)

    # plots
    plot_selection(res, best_k)
    plot_2d(X, labels, "pca", "plot_pca_clusters.png",
            f"Users in PCA space — {best_k} behavioural clusters")
    plot_2d(X, labels, "tsne", "plot_tsne_clusters.png",
            f"Users in t-SNE space — {best_k} behavioural clusters")
    plot_profile_heatmap(X, feat_names, labels)
    plot_sizes(labels)
    if not sent.empty:
        plot_emotion(df, labels, sent)

    # console summary
    print("\n=== cluster profiles ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(profiles.to_string(index=False))
    print("\n=== top terms ===")
    print(terms.to_string(index=False))
    if not overlay.empty:
        print("\n=== manual-label overlay ===")
        print(overlay.to_string(index=False))
    print(f"\nAll outputs written to: {HERE}")


if __name__ == "__main__":
    main()
