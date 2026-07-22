# Iran_POI — זיהוי קבוצת המטרה האיראנית ב-X/Twitter

פרויקט גמר, **מבוא ללמידת מכונה (SE 2026, סמסטר ב')**.
המטרה: לזהות אוטומטית משתמשי X השייכים לאוכלוסיית המטרה האיראנית, באמצעות
**Active Learning**, ולנתח אותם באשכולות.

התיקייה מסודרת **לפי השלבים בהנחיות ה-PDF**. זהו אינדקס הניווט המלא.

## ⚡ תקציר — מה הפרויקט השיג (מבט של 30 שניות)

| שלב | מה עשינו | התוצאה המרכזית |
|---|---|---|
| **1** | בחירת אוכלוסיית יעד | **איראנים** ב-X/Twitter |
| **2** | ניתוח סטטיסטי | 945 יוזרים; חציון עוקבים 68; ותק 2007–2025 + 3 היסטוגרמות |
| **3** | תיוג ידני 100 (כפול) | Cohen's κ = **0.88 / 1.0 / 0.94** (עובר את היעד) |
| **4** | 3 תרשימי החלטה | עצי Yes/No ל-3 משימות הסיווג |
| **5** | אימון מסווגים | **1,296 ניסויים/איטרציה**: 6 אלגוריתמים × K-Fold+LOOCV × 2/3 מחלקות × balanced+unbalanced |
| **6** | Active Learning | 100 → **640** מתויגים לאורך 7 איטרציות |
| **7** | קריטריון עצירה | המודל **נרווה** (Δ<0.5%) → עצירה מוצדקת |
| **7+** | העשרת פיצ'רים | **AUC של target: 0.76 → 0.89** (הפריצה האמיתית) |
| **8-A** | סף ביטחון | סף 0.8 → 173 יוזרים · **Hit Rate 85.7–90%** |
| **8-B** | מול LLM | המודל המאומן **0.95** מול Claude Opus **0.53** |
| **9** | אשכולות | k=8 · 3 על-קבוצות · ARI 0.73 (יציבות) |

> **סדר קריאה מומלץ:** 1→2→3→4→5→6→7→(7+)→8→9. כל שלב בונה על הקודם.

## מפת סטטוס (שלב → תיקייה → סטטוס)

| שלב | נושא | תיקייה | סטטוס |
|----|------|--------|-------|
| **1** | בחירת אוכלוסיית מטרה | `Step1_Population_Choice/` | ✅ Iran |
| **2** | ניתוח סטטיסטי + היסטוגרמות | `Step2_Statistical_Analysis/` | ✅ |
| **3** | תיוג ידני iteration 1 (100, תיוג כפול) | `Classification/Iteration_1/Step3_Manual_Labeling/` | ✅ |
| **4** | תרשימי זרימה להחלטת סיווג | `Step4_Decision_Flowcharts/` | ✅ `user_labeling_decision_flows.pptx` (3 תרשימים) |
| **5** | אימון מסווגים | `Classification/` | ✅ |
| **6** | Active Learning (iterations 2–7, עד 640) | `Classification/` + `Iteration_2..7/` | ✅ |
| **7** | קריטריון עצירה (Stopping Criteria) | `Classification/Step7_Stopping_Criteria/` + `Iteration_7/` | ✅ **בוצע — המודל נרווה** |
| **7+** | **העשרת פיצ'רים** (ספֵּציפיות איראנית + ציוצים) | `Classification/Feature_Enrichment/` | ✅ **AUC 0.76→0.89** |
| **8** | סף ביטחון + השוואת LLM | `Classification/Step8_Confidence_Threshold/` + `LLM_Comparison/` | ✅ **בוצע** — Hit Rate 85–90%; LLM 10 ריצות + majority |
| **9** | חלוקה לאשכולות (Clustering) | `Step9_Clustering/` | ✅ |

> **מספור:** שלבים 3, 5, 6, 7 כולם תחת `Classification/` (דרישת ה-PDF במפורש). מבנה פנימי מפורט — `Classification/README.md`.

---

## אינדקס קבצים מפורט — לפי שלב

### שלב 1 — בחירת אוכלוסייה · `Step1_Population_Choice/`
- `README.md` — Iran נבחרה (אין קוד).

### שלב 2 — ניתוח סטטיסטי · `Step2_Statistical_Analysis/`
- `Step2_Statistical_Analysis.ipynb` — הקוד
- `author_statistics.csv` — טבלת מדדים תיאוריים
- `followers_histogram.png` · `following_histogram.png` · `statuses_histogram.png` — 3 ההיסטוגרמות

### שלב 3 — תיוג ידני iteration 1 · `Classification/Iteration_1/Step3_Manual_Labeling/`
- `step3_agreement.py` — חישוב Cohen's Kappa + קונצנזוס
- `iteration_1_labels_NADAV*.csv` — תיוג **מתייג 1** (שני באצ'ים של 50)
- `iteration_1_labels_*_annotator2.csv` — תיוג **מתייג 2** (נחלק על 7 מהמשתמשים → מזה ה-Kappa)
- `ANNOTATION_NOTE.md` — תיעוד תהליך התיוג (כולל באג-שכפול-ההערות)
- `iteration_1_labels_merged_sidebyside.csv` — מיזוג זה-לצד-זה
- `iteration_1_agreement_report.csv` — κ לכל עמודה · `iteration_1_disagreements.csv` — מחלוקות
- `iteration_1_labels_consensus.csv` — הסימון הסופי
- `iteration_1_{target_population,locals_vs_diaspora,person_vs_organization}.csv` (+ `_summary.csv`) — 3 קובצי היעד

### שלב 4 — תרשימי זרימה · `Step4_Decision_Flowcharts/`
- `user_labeling_decision_flows.pptx` — ✅ 3 תרשימים (target_population / locals_vs_diaspora / person_vs_organization)
- `README.md` — הסבר השלב

### שלב 5 — אימון מסווגים · `Classification/`
- `Step5_Train_Classifiers.ipynb` — notebook האימון
- `Iteration_1/Step5_Analysis/step5_analysis.py` — אימות מנצחים + גרפים
- `Iteration_1/Step5_Analysis/iteration_1_best_models_summary.csv` · `iteration_1_degeneracy_check.csv`
- `Iteration_1/Step5_Analysis/plot_f1_by_algorithm.png` · `plot_f1_by_feature_set.png` · `plot_kfold_vs_loocv.png`
- `Iteration_1/experiments_results_iteration_1.csv` — קובץ הניסויים (**1,296 ניסויים**)

### שלב 6 — Active Learning · `Classification/` + `Iteration_2..7/`
- `active_learning.py` — מנוע ה-AL (Phase A: בחירת לא-ודאיים · Phase C: מיזוג+אימון+השוואה)
- `Step6_Active_Learning.ipynb` — notebook
- `Iteration_N/iteration_N_combined_labeled.csv` — הסט המתויג הגדל (200,300,…,640)
- `Iteration_N/experiments_results_iteration_N.csv` — **1,296 ניסויים** לכל איטרציה
- `Iteration_N/iteration_N_manual_labels_*.csv` — 3 קובצי תיוג לכל איטרציה
- `iteration_comparison_summary.csv` + `plot_iteration_comparison.png` — מגמה בין-איטרציות
- `plot_active_learning_story.png` — **הגרף להצגה:** רוויה על לייבלים → פריצה ע"י פיצ'רים (0.76→0.89)
- `holdout_test_set.csv` · `holdout_improvement_trend.csv` · `plot_holdout_improvement.png` — עקומת שיפור על סט-מבחן קפוע

### שלב 7 — קריטריון עצירה · `Classification/Step7_Stopping_Criteria/` + `Iteration_7/`
- `stopping_criteria.py` — `--phase A` (חיזוי+היסטוגרמה+דגימת 2 רצועות) · `--phase C` (מיזוג 40+אימון+החלטה)
- `stopping_criteria_unlabeled_users_predictions.csv` — חיזוי המודל על הלא-מתויגים
- `stopping_criteria_confidence_histogram.png` — התפלגות ביטחון
- `stopping_criteria_probability_group_samples_for_manual_labeling.csv` — 40 היוזרים שתויגו (20 לא-בטוחים + 20 בטוחים)
- `step7_urls.txt` · `step7_urls_uncertain.txt` — לינקים לגריפה/תיוג
- `stopping_criteria_performance_summary.csv` — ביצועים לפי איטרציה + דלתאות
- `stopping_criteria_final_accuracy_auc_graph.png` — הגרף הסופי
- `stopping_criteria_decision_summary.csv` — **החלטת העצירה** (המודל נרווה: 600→640 ללא שיפור מדיד)
- `../Iteration_7/` — 640 מתויגים: `iteration_7_combined_labeled.csv`, `experiments_results_iteration_7.csv` (**1,296**), `iteration_7_manual_labels_*.csv`

### שלב 7+ — העשרת פיצ'רים · `Classification/`
- `iran_features.py` — לקסיקונים **ספֵּציפיים-איראניים** (גאוגרפיה/פוליטיקה/קמפיינים/מדיה/ספורט-כלכלה + אנטי-סיגנל אפגני/טג'יקי + דגל 🇮🇷), על ביו + ציוצים
- `feature_enrichment_experiment.py` — ניסוי baseline מול enriched (K-Fold)
- `Feature_Enrichment/enrichment_experiments.csv` — כל הניסויים
- `Feature_Enrichment/enrichment_summary.csv` — טבלת before/after לכל משימה
- `Feature_Enrichment/plot_enrichment_before_after.png` — **הגרף:** רוויה → פריצה (target 0.76→0.89)
- `plot_enrichment.py` — קוד הגרף

### שלב 8 — סף ביטחון + LLM · ✅ בוצע
- **8-A** `Classification/Step8_Confidence_Threshold/` — סף 0.8, Hit Rate 85.7%–90% (עובר 85%), 173 באוכלוסיית היעד
- **8-B** `LLM_Comparison/` — Claude Opus (מותר לפי "מודל גבוה יותר"), 10 ריצות + majority; המודל המאומן 0.95 מול LLM 0.53

### שלב 9 — אשכולות · `Step9_Clustering/`
- `clustering.py` — הקוד (k=8) · `REPORT.md` — הדוח
- `cluster_assignments.csv` · `cluster_profiles.csv` · `cluster_top_terms.csv` · `cluster_sentiment_emotion.csv` · `k_search.csv`
- `plot_{elbow,silhouette,pca_clusters,tsne_clusters,cluster_sizes,cluster_profile_heatmap,emotion_by_cluster}.png` — 7 גרפים

---

## נתונים גולמיים · `data/` (משותף, אין לשנות)
- `Candidates_user_data_MERGED.csv` — 946 מועמדים (הפול; פרופילים בלבד)
- `posts.csv` — 21,810 ציוצים (238 יוזרים; כולל `Text_en` מתורגם)
- `Posts_sentiment_emotion_cleaned.csv` — ציוצים עם סנטימנט+רגש (בשימוש שלב 9 + פוטנציאל לפיצ'רים)
- `POIs_candidate_connections_UNIQUE.csv` — הגרף החברתי (קשתות; פוטנציאל לפיצ'רי רשת)
- `POI_twitter_users_data.csv` — עוגני POI ידועים

## מה שנשאר
- **דו"ח Word** — כל 9 השלבים בוצעו; נותר לארוז לדו"ח (מטרה/שיטה/תוצאות/מסקנות) + מצגת
- אופציונלי: פיצ'רי רשת/סנטימנט (שכבת שיפור נוספת), אינטגרציית ההעשרה למנוע הראשי
