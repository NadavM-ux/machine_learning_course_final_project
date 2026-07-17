# Classification — שלבים 3, 5, 6

לפי ה-PDF, שלושת השלבים הבאים נשמרים תחת התיקייה **`Classification/`** (שם זה נדרש במפורש
בהנחיות, ולכן לא שונה). כל שלב ממופה כאן למיקומו המדויק.

## שלב 3 — תיוג ידני (Iteration 1)
📁 `Iteration_1/Step3_Manual_Labeling/`

**אין מחברת לשלב 3 — וזה תקין:** התיוג עצמו הוא **משימה אנושית** (בודקים כל משתמש ידנית ב-X),
ולכן אין קוד אימון. מה שכן קוד — חישוב ה-consensus וה-Cohen's Kappa — הופק פעם אחת והתוצאה
נשמרה כ-CSV. שלב 3 "נכנס לתמונה" כאן כ**קלט לשלב 5**: הקובץ `iteration_1_labels_consensus.csv`
הוא סט האימון של 100 המשתמשים הראשונים, ש-`Step5_Train_Classifiers.ipynb` טוען בתא הראשון.

- `iteration_1_labels_NADAV.csv`, `iteration_1_labels_NADAV_IL.csv` — תיוג של שני המתייגים (תיוג כפול)
- `iteration_1_labels_consensus.csv` — ההכרעה הסופית (עם `consensus_source`)
- `iteration_1_agreement_report.csv` — Percent Agreement + Cohen's Kappa לכל עמודה
- 🐍 `step3_agreement.py` — **הקוד שמחשב את ה-Kappa ובונה את ה-consensus** (שוחזר 2026-07-15;
  מאומת שמשחזר את שני קבצי הפלט זהים בייט-אחר-בייט). קורא מ-`merged_sidebyside`, מריץ
  `cohen_kappa_score` לכל עמודה, וכותב את `agreement_report` + `labels_consensus`.
- `iteration_1_disagreements.csv` — מחלוקות שנפתרו
- `iteration_1_{target_population,locals_vs_diaspora,person_vs_organization}.csv` + קבצי `_summary`
- ⚠️ **לבדיקתך:** קיימים גם `... copy.csv` (של NADAV ו-NADAV_IL) שנבדלים מהמקור ב-5–8 תאים
  בלבד. לא נמחקו כי הם מכילים תיוג — החלט אם למחוק/למזג.

## שלב 5 — אימון מסווגים
- 📓 `Step5_Train_Classifiers.ipynb` — הרצת הניסויים (6 אלגוריתמים × פיצ'רים × KFold/LOOCV × 2/3 מחלקות)
- 📁 `Iteration_1/Step5_Analysis/` — ניתוח + גרפים:
  - `step5_analysis.py`, `iteration_1_consensus_translated.csv` (תרגום לאנגלית)
  - `plot_f1_by_algorithm.png`, `plot_f1_by_feature_set.png`, `plot_kfold_vs_loocv.png`
  - `iteration_1_best_models_summary.csv`, `iteration_1_degeneracy_check.csv`
- 📄 `Iteration_1/experiments_results_iteration_1.csv` — קובץ הניסויים (עמודות חובה לפי ה-PDF)

## שלב 6 — Active Learning (Iterations 2–6)
- 📓 `Step6_Active_Learning.ipynb` — הצינור המלא של iteration 2 (predict → uncertainty → label → retrain)
- ⚙️ `active_learning.py` — **המנוע הכללי** לאיטרציות 3–6. פקודות:
  - `python active_learning.py --iteration N --phase A` → בוחר 100 משתמשים לא-ודאיים לתיוג
  - `python active_learning.py --iteration N --phase C` → ממזג תוויות, מאמן מחדש, מרענן גרפים
  - `python active_learning.py --phase H` → מרענן את גרף מגמת השיפור (holdout)
- 📁 `Iteration_2/ … Iteration_6/` — התוצרים של כל איטרציה (predictions, manual_labels, combined, experiments)
- 📄 תוצרי השוואה ברמת התיקייה:
  - `plot_iteration_comparison.png` + `iteration_comparison_summary.csv`
  - `plot_holdout_improvement.png` + `holdout_improvement_trend.csv`
  - `holdout_test_set.csv` — סט hold-out קבוע להערכה עקבית

## מדריכים
📁 `docs/` — `ITERATIONS_HOWTO.md` (המדריך הסמכותי), `STEP5_GUIDE.md`, `STEP6_GUIDE.md`.

## כלל ברזל (מתוך המדריך)
`balanced=True` תמיד · שלוש המשימות תמיד · גם 2-class וגם 3-class · **אין לתייג משתמשים אוטומטית**
(חובת תיוג אנושי — היוזרים חייבים להיבדק ידנית בפרופיל ב-X).
