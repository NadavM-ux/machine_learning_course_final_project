# שלב 4 — תרשימי זרימה להחלטת סיווג משתמשים

**אין קוד בשלב זה** — התוצר הוא קובץ PowerPoint עם שלושה תרשימי זרימה, שמנסחים באופן פורמלי
את הלוגיקה שבה החלטנו על התיוג הידני (שלב 3).

## תוצר נדרש
קובץ `user_labeling_decision_flows.pptx` הכולל **שלושה תרשימי זרימה**:

1. **תרשים סיווג כללי** (class): target / non_target / unknown
2. **תרשים תת־סיווג** (sub_class): local / diaspora / unknown
3. **תרשים זיהוי סוג חשבון** (account_type): person / organization / unknown

כל תרשים מתאר את רצף השאלות ששימשו בבחינה הידנית, למשל:
אזכור מדינה/לאום בטקסט? מיקום גאוגרפי? שפה? רמזים לארגון מול אדם?

## סטטוס
⬜ **טרם הוגש** — זו משימה אנושית (עריכת שקופיות). כשהקובץ מוכן, יש לשמור אותו כאן.
לפי ה-PDF ניתן גם לשמרו תחת `Manual_Labeling/` — נבחרה כאן תיקייה ייעודית ממוספרת לבהירות.
manual_labaled_process:
    קראתי את המצגת — שלושה תרשימי זרימה, כל אחד מייצג עמודת סיווג אחת. אעבור על כל צומת בנפרד.

שקף 1 — Target population (איראני / לא איראני)
צומת 1: Is the profile suspended/empty/protected (no info)?
האם הפרופיל מושעה / ריק / מוגן — כלומר אין בכלל מידע לעבוד איתו.

Yes → unknown — אין על מה לבסס החלטה, יוצאים מיד.
No → ממשיכים לצומת 2.

זה שער כניסה (gate) — הוא נבדק לפני הכול, כי בלי מידע אין טעם בשאר השאלות.
צומת 2: Does the bio, display name, or username mention Iran / Iranian / Persian / Persia, or an Iranian city?
האם הביו, שם התצוגה או ה-username מזכירים איראן / איראני / פרסי / פרס, או עיר איראנית.

Yes → צומת 3 (בדיקת סתירה).
No → צומת 4 (בדיקת location).

שים לב: זו שאלה על טקסט מוצהר בפרופיל, לא על שפה. העובדה שהוא מצייץ בפרסית לא מפעילה את הצומת הזה.
צומת 3: Clear non-Iranian identity?
זהות לא-איראנית ברורה. זהו צומת הפרכה — הוא קיים כדי לתפוס מקרים כמו "Persian cat lover" או עיתונאי אמריקאי שכותב על איראן.

Yes → Not Target — יש אזכור, אבל הזהות בבירור לא איראנית.
No → Target

צומת 4: Does location indicate Iranian city?
שדה ה-location מצביע על עיר איראנית.

Yes → Target
No → צומת 5.

צומת 5: Personal Posts that indicate the person is Iranian
פוסטים אישיים שמעידים שהאדם איראני (לא ציוצי חדשות/פוליטיקה — אישיים: חיי יומיום, משפחה, מקומות).

Yes → Target
No → Unknown

ההיגיון של השקף: שלושה מסלולים עצמאיים ל-Target (הצהרה בפרופיל / location / התנהגות בפוסטים), עם צומת הפרכה אחד בלבד — ורק על המסלול הראשון. Unknown הוא ברירת המחדל כשכל השלושה נכשלו, לא Not Target.

שקף 2 — Locals vs Diaspora (בתוך איראן / בגולה)
הערה חשובה: השקף הזה רץ רק על מי שסווג Target בשקף 1. אצלך בקובץ הוא נשאר ריק כש-target_population ≠ 1.
צומת 1: Does the location field show an Iranian city?

Yes → Local
No → צומת 2.

צומת 2: Does the location field show a city outside Iran?

Yes → diaspora
No → צומת 3.

שני הצמתים הראשונים הם שדה ה-location בלבד — עדות מוצהרת, הכי חזקה, ולכן ראשונה.
צומת 3: Do recent tweets clearly place the user IN Iran?
האם ציוצים אחרונים ממקמים אותו בבירור בתוך איראן.

Yes → local
No → צומת 4.

צומת 4: Do recent tweets clearly place the user OUTSIDE Iran?
האם ציוצים אחרונים ממקמים אותו בבירור מחוץ לאיראן.

Yes → diaspora
No → צומת 5.

צמתים 3-4 הם עדות התנהגותית — נחותה מ-location מוצהר, ולכן שנייה בתור. המילה clearly היא המפתח: רמז עקיף לא נחשב.
צומת 5: Does the bio/display name suggest diaspora?
האם הביו/שם התצוגה רומז לגולה (למשל "Iranian-American", "Persian in Berlin", דגלים כפולים).

Yes → diaspora
No → Unknown

ההיגיון: היררכיית ראיות ברורה — location מוצהר > ציוצים > רמז בביו. אין כאן צומת הפרכה, כי אין "לא-לוקאל ולא-דיאספורה"; אם כלום לא נתפס → Unknown.

שקף 3 — Person vs Organization
צומת 1: Is the profile suspended/empty/private?

Yes → Unknown
No → צומת 2.

אותו gate כמו בשקף 1.
צומת 2: Does the display name/username clearly indicate an organization?
שם התצוגה/username מצביע בבירור על ארגון (News, Agency, Ministry, Foundation).

Yes → organization
No → צומת 3.

צומת 3: Does the bio describe an institution?
הביו מתאר מוסד — "Official account of...", "Established 1979", כתובת משרדים.

Yes → organization
No → צומת 4.

צומת 4: Is the profile picture a logo / brand mark / non-human image AND Are tweets written in plural/institutional voice
זה הצומת היחיד בכל שלוש התרשימים עם AND — שני התנאים חייבים להתקיים יחד:

תמונת הפרופיל היא לוגו / סמל מותג / תמונה לא-אנושית, וגם
הציוצים כתובים בגוף רבים / בקול מוסדי ("we", "our team", הודעות רשמיות).

ה-AND קיים כי כל תנאי לבדו הוא אינדיקטור חלש: המון אנשים פרטיים שמים לוגו או ציור כתמונת פרופיל, והרבה יחידים כותבים "we" כשהם מדברים על עמם.

Yes → organization
No → צומת 5.

צומת 5: Is there clear evidence this is one individual?
ראיה ברורה שמדובר באדם יחיד — תמונת פנים, סיפור אישי, גוף ראשון יחיד.

Yes → Person
No → Unknown

ההיגיון: התרשים הזה א-סימטרי בכוונה — הוא בודק ארגון שלוש פעמים, ורק בסוף שואל אם זה אדם, ודורש ראיה חיובית ל-Person. כלומר: ברירת המחדל אינה "אדם". זה בדיוק ה-"better not to risk it" שלך — פרופיל עמום נופל ל-Unknown, לא ל-Person.

המיפוי לעמודות ב-CSV
תרשיםעמודהTarget/Local/PersonNot Target/diaspora/organizationUnknownשקף 1target_population102שקף 2locals_vs_diaspora1 (local)0 (diaspora)2שקף 3person_vs_organization1 (person)0 (organization)2

שלוש הערות על התרשימים עצמם

הגייט הראשון של שקף 1 סותר את הכלל שלך. התרשים אומר: מושעה/ריק → unknown, נקודה. אבל הכלל שאתה עובד לפיו הוא שאם פרופיל חסום עדיין מראה אינדיקציה אמיתית לאיראן — מסווגים רגיל. התרשים לא מכיל את החריג הזה. שווה להוסיף יציאה מהגייט.
בשקף 2 אין gate למושעה/ריק. בשקפים 1 ו-3 יש; בשקף 2 חסר. בפועל זה לא שובר כלום (אם target≠1 העמודה ריקה בכל מקרה), אבל זו אי-אחידות.
צומת ההפרכה קיים רק במסלול אחד בשקף 1. אם location = "Tehran" אבל הביו אומר "Israeli journalist covering Iran" — התרשים קופץ ישר ל-Target בלי לעבור דרך Clear non-Iranian identity?. אותו דבר במסלול הפוסטים האישיים. בפועל אתה כנראה מפעיל שיפוט ותופס את זה, אבל התרשים כפי שהוא לא.
 ג