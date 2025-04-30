# -*- coding: utf-8 -*-
"""
Sentiment Analysis - Naive Bayes + Logistic Regression
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import string
import matplotlib.pyplot as plt

# --------------------------
# 1. خواندن داده + چک اولیه
# --------------------------
file_path = 'training.1600000.processed.noemoticon.csv'
column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']

try:
    temp_df = pd.read_csv(file_path, header=None, encoding='latin1', names=column_names, low_memory=False)
    print("✅ مقادیر sentiment در فایل:")
    print(temp_df['sentiment'].value_counts())

    df = temp_df.copy()
except FileNotFoundError:
    print("❌ Dataset file not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# --------------------------
# 2. تمیز کردن ستون sentiment
# --------------------------
print("\n🔧 قبل از فیلتر کردن sentiment:")
print(df['sentiment'].value_counts())

df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
df = df[df['sentiment'].isin(['0', '4'])]

print("\n✅ بعد از فیلتر کردن sentiment (0 و 4):")
print(df['sentiment'].value_counts())

df['sentiment'] = df['sentiment'].replace({'0': 0, '4': 4})
df['sentiment'] = pd.to_numeric(df['sentiment'], errors='raise')

# --------------------------
# 3. پیش‌پردازش متن
# --------------------------
punctuation = string.punctuation

def remove_punctuation(text):
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)

df['text_lower'] = df['text'].str.lower()
df['text_without_punctuation'] = df['text_lower'].apply(remove_punctuation)

# --------------------------
# 4. تقسیم داده‌ها
# --------------------------
X = df['text_without_punctuation']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

print("\n📊 توزیع کلاس‌ها در train:")
print(y_train.value_counts())

print("\n📊 توزیع کلاس‌ها در test:")
print(y_test.value_counts())

# --------------------------
# 5. TF-IDF Vectorizer
# --------------------------
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("\n🔢 تعداد ویژگی‌ها:", len(tfidf_vectorizer.vocabulary_))

# --------------------------
# 6. آموزش مدل Naive Bayes
# --------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# --------------------------
# 7. ارزیابی مدل
# --------------------------
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 دقت مدل Naive Bayes: {accuracy:.4f}")
print("\n📋 گزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred))

# --------------------------
# 8. Confusion Matrix
# --------------------------
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Naive Bayes - Confusion Matrix")
plt.show()

# --------------------------
# 9. Cross Validation
# --------------------------
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("\n🔁 Cross-validation scores:", cv_scores)
print(f"میانگین دقت CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --------------------------
# 10. Optional: Logistic Regression
# --------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"\n🎯 دقت مدل Logistic Regression: {accuracy_lr:.4f}")
print(classification_report(y_test, y_pred_lr))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# --------------------------
# 11. نمودار دقت
# --------------------------
plt.figure(figsize=(6, 4))
plt.bar(['Naive Bayes'], [accuracy], color='skyblue', label='Naive Bayes')
plt.bar(['Logistic Reg.'], [accuracy_lr], color='orange', label='Logistic Reg.')
plt.ylabel('دقت')
plt.title('مقایسه دقت مدل‌ها')
plt.legend()
plt.ylim(0, 1)
plt.show()