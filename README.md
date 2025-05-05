# Sentiment Analysis on Twitter Data using Naive Bayes

This project performs sentiment analysis on Twitter data to classify tweets into two classes:
- Positive (`4`)
- Negative (`0`)

## Steps:

1. **Data Preprocessing:**
   - Removed punctuation from text.
   - Split data into train and test sets with `stratify=y` to ensure balanced distribution of classes.

2. **Feature Extraction:**
   - Used `TfidfVectorizer` to convert text data into numerical features.
   - Applied TF-IDF weighting to highlight important words.

3. **Model Training:**
   - Trained a `MultinomialNB` model on the training data.
   - The model learns to classify tweets based on word frequencies.

4. **Model Evaluation:**
   - Calculated overall accuracy (~77%).
   - Generated a `classification_report` to analyze performance on each class.
   - Plotted a `confusion matrix` to visualize errors.
   - Performed `cross-validation` to ensure model stability.

5. **Results:**
   - Overall accuracy: ~77%
   - Recall for class `4` (Positive): ~7% (Low performance on positive tweets).

6. **Improvement Suggestions:**
   - Use `class_weight='balanced'` to handle class imbalance.
   - Experiment with stronger models like Logistic Regression or SVM.
   - Apply oversampling techniques to increase the number of positive samples.
   - Perform more advanced text preprocessing (e.g., remove stop words, stemming).

## Conclusion:
The Naive Bayes model provides a baseline for sentiment analysis but struggles with imbalanced data. Further improvements can enhance its performance, especially on the minority class (`4`).