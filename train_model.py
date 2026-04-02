import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# --- 1. Load Data ---
print("Loading data...")
fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')

# Assign labels: 0 for Fake News, 1 for Real News
fake_df['label'] = 0  
real_df['label'] = 1  

# Combine and shuffle the data
df = pd.concat([fake_df, real_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# --- 2. Preprocess Text ---
print("Preprocessing text...")
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Apply the cleaning function to the text column
df['text'] = df['text'].apply(clean_text)

x = df['text']
y = df['label']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# --- 3. Vectorization ---
print("Vectorizing text (TF-IDF)...")
vectorization = TfidfVectorizer(max_features=5000)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# --- 4. Train Models ---
print("Training Logistic Regression...")
LR_model = LogisticRegression()
LR_model.fit(xv_train, y_train)
pred_lr = LR_model.predict(xv_test)

print("Training Naive Bayes...")
NB_model = MultinomialNB()
NB_model.fit(xv_train, y_train)
pred_nb = NB_model.predict(xv_test)

print("Training Random Forest (this one takes a little longer)...")
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
RF_model.fit(xv_train, y_train)
pred_rf = RF_model.predict(xv_test)

# --- 5. Save Visualizations ---
print("Generating and saving all graphs...")

# A. Bar Chart
models = ['Log Regression', 'Naive Bayes', 'Random Forest']
accuracies = [accuracy_score(y_test, pred_lr), accuracy_score(y_test, pred_nb), accuracy_score(y_test, pred_rf)]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylim(0.8, 1.0)
plt.savefig('static/bar_chart.png', bbox_inches='tight')
plt.close()

# B. ROC Curve
prob_lr = LR_model.predict_proba(xv_test)[:, 1]
prob_nb = NB_model.predict_proba(xv_test)[:, 1]
prob_rf = RF_model.predict_proba(xv_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_test, prob_nb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Log Reg (AUC = {auc(fpr_lr, tpr_lr):.3f})')
plt.plot(fpr_nb, tpr_nb, color='green', lw=2, label=f'Naive Bayes (AUC = {auc(fpr_nb, tpr_nb):.3f})')
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Mistakes)')
plt.ylabel('True Positive Rate (Correct Guesses)')
plt.title('ROC Curve: Comparing All Models')
plt.legend(loc="lower right")
plt.savefig('static/roc_curve.png', bbox_inches='tight')
plt.close()

# C. Confusion Matrices
def save_conf_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'static/{filename}', bbox_inches='tight')
    plt.close()

save_conf_matrix(y_test, pred_lr, 'Logistic Regression Matrix', 'conf_matrix_lr.png')
save_conf_matrix(y_test, pred_nb, 'Naive Bayes Matrix', 'conf_matrix_nb.png')
save_conf_matrix(y_test, pred_rf, 'Random Forest Matrix', 'conf_matrix_rf.png')

# --- 6. Save AI for Web App ---
print("Saving models and vectorizer for the live website...")
joblib.dump(RF_model, 'model.pkl') 
joblib.dump(vectorization, 'vectorizer.pkl')

print("All done! Graphs and models are successfully saved.")