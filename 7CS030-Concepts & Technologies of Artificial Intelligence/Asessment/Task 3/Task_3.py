#!/usr/bin/env python
# coding: utf-8

# ### Task 3: Classification – NBA Rookie Career Longevity Prediction
# **Summary of Code**
# 
# - This script demonstrates multi-algorithm classification using real NBA rookie performance data from a CSV file named nba_rookie_data.csv.
# - The objective of this task is to predict whether NBA rookies will sustain professional careers lasting five years or longer based on their first-season performance metrics.
# - We implement and compare three distinct machine learning approaches: Logistic Regression, Gaussian Naive Bayes, and Neural Networks with varying architectures, providing comprehensive performance evaluation and decision boundary visualization.

# In[ ]:





# In[3]:


# task3_complete_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# In[ ]:





# In[4]:


# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# In[ ]:





# In[5]:


print("NBA ROOKIE CAREER LONGEVITY PREDICTION ")

# Load and explore data
df = pd.read_csv('nba_rookie_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['TARGET_5Yrs'].value_counts()}")


# In[ ]:





# In[6]:


# Data preprocessing
X = df.drop(['Name', 'TARGET_5Yrs'], axis=1)
y = df['TARGET_5Yrs']

# Handle any missing values
X = X.fillna(X.mean())


# In[ ]:





# In[7]:


# Feature scaling for neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# In[ ]:





# In[8]:


# Initialize and train models
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Gaussian Naive Bayes': GaussianNB(),
    'Neural Network (Simple)': MLPClassifier(hidden_layer_sizes=(), activation='logistic', random_state=42, max_iter=2000),
    'Neural Network (Complex)': MLPClassifier(hidden_layer_sizes=(20, 40, 20), activation='tanh', random_state=42, max_iter=2000)
}


# In[ ]:





# In[9]:


predictions = {}
performance_data = []

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    accuracy = accuracy_score(y_test, y_pred)
    mislabeled = (y_test != y_pred).sum()
    report = classification_report(y_test, y_pred, output_dict=True)
    
    performance_data.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1-Score': report['1']['f1-score'],
        'Mislabeled Points': mislabeled
    })
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Misclassified: {mislabeled}/{len(y_test)}")


# In[ ]:





# In[10]:


# Create performance comparison table
performance_df = pd.DataFrame(performance_data)
print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)
print(performance_df.round(3))


# In[ ]:





# In[12]:


# Feature importance analysis
logreg = models['Logistic Regression']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'logreg_coef': logreg.coef_[0],
    'correlation': [np.corrcoef(X.iloc[:, i], y)[0, 1] for i in range(X.shape[1])]
})
feature_importance['abs_coef'] = np.abs(feature_importance['logreg_coef'])
feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)
print(feature_importance.head(10)[['feature', 'logreg_coef', 'correlation']].round(3))


# In[ ]:





# In[27]:


# VISUALIZATION 1: Confusion Matrices
print("\nGenerating Confusion Matrices...")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                cbar=False, annot_kws={"size": 14})
    axes[idx].set_title(f'{name}\nAccuracy: {performance_df.iloc[idx]["Accuracy"]:.3f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    axes[idx].set_ylabel('True Label', fontsize=10)

plt.suptitle('Confusion Matrices: Model Performance Comparison \n\n', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
#plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[ ]:





# In[21]:


# VISUALIZATION 2: Feature Importance
print("Generating Feature Importance Plot...")
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)

plt.subplot(1, 2, 1)
# Absolute coefficients
plt.barh(range(len(top_features)), top_features['abs_coef'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance (Logistic Regression)\nAbsolute Coefficients', fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
# Correlation with target
plt.barh(range(len(top_features)), top_features['correlation'].abs())
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Absolute Correlation with Target')
plt.title('Feature Importance\nCorrelation with Career Longevity', fontweight='bold')
plt.gca().invert_yaxis()

plt.tight_layout()
#plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[ ]:





# In[22]:


# VISUALIZATION 3: Performance Comparison
print("Generating Performance Comparison Chart...")
plt.figure(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
model_names = list(predictions.keys())
performance_values = performance_df[metrics].values

x_pos = np.arange(len(metrics))
width = 0.2

for i, model_name in enumerate(model_names):
    plt.bar(x_pos + i*width, performance_values[i], width, label=model_name, alpha=0.8)

plt.xlabel('Performance Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison Across Metrics', fontsize=14, fontweight='bold')
plt.xticks(x_pos + width*1.5, metrics)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[ ]:





# In[23]:


# VISUALIZATION 4: Decision Boundaries
print("Generating Decision Boundary Visualization...")

def plot_decision_boundary(model, X, y, feature_names, title, ax):
    """Plot decision boundary for 2D feature space"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                        edgecolor='black', s=30, alpha=0.7)
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title, fontweight='bold')
    
    return scatter


# In[ ]:





# In[25]:


# Select top 2 features for visualization
top_feature_names = feature_importance.head(2)['feature'].tolist()
feature_indices = [list(X.columns).index(feat) for feat in top_feature_names]

X_2d = X_scaled[:, feature_indices]
X_train_2d = X_train[:, feature_indices]

# Retrain models on 2D data
models_2d = {}
for name, model in models.items():
    if 'Neural Network' in name:
        # Use same architecture but retrain on 2D data
        new_model = type(model)(**model.get_params())
    else:
        new_model = type(model)(**model.get_params())
    
    new_model.fit(X_train_2d, y_train)
    models_2d[name] = new_model

# Create decision boundary plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(models_2d.items()):
    accuracy_2d = accuracy_score(y, model.predict(X_2d))
    scatter = plot_decision_boundary(model, X_2d, y, top_feature_names,
                                   f'{name}\n(2D Accuracy: {accuracy_2d:.3f})', 
                                   axes[idx])

plt.suptitle('Decision Boundary Analysis: Games Played vs 3-Point Shooting\n\n', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
#plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[ ]:





# In[26]:


# Final summary
print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("Generated Visualizations:")
#print("1. confusion_matrices.png - Model performance details")
#print("2. feature_importance_analysis.png - Key predictors")
#print("3. performance_comparison.png - Metric comparison") 
#print("4. decision_boundaries.png - Classification boundaries")
print("\nKey Findings:")
print(f"- Best Model: {performance_df.loc[performance_df['Accuracy'].idxmax(), 'Model']}")
print(f"- Highest Accuracy: {performance_df['Accuracy'].max():.3f}")
print(f"- Most Important Feature: {feature_importance.iloc[0]['feature']}")
print(f"- Dataset Size: {df.shape[0]} players, {df.shape[1]-1} features")


# In[ ]:




