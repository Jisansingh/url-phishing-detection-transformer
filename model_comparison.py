"""
Model Comparison: Logistic Regression Baseline vs Transformer

This script generates a side-by-side comparison of both models
evaluated on the same test set.
"""

import pandas as pd

# Results from running both models on the same test set (1500 samples)

comparison_data = {
    'Metric': [
        'Accuracy',
        'Precision (Phishing)',
        'Recall (Phishing)', 
        'F1-Score (Phishing)',
        'ROC-AUC',
        'False Positives',
        'False Negatives',
        'Training Time'
    ],
    'Logistic Regression (TF-IDF n-grams)': [
        '98.80%',
        '100.00%',
        '97.60%',
        '98.79%',
        '0.9952',
        '0',
        '18',
        '~1.5 seconds'
    ],
    'Transformer (Character-level)': [
        '98.87%',
        '99.06%',
        '98.67%',
        '98.86%',
        '0.9965',
        '7',
        '10',
        '~2-3 minutes'
    ],
    'Winner': [
        'Transformer (+0.07%)',
        'Logistic Regression',
        'Transformer (+1.07%)',
        'Transformer (+0.07%)',
        'Transformer (+0.0013)',
        'Logistic Regression',
        'Transformer (-8 FN)',
        'Logistic Regression'
    ]
}

df = pd.DataFrame(comparison_data)

print("\n" + "="*80)
print("MODEL COMPARISON: BASELINE vs TRANSFORMER")
print("="*80)
print("\nTest Set: 1500 samples (750 Legitimate, 750 Phishing)")
print("-"*80)

# Print the table
print(df.to_string(index=False))

# Save to CSV
df.to_csv('model_comparison.csv', index=False)
print("\n" + "-"*80)
print("Comparison saved to 'model_comparison.csv'")

# Analysis
print("\n" + "="*80)
print("ANALYSIS & CONCLUSIONS")
print("="*80)

print("""
KEY FINDINGS:
─────────────
1. OVERALL PERFORMANCE: Both models achieve excellent accuracy (~99%), 
   demonstrating that phishing URL detection is feasible with either approach.

2. TRANSFORMER ADVANTAGE - RECALL:
   • The Transformer catches MORE phishing attacks (98.67% vs 97.60%)
   • 8 fewer False Negatives (10 vs 18) - critical for security
   • In phishing detection, missing an attack is worse than a false alarm

3. LOGISTIC REGRESSION ADVANTAGE - PRECISION:
   • Zero false positives (100% precision on phishing class)
   • Faster training (~100x faster)
   • Simpler to deploy and interpret

4. ROC-AUC: 
   • Transformer: 0.9965 vs Baseline: 0.9952
   • Marginal improvement, but consistent across thresholds

RECOMMENDATION:
───────────────
The Transformer model is preferred for production use because:
• Higher recall means fewer successful phishing attacks get through
• 8 fewer missed attacks per 1500 URLs = significant security improvement
• The 7 false positives are acceptable trade-off for catching more attacks

When to use Logistic Regression baseline:
• Resource-constrained environments
• When interpretability is required
• As a quick fallback or ensemble component
""")

print("="*80)
