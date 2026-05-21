import os
import pandas as pd
from os.path import join
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# fix the seed for reproducibility
np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))

feature_dirs = {
    'REECAP': os.path.join(script_dir, '../data/external_val/processed/REECAP_features'),
    'RETFound': os.path.join(script_dir, '../data/external_val/processed/RETFound_features')
}
phenotypes_path = os.path.join(script_dir, '../data/external_val/retina_age_40_70.csv')

# Pick which model's features to use.
for model in ['REECAP', 'RETFound']:
    basedir = feature_dirs[model]

    embeddings_path = join(basedir, 'embeddings.csv')

    embeddings_df = pd.read_csv(embeddings_path)
    phenotypes_df = pd.read_csv(phenotypes_path)

    # Merge embeddings with phenotypes on file paths
    data_df = pd.merge(embeddings_df, phenotypes_df, left_on='file_path', right_on='Path', how='inner')
    feature_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]

    alphas = np.logspace(-3, 3, 13)

    X = data_df[feature_cols].values
    y = data_df['Age'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    maes = []
    r2s = []
    best_alphas = []

    oof_pred = np.zeros_like(y, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas, cv=5))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        oof_pred[val_idx] = y_pred

        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        maes.append(mae)
        r2s.append(r2)
        best_alphas.append(pipeline.named_steps["ridge"].alpha_)

        print(f"Fold {fold}: MAE={mae:.3f}, R2={r2:.3f}, alpha={best_alphas[-1]}")

    print("\n===== Ridge regression (5-fold CV) =====")
    print(f"MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"R2:  {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    print("Selected alphas:", best_alphas)

    # Save per-fold metrics
    fold_metrics = pd.DataFrame({
        'Fold': np.arange(1, len(maes)+1),
        'MAE': maes,
        'R2': r2s,
        'Alpha': best_alphas
    })

    fold_metrics_path = join(basedir, 'age_prediction_fold_metrics.csv')
    fold_metrics.to_csv(fold_metrics_path, index=False)
    print(f'Per-fold metrics saved to {fold_metrics_path}')

    # Visualize true vs predicted age (OOF predictions)
    y_real = y
    y_pred = oof_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_pred, alpha=0.4)
    plt.plot([y_real.min(), y_real.max()],
            [y_real.min(), y_real.max()],
            'r--', label='Ideal')

    z = np.polyfit(y_real, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_real, p(y_real), "b-", label='Fit')

    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title('True vs Predicted Age (5-fold OOF)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(basedir, 'age_prediction_scatter.png'))
    plt.close()
    print('Age prediction scatter plot saved.')

    with open(join(basedir, 'age_prediction_metrics.txt'), 'w') as f:
        f.write(f'MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}\n')
        f.write(f'R2:  {np.mean(r2s):.3f} ± {np.std(r2s):.3f}\n')
    print('Age prediction metrics saved.')