import pandas as pd
import numpy as np
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data(csv_file, label_col='label'):
    df = pd.read_csv(csv_file)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return df, X, y

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_cnn_cv(X, y, selected_features, epochs=10, batch_size=32, cv_folds=5):
    X_sub = X[selected_features].values
    scaler = StandardScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    X_sub_scaled = np.expand_dims(X_sub_scaled, axis=2)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, val_idx in cv.split(X_sub_scaled, y):
        model = build_cnn_model(input_shape=(X_sub_scaled.shape[1], 1))
        X_train, X_val = X_sub_scaled[train_idx], X_sub_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        accuracies.append(acc)

    return np.mean(accuracies)

def cnn_rfe_shap_optimized(csv_file, label_col='label', epochs=10, batch_size=32,
                           cv_folds=5, step=1, min_features=5, output_prefix='cnn_rfe_shap_opt'):
    df, X, y = load_data(csv_file, label_col)
    selected_features = list(X.columns)
    feature_elimination_log = []

    best_accuracy = 0
    best_features = selected_features.copy()
    best_shap_values = None

    print(f"Starting CNN-RFE with SHAP-based feature ranking ({len(selected_features)} features)")

    while len(selected_features) > min_features:
        print(f"\nEvaluating with {len(selected_features)} features: {selected_features}")

        # Evaluate CV accuracy on current feature subset
        cv_acc = evaluate_cnn_cv(X, y, selected_features, epochs=epochs, batch_size=batch_size, cv_folds=cv_folds)
        print(f"CV Accuracy with {len(selected_features)} features: {cv_acc:.4f}")

        # Prepare data scaled for SHAP
        X_sub = X[selected_features].values
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_sub_scaled_exp = np.expand_dims(X_sub_scaled, axis=2)

        # Train CNN on full dataset for SHAP explanation
        model = build_cnn_model(input_shape=(len(selected_features), 1))
        model.fit(X_sub_scaled_exp, y, epochs=epochs, batch_size=batch_size, verbose=0)

        # Prediction function for SHAP
        def model_predict(data):
            data_exp = np.expand_dims(data, axis=2)
            return model.predict(data_exp).flatten()

        # Compute SHAP values on (a sample of) dataset for efficiency
        background = shap.sample(X_sub_scaled, min(100, X_sub_scaled.shape[0]))
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_sub_scaled, nsamples=200)
        shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

        # Track best feature set and stop if accuracy decreases
        if cv_acc >= best_accuracy:
            best_accuracy = cv_acc
            best_features = selected_features.copy()
            best_shap_values = shap_values_abs_mean.copy()
            print(f"New best accuracy: {best_accuracy:.4f} with {len(best_features)} features")
        else:
            # Accuracy dropped: stop further removal
            print(f"Accuracy decreased from {best_accuracy:.4f} to {cv_acc:.4f} - stopping feature elimination")
            break

        # Rank features by SHAP importance ascending (lowest importance first)
        feature_shap_dict = dict(zip(selected_features, shap_values_abs_mean))
        sorted_features = sorted(feature_shap_dict.items(), key=lambda x: x[1])

        # Remove 'step' least important features
        features_to_remove = [f[0] for f in sorted_features[:step]]
        print(f"Removing feature(s) with lowest SHAP importance: {features_to_remove}")

        for ftr in features_to_remove:
            selected_features.remove(ftr)
            feature_elimination_log.append({
                'Feature_Removed': ftr,
                'CV_Accuracy': cv_acc,
                'Remaining_Features': len(selected_features)
            })

        if len(selected_features) <= min_features:
            print(f"Reached minimum feature count ({min_features}). Stopping RFE.")
            break

    print(f"\nSelected best features ({len(best_features)}) with accuracy {best_accuracy:.4f}: {best_features}")

    # Save elimination log CSV
    log_df = pd.DataFrame(feature_elimination_log)
    log_df.to_csv(f"{output_prefix}_feature_elimination_log.csv", index=False)
    print(f"Saved feature elimination log to {output_prefix}_feature_elimination_log.csv")

    # Save best selected features as CSV with ranks by SHAP importance descending
    shap_features_df = pd.DataFrame({
        'Feature': best_features,
        'Mean_Abs_SHAP_Value': best_shap_values
    }).sort_values(by='Mean_Abs_SHAP_Value', ascending=False)
    shap_features_df['Rank'] = range(1, len(shap_features_df) + 1)
    shap_features_df.to_csv(f"{output_prefix}_best_features_ranking.csv", index=False)
    print(f"Saved best features ranking to {output_prefix}_best_features_ranking.csv")

    # Save reduced dataset with best features + label
    df_selected = df[best_features + [label_col]]
    df_selected.to_csv(f"{output_prefix}_selected_features.csv", index=False)
    print(f"Saved best selected features dataset to {output_prefix}_selected_features.csv")

    # Plot SHAP summary bar plot for best features
    plt.figure(figsize=(10, 6))
    plt.barh(shap_features_df['Feature'], shap_features_df['Mean_Abs_SHAP_Value'])
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Feature Importance for Best Feature Set")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_shap_feature_importance.png")
    plt.show()

    return best_features, best_accuracy, shap_features_df, log_df

# -----------------
# Example usage:
best_feats, best_acc, shap_rank_df, elim_log_df = cnn_rfe_shap_optimized(
    csv_file='FastText_train_CAB.csv',
    label_col='label',
    epochs=10,
    batch_size=500,
    cv_folds=3,
    step=1,
    min_features=50,
    output_prefix='cnn_rfe_shap_opt'
)

print("\nFinal selected features:")
for i, f in enumerate(best_feats, 1):
    print(f"{i}. {f}")
