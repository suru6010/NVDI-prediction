import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from scipy.signal import savgol_filter

# Load datasets
train_df = pd.read_csv("hacktrain.csv")
test_df = pd.read_csv("hacktest.csv")

# Drop unnecessary columns
train_df.drop(columns=[col for col in ['Unnamed: 0', 'ID'] if col in train_df.columns], inplace=True)
test_ids = test_df['ID']
test_df.drop(columns=['ID'], inplace=True)

# Identify NDVI columns (columns starting with year and ending with '_N')
ndvi_cols = [col for col in train_df.columns if col.startswith('20') and col.endswith('_N')]

# Preprocessing function
def preprocess(df, ndvi_cols):
    df = df.copy()

    # Fill missing values and apply smoothing
    for col in ndvi_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].ffill().bfill()
        df[col] = savgol_filter(df[col], window_length=5, polyorder=2, mode='nearest')

    # FFT features
    def fft_features(row):
        series = row[ndvi_cols].values.astype(float)
        fft_vals = np.fft.fft(series)
        fft_mag = np.abs(fft_vals[:10])
        return pd.Series(fft_mag, index=[f'fft_mag_{i}' for i in range(10)])

    fft_df = df.apply(fft_features, axis=1)
    df = pd.concat([df, fft_df], axis=1)

    # Statistical features
    df['ndvi_mean'] = df[ndvi_cols].mean(axis=1)
    df['ndvi_std'] = df[ndvi_cols].std(axis=1)
    df['ndvi_min'] = df[ndvi_cols].min(axis=1)
    df['ndvi_max'] = df[ndvi_cols].max(axis=1)
    df['ndvi_median'] = df[ndvi_cols].median(axis=1)

    return df

# Apply preprocessing to train and test data
train_processed = preprocess(train_df, ndvi_cols)
test_processed = preprocess(test_df, ndvi_cols)

# Encode class labels
label_encoder = LabelEncoder()
train_processed['class'] = label_encoder.fit_transform(train_processed['class'])

# Separate features and target
X = train_processed.drop(columns=['class'])
y = train_processed['class']
feature_cols = X.columns.tolist()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_processed[feature_cols])

# Split training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Train logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=5.0)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# Predict on test data
test_preds = model.predict(test_scaled)
test_labels = label_encoder.inverse_transform(test_preds)

# Create submission file
submission = pd.DataFrame({
    "ID": test_ids,
    "class": test_labels
})
submission.to_csv("submission_optimized.csv", index=False)

