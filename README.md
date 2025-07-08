# Land Cover Classification - Summer Analytics'25

This project focuses on classifying land cover types using NDVI time-series data 
for a Kaggle hackathon under the Summer Analytics'25 program.

We designed a robust preprocessing pipeline involving:
- Linear interpolation and forward/backward filling to handle missing values
- Savitzky-Golay filtering to smooth noisy NDVI time series
- FFT-based frequency feature extraction to capture seasonal patterns
- Statistical feature engineering (mean, std, min, max, median) for signal summarization

These steps ensured high-quality features for our logistic regression model.
