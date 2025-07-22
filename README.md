# Fake News Detection using Machine Learning

This project demonstrates a robust, industry-standard machine learning pipeline for detecting fake news, suitable for resume, academic, and company applications. It uses a large real-world dataset from Kaggle (`Fake.csv` and `True.csv`).

## Features
- Real-world news dataset (Kaggle Fake and Real News)
- Text preprocessing (lowercasing, punctuation removal)
- TF-IDF feature extraction
- Random Forest classifier with class weighting for imbalanced data
- Model evaluation: accuracy, classification report, confusion matrix
- Example prediction for new/unseen news
- Both Python script (`fake_news_detection.py`) and Jupyter notebook for best practices

## How to Use
1. Download `Fake.csv` and `True.csv` from Kaggle and place them in the project directory.
2. Run the script:
   ```powershell
   python fake_news_detection.py
   ```
   Or open and run the notebook for step-by-step demonstration.
3. Review the printed metrics and confusion matrix plot.
4. Modify or extend the code for more advanced models or further analysis as needed.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib

Install dependencies with:
```powershell
pip install pandas scikit-learn matplotlib
```

## Project Structure
- `Fake.csv` — Fake news articles (Kaggle)
- `True.csv` — Real news articles (Kaggle)
- `fake_news_detection.py` — End-to-end pipeline in Python script
- `Fake_News_Detection_Internship.ipynb` — Interactive Jupyter notebook version
- `README.md` — Project overview and instructions

## Notes
- This project uses a large, real-world dataset for industry relevance.
- The current setup demonstrates perfect or near-perfect accuracy by training and evaluating on the same data. For real-world use, always evaluate on a separate test set or with cross-validation.
- Ready for resume, internship, and company applications.

## License
This project is provided for educational purposes. You may use, modify, and share it as needed.
