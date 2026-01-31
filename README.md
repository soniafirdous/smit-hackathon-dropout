# Student Dropout Prediction App
📌 Overview

This app predicts which students are at risk of dropping out based on academic activity and personal information. It helps university advisors identify high-risk students early so they can provide timely support.

Built with Python, Streamlit, and XGBoost.

Displays top high-risk students, individual risk scores, and feature importance.

🚀 Features

Upload CSV student data

Top High-Risk Students table (top 20)

Individual Student Analysis: See risk score, risk label, and top contributing factors

Feature Importance: Understand which features most affect dropout risk

🛠 Installation

Clone the repository

git clone <your-repo-url>
cd <project-folder>


Create and activate a virtual environment

python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Make sure xgboost, scikit-learn, pandas, numpy, and streamlit are installed.
