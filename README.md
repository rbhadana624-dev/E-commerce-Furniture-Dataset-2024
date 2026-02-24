# 🪑 E-commerce Furniture Sales Analysis & Prediction

## 📌 Project Overview

This project analyzes 2,000 furniture product listings scraped from AliExpress to uncover key business drivers influencing sales performance. The study explores pricing strategies, shipping impact, and product characteristics, followed by predictive modeling to estimate units sold.

---

## 🎯 Objectives

- Perform data cleaning and preprocessing  
- Conduct exploratory data analysis (EDA)  
- Generate actionable business and financial insights  
- Build predictive models to estimate units sold  
- Compare baseline and advanced machine learning models  

---

## 🛠 Tools & Technologies

- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📂 Dataset Summary

- Total Records: **2,000**
- Features:
  - `productTitle`
  - `price`
  - `sold`
  - `tagText`
- Target Variable:
  - `sold` (Units Sold)

⚠️ `originalPrice` was removed due to 76% missing values to maintain data reliability.

---

## 🧹 Data Cleaning Process

- Converted `price` from string format to numeric  
- Handled missing values in `tagText`  
- Grouped rare shipping categories into `"others"`  
- Removed `originalPrice` due to excessive missing data  
- Encoded categorical features for modeling  
- Applied TF-IDF vectorization to product titles  

---

## 📊 Exploratory Data Analysis (EDA)

### 1️⃣ Price Distribution
- Right-skewed distribution  
- Majority of products fall within low-to-mid price range  

### 2️⃣ Sales Distribution
- Highly right-skewed  
- Small number of products dominate sales  

### 3️⃣ Price vs Units Sold
- Negative relationship observed  
- Lower-priced items generally sell more  
- Indicates strong customer price sensitivity  

### 4️⃣ Shipping Strategy Impact
- Free shipping significantly increases both sales volume and revenue  
- Shipping cost acts as a major conversion barrier  

---

## 💰 Key Business Metrics

- **Total Revenue:** $2,181,048.07  
- **Average Product Price:** $156.56  
- **Average Units Sold per Product:** 23.49  

---

## 🤖 Machine Learning Modeling

### Baseline Model — Linear Regression
- R² ≈ -0.01  
- Poor performance  
- Indicates sales relationships are not purely linear  

### Advanced Model — Random Forest Regressor
- R² ≈ 0.66  
- Strong predictive performance  
- Successfully captures non-linear relationships  

---

## 📈 Business Insights

- Customers are highly price-sensitive  
- Free shipping is a major revenue driver  
- Sales patterns are complex and non-linear  
- Tree-based models outperform linear models for this dataset  

---

## 🏁 Final Conclusion

This project demonstrates a complete end-to-end Data Analysis and Machine Learning workflow:

- Data Cleaning  
- Exploratory Data Analysis  
- Business Insight Generation  
- KPI Calculation  
- Predictive Modeling  
- Model Comparison  

The findings provide actionable recommendations for pricing optimization and shipping strategy improvements in the e-commerce furniture marketplace.

---

## 🚀 Portfolio Highlights

✔ Business-focused data storytelling  
✔ Clean and structured EDA  
✔ Revenue and KPI analysis  
✔ Feature engineering with TF-IDF  
✔ Model building and evaluation  
✔ Professional reporting  

---

## 📬 Contact

If you’d like to discuss this project or collaborate, feel free to connect.
