import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\rahul\Downloads\ecommerce_furniture_dataset_2024.csv")

# Check first 5 rows
print(df.head())

# Check shape
print(df.shape)

# Check data types and missing values
print(df.info())

# Check missing values count
print(df.isnull().sum())

# Analyze 'tagText' column
print(df['tagText'].value_counts(dropna=False))

#clean 'tagText' column by filling NaN with 'others'
# Fill missing values first
df['tagText'] = df['tagText'].fillna('others')

# Keep only important categories
df['tagText'] = df['tagText'].apply(
    lambda x: x if x in ['Free shipping', '+Shipping: $5.09'] else 'others'
)

# Check result
print(df['tagText'].value_counts())


# Remove $ and commas, then convert to float
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Verify
print(df['price'].head())
print(df['price'].dtype)

#droping original price column
df.drop('originalPrice', axis=1, inplace=True)

print(df.info())

# Analyze price distribution
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Distribution of Product Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Analyzer sales (sold) distribution
plt.figure(figsize=(8,5))
sns.histplot(df['sold'], bins=30, kde=True)
plt.title("Distribution of Units Sold")
plt.xlabel("Units Sold")
plt.ylabel("Count")
plt.show()

#price vs sales relationship
plt.figure(figsize=(8,5))
sns.scatterplot(x='price', y='sold', data=df)
plt.title("Price vs Units Sold")
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.show()

#shipping influence on sales
plt.figure(figsize=(8,5))
sns.boxplot(x='tagText', y='sold', data=df)
plt.title("Impact of Shipping Type on Units Sold")
plt.xlabel("Shipping Type")
plt.ylabel("Units Sold")
plt.show()

# Create revenue column
df['revenue'] = df['price'] * df['sold']

# Check first few rows
print(df.head())

# Total Revenue
total_revenue = df['revenue'].sum()

# Average price
avg_price = df['price'].mean()

# Average units sold
avg_sold = df['sold'].mean()

print("Total Revenue:", total_revenue)
print("Average Price:", avg_price)
print("Average Units Sold:", avg_sold)

#top 10 selling products by revenue
top_sellers = df.sort_values(by='sold', ascending=False)[
    ['productTitle', 'price', 'sold', 'revenue']
].head(10)

print(top_sellers)

#revenue by shipping type
revenue_by_shipping = df.groupby('tagText')['revenue'].sum().sort_values(ascending=False)
print(revenue_by_shipping)

plt.figure(figsize=(8,5))
revenue_by_shipping.plot(kind='bar')
plt.title("Revenue by Shipping Type")
plt.xlabel("Shipping Type")
plt.ylabel("Total Revenue")
plt.show()

# machine learning- encoding 'tagText' column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['tagText_encoded'] = le.fit_transform(df['tagText'])

print(df[['tagText', 'tagText_encoded']].head())

#convert productTitle to numeric using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Limit features to avoid too many columns
tfidf = TfidfVectorizer(max_features=100)

title_tfidf = tfidf.fit_transform(df['productTitle'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(
    title_tfidf.toarray(),
    columns=tfidf.get_feature_names_out()
)

print(title_tfidf_df.head())
print("Number of TF-IDF columns:", title_tfidf_df.shape[1])

# Combine TF-IDF features with original DataFrame# Merge TF-IDF features with main dataframe
df_ml = pd.concat([df, title_tfidf_df], axis=1)

# Drop original text column
df_ml.drop(['productTitle', 'tagText'], axis=1, inplace=True)

print(df_ml.shape)
print(df_ml.head())
print(df_ml.dtypes.value_counts())

#trian-test split
from sklearn.model_selection import train_test_split

# Features and target
X = df_ml.drop('sold', axis=1)
y = df_ml['sold']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train a simple linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize model
lr_model = LinearRegression()

# Train model
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

# Train a Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Initialize model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results")
print("MSE:", mse_rf)
print("R2 Score:", r2_rf)