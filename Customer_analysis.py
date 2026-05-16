import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

possible_files = [
    "sales_data.csv",
    "sales_data (2).csv",
    os.path.join("data", "sales_data.csv"),
    os.path.join("data", "sales_data (2).csv"),
]

sales_file = next((f for f in possible_files if os.path.exists(f)), None)
if sales_file is None:
    raise FileNotFoundError(
        "Could not find sales data file. Please add sales_data.csv or sales_data (2).csv to the project folder."
    )

print(f"Loading data from: {sales_file}")
df = pd.read_csv(sales_file)

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# Normalize date and customer columns for this dataset
if 'Order_Date' in df.columns:
    date_col = 'Order_Date'
elif 'Date' in df.columns:
    date_col = 'Date'
else:
    raise KeyError("No date column found. Expected 'Order_Date' or 'Date'.")

if 'Customer_Name' in df.columns:
    customer_col = 'Customer_Name'
elif 'Customer_ID' in df.columns:
    customer_col = 'Customer_ID'
else:
    raise KeyError("No customer column found. Expected 'Customer_Name' or 'Customer_ID'.")

if 'Category' in df.columns:
    category_col = 'Category'
elif 'Product' in df.columns:
    category_col = 'Product'
else:
    raise KeyError("No category/product column found. Expected 'Category' or 'Product'.")

# Parse dates and add time features
if not np.issubdtype(df[date_col].dtype, np.datetime64):
    df[date_col] = pd.to_datetime(df[date_col])

df['Month'] = df[date_col].dt.month
df['Year'] = df[date_col].dt.year
print("\nSummary Statistics:")
print(df.describe())

top_customers = df.groupby(customer_col)['Total_Sales'].sum() \
    .sort_values(ascending=False).head(10)

print("\nTop 10 Customers:")
print(top_customers)
plt.figure(figsize=(10,5))
top_customers.plot(kind='bar')
plt.title("Top 10 Customers by Sales")
plt.xlabel(customer_col)
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_customers.png")
plt.close()
monthly_sales = df.groupby('Month')['Total_Sales'].sum()

print("\nMonthly Sales:")
print(monthly_sales)
plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.savefig("monthly_sales_trend.png")
plt.close()
category_sales = df.groupby(category_col)['Total_Sales'].sum()

print("\nCategory-wise Sales:")
print(category_sales)
plt.figure(figsize=(8,8))
category_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales by " + category_col)
plt.ylabel("")
plt.savefig("category_sales.png")
plt.close()
top_products = df.groupby('Product')['Quantity'].sum() \
                 .sort_values(ascending=False).head(10)

print("\nTop Selling Products:")
print(top_products)
plt.figure(figsize=(10,5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("top_products.png")
plt.close()
regional_sales = df.groupby('Region')['Total_Sales'].sum()

print("\nRegional Sales:")
print(regional_sales)
plt.figure(figsize=(8,5))
sns.barplot(x=regional_sales.index, y=regional_sales.values)
plt.title("Regional Sales Analysis")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("regional_sales.png")
plt.close()
pivot_table = pd.pivot_table(
    df,
    values='Total_Sales',
    index='Region',
    columns=category_col,
    aggfunc='sum',
    fill_value=0
)

print("\nPivot Table:")
print(pivot_table)

# Heatmap Visualization
plt.figure(figsize=(10,6))
sns.heatmap(pivot_table, annot=True, cmap='Blues')
plt.title(f"Region vs {category_col} Sales Heatmap")
plt.savefig("sales_heatmap.png")
plt.close()
purchase_frequency = df[customer_col].value_counts().head(10)
print("\nCustomer Purchase Frequency:")
print(purchase_frequency)
plt.figure(figsize=(10,5))
sns.barplot(x=purchase_frequency.values,
            y=purchase_frequency.index)

plt.title("Top Repeat Customers")
plt.xlabel("Number of Purchases")
plt.ylabel(customer_col)
plt.tight_layout()
plt.savefig("repeat_customers.png")
plt.close()
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()
highest_sales_category = category_sales.idxmax()
highest_sales_region = regional_sales.idxmax()
best_customer = top_customers.idxmax()

print("\n========== BUSINESS INSIGHTS ==========")

print(f"Highest Sales Category: {highest_sales_category}")
print(f"Top Performing Region: {highest_sales_region}")
print(f"Best Customer: {best_customer}")

print("\nProject Analysis Completed Successfully!")
