# ============================================
# Customer Segmentation Analysis using K-Means
# ============================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load the Given Dataset
data = pd.read_csv("Mall_Customers.csv")

# 3. Display Dataset Information
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# 4. Select Relevant Features
# Using Annual Income and Spending Score for segmentation
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Elbow Method to Find Optimal Number of Clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 7. Apply K-Means Clustering
# Optimal clusters chosen as 5
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 8. Visualize Customer Segments
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='Set1'
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# 9. Cluster-wise Descriptive Statistics
print("\nCluster-wise Analysis:")
print(data.groupby('Cluster').mean())

# 10. Save Final Output
data.to_csv("Customer_Segmentation_Output.csv", index=False)

print("\nCustomer Segmentation completed successfully!")
print("Output file saved as 'Customer_Segmentation_Output.csv'")
