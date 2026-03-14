import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the training data
print("📊 Loading training data...")
df = pd.read_csv('train_data.csv')

# 2. Count the number of samples in each class
class_counts = df['AQI_Class'].value_counts()

# 3. Print the results to the terminal
print("\n" + "="*45)
print("📈 TRAINING DATA CLASS DISTRIBUTION")
print("="*45)
print(class_counts)
print("="*45 + "\n")

# 4. Create a quick bar chart
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Number of Samples per AQI Class (Training Data)')
plt.xlabel('AQI Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save and show the plot
plt.savefig('class_distribution.png')
print("✅ Saved bar chart as 'class_distribution.png'")
plt.show()