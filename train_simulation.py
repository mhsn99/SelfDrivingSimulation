from utlis import *
from sklearn.model_selection import train_test_split

# Step 1: Initialize Data
path = "simulation_data"
data = importDataInfo(path)

# Step 2: Visualization Data
data = balanceData(data, display=False)

# Step 3: Prepare for Processing
imagesPath, steering= loadData(path, data)
#print(imagesPath[0], steering[0])

# Step 4: Split data train and validation
X_train, X_val, y_train, y_val = train_test_split(imagesPath, steering, test_size=0.2, random_state=42)
print("Total training images: ", len(X_train))
print("Total validation images: ", len(X_val))
