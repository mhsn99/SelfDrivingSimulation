import matplotlib.pyplot as plt

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

# Step 5: Data Augmentation
# Step 6: Preprocessing
# Step 7:
# Step 8:

model = createModel()
model.summary()

# Step 9:
history = model.fit(batchGen(X_train,y_train,100,1), steps_per_epoch=100, epochs=10,
          validation_data=batchGen(X_val, y_val, 100, 0), validation_steps=200)

model.save("model.h5")
print("Model Saved")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training", "Validation"])
plt.ylim([0,1])
plt.title("Loss")
plt.xlabel("Epoch")
plt.show()