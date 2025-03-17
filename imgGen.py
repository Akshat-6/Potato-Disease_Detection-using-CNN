#!/usr/bin/env python
# coding: utf-8

# # Potato Disease Classification

# Dataset credits: https://www.kaggle.com/arjuntejaswi/plant-village

# ### Import all the Dependencies

# In[1]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML


# ### Import data into tensorflow dataset object

# Used splitfolders tool to split dataset into training, validation and test directories.
# 
# $ pip install split-folders
# 
# $ splitfolders --ratio 0.8 0.1 0.1 -- ./training/PlantVillage/
# 

# In[2]:


IMAGE_SIZE = 256
CHANNELS = 3


# In[3]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your image size
IMAGE_SIZE = 224  # Set this to your required size (e.g., 224 for many pre-trained models)

# Create an instance of ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=10,  # Randomly rotate images in the range (0 to 10 degrees)
    horizontal_flip=True  # Randomly flip images horizontally
)

# Create a data generator for training images
train_generator = train_datagen.flow_from_directory(
    r'D:\Study\plant disease project\Training\dataset\train',  # Use raw string to handle backslashes
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to this size
    batch_size=32,  # Set the batch size
    class_mode='sparse',  # Use 'sparse' for integer labels (suitable for multi-class classification)
    shuffle=True  # Shuffle the data to ensure randomness
    # save_to_dir=r"C:\Code\potato-disease-classification\training\AugmentedImages"  # Uncomment if you want to save augmented images
)

# Optionally, you can also print the class indices to verify them
print("Class indices:", train_generator.class_indices)


# In[4]:


train_generator.class_indices


# In[5]:


class_names = list(train_generator.class_indices.keys())
class_names


# In[6]:


count=0
for image_batch, label_batch in train_generator:
#     print(label_batch)
    print(image_batch[0])
    break
#     count+=1
#     if count>2:
#         break


# In[7]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your image size
IMAGE_SIZE = 224  # Adjust this size as per your model's requirements

# Create an instance of ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=10,  # (Optional) Randomly rotate images in the range (0 to 10 degrees)
    horizontal_flip=True  # (Optional) Randomly flip images horizontally
)

# Create a data generator for validation images
validation_generator = validation_datagen.flow_from_directory(
    r'D:\Study\plant disease project\Training\dataset\val',  # Use raw string for Windows paths
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to this size
    batch_size=32,  # Set the batch size
    class_mode='sparse',  # Use 'sparse' for integer labels
    shuffle=False  # Do not shuffle validation data
)

# Optionally, print the class indices to verify them
print("Validation class indices:", validation_generator.class_indices)


# In[8]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your image size
IMAGE_SIZE = 224  # Set this to your model's required input size

# Create an instance of ImageDataGenerator for test data
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255  # Only normalize pixel values for testing
)

# Create a data generator for test images
test_generator = test_datagen.flow_from_directory(
    r'D:\Study\plant disease project\Training\dataset\test',  # Use raw string for Windows paths
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to this size
    batch_size=32,  # Set the batch size
    class_mode='sparse',  # Use 'sparse' for integer labels
    shuffle=False  # Do not shuffle test data
)

# Optionally, print the class indices to verify them
print("Test class indices:", test_generator.class_indices)


# In[9]:


for image_batch, label_batch in test_generator:
    print(image_batch[0])
    break


# ## Building the Model
import tensorflow as tf
from tensorflow.keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.Wq = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                  initializer='random_normal', trainable=True)
        self.Wk = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                  initializer='random_normal', trainable=True)
        self.Wv = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                  initializer='random_normal', trainable=True)

    def call(self, inputs):
        # Compute query, key, and value matrices
        Q = tf.matmul(inputs, self.Wq)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)

        # Compute attention scores
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.output_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Compute the attention output
        output = tf.matmul(attention_weights, V)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim

# Example usage of the SelfAttention layer
if __name__ == "__main__":
    # Create a sample input tensor
    sample_input = tf.random.normal((1, 10, 64))  # (batch_size, sequence_length, feature_dim)
    
    # Instantiate and call the SelfAttention layer
    self_attention_layer = SelfAttention(output_dim=32)
    output = self_attention_layer(sample_input)
    
    print("Output shape:", output.shape)  # Should be (1, 10, 32)


# In[10]:


input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])


# In[11]:


model.summary()


# ### Compiling the Model
# We use `adam` Optimizer, `SparseCategoricalCrossentropy` for losses, `accuracy` as a metric

# In[12]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[13]:


1506/32


# In[14]:


215/32


# In[15]:


history = model.fit(
    train_generator,
    steps_per_epoch=47,
    batch_size=32,
    validation_data=validation_generator,
    validation_steps=6,
    verbose=1,
    epochs=50,
)


# In[16]:


scores = model.evaluate(test_generator)


# In[17]:


scores


# Scores is just a list containing loss and accuracy value

# ### Plotting the Accuracy and Loss Curves

# In[18]:


history


# You can read documentation on history object here: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History

# In[19]:


history.params


# In[20]:


history.history.keys()


# **loss, accuracy, val loss etc are a python list containing values of loss, accuracy etc at the end of each epoch**

# In[21]:


type(history.history['loss'])


# In[22]:


len(history.history['loss'])


# In[23]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[24]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[25]:


val_acc


# In[26]:


acc


# In[27]:


EPOCHS = 50

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ### Run prediction on a sample image

# In[28]:


import numpy as np


for image_batch, label_batch in test_generator:
    first_image = image_batch[0]
    first_label = int(label_batch[0])
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(image_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])
    
    break


# ### Write a function for inference

# In[29]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# **Now run inference on few sample images**

# In[30]:


plt.figure(figsize=(15, 15))
for images, labels in test_generator:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        
        predicted_class, confidence = predict(model, images[i])
        actual_class = class_names[int(labels[i])] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
    break


# ### Saving the Model
# 
# Save model in h5 format so that there is just one file and we can upload that to GCP conveniently

# In[31]:


model.save(r"D:\Study\plant disease project\models\final1.keras")


# In[ ]:





# In[33]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Predict the classes
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)  # Get the predicted class indices

# 2. Get true labels
y_true = test_generator.classes  # True class labels from the generator

# 3. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# 4. Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 5. Classification Report (includes precision, recall, F1-score)
print('Classification Report:')
target_names = list(test_generator.class_indices.keys())  # Class labels
print(classification_report(y_true, y_pred, target_names=target_names))

# 6. Accuracy Score
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[ ]:




