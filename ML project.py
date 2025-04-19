import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

# Define dataset paths
train_data_path = 'C:/Users/User/Downloads/Nike_Adidas_converse_Shoes_image_dataset/Nike_Adidas_converse_Shoes_image_dataset/train'
validation_data_path = 'C:/Users/User/Downloads/Nike_Adidas_converse_Shoes_image_dataset/Nike_Adidas_converse_Shoes_image_dataset/validate'


# Data preprocessing with enhanced augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30, 
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.3, 
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_data_path, 
                                               target_size=(240, 240), 
                                               batch_size=32, 
                                               class_mode='categorical')

validation_data = validation_datagen.flow_from_directory(validation_data_path, 
                                                         target_size=(240, 240), 
                                                         batch_size=32, 
                                                         class_mode='categorical')

# Load pre-trained VGG16 model
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(240, 240, 3))

# Fine-tuning: Unfreeze the last few layers
for layer in vgg_base.layers[:-4]:
    layer.trainable = False

# Build the new model with Batch Normalization
model = Sequential([
    vgg_base,
    Flatten(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(3, activation='softmax')  # 3 classes: Nike, Adidas, Converse
])

# Compile the model with a custom learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for better training performance
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(train_data, 
                    epochs=15, 
                    validation_data=validation_data, 
                    callbacks=[reduce_lr, early_stopping])

# Save the model
print("Current Working Directory:", os.getcwd())
model.save('optimized_shoe_classification_model.h5', save_format='h5')

# Test the model
test_loss, test_accuracy = model.evaluate(validation_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
