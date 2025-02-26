import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models


def create_resnet_model(input_shape):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    backbone = ResNet50(
        include_top=False, input_shape=input_shape, weights="imagenet"
    )
    backbone.trainable = True
    for layer in backbone.layers[:-10]:
        layer.trainable = False

    neck = models.Sequential(
        [
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1),
        ]
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = backbone(x)
    outputs = neck(x)
    model = tf.keras.Model(inputs, outputs)

    return model
