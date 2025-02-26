import tensorflow as tf

DATA_DIR = r"C:\Users\user\Projekty\FaceData_2"
BATCH_SIZE = 32
X_SHAPE = (224, 224, 3)
X_TYPE = tf.float32
Y_TYPE = tf.float32

MAX_CLASS_COUNT = 1000

AGE_RANGES = {"all": (0, 90)}
