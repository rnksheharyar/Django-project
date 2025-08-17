# import libraries
## numpy for fast array calculations. Matplot.pyplot to visualize images and masks.
# tensorflow for building and training the model. Tensorflow main deep learning framework.
# keras: Streamlines model building.
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
# Load the Dataset
dataset, info=tfds.load('oxford_iiit_pet:4.*.*', with_info=True)

# Set Constants
## Batch size and buffer control training efficiency fand randomization.
## Width/height standardize images for VGG16
BATCH_SIZE = 64
BUFFER_SIZE = 1000
width, height = 224, 224
TRAIN_LENGTH = info.splits['train'].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
# Data Preprocessing and Augmentation

# We perform the data preprocessing.
 ## Converts image pixels to float and scales between 0-1.
 ## Masks start from zero for correct class indexing.
 ## Resizes images and masks.
 ## Random flip adds variety for robust training.
def normalize(input_image, input_mask):
    img = tf.cast(input_image, dtype=tf.float32) / 255.0
    input_mask -= 1
    return img, input_mask
@tf.function
def load_train_ds(example):
    img = tf.image.resize(example['image'], (width, height))
    mask =tf.image.resize(example['segmentation_mask'], (width, height))
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    img, mask = normalize(img, mask)
    return img, mask
# Build Data Pipeline
## We prepare the data pipelines,
## Map: Applies preprocessing to each sample.
## cache, shuffle, batch, prefetch: Optimize data loading and traning throghput.
train = dataset['train'].map(load_train_ds, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_train_ds)
train_ds = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
# Visualize the data
## We visualize the input, ground-truth mask and prediction side by side for easy comparison.
def display_images(display_list):
    plt.figure(figsize=(15, 15))
    titles = ['Input image', 'True Mask', 'Predicted Mask']
    for i, image in enumerate(display_list):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
    plt.imshow(keras.preprocessing.image.array_to_img(image))
    plt.axis('off')
    plt.show()
for img, mask in train_ds.take(1):
    display_images([img[0], mask[0]])
# Model Construction
base_model = keras.applications.vgg16.VGG16(
    include_top=False, input_shape=(width, height, 3))
layer_names = ['block1_pool', 'block2_pool',
               'block3_pool', 'block4_pool', 'block5_pool']
base_model_outputs = [base_model.get_layer(
    name).output for name in layer_names]
base_model.trainable = False
VGG_16 = keras.Model(inputs=base_model.input, outputs=base_model_outputs)


def fcn8_decoder(convs, n_classes):
    f1, f2, f3, f4, p5 = convs
    n = 4096
    c6 = keras.layers.Conv2D(n, (7, 7), activation='relu', padding='same')(p5)
    c7 = keras.layers.Conv2D(n, (1, 1), activation='relu', padding='same')(c6)
    f5 = c7
    o = keras.layers.Conv2DTranspose(
        n_classes, (4, 4), strides=(2, 2), use_bias=False)(f5)
    o = keras.layers.Cropping2D((1, 1))(o)
    o2 = keras.layers.Conv2D(
        n_classes, (1, 1), activation='relu', padding='same')(f4)
    o = keras.layers.Add()([o, o2])
    o = keras.layers.Conv2DTranspose(
        n_classes, (4, 4), strides=(2, 2), use_bias=False)(o)
    o = keras.layers.Cropping2D((1, 1))(o)
    o2 = keras.layers.Conv2D(
        n_classes, (1, 1), activation='relu', padding='same')(f3)
    o = keras.layers.Add()([o, o2])
    o = keras.layers.Conv2DTranspose(
        n_classes, (8, 8), strides=(8, 8), use_bias=False)(o)
    o = keras.layers.Activation('softmax')(o)
    return o

# Build and compile segmentation model

def segmentation_model():
    inputs = keras.layers.Input(shape=(width, height, 3))
    convs = VGG_16(inputs)
    outputs = fcn8_decoder(convs, 3)
    return keras.Model(inputs, outputs)


model = segmentation_model()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train this model

# EPOCHS = 15
# VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

# model_history = model.fit(
#     train_ds, epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_data=test_ds,
#     validation_steps=VALIDATION_STEPS
# )
# # Model Make predictions and we visualize the it.
## Convert model output to a simple mask for visualization.
## Displays results for sample images to verify segmentation performance.
# def create_mask(pred_mask):
#     pred_mask = tf.argmax(pred_mask, axis=-1)
#     pred_mask = pred_mask[..., tf.newaxis]
#     return pred_mask[0]


# def show_predictions(dataset=None, num=1):
#     for image, mask in dataset.take(num):
#         pred_mask = model.predict(image)
#         display_images([image[0], mask[0], create_mask(pred_mask)])

def compute_metrics(y_true, y_pred):
    class_wise_iou, class_wise_dice_score = [], []
    smooth = 1e-5
    for i in range(3):
        intersection = np.sum((y_pred == i) & (y_true == i))
        y_true_area = np.sum(y_true == i)
        y_pred_area = np.sum(y_pred == i)
        combined_area = y_true_area + y_pred_area
        iou = (intersection + smooth) / (combined_area - intersection + smooth)
        dice = 2 * (intersection + smooth) / (combined_area + smooth)
        class_wise_iou.append(iou)
        class_wise_dice_score.append(dice)
    return class_wise_iou, class_wise_dice_score
