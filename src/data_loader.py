import tensorflow as tf
import os
import numpy as np
from tensorflow import keras

def make_dataset(partition, nutrition5k_path, dishes, nutr_names, preprocess_fn, crop_size, img_size, batch_size):
    preprocessing = keras.Sequential(
        [
            keras.layers.RandomCrop(*crop_size),
            keras.layers.RandomFlip("vertical"),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1)
        ]
    ) if partition == 'train' else keras.layers.CenterCrop(*crop_size)

    def get_nutr(image_path):
        image_id = f'dish_{tf.strings.split(image_path, "_")[1].numpy().decode("UTF-8")}'
        nutr = dishes[image_id]['nutr_values_per100g']
        return tf.constant([nutr[name] for name in nutr_names], dtype=tf.float32)

    def process_path(file_path):
        label = tf.py_function(get_nutr, [tf.strings.split(file_path, os.sep)[-1]], tf.float32)
        file_content = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(file_content)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        if width > height:
            image = tf.image.resize(image, (img_size, int(float(img_size * width) / float(height))))
        else:
            image = tf.image.resize(image, (int(float(img_size * height) / float(width)), img_size))

        image = preprocess_fn(image)
        image = preprocessing(image)
        return image, label

    return tf.data.Dataset.list_files(f'{nutrition5k_path}/images/{partition}/*')\
        .map(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)\
        .shuffle(1024 if partition == 'train' else 1)\
        .batch(batch_size)

def make_ingredients_dataset(partition, seq_length, vocab_size, nutrition5k_path, dishes_sorted, 
                                preprocess_fn, crop_size, img_size, batch_size):
    preprocessing = keras.Sequential(
        [
            keras.layers.RandomCrop(*crop_size),
            keras.layers.RandomFlip("vertical"),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1)
        ]
    ) if partition == 'train' else keras.layers.CenterCrop(*crop_size)

    def get_ingr(image_path, shifted):
        image_id = f'dish_{tf.strings.split(image_path, "_")[1].numpy().decode("UTF-8")}'
        ingrs = [vocab_size - 2] + [int(ingr) for ingr in dishes_sorted[image_id]['ingredients']][:seq_length - 1] + [vocab_size - 1]
        ingrs = np.pad(ingrs, (0, seq_length + 1 - len(ingrs)))
        if shifted:
            return tf.constant(ingrs[1:], dtype=tf.int16)
        return tf.constant(ingrs[:-1], dtype=tf.int16)

    def process_path(file_path):
        image_path = tf.strings.split(file_path, os.sep)[-1]
        decoder_input = tf.py_function(get_ingr, [image_path, False], tf.int16)
        decoder_input.set_shape((seq_length,))
        target = tf.py_function(get_ingr, [image_path, True], tf.int16)
        target.set_shape((seq_length,))

        file_content = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(file_content)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        if width > height:
            image = tf.image.resize(image, (img_size, int(float(img_size * width) / float(height))))
        else:
            image = tf.image.resize(image, (int(float(img_size * height) / float(width)), img_size))

        image = preprocess_fn(image)
        image = preprocessing(image)
        return (image, decoder_input), target

    return tf.data.Dataset.list_files(f'{nutrition5k_path}/images/{partition}/*')\
        .map(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)\
        .shuffle(1024 if partition == 'train' else 1)\
        .batch(batch_size)

def make_portions_dataset(partition, seq_length, vocab_size, nutrition5k_path, dishes_2plusing, 
                                preprocess_fn, crop_size, img_size, batch_size):
    preprocessing = keras.Sequential(
        [
            keras.layers.RandomCrop(*crop_size),
            keras.layers.RandomFlip("vertical"),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1)
        ]
    ) if partition == 'train' else keras.layers.CenterCrop(*crop_size)

    def contains_dish(file_path):
        filename = file_path = tf.strings.split(file_path, os.sep)[-1]
        dish_id = f'dish_{tf.strings.split(filename, "_")[1].numpy().decode("UTF-8")}'
        return tf.constant(dish_id in dishes_2plusing)

    def get_ingr(image_path):
        image_id = f'dish_{tf.strings.split(image_path, "_")[1].numpy().decode("UTF-8")}'
        ingrs = [int(ingr) for ingr in dishes_2plusing[image_id]['ingredients']][:seq_length - 1] + [vocab_size - 1]
        ingrs = np.pad(ingrs, (0, seq_length - len(ingrs)))
        return tf.constant(ingrs, dtype=tf.int16)

    def get_portions(image_path, shifted):
        image_id = f'dish_{tf.strings.split(image_path, "_")[1].numpy().decode("UTF-8")}'
        portions = [-1] + [weight for weight in dishes_2plusing[image_id]['weight_per_ingr']][:seq_length - 1]
        portions = np.pad(portions, (0, seq_length + 1 - len(portions)))
        if shifted:
            return tf.constant(portions[1:], dtype=tf.float32)
        return tf.constant(portions[:-1], dtype=tf.float32)

    def process_path(file_path):
        image_path = tf.strings.split(file_path, os.sep)[-1]
        ingredients = tf.py_function(get_ingr, [image_path], tf.int16)
        ingredients.set_shape((seq_length,))
        decoder_input = tf.py_function(get_portions, [image_path, False], tf.float32)
        decoder_input.set_shape((seq_length,))
        target = tf.py_function(get_portions, [image_path, True], tf.float32)
        target.set_shape((seq_length,))

        file_content = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(file_content)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        if width > height:
            image = tf.image.resize(image, (img_size, int(float(img_size * width) / float(height))))
        else:
            image = tf.image.resize(image, (int(float(img_size * height) / float(width)), img_size))

        image = preprocess_fn(image)
        image = preprocessing(image)
        return (image, ingredients, decoder_input), target

    return tf.data.Dataset.list_files(f'{nutrition5k_path}/images/{partition}/*')\
        .filter(lambda filename: tf.py_function(contains_dish, [filename], tf.bool))\
        .map(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)\
        .shuffle(1024 if partition == 'train' else 1)\
        .batch(batch_size)
