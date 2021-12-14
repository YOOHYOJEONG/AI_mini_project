import tensorflow as tf

resnet = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet')

def _make_deconv_layer(num_deconv_layers):
    seq_model = tf.keras.models.Sequential()

    # [[YOUR CODE]]

    return seq_model

upconv = _make_deconv_layer(3)

final_layer = # [[YOUR CODE]]


def Simplebaseline(input_shape=(256, 256, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # [[YOUR CODE]]

    model = tf.keras.Model(inputs, out, name='simple_baseline')
    return model