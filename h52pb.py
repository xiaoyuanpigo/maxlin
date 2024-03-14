#import keras
import keras2onnx
#import tensorflow.keras
import numpy as np
import tensorflow as tf
import onnx
from tensorflow.contrib.keras.api.keras.models import load_model


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


# start-transformation

model_pyt.layer1.weight.data = torch.tensor(
    model_keras.layers[0].get_weights()[0].T)
model_pyt.layer1.bias.data = torch.tensor(
    model_keras.layers[0].get_weights()[1])
from tensorflow.contrib.keras.api.keras.models import load_model
import tensorflow as tf
import os
from tensorflow.contrib.keras import backend as K
from tensorflow.python.framework import graph_util, graph_io
import tensorflow as tf
from tensorflow.python.platform import gfile


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


def h5_to_pb(h5_weight_path,
             output_dir,
             out_prefix="output_",
             log_tensorboard=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    h5_model = load_model(h5_weight_path, custom_objects={'fn': fn})
    h5_model.summary()
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    model_name = os.path.splitext(os.path.split(h5_weight_path)[-1])[0] + '.pb'

    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(
        sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph,
                         output_dir,
                         name=model_name,
                         as_text=False)


#
#!!!!!
h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/cifar_cnn_4layer',
         output_dir='/MaxLin/3dcertify/models')
# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/cifar_cnn_5layer', output_dir='/MaxLin/3dcertify/models')
# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/cifar_cnn_6layer', output_dir='/MaxLin/3dcertify/models')

# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/mnist_cnn_4layer', output_dir='/MaxLin/3dcertify/models')
# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/mnist_cnn_5layer', output_dir='/MaxLin/3dcertify/models')
# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/mnist_cnn_6layer', output_dir='/MaxLin/3dcertify/models')

# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/mnist_cnn_lenet_sigmoid',output_dir='/MaxLin/3dcertify/models')
# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/mnist_cnn_lenet',output_dir='/MaxLin/3dcertify/models')
# h5_to_pb(h5_weight_path='/root/Ti-Lin/Ti-Lin/models/mnist_cnn_lenet_tanh',output_dir='/MaxLin/3dcertify/models')
