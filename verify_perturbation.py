'''verify_perturbation.py'''

import argparse
from timeit import default_timer as timer

import numpy as np
import onnx
import onnxruntime
import torch
import os


from tensorflow.keras.models import load_model
import sys
sys.path.insert(0, './ERAN/ELINA/python_interface/')
sys.path.insert(0, './ERAN/tf_verify/')

from eran import ERAN

#from data_processing import datasets
#from pointnet.model import PointNet
import onnxruntime as ort


from relaxations.interval import Interval
#from util import onnx_converter
from util.argparse import absolute_path
from util.experiment import Experiment
from util.math import set_random_seed, DEFAULT_SEED
from verifier.eran_verifier import EranVerifier
import csv


def load_pb_model(sess, save_path):
    with tf.gfile.FastGFile(save_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')  


def get_tests(dataset):
    if (dataset == 'cifar'):
        csvfile = open('/root/3dcertify/ERAN/data/{}10_test.csv'.format(dataset), 'r')
        tests = csv.reader(csvfile, delimiter=',')
    else:
        csvfile = open('/root/3dcertify/ERAN/data/{}_test.csv'.format(dataset), 'r')
        tests = csv.reader(csvfile, delimiter=',')
    return tests
def get_out_tensors(out_names):
    return [sess.graph.get_tensor_by_name(name[7:]) for name in out_names]

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=absolute_path, required=True, help="Path to the model to verify (.pth)")
parser.add_argument('--num_points', type=int, default=1024, help="Number of points per point cloud")
parser.add_argument('--eps', type=float, default=0.01, help="Epsilon-box to certify around the input point")
parser.add_argument('--pooling', choices=['improved_max', 'max', 'avg'], default='improved_max', help='The pooling function to use')
parser.add_argument('--max_features', type=int, default=1024, help='The number of global features')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='seed for random number generator')
parser.add_argument('--experiment', type=str, help='name of the experiment')

settings = parser.parse_args()

experiment = Experiment(settings)
logger = experiment.logger
set_random_seed(settings.seed)

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)

if "onnx" in str(settings.model):
    
    model_ = onnx.load(settings.model)
    onnx.checker.check_model(model_)
    ort_session = ort.InferenceSession(model_.SerializeToString())
    eran = EranVerifier(model_)


elif "pb" in str(settings.model):
    import tensorflow.compat.v1 as tf

    sess = tf.Session()

    load_pb_model(sess, str(settings.model))

    graph = tf.get_default_graph()
    ops = sess.graph.get_operations()
    last_layer_index = -1
    out_tensor = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')
    eran = ERAN(out_tensor, sess)
    sess.run(tf.global_variables_initializer())
else:#h5 model
    import tensorflow as tf
    model = load_model(str(settings.model), custom_objects={'fn': fn,'tf': tf })
    eran = EranVerifier(model)

if "cifar" in str(settings.model):
    dataset='cifar'
    tests=get_tests('cifar')
else:
    dataset='mnist'
    tests=get_tests('mnist')

correct_predictions = 0
verified_same = 0
verified_different = 0
not_verified = 0
iterations = 0

total_time = 0
steps=15
eps_0=0.005


summation=0
truenum=0
for i,test in enumerate(tests):
    if dataset=='cifar':
        if "onnx" in str(settings.model): 
            image= np.float32(test[1:len(test)]).reshape(3,32,32)/np.float32(255)
        elif "pb" in str(settings.model): 
            image= np.float32(test[1:len(test)]).reshape(32,32,3)/np.float32(255)
            image=np.expand_dims(image.copy(),axis=0)
    else:
        if "onnx" in str(settings.model): 
            image=np.float32(test[1:len(test)]).reshape(1,28,28)/np.float32(255)
        elif "pb" in str(settings.model): 
            image=np.float32(test[1:len(test)]).reshape(28,28,1)/np.float32(255)
            image=np.expand_dims(image.copy(),axis=0)


    true_label=test[0]
    iterations += 1
    if iterations<=10:
        continue
    if "onnx" in str(settings.model): 
        input={ort_session.get_inputs()[0].name: np.expand_dims(image.copy(),axis=0)}
        predict_label=ort_session.run(None,input)
        print(np.argmax(predict_label))
        print(true_label)
        print('num image',iterations)
    elif "pb" in str(settings.model): 
        x = 'input_1:0'
        pred = 'output_1:0' 
        predict_label=sess.run(pred, {sess.graph.get_operations()[0].name + ':0': image})
    else:
        
        predict_label=model.predict(np.expand_dims(image.copy(),axis=0))
        print(np.argmax(predict_label))
        print(true_label)
        print('num image',iterations)

    if int(np.argmax(predict_label)) != int(true_label):
        print('wrong_predict')
        continue
    else:
        truenum+=1

    correct_predictions += 1
    logger.info("Verifying onnx model...")
    logger.info("Solving network...")

    log_eps=np.log(eps_0)
    log_eps_min=-np.inf
    log_eps_max=np.inf
    start = timer()
    for j in range(steps):
        print('current steps=',j)
        lower_bound = image -np.exp(log_eps) 
        upper_bound = image +np.exp(log_eps) 
        assert np.all(lower_bound <= upper_bound)

        if "pb" in str(settings.model): 
            (dominant_class,_, nlb, nub,_) =eran.analyze_box(
                specLB=lower_bound,
                specUB=upper_bound,
                domain='deeppoly',
                timeout_lp=1_000_000,
                timeout_milp=1_000_000,
                use_default_heuristic=True,
                testing=True
            )
        else: 
            (dominant_class, nlb, nub) = eran.analyze_classification_box(Interval(lower_bound, upper_bound))
        certified = int(dominant_class) == int(true_label)
        print('certified',certified)
        if certified:
            log_eps_min = log_eps
            log_eps = np.minimum(log_eps*2, (log_eps_max+log_eps_min)/2)

        else:
            log_eps_max = log_eps
            log_eps = np.maximum(log_eps/2, (log_eps_max+log_eps_min)/2)
    summation+=np.exp(log_eps_min)
    print('eps=',np.exp(log_eps_min))
    
    end = timer()
    elapsed = end - start
    total_time += elapsed


avg_time=total_time/truenum
eps_avg=summation/truenum




logger.info(f"Time for this round: {elapsed}s. Total time: {total_time}s. Average Time:{avg_time}.")
logger.info(f"Tested {iterations} data points out of which {correct_predictions} were correctly predicted.")
logger.info(f"avg robustness bound:{eps_avg}.")
print("avg={:.6f}".format(eps_avg))



