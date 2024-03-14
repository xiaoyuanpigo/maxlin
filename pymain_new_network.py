import subprocess
import numpy as np
from cnn_bounds_full import run as run_cnn_full
from cnn_bounds_full_core import run as run_cnn_full_core
#from cnn_bounds_full_core_not_zero import run as run_cnn_full_core
from Attacks.cw_attack import cw_attack
from tensorflow.contrib.keras.api.keras import backend as K

import time as timing
import datetime

ts = timing.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')


#Prints to log file
def printlog(s):
    print(s, file=open("log_pymain_new_network_" + timestr + ".txt", "a"))


#Runs command line command
def command(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          shell=True).stdout.decode('utf-8')


#Runs Fast-Lin with specified parameters
def run(hidden,
        numlayer,
        numimage,
        norm,
        filename='',
        layers=None,
        lp=False,
        lpfull=False,
        dual=False,
        sparse=False,
        spectral=False,
        cifar=False,
        cnnmodel=False,
        tinyimagenet=False):
    if sparse:
        cmd = 'python3 Fast-Lin/main_sparse.py '
    else:
        cmd = 'python3 Fast-Lin/main.py '
    if cifar:
        cmd += '--model cifar '
    if tinyimagenet:
        cmd += '--model tiny '
    if spectral:
        cmd += '--method spectral '
    if cnnmodel:
        cmd += '--cnnmodel '
    cmd += '--hidden ' + str(hidden) + ' '
    cmd += '--numlayer ' + str(numlayer) + ' '
    cmd += '--numimage ' + str(numimage) + ' '
    cmd += '--norm ' + str(norm) + ' '
    if lp:
        cmd += '--LP '
    if lpfull:
        cmd += '--LPFULL '
    if dual:
        cmd += '--dual '
    if filename:
        cmd += '--filename ' + str(filename) + ' '
        cmd += '--layers ' + ' '.join(str(l) for l in layers) + ' '
    cmd += '--eps 0.05 --warmup --targettype random'
    printlog("cmd: " + str(cmd))
    result = command(cmd)
    result = result.rsplit('\n', 2)[-2].split(',')
    LB = result[1].strip()[20:]
    time = result[3].strip()[17:]
    return float(LB), float(time)


#Runs Ti-Lin with specified parameters
def run_cnn(file_name,
            n_samples,
            norm,
            core=True,
            activation='relu',
            cifar=False,
            tinyimagenet=False):
    if core:
        if norm == 'i':
            #run_cnn_full_core(file_name, n_samples, 105, 1, activation, cifar, tinyimagenet)
            return run_cnn_full_core(file_name, n_samples, 105, 1, activation,
                                     cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full_core(file_name, n_samples, 2, 2, activation,
                                     cifar, tinyimagenet)
        if norm == '1':
            return run_cnn_full_core(file_name, n_samples, 1, 105, activation,
                                     cifar, tinyimagenet)
    else:
        if norm == 'i':
            return run_cnn_full(file_name, n_samples, 105, 1, activation,
                                cifar, tinyimagenet)
        elif norm == '2':
            return run_cnn_full(file_name, n_samples, 2, 2, activation, cifar,
                                tinyimagenet)
        if norm == '1':
            return run_cnn_full(file_name, n_samples, 1, 105, activation,
                                cifar, tinyimagenet)


#Runs all Fast-Lin and Ti-Lin variations
def run_all_relu(layers,
                 file_name,
                 mlp_file_name,
                 cifar=False,
                 num_image=10,
                 flfull=False,
                 nonada=False):
    if len(file_name.split('_')) == 5:
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        if nonada:  #Run non adaptive Ti-Lin bounds
            LB, time = run_cnn(file_name, num_image, norm, cifar=cifar)
            printlog("Ti-Lin-relu")
            if filters:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                    .format(file_name,
                            len(layers) + 1, num_image, norm, filters,
                            kernel_size))
            else:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
                    .format(file_name,
                            len(layers) + 1, num_image, norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time) + " sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LB, time = run_cnn(file_name,
                           num_image,
                           norm,
                           activation='ada',
                           cifar=cifar)
        printlog("Ti-Lin-Ada, ReLU activation")
        if filters:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                .format(file_name,
                        len(layers) + 1, num_image, norm, filters,
                        kernel_size))
        else:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
                .format(file_name,
                        len(layers) + 1, num_image, norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LB, time = run(999,
                       len(layers) + 1,
                       num_image,
                       norm,
                       mlp_file_name,
                       layers,
                       sparse=True,
                       cifar=cifar)
        printlog("Fast-Lin, Sparse")
        if filters:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                .format(file_name,
                        len(layers) + 1, num_image, norm, filters,
                        kernel_size))
        else:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
                .format(file_name,
                        len(layers) + 1, num_image, norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        if flfull:  #Run full matrix version of Fast-Lin
            LB, time = run(999,
                           len(layers) + 1,
                           num_image,
                           norm,
                           mlp_file_name,
                           layers,
                           cifar=cifar)
            printlog("Fast-Lin")
            if filters:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                    .format(file_name,
                            len(layers) + 1, num_image, norm, filters,
                            kernel_size))
            else:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
                    .format(file_name,
                            len(layers) + 1, num_image, norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time) + " sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LB, time = run(999,
                       len(layers) + 1,
                       num_image,
                       norm,
                       mlp_file_name,
                       layers,
                       spectral=True,
                       cifar=cifar)
        printlog("Global-Lips (spectral)")
        if filters:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                .format(file_name,
                        len(layers) + 1, num_image, norm, filters,
                        kernel_size))
        else:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
                .format(file_name,
                        len(layers) + 1, num_image, norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times


#Runs all general activation function versions of Ti-Lin
# 2022/10/29 added new params: tiny_imagenet relative to def run_cnn(tinyimagenet, ...)
def run_all_general(file_name,
                    num_image=10,
                    core=True,
                    cifar=False,
                    tiny_imagenet=False,
                    ada=True,
                    onlyrelu=False,
                    skipsigmoid=False,
                    no_activate=True):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        if no_activate:
            LB, time = run_cnn(file_name,
                               num_image,
                               norm,
                               core=core,
                               activation='relu',
                               cifar=cifar,
                               tinyimagenet=tiny_imagenet)
            printlog("Ti-Lin-relu")
            if filters:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                    .format(file_name, nlayer, num_image, norm, filters,
                            kernel_size))
            else:
                printlog(
                    "model name = {0}, numimage = {1}, norm = {2}, targettype = random"
                    .format(file_name, num_image, norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time) + " sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        if ada:

            LB, time = run_cnn(file_name,
                               num_image,
                               norm,
                               core=core,
                               activation='ada',
                               cifar=cifar,
                               tinyimagenet=tiny_imagenet)
            printlog("Ti-Lin-Ada, ReLU activation")
            if filters:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                    .format(file_name, nlayer, num_image, norm, filters,
                            kernel_size))
            else:
                printlog(
                    "model name = {0}, numimage = {1}, norm = {2}, targettype = random"
                    .format(file_name, num_image, norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time) + " sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        if not onlyrelu:
            if not skipsigmoid:
                LB, time = run_cnn(file_name + '_sigmoid',
                                   num_image,
                                   norm,
                                   core=core,
                                   activation='sigmoid',
                                   cifar=cifar,
                                   tinyimagenet=tiny_imagenet)
                printlog("Ti-Lin-Ada, Sigmoid activation")
                if filters:
                    printlog(
                        "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                        .format(file_name, nlayer, num_image, norm, filters,
                                kernel_size))
                else:
                    printlog(
                        "model name = {0}, numimage = {1}, norm = {2}, targettype = random"
                        .format(file_name, num_image, norm))
                printlog("avg robustness = {:.5f}".format(LB))
                printlog("avg run time = {:.2f}".format(time) + " sec")
                printlog("-----------------------------------")
                LBss.append(LB)
                timess.append(time)
            LB, time = run_cnn(file_name + '_tanh',
                               num_image,
                               norm,
                               core=core,
                               activation='tanh',
                               cifar=cifar,
                               tinyimagenet=tiny_imagenet)
            printlog("Ti-Lin-Ada, Tanh activation")
            if filters:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                    .format(file_name, nlayer, num_image, norm, filters,
                            kernel_size))
            else:
                printlog(
                    "model name = {0}, numimage = {1}, norm = {2}, targettype = random"
                    .format(file_name, num_image, norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time) + " sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
            LB, time = run_cnn(file_name + '_atan',
                               num_image,
                               norm,
                               core=core,
                               activation='arctan',
                               cifar=cifar,
                               tinyimagenet=tiny_imagenet)
            printlog("Ti-Lin-Ada, Arctan activation")
            if filters:
                printlog(
                    "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                    .format(file_name, nlayer, num_image, norm, filters,
                            kernel_size))
            else:
                printlog(
                    "model name = {0}, numimage = {1}, norm = {2}, targettype = random"
                    .format(file_name, num_image, norm))
            printlog("avg robustness = {:.5f}".format(LB))
            printlog("avg run time = {:.2f}".format(time) + " sec")
            printlog("-----------------------------------")
            LBss.append(LB)
            timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times


#Runs Dual LP version of Fast-Lin
def run_LP(layers, mlp_file_name, num_image=10, core=True, cifar=False):
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = run(999,
                       len(layers) + 1,
                       num_image,
                       norm,
                       mlp_file_name,
                       layers,
                       lp=True,
                       dual=True,
                       cifar=cifar)
        printlog("Dual-LP")
        printlog(
            "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
            .format(mlp_file_name,
                    len(layers) + 1, num_image, norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times


#Runs global Lips bound
def run_global(file_name,
               num_layers,
               num_image=10,
               cifar=False,
               tinyimagenet=False):
    if len(file_name.split('_')) == 5:
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    LBs = []
    times = []
    for norm in ['i', '2', '1']:
        LBss = []
        timess = []
        LB, time = run(999,
                       num_layers + 1,
                       num_image,
                       norm,
                       file_name, [1 for i in range(num_layers + 1)],
                       spectral=True,
                       dual=True,
                       cifar=cifar,
                       cnnmodel=True,
                       tinyimagenet=tinyimagenet)
        printlog("Global-Lips (spectral)")
        if filters:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                .format(file_name,
                        len(layers) + 1, num_image, norm, filters,
                        kernel_size))
        else:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random"
                .format(file_name,
                        len(layers) + 1, num_image, norm))
        printlog("avg robustness = {:.5f}".format(LB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        LBss.append(LB)
        timess.append(time)
        LBs.append(LBss)
        times.append(timess)
    return LBs, times


#Run all norm attacks
def run_attack(file_name, sess, num_image=10, cifar=False, tinyimagenet=False):
    if len(file_name.split('_')) == 5:
        nlayer = file_name.split('_')[-3][0]
        filters = file_name.split('_')[-2]
        kernel_size = file_name.split('_')[-1]
    else:
        filters = None
    UBs = []
    times = []
    for norm in ['i', '2', '1']:
        UB, time = cw_attack(file_name, norm, sess, num_image, cifar,
                             tinyimagenet)
        printlog("CW/EAD")
        if filters:
            printlog(
                "model name = {0}, numlayer = {1}, numimage = {2}, norm = {3}, targettype = random, filters = {4}, kernel size = {5}"
                .format(file_name, nlayer, num_image, norm, filters,
                        kernel_size))
        else:
            printlog(
                "model name = {0}, numimage = {1}, norm = {2}, targettype = random"
                .format(file_name, num_image, norm))
        printlog("avg robustness = {:.5f}".format(UB))
        printlog("avg run time = {:.2f}".format(time) + " sec")
        printlog("-----------------------------------")
        UBs.append([UB])
        times.append([time])
    return UBs, times


if __name__ == '__main__':
    LB = []
    time = []

    table = 1
    print("==================================================")
    print("================ Running Table {} ================".format(table))
    print("==================================================")
    printlog("Table {} result".format(table))
    printlog("-----------------------------------")


    if table == 1:
        LBs, times = run_all_general(
            'models/mnist_cnn_4layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)

        LBs, times = run_all_general(
            'models/mnist_cnn_5layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/mnist_cnn_6layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/mnist_cnn_7layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/mnist_cnn_8layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/cifar_cnn_4layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=True,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)

        LBs, times = run_all_general(
            'models/cifar_cnn_5layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=True,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/cifar_cnn_6layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=True,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/cifar_cnn_7layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=True,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            'models/cifar_cnn_8layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=True,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)

        #relu
        LBs, times = run_all_general(
            'models/mnist_cnn_lenet',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True)
        LB.append(LBs)
        time.append(times)
        #sigmoid,tanh,arctan
        LBs, times = run_all_general(
            'models/mnist_cnn_lenet',
            no_activate=False,
            core=False,
            ada=False,
            cifar=False,
            onlyrelu=False)
        LB.append(LBs)
        time.append(times)
        LBs, times = run_all_general(
            '/root/Ti-Lin/Ti-Lin/models/tiny_cnn_7layer',
            no_activate=False,
            core=True,
            ada=True,
            cifar=False,
            onlyrelu=True,
            tiny_imagenet=True)
        LB.append(LBs)
        time.append(times)
