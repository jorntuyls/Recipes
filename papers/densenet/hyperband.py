#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimizes hyperparameters of DenseNet on CIFAR-10.

For usage information, call with --help.

Author: Jorn Tuyls
"""

import numpy as np

import os
import math
import time

save_weights=None
save_errors=None
augment=False

def generate_in_background(generator, num_cached=10):
    """
    Runs a generator in a background thread, caching up to `num_cached` items.
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        item = queue.get()

def get_random_hyperparameter_configuration():
    x = np.random.uniform(0,1,5)

    k = 6 + int(18*x[0])                                #   growth rate in [6, 24]
    L = 3*(13+int(50*x[0])) + 1                          #   depth in [40, 190] and equal to num_blocks(=3) * n + 1 for some n
    batch_size_train = int(pow(2.0, 4.0 + 4.0*x[2]))    #   in [2^4, 2^8] = [16, 256]
    LR = float(pow(10.0, -2 + 1.5*x[3]))                #   learning rate in [10^-2, 10^-0.5] = [0.01, ~0.31]
    d = 0.05 + int(0.25*x[4])                           #   dropout in [0.05, 0.3]

    return k, L, batch_size_train, LR, d

def run_then_return_val_loss(hyperparameters, time_limit=100000, nepochs=100000):
    k = hyperparameters[0]
    L = hyperparameters[1]
    batch_size_train = hyperparameters[2]
    LR = hyperparameters[3]
    d = hyperparameters[4]

    results = train_densenet(depth=L, growth_rate=k, dropout=d, augment=augment, eta=LR, save_weights=save_weights,
                    save_errors=save_errors, batchsize=batch_size_train, nepochs=nepochs, time_limit=time_limit)

    return results

def hyperband( max_iter=81, eta=3, unit=1, resource="nepochs"):
    logeta = lambda x: math.log(x)/math.log(eta)
    s_max = int(logeta(max_iter))   # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter          # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    #noiselevel = 0.2  # noise level of the objective function
    # Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.

    nruns = 1       # set it to e.g. 10 when testing hyperband against randomsearch
    for irun in range(0, 10):
        start_time = time.time()
        hband_results_filename = "stat_hyperband/hyperband_{}.txt".format(irun)
        hband_file = open(hband_results_filename, 'w+', 0)

        x_best_observed = []
        x_best_observed_nep = 0
        y_best_observed = 0
        best_observed_err = 0

        nevals = 0       # total number of full (with max_iter nepochs) evaluations used so far

        for s in reversed(range(s_max+1)):

            stat_filename = "stat_hyperband/hband_benchmark_{}_{}.txt".format(irun,s)
            stat_file = open(stat_filename, 'w+', 0)

            n = int(math.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
            r = max_iter*eta**(-s)      # initial number of iterations to run configurations for

            # Begin Finite Horizon Successive Halving with (n,r)
            T = [ get_random_hyperparameter_configuration() for i in range(n) ]
            for i in range(s+1):
                print("RUN: {}, {}, {}".format(irun, s, i))
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = n*eta**(-i)
                r_i = r*eta**(i)
                if resource == "nepochs":
                    nnepochs = r_i*unit
                    time_limit = 100000
                elif resource == "time":
                    nnepochs = 100000
                    time_limit = r_i*unit
                else:
                    raise ValueError("resource should be either 'nepochs' or 'time'")

                results = [ run_then_return_val_loss(hyperparameters=t, time_limit=time_limit, nepochs=nnepochs) for t in T ]

                val_losses = [ result[0] for result in results]
                val_errs = [ result[1] for result in results]

                nevals = nevals + len(T) * r_i / max_iter
                argsortidx = np.argsort(val_losses)

                if (x_best_observed == []):
                    x_best_observed = T[argsortidx[0]]
                    y_best_observed = val_losses[argsortidx[0]]
                    best_observed_err = val_errs[argsortidx[0]]
                    x_best_observed_nep = r_i
                # only if better AND based on >= number of nepochs, the latter is optional
                if (val_losses[argsortidx[0]] < y_best_observed):# and (r_i >= x_best_observed_nep):
                    x_best_observed_nep = r_i
                    y_best_observed = val_losses[argsortidx[0]]
                    best_observed_err = val_errs[argsortidx[0]]
                    x_best_observed = T[argsortidx[0]]

                for j in range(0, len(T)):
                    stat_file.write("{}\t{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(
                                        T[j][0], T[j][1],T[j][2],T[j][3],r_i,val_losses[j]))
                T = [ T[i] for i in argsortidx[0:int( n_i/eta )] ]

                # suppose the current best solution w.r.t. validation loss is our recommendation
                # then let's evaluate it in noiseless settings (~= averaging over tons of runs)
                # if (len(T)):
                #    f_recommendation = self.run_then_return_val_loss_parallel(81, [x_best_observed]) # 81 nepochs and 1e-10 ~= zero noise
                hband_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                nevals, time.time()-start_time, x_best_observed[0], x_best_observed[1], x_best_observed[2],
                                x_best_observed[3], x_best_observed[4], x_best_observed_nep, y_best_observed, best_observed_err))
            # End Finite Horizon Successive Halving with (n,r)

            stat_file.close()
        hband_file.close()

def train_densenet(depth, growth_rate, dropout, augment, eta, save_weights,
                save_errors, batchsize=64, nepochs=100000, time_limit=100000):
    # import (deferred until now to make --help faster)
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne

    import densenet_fast as densenet  # or "import densenet" for slower version
    import cifar10
    import progress

    batch_size_valid = 512 # batch size for validation is always 512

    # instantiate network
    print("Instantiating network...")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = densenet.build_densenet(input_var=input_var, depth=depth,
                                      growth_rate=growth_rate, dropout=dropout)
    print("%d layers with weights, %d parameters" %
          (sum(hasattr(l, 'W')
               for l in lasagne.layers.get_all_layers(network)),
           lasagne.layers.count_params(network, trainable=True)))

    # load dataset
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = cifar10.load_dataset(
            path=os.path.join(os.path.dirname(__file__), 'data'))
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]
    # !! We only using training and validation set for optimization
    print(X_train.shape)
    print(X_val.shape)
    X_train = X_train[:500]
    y_train = y_train[:500]
    X_val = X_val[:500]
    y_val = y_val[:500]

    # define training function
    print("Compiling training function...")
    prediction = lasagne.layers.get_output(network)
    # note: The Keras implementation clips predictions for the categorical
    #       cross-entropy. This doesn't seem to have a positive effect here.
    # prediction = T.clip(prediction, 1e-7, 1 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                       target_var).mean()
    # note: The paper says 1e-4 decay, but 1e-4 in Torch is 5e-5 elsewhere.
    #       However, 1e-4 seems to work better than 5e-5, so we use 1e-4.
    # note: Torch includes biases in L2 decay. This seems to be important! So
    #       we decay all 'trainable' parameters, not just 'regularizable' ones.
    l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2, {'trainable': True})
    params = lasagne.layers.get_all_params(network, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(eta), name='eta')
    updates = lasagne.updates.nesterov_momentum(
            loss + l2_loss, params, learning_rate=eta, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    l2_fn = theano.function([], l2_loss)

    # define validation/testing function
    print("Compiling testing function...")
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var).mean()
    test_err = 1 - lasagne.objectives.categorical_accuracy(test_prediction,
                                                           target_var).mean()
    test_fn = theano.function([input_var, target_var], [test_loss, test_err])

    # Finally, launch the training loop.
    print("Starting training...")
    start_time = time.time()
    best_val_loss = 0
    best_val_acc = 0
    best_val_err = 0
    if save_errors:
        errors = []
    for epoch in range(nepochs):
        # shrink learning rate at 50% and 75% into training
        if epoch == (nepochs // 2) or epoch == (nepochs * 3 // 4):
            eta.set_value(eta.get_value() * lasagne.utils.floatX(0.1))

        # In each epoch, we do a full pass over the training data:
        train_loss = 0
        train_batches = len(X_train) // batchsize
        batches = cifar10.iterate_minibatches(X_train, y_train, batchsize,
                                              shuffle=True)
        if augment:
            batches = cifar10.augment_minibatches(batches)
            batches = generate_in_background(batches)
        batches = progress.progress(
                batches, desc='Epoch %d/%d, Batch ' % (epoch + 1, nepochs),
                total=train_batches)
        for inputs, targets in batches:
            train_loss += train_fn(inputs, targets)

        val_loss = 0
        val_err = 0
        val_batches = len(X_val) // batch_size_valid
        for inputs, targets in cifar10.iterate_minibatches(X_val, y_val,
                                                           batch_size_valid,
                                                           shuffle=False):
            loss, err = test_fn(inputs, targets)
            val_loss += loss
            val_err += err

        # Then we print the results for this epoch:
        train_loss /= train_batches
        l2_loss = l2_fn()
        print("  training loss:\t%.6f" % train_loss)
        print("  L2 loss:      \t%.6f" % l2_loss)
        if save_errors:
            errors.extend([train_loss, l2_loss])
        val_loss /= val_batches
        val_err /= val_batches
        val_acc = 1 - val_err
        print("  validation loss:\t%.6f" % val_loss)
        print("  validation error:\t%.2f%%" % (val_err * 100))
        if save_errors:
            errors.extend([val_loss, val_err])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_err = val_err
            best_val_loss = val_loss

        if time.time() - start_time > time_limit:
            break

    # Optionally, we dump the network weights to a file
    if save_weights:
        np.savez(save_weights, *lasagne.layers.get_all_param_values(network))

    # Optionally, we dump the learning curves to a file
    if save_errors:
        errors = np.asarray(errors).reshape(nepochs, -1)
        np.savez(save_errors, errors=errors)

    return (best_val_loss, best_val_err)

if __name__ == "__main__":
    hyperband(max_iter=60, eta=3, unit=5, resource="time")
