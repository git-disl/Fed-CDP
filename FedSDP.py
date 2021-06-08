from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import time
import struct
import pickle
import copy


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



rounds = 102
batch_size = 600

l2_norm_clip = 4

per_clip = l2_norm_clip/batch_size


noise_multiplier = 0.5
learning_rate = 0.05

# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
# from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(16, 3,
#                            strides=2,
#                            padding='same',
#                            activation='relu',
#                            input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Conv2D(32, 3,
#                            strides=2,
#                            padding='valid',
#                            activation='relu'),
#     tf.keras.layers.MaxPool2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


# ######################## without bias
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",use_bias=False),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax",use_bias=False),
])



# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(10, activation="softmax"),
# ])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(600, activation='relu',use_bias=False),
#     tf.keras.layers.Dense(100, activation='relu',use_bias=False),
#     tf.keras.layers.Dense(10, activation='softmax',use_bias=False)
# ])


def read(dataset = "training", path = "."):

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")

    print(fname_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


    # Reshape and normalize

    img = np.reshape(img, [img.shape[0], img.shape[1], img.shape[2],1])*1.0/255.0
    #img = np.reshape(img, [img.shape[0], img.shape[1]* img.shape[2]]) * 1.0 / 255.0

    return img, lbl



def get_data():
    # load the data
    x_train, y_train = read('training', './MNIST_original')
    x_test, y_test = read('testing', './MNIST_original')

    # create validation set
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # create test_set
    x_train = x_train[:50000].astype(float)
    y_train = y_train[:50000].astype(float)

    # sort test set (to make federated learning non i.i.d.)
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train])
    sorted_y_train = list(y_train[indices_train])

    # create a test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test



total_client_num=1000
parti_client_num=100
client_batch=5
total_local_iter=100
client_set = pickle.load(open('./clients/' + str(total_client_num) + '_clients.pkl', 'rb'))
sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test = get_data()

data_set_asarray = np.asarray(sorted_x_train)
label_set_asarray = np.asarray(sorted_y_train)


#y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=10)
#y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=10)


accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
accuracy_test = tf.keras.metrics.SparseCategoricalAccuracy()

######### for private mode;
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
######## for benign model
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.05,momentum=0.9)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)



# dp_optimizer = DPGradientDescentGaussianOptimizer(
#     l2_norm_clip=l2_norm_clip,
#     noise_multiplier=noise_multiplier,
#     num_microbatches=batch_size,
#     learning_rate=learning_rate)
# dp_loss = tf.keras.losses.CategoricalCrossentropy(
#     from_logits=True, reduction=tf.losses.Reduction.NONE)
#

model.build(np.asarray(x_test).shape)


new_global_model = copy.deepcopy(model)
starting_time = time.time()
for round in range(rounds):
    # Iterate over the batches of a dataset.

    perm = np.random.permutation(total_client_num)
    s = perm[0:parti_client_num].tolist()

    participating_clients_data = [client_set[k] for k in s]



    old_global_model = copy.deepcopy(new_global_model)



    for k_t in range(parti_client_num):
        #data_ind = np.split(np.asarray(participating_clients_data[k_t]), client_batch, 0)
        data_ind = np.split(np.asarray(participating_clients_data[k_t]), 100, 0)
        #data_ind = np.split(np.asarray(participating_clients_data[k_t]), 50, 0)

        model =copy.deepcopy(old_global_model)
        #for local_iter in range(total_local_iter):
        l2_norm_list_l1=[]
        l2_norm_list_l2=[]
        l2_norm_list_l3=[]

        for local_iter in range(len(data_ind)):

            batch_ind=data_ind[local_iter]

            x = data_set_asarray[[int(j) for j in batch_ind]]
            y = label_set_asarray[[int(j) for j in batch_ind]]



            #######################################################################
            ##  benign model and fed-sdp (client level differential privacy protection)
            ## to make it benign, comment the lines for clipping and add a 0 to noise injection
            with tf.GradientTape(persistent=True) as tape:
                    logits_all = model(x)
                    loss_value = loss_fn(y, logits_all)
                    gradients = tape.gradient(loss_value, model.trainable_weights)
                    Sanitized_gradients=gradients

            ####################################
                    ######print noisy gradients

            for i, item in enumerate(Sanitized_gradients):
                l2norm = tf.norm(tf.reshape(item, [1, -1]))
                #l2norm_mean = tf.reduce_mean(tf.reshape(item, [1, -1]))
                #l2norm_std = tf.math.reduce_std(tf.reshape(item, [1, -1]))

                if i==0:
                    l2_norm_list_l1.append(l2norm.numpy())
                if i == 1:
                    l2_norm_list_l2.append(l2norm.numpy())
                if i == 2:
                    l2_norm_list_l3.append(l2norm.numpy())


            optimizer.apply_gradients(zip(Sanitized_gradients, model.trainable_weights))

        if k_t==0:
            l2_norm_list_all_l1= np.asarray(l2_norm_list_l1)
            l2_norm_list_all_l2= np.asarray(l2_norm_list_l2)
            l2_norm_list_all_l3= np.asarray(l2_norm_list_l3)
        else:
            l2_norm_list_all_l1 = np.vstack([np.asarray(l2_norm_list_all_l1),np.asarray(l2_norm_list_l1)])
            l2_norm_list_all_l2 = np.vstack([np.asarray(l2_norm_list_all_l2), np.asarray(l2_norm_list_l2)])
            l2_norm_list_all_l3 = np.vstack([np.asarray(l2_norm_list_all_l3),np.asarray(l2_norm_list_l3)])

        num_weights = len(gradients)

        delta_model =  [model.get_weights()[i] - old_global_model.get_weights()[i] for i in range(num_weights)]

        # ############if clipping
        # norms = [tf.norm(delta_model[i]) for i in range(num_weights)]
        # factors = [norms[i] / l2_norm_clip for i in range(num_weights)]
        # delta_model = [delta_model[i] / np.max([1, factors[i]]) for i in range(num_weights)]

        if k_t == 0:
            model_update= delta_model
        else:
            model_update = [model_update[i] + delta_model[i] for i in range(num_weights)]

    meanclippedupdate = [model_update[i]/k_t for i in range(num_weights)]



    GaussianNoises = [0*1/k_t * np.random.normal(loc=0.0, scale=float(noise_multiplier * l2_norm_clip),
                                                    size=gradients[i].shape) for i in
                range(num_weights)]  # layerwise gaussian noise



    Sanitized_updates = [meanclippedupdate[i] + GaussianNoises[i] for i in range(num_weights)]  # add gaussian noise


    new_global_model.set_weights([old_global_model.get_weights()[i] + Sanitized_updates[i] for i in range(num_weights)])

    if round==0:
        l2_norm_list_mean_l1 = np.mean(l2_norm_list_all_l1, axis=0)
        l2_norm_list_mean_l2 = np.mean(l2_norm_list_all_l2, axis=0)
        l2_norm_list_mean_l3 = np.mean(l2_norm_list_all_l3, axis=0)
    else:
        l2_norm_list_mean_l1 = np.concatenate((l2_norm_list_mean_l1, np.mean(l2_norm_list_all_l1, axis=1)),axis=None)
        l2_norm_list_mean_l2 = np.concatenate((l2_norm_list_mean_l2, np.mean(l2_norm_list_all_l2, axis=1)),axis=None)
        l2_norm_list_mean_l3 = np.concatenate((l2_norm_list_mean_l3, np.mean(l2_norm_list_all_l3, axis=1)),axis=None)

    print (l2_norm_list_mean_l1.shape)



    cur_time = time.time()
    duration = cur_time-starting_time
    if local_iter % 1 == 0:
        # for Sanitized_gradient in Sanitized_gradients:
        #    print(anitized_gradient)
        # for item in norms:
        #   print(item)
        # print (l2_norm_clip)

        # Update the state of the `accuracy` metric.
        logits_all = new_global_model(np.asarray(sorted_x_train)[:10000], 0)
        accuracy.update_state(np.asarray(sorted_y_train)[:10000],logits_all)


        #accuracy.update_state(y, logits_all)
        print("round:", round, "local iter:", local_iter)
        print("Training accuracy: %.5f" % accuracy.result())
        with open('training.csv', mode='a') as train_file:
            writer_train = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer_train.writerow([round,local_iter,accuracy.result(),duration])

    if local_iter % 1 == 0:
        #logits_test = new_global_model(np.asarray(x_test))
        #accuracy_test.update_state(np.asarray(y_test), logits_test)
        logits_test = new_global_model(np.asarray(x_vali))
        accuracy_test.update_state(np.asarray(y_vali), logits_test)

        print("round:", round, "local iter:", local_iter)
        print("Test accuracy: %.5f" % accuracy_test.result())

        with open('test.csv', mode='a') as test_file:
            writer_test = csv.writer(test_file, delimiter=',')

            writer_test.writerow([round,local_iter,accuracy_test.result()])

    with open('sigma.csv', mode='a') as sigma_file:
        writer_sigma = csv.writer(sigma_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer_sigma.writerow([epoch, step, 24/sqrt(sum(n*n for n in l2_norm_clip)/len(l2_norm_clip))])  #if per exampl
        writer_sigma.writerow([round, local_iter, 24.0 / l2_norm_clip])  # if batch

            # Reset the metric's state at the end of an global iter
    accuracy.reset_states()
    accuracy_test.reset_states()



for i in range(3):
    with open('beforesanitized_%s.csv' % i, mode='a') as norm_sani_file:
        writer_train = csv.writer(norm_sani_file, delimiter=',', quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        if i==0:
            for idx in range(l2_norm_list_mean_l1.shape[0]):
                writer_train.writerow([idx,l2_norm_list_mean_l1[idx]])
        if i == 1:
            for idx in range(l2_norm_list_mean_l2.shape[0]):
                writer_train.writerow([idx, l2_norm_list_mean_l2[idx]])
        if i == 2:
            for idx in range(l2_norm_list_mean_l3.shape[0]):
                writer_train.writerow([idx, l2_norm_list_mean_l3[idx]])



















