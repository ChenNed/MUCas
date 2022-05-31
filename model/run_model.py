import numpy as np
import math
import tensorflow as tf
from model.mucas import Model
import joblib
import time
import random

tf.set_random_seed(0)

n_time_interval = 5
n_steps = n_time_interval
n_nodes = 100
max_order = 3
iteration_number = 4
emb_size = 50

tf.flags.DEFINE_integer("batch_size", 64, "batch size.")
tf.flags.DEFINE_integer("n_steps", n_steps, "num of step.")
tf.flags.DEFINE_integer("n_time_interval", n_time_interval, "the number of  time interval.")
tf.flags.DEFINE_integer("num_nodes", n_nodes, "number of max nodes in cascade.")
tf.flags.DEFINE_integer("emb_size", emb_size, "embedding size of position embedding.")
tf.flags.DEFINE_string('order_level', 'graph-level', 'order mask.')

# MGCN
tf.flags.DEFINE_integer("feat_in", n_nodes, "num of feature in.")
tf.flags.DEFINE_integer("feat_out", 60, "num of feature out")
tf.flags.DEFINE_integer("max_order", max_order, "num of step.")
tf.flags.DEFINE_bool('whether_mask', False, 'whether use order mask')

# node-level capsule
tf.flags.DEFINE_integer("node_emb_size", 30, "nodes embedding size.")
tf.flags.DEFINE_integer("node_iter", iteration_number, "num of iteration in node-level capsule.")

# graph-level capsule
tf.flags.DEFINE_integer("graph_emb_size", 8, "graph embedding size.")
tf.flags.DEFINE_integer("graph_iter", iteration_number, "num of iteration in graph-level capsule.")

# final capsule
tf.flags.DEFINE_integer("final_size", 16, "final embedding size.")
tf.flags.DEFINE_integer("final_iter", iteration_number, "num of iteration in final capsule.")

# prediction layer
tf.flags.DEFINE_integer("n_hidden_dense1", 8, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")

# loss
tf.flags.DEFINE_float("lambda_val", 1, "Lambda factor for margin loss.")
tf.flags.DEFINE_float("reg_scale", 0.1, "Regualar scale. beta")

tf.flags.DEFINE_integer("display_step", 100, "display step.")

tf.flags.DEFINE_float("learning_rate", 0.0005, "learning_rate.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", 1e-3, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_integer("training_iters", 200 * 3200 + 1, "max training iters.")
tf.flags.DEFINE_integer("cl_decay_steps", 1000, "cl_decay_steps .")
tf.flags.DEFINE_string("version", "v1", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 5, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")

data = 'Weibo'
#data = 'APS'
time_label = 0.5
# paths
tf.flags.DEFINE_string('input', '../data/' + data + '/' + str(time_label) + '/', 'Pre-training data path.')

if data == 'Weibo':
    tf.flags.DEFINE_integer('observation', int(time_label * 3600) - 1, 'Observation time.')
elif data == 'APS':
    tf.flags.DEFINE_integer('observation', time_label * 365, 'Observation time.')

config = tf.flags.FLAGS

print("The learning rate is: ", config.learning_rate)


def get_batch(id_list, x, support, adj, pos, y, y_c, time_interval, k, rnn_index, max_order, order_level, step,
              batch_size):
    batch_y = np.zeros(shape=(batch_size, 1))
    batch_x = []
    batch_support = []
    batch_rnn_index = []
    batch_time_interval = []
    batch_k = []
    batch_adj = []
    batch_y_c = []
    batch_id = []
    batch_pos = []
    start = step * batch_size % len(x)
    for i in range(batch_size):
        id = (i + start) % len(x)
        batch_y[i, 0] = y[id]
        batch_id.append(id_list[id])
        batch_y_c.append(y_c[id])
        L = []
        for s in support[id]:
            L.append(s.todense())
        batch_support.append(L)
        batch_adj.append(adj[id].todense())
        temp_x = []
        for m in range(len(x[id])):
            temp_x.append(x[id][m].todense())

        temp_p = []
        for p in range(len(pos[id])):
            temp_p.append(pos[id][p].todense())

        batch_x.append(temp_x)
        batch_pos.append(temp_p)

        batch_rnn_index.append(rnn_index[id])
        batch_time_interval.append(time_interval[id])
        k_ = []
        if order_level == 'graph-level':
            for j in k[id]:
                k_0 = np.zeros(max_order)
                index = int(j)
                if index > max_order:
                    index = max_order
                k_0[:index] = 1
                k_.append(k_0)
        else:
            for j in k[id]:
                o = []
                for m in j:
                    k_0 = np.zeros(max_order)
                    index = int(m)
                    if index > max_order:
                        index = max_order
                    k_0[:index] = 1
                    o.append(k_0)
                k_.append(o)
        batch_k.append(k_)  # graph-level(B,T,M), nodes-level(B,T,N,M)

    return batch_x, batch_support, batch_y, batch_y_c, batch_rnn_index, batch_time_interval, batch_k, batch_adj, batch_pos, batch_id


version = config.version

id_train, x_train, support_train, y_train, y_c_train, rnn_train, order_train, time_train, adj_train, pos_train = joblib.load(
    open(config.input + 'train_' + str(config.emb_size) + '_' + str(n_time_interval) + '.pkl', 'rb'))
id_test, x_test, support_test, y_test, y_c_test, rnn_test, order_test, time_test, adj_test, pos_test = joblib.load(
    open(config.input + 'test_' + str(config.emb_size) + '_' + str(n_time_interval) + '.pkl', 'rb'))
id_val, x_val, support_val, y_val, y_c_val, rnn_val, order_val, time_val, adj_val, pos_val = joblib.load(
    open(config.input + 'val_' + str(config.emb_size) + '_' + str(n_time_interval) + '.pkl', 'rb'))


def shuffle_dataset(id_train, x_train, support_train, y_train, y_c_train, rnn_train, order_train, time_train, adj_train,
                    pos_train):
    couple = list(
        zip(id_train, x_train, support_train, y_train, y_c_train, rnn_train, order_train, time_train, adj_train,
            pos_train))
    random.shuffle(couple)
    return zip(*couple)


training_iters = config.training_iters
batch_size = config.batch_size
display_step = min(config.display_step, len(id_train) / batch_size)

# determine the way floating point numbers,arrays and other numpy object are displayed
np.set_printoptions(precision=2)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start_all = time.time()
model = Model(config, sess)
sess.graph.finalize()

step = 0
best_val_loss = 1000
best_test_loss = 1000
best_test_mape = 1000
best_test_r2 = 1000

train_writer = tf.summary.FileWriter("./train", sess.graph)

# Keep training until reach max iterations or max_try
max_try = 10
patience = max_try

train_drop = 0.5
else_drop = 0
epochs = 1000

save_predictions = list()
save_labels = list()
save_representation = list()

for epoch in range(epochs):
    start = time.time()
    train_loss = []
    train_loss_total = []
    id_train, x_train, support_train, y_train, y_c_train, rnn_train, order_train, time_train, adj_train, pos_train = shuffle_dataset(
        id_train, x_train, support_train, y_train, y_c_train, rnn_train, order_train, time_train, adj_train, pos_train
    )
    for step in range(int(len(x_train) / batch_size)):
        batch_x, batch_s, batch_y, batch_y_c, batch_rnn_index, batch_time_train, batch_k_train, batch_adj_train, batch_pos_train, _ = get_batch(
            id_train,
            x_train,
            support_train,
            adj_train,
            pos_train,
            y_train,
            y_c_train,
            time_train,
            order_train,
            rnn_train,
            max_order,
            config.order_level,
            step,
            batch_size)
        # time_decay = model.train_batch(batch_x, batch_s, batch_y, batch_y_c, batch_k_train, batch_time_train,
        #                                batch_rnn_index, batch_adj_train, train_drop)
        model.train_batch(batch_x, batch_s, batch_y, batch_y_c, batch_k_train, batch_time_train, batch_rnn_index,
                          batch_adj_train, batch_pos_train, train_drop)
        train_loss.append(
            model.get_error(batch_x, batch_s, batch_y, batch_y_c, batch_k_train, batch_time_train, batch_rnn_index,
                            batch_adj_train, batch_pos_train, train_drop)[0])
        train_loss_total.append(
            model.get_loss(batch_x, batch_s, batch_y, batch_y_c, batch_k_train, batch_time_train, batch_rnn_index,
                           batch_adj_train, batch_pos_train, train_drop))

    val_loss = []
    for val_step in range(int(len(y_val) / batch_size)):
        val_x, val_s, val_y, val_y_c, val_rnn_index, val_time, val_k, val_adj, val_pos, _ = get_batch(
            id_val,
            x_val,
            support_val,
            adj_val,
            pos_val,
            y_val,
            y_c_val,
            time_val,
            order_val,
            rnn_val,
            max_order,
            config.order_level,
            val_step,
            batch_size)
        val_loss.append(
            model.get_error(val_x, val_s, val_y, val_y_c, val_k, val_time, val_rnn_index, val_adj, val_pos,
                            else_drop)[0])

    # test
    test_loss = []
    test_mape = []
    test_r2 = []
    predict_result = []
    represention = []
    y_list = []
    for test_step in range(int(len(y_test) / batch_size + 1)):
        test_x, test_s, test_y, test_y_c, test_rnn_index, test_time, test_k, test_adj, test_pos, test_id = get_batch(
            id_test,
            x_test,
            support_test,
            adj_test,
            pos_test,
            y_test,
            y_c_test,
            time_test,
            order_test,
            rnn_test,
            max_order,
            config.order_level,
            test_step,
            batch_size)

        predict_result.extend(
            model.predict(test_x, test_s, test_y, test_y_c, test_k, test_time, test_rnn_index, test_adj,
                          test_pos, else_drop))
        y_list.extend(test_y)
        represention.extend(
            model.get_represenattion(test_x, test_s, test_y, test_y_c, test_k, test_time, test_rnn_index, test_adj,
                                     test_pos, else_drop))

    report_loss = np.mean(np.square(np.array(predict_result) - np.array(y_list)))
    report_mape = np.mean(np.abs(np.subtract(predict_result, y_list)) / (
            (np.abs(predict_result) + np.abs(y_list)) / 2))
    rss = np.mean(np.square(np.subtract(predict_result, y_list)))
    tss = np.mean(np.square(y_list - np.mean(y_list)))
    r2_score = 1 - rss / tss

    template = 'MUCas: Training Epoch {:3}, Time: {:.3f}s, Train---Total: {:.3f} / Loss: {:.3f}, Val Loss: {:.3f}, ' \
               'Test Loss: {:.3f}, MAPE: {:.3f}, R2: {:.3f}'
    print(template.format(epoch + 1, time.time() - start,
                          np.mean(train_loss_total), np.mean(train_loss), np.mean(val_loss), report_loss, report_mape, r2_score))

    if np.mean(val_loss) < best_val_loss:
        best_val_loss = np.mean(val_loss)
        patience = max_try

        save_predictions = predict_result
        save_labels = y_list
        save_representation = represention
        test_loss = []
        test_mape = []
        test_r2 = []
        cap_rep = []
        id_list = []

    if patience == 0:
        report_loss = np.mean(np.square(np.array(save_predictions) - np.array(save_labels)))
        report_mape = np.mean(np.abs(np.subtract(save_predictions, save_labels)) / (
                (np.abs(save_predictions) + np.abs(save_labels)) / 2))
        rss = np.mean(np.square(np.subtract(save_predictions, save_labels)))
        tss = np.mean(np.square(save_labels - np.mean(save_labels)))
        r2_score = 1 - rss / tss
        joblib.dump((id_test, save_predictions, save_labels, save_representation), open(
            config.input + "prediction_result_" + str(
                config.batch_size) + "_MuCas_" + config.version + '_' + str(
                max_order), 'wb'))

        print(
            'Predictions saved! Best Test MSLE: {:3}, MAPE: {:3}, R2: {:3}'.format(report_loss, report_mape, r2_score))
        break
    else:
        patience -= 1

print("Finished!\n----------------------------------------------------------------")
print('Finished! Time used: {:.3f}min'.format((time.time() - start_all) / 60))
