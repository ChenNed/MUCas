"""
Multi-scale cascade modeling
change 1: higher order GCN kernel
=>graph-level
change 2: interval-based sampling method
change 3: additional position information
change in capsule: norm() replace squash() before the last iteration / the value norm for margin loss
"""

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim
epsilon = 1e-11
m_plus = 0.9
m_minus = 0.1
regular_sc = 1e-7
defaut_initializer = tf.contrib.layers.xavier_initializer()
constant_initializer = tf.constant_initializer(0.1)


def MGN(inputs, supports_batch, output_size, p_e, mask_k, num_kernel=1, drop_out=0.0, whether_mask=True):
    batch_size, num_nodes, input_size = inputs.get_shape()
    supports_ = tf.unstack(supports_batch, batch_size, 0)  # B [2, N, N] [in/out]
    x_l = []
    # node feature
    x = inputs
    X0 = tf.unstack(x, batch_size, 0)  # B[N, F_in] input feature
    # position embedding
    pe = tf.unstack(p_e, batch_size, 0)  # B[N, F_in]
    mask = tf.unstack(mask_k, batch_size, 0)  # B[K, 1]
    # for each graph
    for supports, x0, p, m in zip(supports_, X0, pe, mask):
        # supports [2, N, N], x0 [N, F_in]
        x0 = tf.nn.dropout(x0, 1 - drop_out)
        base_features = x0
        support = tf.unstack(supports, 2, 0)  # 2[N, N]
        x_d = []

        # print('mask shape:', m.shape)
        m = m[tf.newaxis, :, :]  # [1, K, 1]
        # print('mask shape:', m.shape)
        m = tf.tile(m, [num_nodes, 1, 1])  # [N, K, 1]

        for s in range(len(support)):
            x_k = []
            for i in range(num_kernel):
                if i == 0:
                    A_0 = tf.eye(int(num_nodes))
                    base_features = tf.matmul(A_0, base_features)  # base_features = A_0 * x_0
                else:
                    base_features = tf.matmul(support[s], base_features)  # base_features = base_features * Adj
                net_p = tf.layers.dense(base_features,
                                        output_size,
                                        activation=tf.nn.relu,
                                        use_bias=False)

                x_k.append(net_p)  # K[N, F_out]
            # (((A_0 * x0) * adj))*adj)*adj
            x_k = tf.stack(x_k)  # [K, N, F_out]
            x_k = tf.transpose(x_k, [1, 0, 2])  # [N, K, F_in]
            if whether_mask:
                # m = m[tf.newaxis, :, :]  # [1, K, 1]
                # m = tf.tile(m, [num_nodes, 1, 1])  # [N, K, 1]
                x_k = x_k * m  # mask order
            x_d.append(x_k)  # 2[N, K, F_in]
        # positional encoding concatenate
        p = p[:, tf.newaxis, :]  # [ N, K, F]
        p = tf.tile(p, [1, num_kernel, 1])
        x_c = tf.concat((x_d[0], x_d[1], p), 2)  # [N, K, 2F_out +  F_e]
        x_l.append(x_c)  # B[N, K, 2F_o + F_e]
    out = tf.stack(x_l)  # [B, N, K, 2F_out]

    return out


def MGN1(inputs, supports_batch, output_size, p_e, mask_k, num_kernel=1, drop_out=0.0, whether_mask=True):
    batch_size, num_nodes, input_size = inputs.get_shape()
    batch_size_, num_nodes_, emb_size = p_e.get_shape()

    supports_ = tf.unstack(supports_batch, batch_size, 0)  # B [2, N, N] [in/out]
    x_l = []
    # node feature
    x = inputs
    X0 = tf.unstack(x, batch_size, 0)  # B[N, F_in] input feature
    # position embedding
    pe = tf.unstack(p_e, batch_size, 0)  # B[N, F_in]
    mask = tf.unstack(mask_k, batch_size, 0)  # B[K, 1]
    with tf.variable_scope('MGN') as scope:
        # for each graph
        try:
            Wp = tf.get_variable("Wp", [emb_size, output_size], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            Wk = tf.get_variable("Wk", [num_kernel, input_size, output_size], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        except ValueError:
            scope.reuse_variables()
            Wp = tf.get_variable("Wp", [emb_size, output_size], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            Wk = tf.get_variable("Wk", [num_kernel, input_size, output_size], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        for supports, x0, p, m in zip(supports_, X0, pe, mask):
            # supports [2, N, N], x0 [N, F_in]
            x0 = tf.nn.dropout(x0, 1 - drop_out)
            base_features = x0
            support = tf.unstack(supports, 2, 0)  # 2[N, N]
            x_d = []

            # print('mask shape:', m.shape)
            m = m[tf.newaxis, :, :]  # [1, K, 1]
            # print('mask shape:', m.shape)
            m = tf.tile(m, [num_nodes, 1, 1])  # [N, K, 1]

            for s in range(len(support)):
                x_k = []
                for i in range(num_kernel):
                    if i == 0:
                        A_0 = tf.eye(int(num_nodes))
                        base_features = tf.matmul(A_0, base_features)  # base_features = A_0 * x_0
                    else:
                        base_features = tf.matmul(support[s], base_features)  # base_features = base_features * Adj
                    x_k.append(base_features)  # K[N, F_in]
                # (((A_0 * x0) * adj))*adj)*adj
                x_k = tf.stack(x_k)  # [K, N, F_in]
                #print(x_k.shape)
                x_k = tf.matmul(x_k, Wk) # [K, N, F_out]
                x_k = tf.transpose(x_k, [1, 0, 2])  # [N, K, F_in]
                if whether_mask:
                    # m = m[tf.newaxis, :, :]  # [1, K, 1]
                    # m = tf.tile(m, [num_nodes, 1, 1])  # [N, K, 1]
                    x_k = x_k * m  # mask order
                x_d.append(x_k)  # 2[N, K, F_in]
            # positional encoding concatenate
            p = tf.matmul(p, Wp)
            p = p[:, tf.newaxis, :]  # [ N, K, F]
            p = tf.tile(p, [1, num_kernel, 1])
            x_c = tf.concat((x_d[0], x_d[1], p), 2)  # [N, K, 2F_out +  F_e]
            x_l.append(x_c)  # B[N, K, 2F_o + F_e]
        out = tf.stack(x_l)  # [B, N, K, 2F_out]

    return tf.nn.relu(out)


def node_level_capsule(inputs, Ci, Co, in_emb_size, node_emb_size, iterations,
                       batch_size, coordinate):
    """
    :param inputs: (B, N, K, di)
    :param Ci:M input capsule
    :param Co:1 output capsule
    :param in_emb_size: di
    :param out_emb_size: d
    :param iterations:
    :param position_emb_size: d'
    :param batch_size:
    :param name:
    :param shared:
    :param with_position: bool
    :return: (B, N, do)
    """

    inputs = tf.expand_dims(inputs, 2)  # (B, N, 1, K, di)
    inputs_poses = tf.reshape(inputs, [-1, 1, Ci, in_emb_size])  # (B*N, 1, K, di)
    i_size = Ci  # K
    out_emb_size = node_emb_size  # do
    o_size = Co  # 1
    in_emb_size = in_emb_size  # di
    N = tf.shape(inputs)[1]  # N
    batch_ = tf.shape(inputs_poses)[0]  # B*N

    with tf.variable_scope('votes'):
        votes = mat_transform_with_coordinate(
            input=inputs_poses,
            Co=Co,
            in_emb_size=in_emb_size,
            out_emb_size=out_emb_size,
            batch_size=batch_,
            num_node=1,
            coordinate=coordinate
        )  # (B*N, 1, 1, 1, K, 1, do) # calculating votes
        votes = tf.reshape(votes,
                           shape=[batch_, 1, 1, tf.shape(votes)[3] * tf.shape(votes)[4], o_size,
                                  out_emb_size])  # (B*N, 1, 1, K, 1, do)

    with tf.variable_scope('routing'):
        scope = tf.get_variable_scope()
        try:
            bias = slim.variable(
                'bias',
                shape=[1, 1, 1, 1, Co, out_emb_size], dtype=tf.float32,
                initializer=constant_initializer)  # (?, 1, 1, 1, Co, do)
        except ValueError:
            scope.reuse_variables()
            bias = slim.variable(
                'bias',
                shape=[1, 1, 1, 1, Co, out_emb_size], dtype=tf.float32,
                initializer=constant_initializer)  # (1, 1, 1, Ci, Co, di, do)
        bias_ = tf.tile(bias, [batch_, 1, 1, 1, 1, 1])
        b_IJ = tf.zeros(shape=[batch_, 1, 1, 1 * i_size, Co, 1], dtype=tf.float32)  # (B*N, 1, 1, K, 1, 1)
        v_j, a_j = routing_graph(
            votes=votes,
            b_Ij=b_IJ,
            num_nodes=1,
            bias=bias_,
            iterations=iterations)  # (?, 1, 1, 1, 1, do) # routing aggregation
        v_j = tf.reshape(v_j, shape=[batch_size, N, Co * out_emb_size])  # (B, N, do)

    return v_j


def graph_level_capsule(inputs, Ci, Co, in_emb_size, graph_emb_size, iterations, position_emb_size, nodes_indicator,
                        batch_size, coordinate):
    """
    :param inputs: (?, 1, N, di)
    :param Ci: N
    :param Co: 1
    :param in_emb_size:
    :param out_emb_size:
    :param iterations:
    :param position_emb_size:
    :param nodes_indicator: (?, N, 1)
    :param batch_size:
    :param name:
    :param shared:
    :param with_position: bool
    :return:
    """
    inputs_poses = inputs[:, tf.newaxis, :, :]
    if coordinate:
        i_size = Ci - 1
        out_emb_size = graph_emb_size + position_emb_size
    else:
        i_size = Ci  # N
        out_emb_size = graph_emb_size
    o_size = Co  # Co 1
    in_emb_size = in_emb_size  # di
    N = tf.shape(inputs_poses)[1]

    with tf.variable_scope('votes'):
        votes = mat_transform_with_coordinate(
            input=inputs_poses,
            Co=Co,
            in_emb_size=in_emb_size,
            out_emb_size=graph_emb_size,
            batch_size=batch_size,
            num_node=N,
            position_emb_size=position_emb_size,
            coordinate=coordinate
        )  # (batch, 1, 1, N, i_size, Co, d)
        votes = tf.reshape(votes,
                           shape=[batch_size, 1, 1, tf.shape(votes)[3] * tf.shape(votes)[4], o_size,
                                  out_emb_size])  # (batch, 1, 1, N*i_size, Co, d)

    with tf.variable_scope('routing'):
        scope = tf.get_variable_scope()
        nodes_indicator = tf.reduce_max(nodes_indicator, axis=-1, keep_dims=True)  # (?, N, 1)
        num_nodes = tf.reduce_sum(nodes_indicator, axis=1, keep_dims=True)  # (?, #1, 1)
        num_nodes = num_nodes[:, tf.newaxis, tf.newaxis, :, tf.newaxis, :]  # (?, 1, 1, #1, 1, 1)
        try:
            bias = slim.variable(
                'bias',
                shape=[1, 1, 1, 1, Co, out_emb_size], dtype=tf.float32,
                initializer=constant_initializer)  # (?, 1, 1, 1, Co, do)
        except ValueError:
            scope.reuse_variables()
            bias = slim.variable(
                'bias',
                shape=[1, 1, 1, 1, Co, out_emb_size], dtype=tf.float32,
                initializer=constant_initializer)  # (1, 1, 1, Ci, Co, di, do)
        bias_ = tf.tile(bias, [batch_size, 1, 1, 1, 1, 1])
        b_IJ = tf.zeros(shape=[batch_size, 1, 1, N * i_size, Co, 1], dtype=tf.float32)  # (?, 1, 1, N*Ci, Co, 1)
        v_j, a_j = routing_graph(
            votes=votes,
            b_Ij=b_IJ,
            num_nodes=num_nodes,
            bias=bias_,
            iterations=iterations)  # (?, 1, 1, 1, Co, d)
        v_j = tf.reshape(v_j, shape=[batch_size, -1, out_emb_size])  # (?, 1, Co, d)
        a_j = tf.reshape(a_j, shape=[batch_size, 1, Co, 1])

    return v_j, a_j


def class_capsules(inputs_poses, graph_emb_size, num_classes, iterations, batch_size):
    """
    :param inputs: (?, 1, C, d)
    :param num_classes: o
    :param iterations: 3
    :param batch_size: ?
    :param name:
    :return poses, activations: poses (?, num_classes, 1, d), activation (?, num_classes).
    """
    inputs_poses = inputs_poses
    inputs_shape = inputs_poses.get_shape()
    in_emb_size = int(inputs_shape[-1])  # d
    N = tf.shape(inputs_poses)[1]
    i_size = tf.shape(inputs_poses)[2]
    with tf.variable_scope('votes'):
        votes = mat_transform_with_coordinate(
            input=inputs_poses,
            Co=num_classes,
            in_emb_size=in_emb_size,
            out_emb_size=graph_emb_size,
            batch_size=batch_size,
            num_node=N
        )  # (batch, 1, 1, 1, Ci, Co, d)
        votes = tf.squeeze(votes, axis=3)  # (?, 1, 1, Ci, Co, d)

    with tf.variable_scope('routing'):
        scope = tf.get_variable_scope()
        try:
            bias = slim.variable(
                'bias',
                shape=[1, 1, 1, 1, num_classes, graph_emb_size], dtype=tf.float32,
                initializer=constant_initializer)  # (?, 1, 1, 1, Co, do)
        except ValueError:
            scope.reuse_variables()
            bias = slim.variable(
                'bias',
                shape=[1, 1, 1, 1, num_classes, graph_emb_size], dtype=tf.float32,
                initializer=constant_initializer)  # (1, 1, 1, Ci, Co, di, do)
        bias_ = tf.tile(bias, [batch_size, 1, 1, 1, 1, 1])
        num_nodes = tf.ones(shape=[batch_size, 1, 1, 1, 1, 1], dtype=tf.float32)
        b_IJ = tf.zeros(shape=[batch_size, 1, 1, i_size, num_classes, 1],
                        dtype=tf.float32)  # (?, 1, 1, N*Ci, Co, 1)
        v_j, a_j = routing_graph(votes=votes, b_Ij=b_IJ, num_nodes=num_nodes, bias=bias_,
                                 iterations=iterations)  # (?, 1, 1, 1, Co, d)
        v_j = tf.reshape(v_j, shape=[batch_size, 1, num_classes, graph_emb_size])  # (?, 1, Co, d)
        a_j = tf.reshape(a_j, shape=[batch_size, 1, num_classes, 1])
    return v_j, a_j


def mat_transform_with_coordinate(input, Co, in_emb_size, out_emb_size, batch_size, num_node, position_emb_size=0,
                                  coordinate=False):
    """
    :param input: (?, N, Ci, d)
    :param Co:
    :param in_emb_size:
    :param out_emb_size:
    :param batch_size:
    :param num_node:
    :param position_emb_size:
    :param corordinate:
    :return: (batch, 1, 1, N, Ci, Co, d)
    """

    if coordinate:
        # aims to preserve positional information
        input_shape = input.get_shape()
        Ci = input_shape[2]
        output = input[:, :, tf.newaxis, :, tf.newaxis, tf.newaxis, :]  # (batch, N, 1, Ci, 1, 1, 16)
        properties = output[:, :, :, :-1, :, :, :]  # (batch, N, 1, Ci-1 , 1, 1, 16)
        position = output[:, :, :, -1, :, :, :]  # (batch, N, 1, 1, 1, 16)
        position = tf.expand_dims(position, axis=3)  # (batch, N, 1, 1, 1, 1, 16)

        scope = tf.get_variable_scope()
        try:
            w_pro = slim.variable(
                'w_pro',
                shape=[1, 1, 1, Ci - 1, Co, in_emb_size, out_emb_size], dtype=tf.float32,
                initializer=defaut_initializer,
                regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, Ci-1, Co, d, d)
            w_pos = slim.variable(
                'w_pos',
                shape=[1, 1, 1, 1, Co, in_emb_size, position_emb_size], dtype=tf.float32,
                initializer=defaut_initializer,
                regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, 1, Co, d, d)
        except ValueError:
            scope.reuse_variables()
            w_pro = slim.variable(
                'w_pro',
                shape=[1, 1, 1, Ci - 1, Co, in_emb_size, out_emb_size], dtype=tf.float32,
                initializer=defaut_initializer,
                regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, Ci-1, Co, d, d)
            w_pos = slim.variable(
                'w_pos',
                shape=[1, 1, 1, 1, Co, in_emb_size, position_emb_size], dtype=tf.float32,
                initializer=defaut_initializer,
                regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, 1, Co, d, d)

        w_pro = tf.tile(w_pro, [batch_size, num_node, 1, 1, 1, 1, 1])  # (batch, N, 1, Ci-1, Co, d, d)

        w_pos = tf.tile(w_pos, [batch_size, num_node, 1, 1, 1, 1, 1])  # (batch, N, 1, 1, Co, d, d)

        properties = tf.tile(properties, [1, 1, 1, 1, Co, 1, 1])  # (batch, N, 1, Ci-1, Co, 1, d)
        position = tf.tile(position, [1, 1, 1, 1, Co, 1, 1])  # (batch, N, 1, 1, Co, 1, d)
        votes_properties = tf.matmul(properties, w_pro, name='Trans')  # (batch, N, 1, Ci-1 , Co, 1, d)
        votes_positions = tf.tile(tf.matmul(position, w_pos),
                                  multiples=[1, 1, 1, Ci - 1, 1, 1, 1])  # (batch, N, 1, Ci-1, Co, 1, d)
        votes = tf.concat([votes_properties, votes_positions], axis=-1)
        votes = tf.reshape(votes, [batch_size, 1, 1, num_node, Ci - 1, Co,
                                   position_emb_size + out_emb_size])  # (batch, 1, 1, N, Ci-1, Co, d_pro+d_pos)
    else:
        input_shape = input.get_shape()
        Ci = input_shape[2]  # M
        output = input[:, :, tf.newaxis, :, tf.newaxis, tf.newaxis, :]  # (batch, N, 1, Ci, 1, 1, d)

        scope = tf.get_variable_scope()
        try:
            w = slim.variable(
                'w',
                shape=[1, 1, 1, Ci, Co, in_emb_size, out_emb_size], dtype=tf.float32,
                initializer=defaut_initializer,
                regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, Ci, Co, di, do)
        except ValueError:
            scope.reuse_variables()
            w = slim.variable(
                'w',
                shape=[1, 1, 1, Ci, Co, in_emb_size, out_emb_size], dtype=tf.float32,
                initializer=defaut_initializer,
                regularizer=slim.l2_regularizer(regular_sc))  # (1, 1, 1, Ci, Co, di, do)

        w = tf.tile(w, [batch_size, num_node, 1, 1, 1, 1, 1])  # (batch, N, 1, Ci, Co, d, d)

        output = tf.tile(output, [1, 1, 1, 1, Co, 1, 1])  # (batch, N, 1, Ci, Co, 1, d)

        votes = tf.matmul(output, w, name='Trans')  # (batch, N, 1, Ci, Co, 1, d)
        votes = tf.reshape(votes, [batch_size, 1, 1, num_node, Ci, Co, out_emb_size])  # (batch, 1, 1, N, Ci, Co, d)

    return votes


def routing_graph(votes, b_Ij, num_nodes, bias, iterations=3):
    """
    :param votes: (?, 1, 1, Ci, Co, d)
    :param b_Ij: (?, 1, 1, Ci, Co, 1)
    :param num_nodes: (?, 1, 1, #1, 1, 1)
    :param iterations: 3
    :return:
    """
    u_hat = votes
    u_hat_stopped = tf.stop_gradient(u_hat)
    for r_iter in range(iterations):
        with tf.variable_scope('iter_' + str(r_iter)) as scope:
            c_ij = tf.nn.softmax(b_Ij, dim=4)  # (?, 1, 1, Ci, Co, 1)
            if r_iter == iterations - 1:
                s_j = tf.multiply(c_ij, u_hat)
                s_j = tf.reduce_sum(s_j, axis=3, keep_dims=True) / num_nodes + bias
                v_j, a_j = squash(s_j)  # (?, 1, 1, 1, Co, d)
            elif r_iter < iterations - 1:
                s_j = tf.multiply(c_ij, u_hat_stopped)
                s_j = tf.reduce_sum(s_j, axis=3, keep_dims=True) / num_nodes + bias
                # v_j = tf.nn.l2_normalize(s_j, axis=-1)  # (?, 1, 1, 1, Co, d)
                v_j, _ = squash(s_j)  # (?, 1, 1, 1, Co, d)
                v_j = tf.tile(v_j, [1, 1, 1, tf.shape(votes)[3], 1, 1])  # (?, 1, 1, Ci, Co, d)
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_j, axis=5,
                                            keep_dims=True)  # (?, 1, 1, Ci, Co, 1)
                b_Ij += u_produce_v

    return v_j, a_j


def squash(v_j, dim=-1):
    """
    :param v_j: (?, 1, 1, 1, Co, d)
    :param dim:
    :return:
    """
    vec_squared_norm = tf.reduce_sum(tf.square(v_j), dim, keep_dims=True)  # ||v||^2
    a_j = vec_squared_norm / (1 + vec_squared_norm)
    scalar_factor = a_j / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * v_j  # element-wise
    return vec_squashed, a_j


def norm(v_j, dim=-1):
    """
    :param v_j: (?, 1, 1, 1, Co, d)
    :param dim:
    :return: ｜｜v_j｜|
    """
    vec_squared_norm = tf.reduce_sum(tf.square(v_j), dim, keep_dims=True)  # ||v||^2
    v_norm = tf.sqrt(vec_squared_norm + epsilon)
    return v_norm  # a_j


class Model(object):
    """
    Defined:
        Placeholder
        Model architecture
        Train / Test function
    """

    def __init__(self, config, sess):
        # input
        self.batch_size = config.batch_size  # bach size
        self.num_nodes = config.num_nodes  # each sample has num_nodes
        self.n_time_interval = config.n_time_interval  # number of time intervals
        self.n_steps = config.n_steps  # "n_steps" equals to "n_time_interval" --> the sampling method based on time-interval
        self.emb_size = config.emb_size  # positional embedding size

        # MGCN
        self.feat_in = config.feat_in  # number of feature
        self.feat_out = config.feat_out  # number of output feature
        self.max_order = config.max_order  # max-order in datasets
        self.whether_mask = config.whether_mask

        # node-level capsule
        self.node_emb_size = config.node_emb_size  # node capsule size
        self.node_iter = config.node_iter  # iteration number

        # graph-level capsule
        self.graph_emb_size = config.graph_emb_size  # graph capsule size
        self.graph_iter = config.graph_iter  # iteration number

        # class-capsule
        self.final_size = config.final_size  # final capsule size
        self.final_iter = config.final_iter  # iteration number

        # loss
        self.lambda_val = config.lambda_val
        self.reg_scale = config.reg_scale

        self.sess = sess

        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu

        self.max_grad_norm = config.max_grad_norm

        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2

        self.learning_rate = config.learning_rate
        self.scale1 = config.l1
        self.scale2 = config.l2

        self.scale = config.l1l2
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.initializer2 = tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)

        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)

        self._build_placeholders()
        self._build_var()
        self._build_model()

        # Define loss and optimizer
        self.loss = self.reg_scale * self.margin_loss + (1 - self.reg_scale) * self.reg_loss + self.scale * tf.add_n(
            [self.regularizer(var) for var in tf.trainable_variables()])
        self.loss_total = self.reg_scale * self.margin_loss + (1 - self.reg_scale) * self.reg_loss

        var_list = tf.trainable_variables()

        total_para = 0
        var_list1 = [v for v in var_list if "order_level" in v.name]
        for v in var_list:
            shape = v.get_shape()
            vp = 1
            for dim in shape:
                vp *= dim.value
            total_para += vp
        print(total_para)

        # learning rate decay
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step, 20000, 0.9,
                                                   staircase=True)  # linear decay over time

        # optimization
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = tf.gradients(self.loss, var_list)
        grads_c = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads]  # 防止梯度爆炸
        self.train_op = opt.apply_gradients(zip(grads_c, var_list), global_step=self.global_step, name='train_op')
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _build_placeholders(self):

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_steps, self.num_nodes, self.feat_in],
                                name="x")  # input structure features [B, T, N, F_stu]

        self.pos_emb = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_steps, self.num_nodes, self.emb_size],
                                      name="pos_emb")  # node positional embedding [B, T, N, F_pos]

        self.supports = tf.placeholder(tf.float32, shape=[self.batch_size, 2, self.num_nodes, self.num_nodes],
                                       name="supports")  # in- / out- normalized adj [B, 2, N, N]

        self.k_order = tf.placeholder(tf.float32,
                                      shape=[self.batch_size, self.n_steps, self.max_order],
                                      name='k_order')  # order_mask [B, T, K]

        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="y")  # popularity prediction

        self.y_c = tf.placeholder(tf.int64, shape=[self.batch_size, ], name='y_c')  # outbreak prediction

        self.time_interval = tf.placeholder(tf.int32, shape=[self.batch_size, self.n_steps],
                                            name="time_interval")  # [B, T] for influence attention

        self.drop_out = tf.placeholder(tf.float32, shape=[], name="drop_out")  # MGCN drop_out

        # not use in this work
        self.adj = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_nodes, self.num_nodes],
                                  name="adj")  # normal adj
        #
        self.rnn_index = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_steps], name="rnn_index")  # [B, T]

    def _build_var(self, reuse=None):
        with tf.variable_scope('dense'):
            self.weights = {
                'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([self.final_size,
                                                                                         self.n_hidden_dense1])),
                # 'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([self.n_hidden_dense1,
                #                                                                          self.n_hidden_dense2])),
                'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense1, 1]))
            }
            self.biases = {
                'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.n_hidden_dense1])),
                # 'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense2])),
                'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
            }


        with tf.variable_scope('time_decay'):
            self.time_lambda = tf.get_variable('time_lambda', [self.n_time_interval + 1, self.node_emb_size],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               dtype=tf.float32)  # ,

    def _build_model(self, reuse=None):
        with tf.variable_scope('mucas') as scope:
            with tf.variable_scope('order_level'):
                k_order = tf.reshape(self.k_order,
                                     [self.batch_size, self.n_steps, self.max_order, -1])  # [B, T, K, 1]
                # print(k_order.shape)
                x_vector = tf.unstack(self.x, self.n_steps, 1)  # T[B, N, D]
                k_order = tf.unstack(k_order, self.n_steps, 1)  # T[B, K, 1]
                p_e = tf.unstack(self.pos_emb, self.n_steps, 1)  # T[B, N, D]
                x_T = []

                # for each sub-graph
                for x, k, p in zip(x_vector, k_order, p_e):
                    # p = tf.layers.dense(p, self.feat_out, activation=tf.nn.relu, use_bias=False)  # [B, N, F] op: WP
                    # p = tf.matmul(p, self.Wp)
                    # print('positional-encodding shape {}'.format(p.shape))

                    node_hidden = MGN1(inputs=x, supports_batch=self.supports, output_size=self.feat_out, p_e=p,
                                       mask_k=k, num_kernel=self.max_order, drop_out=self.drop_out,
                                       whether_mask=self.whether_mask)
                    # [B, N, K, 3F] -> direction-scale (in and out), position-scale

                    with tf.variable_scope("node_capsule"):
                        node_capsule = node_level_capsule(inputs=node_hidden, Ci=self.max_order, Co=1,
                                                          in_emb_size=3 * self.feat_out,
                                                          node_emb_size=self.node_emb_size,
                                                          batch_size=self.batch_size,
                                                          iterations=self.node_iter,
                                                          coordinate=False)  # (B, N, d)
                        x_T.append(node_capsule)  # T[B, N, d] aggregate order-level to form graph-level capsuel

            with tf.variable_scope('node_level'):
                G_i = []
                for x_n in x_T:
                    nets, a_j = graph_level_capsule(inputs=x_n, Ci=self.num_nodes, Co=1, in_emb_size=self.node_emb_size,
                                                    graph_emb_size=self.graph_emb_size, iterations=self.graph_iter,
                                                    nodes_indicator=self.adj, batch_size=self.batch_size,
                                                    position_emb_size=0,
                                                    coordinate=False)
                    G_i.append(nets)
                G_i_stack = tf.stack(G_i)
                print("after graph capsule: {}".format(G_i_stack.shape))  # [T, B, 1, d_g]
                G_i_stack_trans = tf.transpose(G_i_stack, [1, 2, 0, 3])  # [B, 1, T, d_g]

            with tf.variable_scope('time_decay'):
                self.time_weight = tf.nn.embedding_lookup(self.time_lambda, self.time_interval)
                time_influence = tf.layers.dense(self.time_weight, self.graph_emb_size, tf.nn.elu,
                                                 'time_influence')  # [B, T, 1]
                print(time_influence.shape)
                G_a = tf.reshape(G_i_stack_trans, [self.batch_size, -1, self.graph_emb_size])  # [B, T, d_g]
                influence_atten = tf.layers.dense(time_influence * G_a, 1)
                atten_score = tf.nn.softmax(influence_atten, 1)
                print("attention score: ", atten_score.shape)
                G_inp = G_a * atten_score
                G_inp = tf.reshape(G_inp, [self.batch_size, 1, -1, self.graph_emb_size])
                print("graph capsule input: ", G_inp.shape)

            with tf.variable_scope('graph_capsule'):
                nets_2, a_j_2 = class_capsules(
                    G_inp,
                    num_classes=2, iterations=self.final_iter, graph_emb_size=self.final_size,
                    batch_size=self.batch_size)  # (?, num_class, 1, d_final)/（？，num_class）
                print("after final capsule:{}/{} ".format(nets_2.shape, a_j_2.shape))

            # classification loss
            with tf.variable_scope('margin_loss'):
                a_j_2 = tf.reshape(a_j_2, shape=[self.batch_size, 2, 1, 1])  # ||*|| norm

                # calculate margin loss
                max_l = tf.square(tf.maximum(0., m_plus - a_j_2))
                max_r = tf.square(tf.maximum(0., a_j_2 - m_minus))

                max_l = tf.reshape(max_l, shape=(self.batch_size, 2))
                max_r = tf.reshape(max_r, shape=(self.batch_size, 2))

                T_c = tf.one_hot(self.y_c, depth=2)
                L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r
                # margin loss
                self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

                result = tf.reshape(a_j_2, shape=[self.batch_size, 2])
                # print(result.shape)
                pred_class = tf.argmax(result, axis=1)
                self.error_class = tf.cast(tf.not_equal(pred_class, self.y_c), dtype=tf.int32)

            # regression loss
            with tf.variable_scope('populrtity_regression'):
                v_j = tf.reshape(nets_2, shape=[self.batch_size, 2, -1])
                T_c_ = result[:, :, tf.newaxis]
                correct_output = tf.multiply(v_j, T_c_)
                correct_output = tf.reduce_sum(correct_output, axis=1)

                self.cap_states = correct_output

                dense1 = self.activation(
                    tf.add(tf.matmul(correct_output, self.weights['dense1']), self.biases['dense1']))
                # dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
                self.pred = self.activation(tf.add(tf.matmul(dense1, self.weights['out']), self.biases['out']))
                self.reg_loss = tf.reduce_mean(tf.pow(self.pred - self.y, 2))
                self.error = tf.reduce_mean(tf.pow(self.pred - self.y, 2))  # msle
                # r2 score
                RSS = tf.reduce_mean(tf.square(self.pred - self.y))
                TSS = tf.reduce_mean(tf.square(self.y - tf.reduce_mean(self.y)))
                self.r2_score = 1 - RSS / TSS
                # mape
                pred = self.pred
                y = self.y
                one1 = tf.ones_like(pred)
                pred = tf.where(pred < one1, one1, pred)
                one2 = tf.ones_like(y)
                truth = tf.where(y < one2, one2, y)
                # self.error_mape = tf.reduce_mean(tf.abs(pred - truth) / truth)  # mape
                self.error_mape = tf.reduce_mean(
                    tf.abs(self.pred - self.y) / ((tf.abs(self.pred) + tf.abs(self.y)) / 2))  # mape

    def train_batch(self, x, L, y, y_c, K, time_interval, rnn_index, adj, pos, drop_out):
        _, time_lambda = self.sess.run([self.train_op, self.time_lambda], feed_dict={
            self.x: x,
            self.supports: L,
            self.y: y,
            self.y_c: y_c,
            self.k_order: K,
            self.time_interval: time_interval,
            self.rnn_index: rnn_index,
            self.adj: adj,
            self.pos_emb: pos,
            self.drop_out: drop_out})
        return time_lambda

    def get_error(self, x, L, y, y_c, K, time_interval, rnn_index, adj, pos, drop_out):
        return self.sess.run([self.error, self.error_mape, self.r2_score], feed_dict={
            self.x: x,
            self.supports: L,
            self.y: y,
            self.y_c: y_c,
            self.k_order: K,
            self.time_interval: time_interval,
            self.rnn_index: rnn_index,
            self.adj: adj,
            self.pos_emb: pos,
            self.drop_out: drop_out})

    def get_loss(self, x, L, y, y_c, K, time_interval, rnn_index, adj, pos, drop_out):
        return self.sess.run(self.loss_total, feed_dict={
            self.x: x,
            self.supports: L,
            self.y: y,
            self.y_c: y_c,
            self.k_order: K,
            self.time_interval: time_interval,
            self.rnn_index: rnn_index,
            self.adj: adj,
            self.pos_emb: pos,
            self.drop_out: drop_out})

    def predict(self, x, L, y, y_c, K, time_interval, rnn_index, adj, pos, drop_out):

        return self.sess.run(self.pred, feed_dict={
            self.x: x,
            self.supports: L,
            self.y: y,
            self.y_c: y_c,
            self.k_order: K,
            self.time_interval: time_interval,
            self.rnn_index: rnn_index,
            self.adj: adj,
            self.pos_emb: pos,
            self.drop_out: drop_out})

    def get_represenattion(self, x, L, y, y_c, K, time_interval, rnn_index, adj, pos, drop_out):
        cap_rep = self.sess.run(self.cap_states, feed_dict={
            self.x: x,
            self.supports: L,
            self.y: y,
            self.y_c: y_c,
            self.k_order: K,
            self.time_interval: time_interval,
            self.rnn_index: rnn_index,
            self.adj: adj,
            self.pos_emb: pos,
            self.drop_out: drop_out})
        return cap_rep
