import time
import networkx as nx
from absl import app, flags
import numpy as np
import math
from functools import cmp_to_key
import scipy.sparse as sp
import joblib

# flags
FLAGS = flags.FLAGS

data = 'Weibo'
#data = 'APS'
time_label = 0.5
n_time_interval = 5

# paths
flags.DEFINE_string('input', '../data/' + data + '/' + str(time_label) + '/', 'Pre-training data path.')

if data == 'Weibo':
    flags.DEFINE_integer('observation', int(time_label * 3600), 'Observation time.')
elif data == 'APS':
    flags.DEFINE_integer('observation', time_label * 365, 'Observation time.')

flags.DEFINE_integer('emb_dim', 50, 'Embedding dimension for position encoding.')
flags.DEFINE_integer('max_node', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('n_time_interval', n_time_interval, 'number of time interval.')
flags.DEFINE_string('pad_method', 'insert_padding', 'padding method.')
flags.DEFINE_bool('self_loop', True, 'whether use self-loop')

flags.DEFINE_bool('initial_source', False,
                  'whether consider the first source node as a sperate interval 0,if True, '
                  'total interval = n_time_interval+1 else total interval = n_time_interval')


def main(argv):
    # hyper-params
    emb_dim = FLAGS.emb_dim
    max_node = FLAGS.max_node
    observation_time = FLAGS.observation
    n_time_interval = FLAGS.n_time_interval
    time_interval = math.ceil((observation_time + 1) * 1.0 / n_time_interval)

    # hyper-params

    def get_nodes(graph):
        # id_list 是新的坐标index
        nodes = {}  # 为了建立topo图    nodes2 = {}#与生成的index进行转换
        j = 0
        for walks in graph:
            # walks[0]=time,walks[1]=path
            for walk in walks[1]:
                # print(walk)
                for node in walk:
                    # print(node)
                    if node not in nodes.keys():
                        nodes[node] = j
                        j = j + 1
        return nodes

    def seq2graph_interval_aware(filename, num_interval, time_interval, max_node):
        """
        :param filename:
        :param num_interval
        :returns: graphs, k_order
        """

        graphs = {}  # seq to graph
        k_order = {}
        order_dict = {}

        for i in range(num_interval):
            order_dict[i] = np.zeros(max_node)

        with open(filename, 'r') as f:
            for line in f:
                walks = line.strip().split('\t')
                # 0 - cascade id
                # 1 - size
                # 2 - label
                # 3:-1 - paths
                graphs[walks[0]] = {}  # walk[0] = cascadeID
                paths = walks[3:][:max_node]  # keep only max-size seq
                for path in paths:
                    parts = path.strip().split(':')
                    t = parts[1]  # time
                    if FLAGS.initial_source:
                        # keep the source node in the first interval
                        if float(t) == 0.0:
                            graphs[walks[0]][str(t)] = []
                        else:
                            k = int(math.floor(float(t) / time_interval)) + 1
                            graphs[walks[0]][str(k)] = []
                    else:
                        k = int(math.floor(float(t) / time_interval))
                        graphs[walks[0]][str(k)] = []

                for path in paths:
                    parts = path.strip().split(':')
                    t = parts[1]  # time
                    s = parts[0]  # node
                    if FLAGS.initial_source:
                        if float(t) == 0.0:
                            graphs[walks[0]][str(t)].append([str(xx) for xx in s.split(",")])
                        else:
                            k = int(math.floor(float(t) / time_interval)) + 1
                            graphs[walks[0]][str(k)].append([str(xx) for xx in s.split(",")])
                    else:
                        k = int(math.floor(float(t) / time_interval))
                        graphs[walks[0]][str(k)].append([str(xx) for xx in s.split(",")])

                key1 = cmp_to_key(lambda x, y: float(x[0]) - float(y[0]))
                graphs[walks[0]] = sorted(graphs[walks[0]].items(), key=key1)

                k_order[walks[0]] = np.zeros(num_interval)
                for g in graphs[walks[0]]:
                    m = 0
                    for i in g[1]:
                        m = max(m, len(i))
                    k_order[walks[0]][int(g[0])] = m

        return graphs, k_order

    # read label and size from cascade file
    def read_labelANDsize(filename):
        labels = {}
        sizes = {}
        with open(filename, 'r') as f:
            for line in f:
                # 0 - cascade id
                # 1 - size
                # 2 - label
                # 3:-1 - paths
                profile = line.strip().split('\t')
                sizes[profile[0]] = int(profile[1])  # current size
                labels[profile[0]] = profile[2]  # label
        return labels, sizes

    def position_encoding(n_position, d_hid, padding_idx=None):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_pos_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(n_position)])
        # 将 sin 应用于数组中的偶数索引（indices）；2i
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        # 将 cos 应用于数组中的奇数索引；2i+1
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        if padding_idx is not None:
            sinusoid_table[padding_idx] = 0
        return sinusoid_table

    def normalize_adj_dir(adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -1).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).tocoo()

    def preprocess_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj = sp.coo_matrix(adj)
        adj_normalized = normalize_adj_dir(adj + sp.eye(adj.shape[0]))
        return adj_normalized

    def randomedge_sampler(adj, percent):
        """
        Randomly drop edge and preserve percent% edges.
        """
        "Opt here"

        nnz = adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((adj.data[perm],
                               (adj.row[perm],
                                adj.col[perm])),
                              shape=adj.shape)
        return r_adj

    def write_cascade(graphs, labels, sizes, pos_emb, k_order, interval_num, embeding_size, max_num, pad_method,
                      self_loop, percent, filename):
        id_data = []  # store the cascade id
        x_data = []  # sub-graph sequence based on time-interval
        lap_data = []  # the in- and out- adj sequence for each cascade
        pos_input = []
        orders = []  # record the max-order for graph
        time = []
        A_list = []  # adj list

        rnn_index = []

        y_data = []  # label data
        y_class = []  # out-break prediction

        # padding
        pad_t = np.zeros(interval_num)
        order_padding = np.zeros(max_num)
        padd_f = np.zeros(shape=(max_num, max_num))
        padd_pos = np.zeros(shape=(max_num, emb_dim))

        for key, graph in graphs.items():
            id = key  #
            order = k_order[id]  # graph-level [o1, o2, ..., on] # node-level [[n1,n2,...],[...],[...]]

            # store label information
            label = labels[key]
            y = int(label)  # label
            size = sizes[key]
            # outbreak label
            # size = min(size, max_num)

            if size > max_num:
                size_ = max_num
                plus = size - max_num
            else:
                size_ = size
                plus = 0
            y += plus

            inc = math.floor(y / size_)
            if inc >= 1:
                out_break = 1
            else:
                out_break = 0

            temp_fea = dict()  # features---A+I
            temp_pos = dict()  # position embedding

            temp_fea_2 = list()
            temp_pos_2 = list()

            # construct graph
            nodes_items = get_nodes(graph)  # list 点坐标{key=original id,value = new id} from 0-n-1
            nodes_list = list(nodes_items.values())

            if len(nodes_list) > max_num:
                print(id)
                continue

            nx_G = nx.DiGraph()  # create graph
            nx_G.add_nodes_from(nodes_list)  # 创建图,建立一个NxN的图 nodes_list=>new id

            time_interval = np.zeros(shape=(interval_num))

            node_seq = list()  # record the node->feature matrix

            for walks in graph:

                time_interval[int(walks[0])] = 1  # time interval

                list_edge = list()

                f = np.zeros(shape=(max_num, embeding_size))  # (N, F)

                if FLAGS.initial_source:
                    # remove
                    if int(walks[0]) == 0:  # if the first path and the time stamp is 0
                        node_seq.append(nodes_items.get(walks[1][-1][-1]))
                    else:
                        for walk in walks[1]:
                            for i in range(len(walk) - 1):
                                if nodes_items.get(walk[i]) not in node_seq:
                                    node_seq.append(nodes_items.get(walk[i]))
                                if nodes_items.get(walk[i + 1]) not in node_seq:
                                    node_seq.append(nodes_items.get(walk[i + 1]))
                                nx_G.add_edge(nodes_items.get(walk[i]), nodes_items.get(walk[i + 1]))
                else:
                    for walk in walks[1]:
                        if len(walk) == 1:
                            # source node
                            node_seq.append(nodes_items.get(walk[-1]))
                            edge = (nodes_items.get(walk[-1]), nodes_items.get(walk[-1]))
                            list_edge.append(edge)
                            # continue
                        else:
                            node_seq.append(nodes_items.get(walk[-1]))
                            edge = (nodes_items.get(walk[-2]), nodes_items.get(walk[-1]))
                            list_edge.append(edge)

                # position embedding matrix for node
                # print(node_seq)
                for ni in node_seq:
                    f[ni] = pos_emb[ni]
                # print(f)

                # add edge from edge list
                nx_G.add_edges_from(list_edge)

                temp_adj = nx.to_pandas_adjacency(nx_G)
                N = len(temp_adj)

                if self_loop:
                    I = np.eye(N)
                    temp_adj = temp_adj + I
                    temp_adj[temp_adj > 0] = 1

                if N < max_num:
                    col_padding = np.zeros(shape=(N, max_num - N))
                    A_col_padding = np.column_stack((temp_adj, col_padding))
                    row_padding = np.zeros(shape=(max_num - N, max_num))
                    temp_adj = np.row_stack((A_col_padding, row_padding))

                temp_adj = sp.coo_matrix(temp_adj, dtype=np.float32)
                temp_fea[walks[0]] = temp_adj  # stru_input

                f = sp.coo_matrix(f, dtype=np.float32)
                temp_pos[walks[0]] = f

            # caculate normalized adj
            if percent >= 1:
                A_in = nx.to_scipy_sparse_matrix(nx_G, format='coo')
                A_out = sp.coo_matrix.transpose(A_in)
            else:
                A_in = nx.to_scipy_sparse_matrix(nx_G, format='coo')
                A_out = sp.coo_matrix.transpose(A_in)
                A_in = randomedge_sampler(A_in, percent)
                A_out = randomedge_sampler(A_out, percent)

            A = nx.to_pandas_adjacency(nx_G)  # original adj

            A_in_dense = A_in.todense()
            A_out_dense = A_out.todense()
            L1 = preprocess_adj(A_in_dense)
            L2 = preprocess_adj(A_out_dense)
            M, M = A_in_dense.shape
            M = int(M)

            L1 = L1.todense()
            L2 = L2.todense()

            L = []
            if M < max_num:
                col_padding_L = np.zeros(shape=(M, max_num - M))
                L1_col_padding = np.column_stack((L1, col_padding_L))
                L2_col_padding = np.column_stack((L2, col_padding_L))
                A_col_padding = np.column_stack((A, col_padding_L))

                row_padding = np.zeros(shape=(max_num - M, max_num))
                L1_col_row_padding = np.row_stack((L1_col_padding, row_padding))
                L2_col_row_padding = np.row_stack((L2_col_padding, row_padding))
                A_col_row_padding = np.row_stack((A_col_padding, row_padding))

                Laplacian1 = sp.coo_matrix(L1_col_row_padding, dtype=np.float32)
                Laplacian2 = sp.coo_matrix(L2_col_row_padding, dtype=np.float32)
                A_ = sp.coo_matrix(A_col_row_padding, dtype=np.float32)

            else:
                Laplacian1 = sp.coo_matrix(L1, dtype=np.float32)
                Laplacian2 = sp.coo_matrix(L2, dtype=np.float32)
                A_ = sp.coo_matrix(A, dtype=np.float32)

            L.append(Laplacian1)
            L.append(Laplacian2)

            # record the fact number of graph
            rnn_index_temp = np.zeros(shape=(interval_num))
            rnn_index_temp[:len(temp_fea)] = 1

            t_i = []  # []
            t_it = []  #
            o = np.zeros(interval_num)  # record order

            if pad_method == 'insert_padding':
                for i in range(len(time_interval)):
                    t = np.zeros(interval_num)  # [n_time_interval]

                    if time_interval[i] == 1:
                        t[i] = 1
                        t_it.append(i)

                        temp_fea_2.append(temp_fea[str(i)])
                        temp_pos_2.append(temp_pos[str(i)])

                        o[i] = order[i]
                    else:
                        t[i - 1] = 1

                        t_it.append(t_it[i - 1])

                        temp_fea_2.append(temp_fea_2[i - 1])  # use previous status
                        temp_pos_2.append(temp_pos_2[i - 1])

                        o[i] = order[i - 1]

                    t_i.append(t)
            else:
                for i in range(len(time_interval)):
                    t = np.zeros(interval_num)
                    if time_interval[i] == 1:
                        t[i] = 1
                        t_it.append(i)
                        o[i] = order[i]
                    else:
                        t_it.append(0)
                    t_i.append(t)
                temp_fea_2.append(x for key, x in temp_fea.items())
                temp_pos_2.append(z for key, z in temp_pos.items())
                for i in range(int(interval_num - list(time_interval).count(1))):
                    temp_fea_2.append(padd_f)
                    temp_pos_2.append(sp.coo_matrix(padd_pos, dtype=np.float32))
                    t_i.append(pad_t)

            id_data.append(id)  # [?, T]
            # label
            y_data.append(np.log(y + 1.0) / np.log(2.0))  # log normalized #[?, T]
            y_class.append(out_break)

            #
            x_data.append(temp_fea_2)  # [?, T, N, F]
            pos_input.append(temp_pos_2)
            # print(temp_pos_2)
            # normalized adj
            lap_data.append(L)  # [?, 2, N, N]

            rnn_index.append(rnn_index_temp)  # [?, T]

            orders.append(o)  # [?, T]

            time.append(t_it)  # [?, T] record the time interval for each sub-graph

            A_list.append(A_)  # [?, N. N] record the original adj
        print("----------Write!------------")
        joblib.dump((id_data, x_data, lap_data, y_data, y_class, rnn_index, orders, time, A_list, pos_input),
                    open(filename, 'wb'))

    time_start = time.time()

    # get the information of nodes/users of cascades
    graphs_train, train_order = seq2graph_interval_aware(FLAGS.input + 'train.txt', n_time_interval, time_interval,
                                                         max_node)
    # print(graphs_train['77211'])
    graphs_val, val_order = seq2graph_interval_aware(FLAGS.input + 'val.txt', n_time_interval, time_interval, max_node)
    graphs_test, test_order = seq2graph_interval_aware(FLAGS.input + 'test.txt', n_time_interval, time_interval,
                                                       max_node)
    #
    # # get the information of labels and sizes of cascades
    labels_train, sizes_train = read_labelANDsize(FLAGS.input + 'train.txt')
    labels_val, sizes_val = read_labelANDsize(FLAGS.input + 'val.txt')
    labels_test, sizes_test = read_labelANDsize(FLAGS.input + 'test.txt')
    #
    print('generate position embedding!')
    p_e = position_encoding(max_node, emb_dim)
    print(p_e.shape)
    #
    print("Start writing train set into file.")

    write_cascade(graphs_train, labels_train, sizes_train, p_e, train_order, n_time_interval, emb_dim,
                  max_node, FLAGS.pad_method, FLAGS.self_loop, 0.7,
                  FLAGS.input + 'train_' + str(emb_dim) + '_' + str(n_time_interval) + '.pkl')
    #
    print("Start writing validation set into file.")
    write_cascade(graphs_val, labels_val, sizes_val, p_e, val_order, n_time_interval, emb_dim,
                  max_node, FLAGS.pad_method, FLAGS.self_loop, 1,
                  FLAGS.input + 'val_' + str(emb_dim) + '_' + str(n_time_interval) + '.pkl')
    #
    print("Start writing test set into file.")
    write_cascade(graphs_test, labels_test, sizes_test, p_e, test_order, n_time_interval, emb_dim,
                  max_node, FLAGS.pad_method, FLAGS.self_loop, 1,
                  FLAGS.input + 'test_' + str(emb_dim) + '_' + str(n_time_interval) + '.pkl')

    time_end = time.time()
    print("Processing time: {0:.2f}s".format(time_end - time_start))


if __name__ == "__main__":
    app.run(main)
