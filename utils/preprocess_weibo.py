'''
preprocess the original weibo dataset
1. Reorder the paths into chronological order
2. Delete redundant paths.
'''
import random
import time
import datetime
from absl import app, flags

# flags
FLAGS = flags.FLAGS
# observation and prediction time settings:
# for weibo dataset, we use 3600 (1 hour) and 3600*24 (86400, 1 day)
# for aps dataset, we use 365*3 (1095, 3 years) and 365*20+5 (7305, 20 years)

# paths
data = 'Weibo'
# data = 'APS'
flags.DEFINE_string('input', '../data/' + data + '/', 'Dataset path.')


def proprocess_weibo(filename, file_write):
    # a list to save the cascades
    filtered_data = list()
    with open(filename) as file, open(file_write, 'w') as data_write:

        cascade_total = 0

        for line in file:
            # split the cascades into 5 parts
            # 1: cascade id
            # 2: user/item id
            # 3: publish date/time
            # 4: number of adoptions
            # 5: a list of adoptions
            cascade_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]
            user_id = parts[1]
            msg_time = parts[2]
            num_adoptions = parts[3]

            if len(parts) != 5:
                print('wrong format!')
                continue

            # a list of adoptions
            paths = parts[4].strip().split(' ')

            observation_path = list()

            ignore = False

            for path in paths:
                # observed adoption/participant
                p = path.strip().split(':')
                nodes = p[0].split('/')
                time_now = float(p[1])

                if '-1' in nodes:
                    ignore = True
                # save observed adoption/participant into 'observation_path'
                observation_path.append((nodes, time_now))

            # filter cascades which observed popularity less than 10
            if ignore:
                continue

            # sort list by their publish time/date
            observation_path.sort(key=lambda tup: tup[1])

            o_path = list()

            exist_edge = list()
            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                wrong = False
                for n in range(len(nodes) - 1):
                    for j in range(1, len(nodes)):
                        if nodes[n] == nodes[j]:
                            wrong = True
                            continue
                if wrong:
                    continue

                nodes_str = '/'.join(nodes)

                if nodes_str not in exist_edge:
                    exist_edge.append(nodes_str)
                else:
                    continue

                o_path.append(nodes_str + ':' + str(t))

            # write data into the targeted file, if they are not exclude

            data_write.write(parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                             + parts[3] + '\t' + ' '.join(o_path) + '\n')
            cascade_total += 1
        print('after preprocessing, preserved {} cascades!'.format(cascade_total))


def main(argv):
    time_start = time.time()
    print('Start to generate cascades!\n')
    print('Dataset path: ', FLAGS.input)

    proprocess_weibo(
        FLAGS.input + 'dataset_old.txt',
        FLAGS.input + 'dataset.txt'
    )

    time_end = time.time()
    print('Processing Time: {:.2f}s'.format(time_end - time_start))


if __name__ == "__main__":
    app.run(main)
