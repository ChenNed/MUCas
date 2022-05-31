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

time_label = 15
if data == 'Weibo':
    flags.DEFINE_integer('observation_time', int(time_label * 3600) - 1, 'Observation time.')
    flags.DEFINE_integer('prediction_time', 3600 * 24, 'Prediction time.')
elif data == 'APS':
    flags.DEFINE_integer('observation_time', time_label * 365, 'Observation time.')
    flags.DEFINE_integer('prediction_time', 365 * 20 + 5, 'Prediction time.')


def generate_cascades(observation_time, prediction_time, filename, file_train, file_val, file_test,
                      seed='mucas'):
    # a list to save the cascades
    filtered_data = list()
    with open(filename) as file:

        cascades_type = dict()  # 0 for train, 1 for val, 2 for test
        cascades_time_dict = dict()
        cascade_total = 0
        cascade_valid_total = 0

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

            # filter cascades by their publish date/time
            # if criterion satisfied, put cascades into labeled set, otherwise unlabeled set
            if 'Weibo' in FLAGS.input:
                # timezone invariant
                if len(parts) != 5:
                    print('wrong format!')
                    continue

                hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8 # change to Chinese Standard Time（CST）
                # keep cascades post between 9am and 21pm 8，20
                if hour < 8 or hour >= 18:
                    continue

            elif 'APS' in FLAGS.input:
                pub_time = parts[2]
                year = datetime.datetime.strptime(pub_time, "%Y-%m-%d").year
                year = int(year)
                # keep cascades publicate between 1893 and 1997
                if year < 1893 or year > 1997:
                    continue

            # a list of adoptions
            paths = parts[4].strip().split(' ')

            observation_path = list()
            # number of observed popularity
            p_o = 0
            for p in paths:
                # observed adoption/participant
                nodes = p.split(':')[0].split('/')
                time_now = float(p.split(':')[1])
                if time_now < observation_time:
                    p_o += 1
                    # save observed adoption/participant into 'observation_path'
                observation_path.append((nodes, time_now))

            # filter cascades which observed popularity less than 10

            if p_o < 10:
                continue

            # sort list by their publish time/date
            observation_path.sort(key=lambda tup: tup[1])

            # for each cascade, save its publish time into a dict
            if 'APS' in FLAGS.input:
                cascades_time_dict[cascade_id] = int(0)
            else:
                cascades_time_dict[cascade_id] = int(parts[2])

            o_path = list()

            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                o_path.append('/'.join(nodes) + ':' + str(t))

            # write data into the targeted file, if they are not excluded

            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                   + parts[3] + '\t' + ' '.join(o_path) + '\n'
            filtered_data.append(line)
            cascade_valid_total += 1

    with open(file_train, 'w') as data_train, \
            open(file_val, 'w') as data_val, \
            open(file_test, 'w') as data_test:

        def shuffle_cascades():
            # shuffle all cascades
            shuffled_time = list(cascades_time_dict.keys())
            random.seed(seed)
            random.shuffle(shuffled_time)

            count = 0
            # split datasets
            for key in shuffled_time:
                if count < cascade_valid_total * .7:
                    cascades_type[key] = 0  # training set, 50%
                elif count < cascade_valid_total * .85:
                    cascades_type[key] = 1  # validation set, 10%
                else:
                    cascades_type[key] = 2  # test set, 40%
                count += 1

        shuffle_cascades()

        # number of valid cascades
        print("Number of kept cascades: {}/{}".format(cascade_valid_total, cascade_total))

        # 3 list to save the filtered sets
        filtered_data_train = list()
        filtered_data_val = list()
        filtered_data_test = list()
        for line in filtered_data:
            cascade_id = line.split('\t')[0]
            if cascades_type[cascade_id] == 0:
                filtered_data_train.append(line)
            elif cascades_type[cascade_id] == 1:
                filtered_data_val.append(line)
            elif cascades_type[cascade_id] == 2:
                filtered_data_test.append(line)
        print("Number of valid train cascades: {}".format(len(filtered_data_train)))
        print("Number of valid   val cascades: {}".format(len(filtered_data_val)))
        print("Number of valid  test cascades: {}".format(len(filtered_data_test)))

        def file_write(file_name):
            # write file, note that compared to the original 'dataset_old.txt', only cascade_id and each of the
            # observed adoptions are saved, plus label information at last
            file_name.write(cascade_id + '\t' + size + '\t' + label + '\t' + '\t'.join(observation_path) + '\n')

        # write cascades into files
        for line in filtered_data_train + filtered_data_val + filtered_data_test:
            # split the cascades into 5 parts
            parts = line.split('\t')
            cascade_id = parts[0]
            observation_path = list()
            label = int()
            paths = parts[4].split(' ')

            for p in paths:
                nodes = p.split(':')[0].split('/')
                time_now = float(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                # add label information depends on prediction_time, e.g., 24 hours for Weibo dataset
                if time_now < prediction_time:
                    label += 1

            # calculate the incremental prediction
            label = str(label - len(observation_path))
            size = str(len(observation_path))

            # write files by cascade type
            # 0 to train, 1 to validate, 2 to test
            if cascade_id in cascades_type and cascades_type[cascade_id] == 0:
                file_write(data_train)
            elif cascade_id in cascades_type and cascades_type[cascade_id] == 1:
                file_write(data_val)
            elif cascade_id in cascades_type and cascades_type[cascade_id] == 2:
                file_write(data_test)


def main(argv):
    time_start = time.time()
    print('Start to generate cascades!\n')
    print('Dataset path: ', FLAGS.input)

    generate_cascades(FLAGS.observation_time,
                      FLAGS.prediction_time,
                      FLAGS.input + 'dataset.txt',
                      FLAGS.input + str(time_label) + '/train.txt',
                      FLAGS.input + str(time_label) + '/val.txt',
                      FLAGS.input + str(time_label) + '/test.txt',
                      'mucas'
                      # 0
                      # special caveat: because of some... historical reasons about the codes,
                      # for weibo, acm, and dblp datasets, the seed is set to 'xovee' (string),
                      # and for twitter and aps datasets, the seed is set to 0 (integer).
                      )

    time_end = time.time()
    print('Processing Time: {:.2f}s'.format(time_end - time_start))


if __name__ == "__main__":
    app.run(main)
