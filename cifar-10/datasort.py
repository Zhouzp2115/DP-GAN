import pickle
import os

def unPickle(fileDir):
    fo = open(fileDir, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    return dict


def saveData(dir, filename, dict):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        os.system(r'touch {}'.format(dir + filename))

    file = open(dir + filename, 'wb')
    pickle.dump(dict, file)
    print('save file..........ok')


def dataSort(root):
    data = []
    for i in range(1, 6):
        data.append(unPickle(root + 'data_batch_' + str(i)))

    res = []
    for i in range(10):
        res.append({'data': [], 'labels': []})

    for i in range(5):
        for j in range(len(data[i]['data'])):
            x = data[i]['data'][j]
            label = data[i]['labels'][j]
            res[label]['data'].append(x)
            res[label]['labels'].append(label)

    for i in range(10):
        saveData(root + '../sorted', 'train_' + str(i), res[i])

if __name__ == '__main__':
    dataSort('../data/cifar-10/cifar-10-batches-py/')