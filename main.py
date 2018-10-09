import numpy as np
import csv

np.random.seed(9)


def main():
    label_to_idx = {'Iris-setosa': 0,
                    'Iris-versicolor': 1,
                    'Iris-virginica': 2}
    X = []
    T = []
    with open('bezdekIris.data', 'r') as csvf:
        iris_reader = csv.reader(csvf, delimiter=',')
        iris_data = list(iris_reader)
        # Shuffle the data
        for row in np.random.permutation(iris_data):
            if len(row) == 0:
                continue
            x_i = np.array(row[:-1]).astype(float)
            X.append(x_i)

            t_i = np.zeros(3)
            t_i[label_to_idx[row[-1].strip()]] = 1
            T.append(t_i)

    X = np.array(X)
    T = np.array(T)
    for train_split in (0.12, 0.30, 0.50):
        num_train = int(np.ceil(train_split * float(X.shape[0])))

        X_tr = X[:num_train]
        T_tr = T[:num_train]
        X_ts = X[num_train:]
        T_ts = T[num_train:]

        # Now do the least squares approximation. Use formula from Bishop:
        # W = pseudo_inverse(X) * T  (where pseudo_inverse uses SVD)
        # Then f(x) = W^T * x
        W = np.dot(np.linalg.pinv(X_tr), T_tr)
        f = lambda x: np.dot(W.transpose(), x)

        conf_matrix = np.zeros((len(label_to_idx.keys()), len(label_to_idx.keys())))

        print('train: {}, test: {}\n'.format(train_split, 1-train_split))

        for x_i, t_i in zip(X_ts, T_ts):
            pred_i = f(x_i).argmax()
            targ_i = t_i.argmax()
            conf_matrix[pred_i, targ_i] += 1

        class2acc = {}
        print('      actual    ')
        for i, pred in enumerate(conf_matrix):
            print(str(pred) + ' pred={}'.format(i))
            class2acc[i] = conf_matrix[i, i] / sum(conf_matrix[:, i])

        for c in class2acc.keys():
            print('class={} acc. : {}'.format(c, class2acc[c]))

        print('parameters:')
        print(W)
        print('----------\n')


if __name__ == '__main__':
    main()
