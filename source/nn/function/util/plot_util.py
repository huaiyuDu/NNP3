import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from nn.function.util import polynomial_util as poly
def plot_cost(costs, axis, i, j , title):
    axis[i,j].plot(range(len(costs)), costs)
    #axis[i,j].xlabel('Grident steps')
    #axis[i,j].xlabel('costs')
    axis[i,j].set_title(title)

def plot_multipl_decision_region_axis(X, y, num_label ,classifier, axis, i, j , title, resolution=0.02):
    markers = ('s', 'x', 'v','<')
    colors = ('red', 'blue' , 'yellow','black')
    fack_labels = np.arange(num_label)

    cmap = ListedColormap(colors[:num_label])

    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # print(x1_min, x1_max)

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print(x2_min, x2_max)


    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    xAera = np.array([xx1.ravel(), xx2.ravel()])
    print(xAera.shape)
    Z = classifier.test(xAera)
    Z  = np.where(Z  > 0.5, 1, 0)
    Z_new = np.dot(fack_labels,Z)
    Z_new= Z_new.reshape(xx1.shape)

    axis[i,j].contourf(xx1, xx2, Z_new, alpha=0.3, cmap=cmap)
    #axis[i,j].xlim(xx1.min(), xx1.max())
    #axis[i,j].ylim(xx2.min(), xx2.max())
    axis[i,j].set_title(title)
    y_unique_label = np.dot(fack_labels,y)
    print(y_unique_label)
    #plot class samples
    for idx, cl in enumerate(fack_labels):
        axis[i,j].scatter(x=X[0,(y_unique_label == cl)],
                    y=X[1,(y_unique_label == cl)],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')

def plot_regression_curl_axis(X, y ,X_test, y_test ,classifier,axis, i, j , title,  resolution=0.02):
    markers = ('s', 'x', 'v','<')
    colors = ('red', 'blue' , 'yellow','black')


    # plot the decision surface
    x1_min, x1_max = X.min() - 1, X.max() + 1
    # print(x1_min, x1_max)

    #x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print(x2_min, x2_max)

    x_line = np.arange(x1_min, x1_max, resolution)
    x_line = x_line.reshape((1,x_line.shape[0]))

    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                        np.arange(x2_min, x2_max, resolution))
    # xAera = np.array([xx1.ravel(), xx2.ravel()])
    # print(xAera.shape)

    y_line = classifier.test(x_line)

    # plt.contourf(xx1, xx2, Z_new, alpha=0.3, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    axis[i,j].plot(x_line.ravel(), y_line.ravel(), color='r')
    axis[i,j].set_title(title)

    axis[i,j].scatter(X,
                y,
                alpha=0.8,
                c=colors[0],
                marker=markers[0],
                edgecolors='black')
    axis[i,j].scatter(X_test,
                y_test,
                alpha=0.8,
                c=colors[1],
                marker=markers[1],
                edgecolors='black')

def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # print(x1_min, x1_max)

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print(x2_min, x2_max)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    xAera = np.array([xx1.ravel(), xx2.ravel()])
    print(xAera.shape)
    Z = classifier.test(xAera)
    Z  = np.where(Z  > 0.5, 1, 0)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[0,(y == cl).ravel()],
                    y=X[1,(y == cl).ravel()],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')
    plt.show()

def plot_multipl_decision_region(X, y, num_label ,classifier, resolution=0.02):
    markers = ('s', 'x', 'v','<')
    colors = ('red', 'blue' , 'yellow','black')
    fack_labels = np.arange(num_label)
    cmap = ListedColormap(colors[:num_label])

    # plot the decision surface
    x1_min, x1_max = X[0,:].min() - 0.2, X[0,:].max() + 0.2
    # print(x1_min, x1_max)

    x2_min, x2_max = X[1,:].min() - 0.2, X[1,:].max() + 0.2
    # print(x2_min, x2_max)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    xAera = np.array([xx1.ravel(), xx2.ravel()])
    print(xAera.shape)
    Z = classifier.test(xAera)
    Z  = np.where(Z  > 0.5, 1, 0)
    Z_new = np.dot(fack_labels,Z)
    Z_new= Z_new.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z_new, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    y_unique_label = np.dot(fack_labels,y)
    print(y_unique_label)
    for idx, cl in enumerate(fack_labels):
        plt.scatter(x=X[0,(y_unique_label == cl)],
                    y=X[1,(y_unique_label == cl)],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')
    plt.show()


def plot_regression_curl(X, y ,X_test, y_test ,classifier, resolution=0.02):
    markers = ('s', 'x', 'v')
    colors = ('red', 'blue', 'yellow')


    # plot the decision surface
    x1_min, x1_max = X.min() - 1, X.max() + 1
    # print(x1_min, x1_max)

    #x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print(x2_min, x2_max)

    x_line = np.arange(x1_min, x1_max, resolution)
    x_line = x_line.reshape((1,x_line.shape[0]))

    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                        np.arange(x2_min, x2_max, resolution))
    # xAera = np.array([xx1.ravel(), xx2.ravel()])
    # print(xAera.shape)

    y_line = classifier.test(x_line)

    # plt.contourf(xx1, xx2, Z_new, alpha=0.3, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    plt.plot(x_line.ravel(), y_line.ravel(), color='r')


    plt.scatter(X,
                y,
                alpha=0.8,
                c=colors[0],
                marker=markers[0],
                edgecolors='black')
    plt.scatter(X_test,
                y_test,
                alpha=0.8,
                c=colors[1],
                marker=markers[1],
                edgecolors='black')
    plt.show()
    # print(y_unique_label)
    # #plot class samples
    # for idx, cl in enumerate(fack_labels):
    #     plt.scatter(x=X[0,(y_unique_label == cl)],
    #                 y=X[1,(y_unique_label == cl)],
    #                 alpha=0.8,
    #                 c=colors[idx],
    #                 marker=markers[idx],
    #                 label=cl,
    #                 edgecolors='black')

def plot_regression_curl_extend(X, y ,X_test, y_test , degree, classifier, resolution=0.02):
    markers = ('s', 'x', 'v')
    colors = ('red', 'blue', 'yellow')


    # plot the decision surface
    x1_min, x1_max = X.min() - 1, X.max() + 1
    # print(x1_min, x1_max)

    #x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print(x2_min, x2_max)

    x_line = np.arange(x1_min, x1_max, resolution)
    x_line = x_line.reshape((1,x_line.shape[0]))
    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                        np.arange(x2_min, x2_max, resolution))
    # xAera = np.array([xx1.ravel(), xx2.ravel()])
    # print(xAera.shape)
    x_line_extend = poly.expend_to_polynomial(x_line, degree)
    y_line = classifier.test(x_line_extend)

    # plt.contourf(xx1, xx2, Z_new, alpha=0.3, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())
    plt.plot(x_line.ravel(), y_line.ravel(), color='r')


    plt.scatter(X,
                y,
                alpha=0.8,
                c=colors[0],
                marker=markers[0],
                edgecolors='black')
    plt.scatter(X_test,
                y_test,
                alpha=0.8,
                c=colors[1],
                marker=markers[1],
                edgecolors='black')
    plt.show()

def unroll_weight_his(weigh_his):
    result = None
    #unroll weight_matrix
    for record in weigh_his:
        row = np.array([])
        for i in range(1, len(record)+1):
            row = np.hstack((row,record[i].flatten()))
        #print(row.shape)
        if result is None:
            result = row
        else:
            result = np.vstack((result,row))
    return result