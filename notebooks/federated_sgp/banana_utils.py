import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(name, train_proportion, data_base_dir):
    filename = os.path.join(data_base_dir, name, "banana.csv")
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    x = data[:, :2]
    y = data[:, -1]

    # Replace 1's with 0's and -1's with 1's.
    y[y == 1] = 0
    y[y == -1] = 1

    N = x.shape[0]
    N_train = int(np.ceil(train_proportion * N))

    x_train = x[0:N_train]
    y_train = y[0:N_train]
    x_test = x[N_train:]
    y_test = y[N_train:]

    training_set = {
        "x": x_train,
        "y": y_train,
    }

    test_set = {"x": x_test, "y": y_test}

    D = x_test.shape[1]

    del data

    return training_set, test_set, D


def generate_clients_data(x, y, M, dataset_seed):
    random_state = np.random.get_state()

    if dataset_seed is not None:
        np.random.seed(dataset_seed)

    if M == 1:
        client_data = [{"x": x, "y": y}]
        N_is = [x.shape[0]]
        props_positive = [np.mean(y > 0)]

        return client_data, N_is, props_positive, M

    N = x.shape[0]
    client_size = int(np.floor(N / M))

    class_balance = np.mean(y == 0)

    pos_inds = np.where(y > 0)
    zero_inds = np.where(y == 0)

    assert (len(pos_inds[0]) + len(zero_inds[0])) == len(
        y
    ), "Some indeces missed."

    print(f"x shape {x.shape}")

    y_pos = y[pos_inds]
    y_neg = y[zero_inds]

    x_pos = x[pos_inds]
    x_neg = x[zero_inds]

    client_data = []

    # Recombine remaining data and shuffle.
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])

    # As in Bui et al, order according to x1 value.
    inds = np.argsort(x[:, 0])

    x = x[inds]
    y = y[inds]

    # Distribute among clients.
    for i in range(M):
        client_x = x[:client_size]
        client_y = y[:client_size]

        x = x[client_size:]
        y = y[client_size:]

        client_data.append({"x": client_x, "y": client_y})

    N_is = [data["x"].shape[0] for data in client_data]
    props_positive = [np.mean(data["y"] > 0) for data in client_data]

    np.random.set_state(random_state)

    return client_data, N_is, props_positive, M


def plot_data(x, y, ax=None):
    x1_min, x1_max = -3.0, 3.0
    x2_min, x2_max = -3.0, 3.0

    x1x1, x2x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )

    if ax is None:
        plt.figure(figsize=(8, 6), dpi=200)
        ax = plt.gca()

    ax.plot(
        x[y == 0, 0],
        x[y == 0, 1],
        "o",
        color="C1",
        label="Class 1",
        alpha=0.5,
        zorder=1,
    )
    ax.plot(
        x[y == 1, 0],
        x[y == 1, 1],
        "o",
        color="C0",
        label="Class 2",
        alpha=0.5,
        zorder=1,
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(x1x1.min(), x1x1.max())
    ax.set_ylim(x2x2.min(), x2x2.max())
    ax.legend(loc="upper left", scatterpoints=1, numpoints=1)

    return x1x1, x2x2


def acc_and_ll(pred_probs, x, y):

    acc = np.mean((pred_probs > 0.5) == y)

    probs = torch.clip(torch.tensor(pred_probs), 0.0, 1.0)
    loglik = torch.distributions.Bernoulli(probs=probs).log_prob(
        torch.tensor(y)
    )
    loglik = loglik.mean().numpy()

    return acc, loglik


def plot_predictive_distribution(
    x, y, z, model=None, q=None, ax=None, x_old=None, y_old=None, z_old=None
):
    x1x1, x2x2 = plot_data(x, y, ax)
    ax.scatter(z[:, 0], z[:, 1], color="r", marker="o", zorder=3)

    if ax is None:
        ax = plt.gca()

    if model is not None and q is not None:
        x_predict = np.concatenate(
            (x1x1.ravel().reshape(-1, 1), x2x2.ravel().reshape(-1, 1)), 1
        )

        with torch.no_grad():
            y_predict = model(torch.tensor(x_predict), q=q, diag=True)
            y_predict = y_predict.mean.numpy().reshape(x1x1.shape)

        cs2 = ax.contour(
            x1x1,
            x2x2,
            y_predict,
            colors=["k"],
            levels=[0.2, 0.5, 0.8],
            zorder=2,
        )
        ax.clabel(cs2, fmt="%2.1f", colors="k", fontsize=14)

    if x_old is not None and y_old is not None:
        ax.plot(
            x_old[y_old == 0, 0],
            x_old[y_old == 0, 1],
            "o",
            color="C1",
            label="Class 1",
            alpha=0.25,
        )
        ax.plot(
            x_old[y_old == 1, 0],
            x_old[y_old == 1, 1],
            "o",
            color="C0",
            label="Class 2",
            alpha=0.25,
        )

        if z_old is not None:
            ax.scatter(
                z_old[:, 0], z_old[:, 1], color="r", marker="o", alpha=0.25
            )
