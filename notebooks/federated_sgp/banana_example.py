from pvi.models import SparseGaussianProcessClassification
from pvi.distributions import (
    MultivariateGaussianDistributionWithZ,
    MultivariateGaussianFactorWithZ,
)
from pvi.models.sgp.kernels import RBFKernel
from pvi.servers.federated_sgp_extra import SequentialSGPServerNoProjection
from pvi.clients.federated_sgp_extra import FederatedSGPClientNoProjection
from pvi.utils.training_utils import EarlyStopping
from banana_utils import (
    load_data,
    generate_clients_data,
    plot_data,
    plot_predictive_distribution,
    acc_and_ll,
)
import torch
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger(matplotlib.__name__).setLevel(logging.WARNING)
logging.basicConfig(filename="banana_example.log", level=logging.DEBUG)


# load and split data
name = "banana"
data_base_dir = "/home/thangbui/proj/working/pvi/datasets"
train_proportion = 0.08
training_set, test_set, D = load_data(name, train_proportion, data_base_dir)
M = 3
clients_data, nis, prop_positive, M = generate_clients_data(
    training_set["x"],
    training_set["y"],
    M=M,
    dataset_seed=0,
)

# configs
num_inducing = 10
# Shared across all clients.
model_config = {
    "D": D,
    "num_inducing": num_inducing,
    "kernel_class": lambda **kwargs: RBFKernel(**kwargs),
    "kernel_params": {"ard_num_dims": D, "train_hypers": True},
    "num_predictive_samples": 100,
}

client_config = {
    "optimiser": "Adam",
    "optimiser_params": {"lr": 1e-2},
    "epochs": 2000,
    "batch_size": len(clients_data[0]["x"]),
    "num_elbo_samples": 10,
    "num_elbo_hyper_samples": 2,
    "num_predictive_samples": 100,
    "train_model": False,
    "damping_factor": 0.25,
    "valid_factors": False,
    "early_stopping": EarlyStopping(50),
}

server_config = {
    "max_iterations": 50,
    "train_model": False,
    "hyper_optimiser": "SGD",
    "hyper_optimiser_params": {"lr": 1},
    "hyper_updates": 10,
    "optimiser": "Adam",
    "optimiser_params": {"lr": 1e-2},
    "epochs": 2000,
    "early_stopping": EarlyStopping(25, delta=1e-2),
    "num_elbo_samples": 10,
}

init_nat_params = {
    "np1": torch.zeros(model_config["num_inducing"]),
    "np2": torch.zeros(model_config["num_inducing"]).diag_embed(),
}

prior_nat_params = {
    "np1": torch.zeros(model_config["num_inducing"]),
    "np2": -0.5 * torch.ones(model_config["num_inducing"]).diag_embed(),
}


# Construct clients.
clients = []
z_is = []
for i in range(M):
    model_i = SparseGaussianProcessClassification(config=model_config)
    data_i = clients_data[i]

    # Randomly initialise private inducing points.
    perm = torch.randperm(len(data_i["x"]))
    idx = perm[: model_config["num_inducing"]]
    z_i = torch.tensor(data_i["x"][idx]).double()
    z_is.append(z_i)

    # Convert to torch.tensor.
    for k, v in data_i.items():
        data_i[k] = torch.tensor(v).double()

    t = MultivariateGaussianFactorWithZ(
        nat_params=init_nat_params,
        inducing_locations=z_i,
        train_inducing=True,
    )

    clients.append(
        FederatedSGPClientNoProjection(
            data=data_i, model=model_i, t=t, config=client_config
        )
    )

# Construct global model and server.
model = SparseGaussianProcessClassification(config=model_config)

# Union of z_is.
z = torch.cat(z_is)
kzz = model.kernel(z, z)
q = MultivariateGaussianDistributionWithZ(
    std_params={
        "loc": torch.zeros(z.shape[0]),
        "covariance_matrix": kzz,
    },
    inducing_locations=z,
    train_inducing=True,
)

# Randomly initialise global inducing points.
perm = torch.randperm(len(training_set["x"]))
idx = perm[:10]
z = torch.tensor(training_set["x"][idx])
kzz = model.kernel(z, z)
q = MultivariateGaussianDistributionWithZ(
    std_params={
        "loc": torch.zeros(z.shape[0]),
        "covariance_matrix": kzz,
    },
    inducing_locations=z,
    train_inducing=True,
)

server = SequentialSGPServerNoProjection(
    model=model,
    p=q,
    clients=clients,
    config=server_config,
    maintain_inducing=False,  # Set to False to use union of inducing points.
)


# Obtain predictions.
# pp = server.model_predict(torch.tensor(test_set["x"]))

# preds = pp.mean.detach().numpy()
# test_acc, test_mll = acc_and_ll(preds, test_set["x"], test_set["y"])

# print(test_acc)
# print(test_mll)

# fig = plt.figure(figsize=(12, 6), dpi=100, constrained_layout=True)
# gs = fig.add_gridspec(2, 3)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 1])

# for i, (client, ax) in enumerate(zip(clients, [ax1, ax2, ax3])):
#     plot_predictive_distribution(
#         client.data["x"],
#         client.data["y"],
#         z=client.t.inducing_locations,
#         ax=ax,
#     )
#     ax.set_title("Client {}".format(i))

# plot_predictive_distribution(
#     training_set["x"],
#     training_set["y"],
#     z=server.q.inducing_locations,
#     model=server.model,
#     q=server.q,
#     ax=ax4,
# )

# plt.show()


while not server.should_stop():
    server.tick()

    # Obtain predictions.
    pp = server.model_predict(torch.tensor(test_set["x"]))

    preds = pp.mean.detach().numpy()
    test_acc, test_mll = acc_and_ll(preds, test_set["x"], test_set["y"])

    print(test_acc)
    print(test_mll)

    fig = plt.figure(figsize=(12, 6), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 1])

    for i, (client, ax) in enumerate(zip(clients, [ax1, ax2, ax3])):
        plot_predictive_distribution(
            client.data["x"],
            client.data["y"],
            z=client.t.inducing_locations,
            ax=ax,
        )
        ax.set_title("Client {}".format(i))

    plot_predictive_distribution(
        training_set["x"],
        training_set["y"],
        z=server.q.inducing_locations,
        model=server.model,
        q=server.q,
        ax=ax4,
    )

    plt.show()

pdb.set_trace()
