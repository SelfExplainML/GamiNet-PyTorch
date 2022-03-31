import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from itertools import combinations
from matplotlib import pylab as plt
from joblib import Parallel, delayed
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

from .layers import *
from .dataloader import FastTensorDataLoader
from .interpret import EBMPreprocessor, InteractionDetector


class GAMINet(BaseEstimator, metaclass=ABCMeta):
    """
        GAMINet Pytorch Version
        
        Parameters
        ----------
        meta_info : None or dict
            A dict of feature_name: feature_type pairs, by default None.
            If not None, its length should be the same as the number of features.
            E.g., {"X1": "categorical", "X2": "continuous"}
        interact_num : int
            The max number of interactions to be included in the second stage training, by default 10.
        subnet_size_main_effect : list of int
            The hidden layer architecture of each subnetwork in the main effect block, by default [100].
        subnet_size_interaction : list of int
            The hidden layer architecture of each subnetwork in the interaction block, by default [200].
        activation_func : torch funciton
            The activation function, by default torch.nn.ReLU().
        max_epochs : list of int
            The max number of epochs in the first (main effect training), second (interaction training), and third (fine tuning) stages, respectively, by default [1000, 1000, 1000].
        learning_rates : list of float
            The initial learning rates of Adam optimizer in the first (main effect training), second (interaction training), and third (fine tuning) stages, respectively, by default [1e-3, 1e-3, 1e-3].
        early_stop_thres : list of int or "auto"
            The early stopping threshold in the first (main effect training), second (interaction training), and third (fine tuning) stages, respectively, by default ["auto", "auto", "auto"].
            In auto mode, the value is set to min(80000 / (max_iter_per_epoch * batch_size), 100).
        batch_size : int
            The batch size, by default 1000.
            Note that it should not be larger than the training size * (1 - validation ratio). 
        batch_size : int
            The batch size used in the inference stage by default 10000.
            It is imposed to avoid out-of-memory issue when dealing very large dataset.
        max_iter_per_epoch : int
            The max number of iterations per epoch, by default 100.
            In the init stage of model fit, its value will be clipped by min(max_iter_per_epoch, int(sample_size / batch_size)).
            For each epoch, the data would be reshuffled and only the first "max_iter_per_epoch" batches would be used for training. 
            It is imposed to make the training scalable for very large dataset.
        val_ratio : float
            The validation ratio, by default 0.2.
            It is used together with max_val_size, with validation size = min(train_x.shape[0] * val_ratio, max_val_size)
        max_val_size : int
            The max size of validation set, by default 10000.
            It is used together with val_ratio, with validation size = min(train_x.shape[0] * val_ratio, max_val_size)
        warm_start : bool
            Initialize the network by fitting a rough B-spline based GAM model with tensor product interactions, by default True.
            The initialization is performed by, 1) fit B-spline GAM as teacher model, 2) generate random samples from the teacher model, 3) fit each subnetwork using the generated samples. And it is used for both main effect and interaction subnetwork initialization.
        gam_sample_size : int
            The sub-sample size for GAM fitting as warm_start=True, by default 5000.
        mlp_sample_size : int
            The generated sample size for individual subnetwork fitting as warm_start=True, by default 1000.
        heredity : bool
            Whether to perform interaction screening subject to heredity constraint, by default True.
        loss_threshold : float
            The loss tolerance threshold for selecting fewer main effects or interactions, according to the validation performance, by default 0.0.
            For instance, assume the best validation performance is achived when using 10 main effects; if only use the top 5 main effects also gives similar validation performance, we could prune the last 5 by setting this parameter to be positive.
        reg_clarity : float
            The regularization strength of marginal clarity constraint, by default 0.1.
        reg_mono : float
            The regularization strength of monotonicity constraint, by default 0.1.
        mono_sample_size : int
            As monotonicity constraint is used, we would generate some data points uniformly within the feature spacec per epoch, to impose the monotonicity regularization in addition to original training samples, by default 200.
        mono_increasing_list : None or list
            The feature index list with monotonic increasing constraint, by default None.
        mono_decreasing_list : None or list
            The feature index list with monotonic decreasing constraint, by default None.
        boundary_clip : bool
            In the inference stage, whether to clip the feature values by their min and max values in the training data, by default True.
        normalize : bool
            Whether to to normalize the data before inputing to the network, by default True.
        verbose : bool
            Whether to output the training logs, by default False.
        device : string
            The hard device name used for training, by default "cpu".
        random_state : int
            The random seed, by default 0.
     """

    def __init__(self, loss_fn,
                 meta_info=None,
                 interact_num=10,
                 subnet_size_main_effect=[100],
                 subnet_size_interaction=[200],
                 activation_func=torch.nn.ReLU(),
                 max_epochs=[1000, 1000, 1000],
                 learning_rates=[1e-3, 1e-3, 1e-3],
                 early_stop_thres=["auto", "auto", "auto"],
                 batch_size=1000,
                 batch_size_inference=1000,
                 max_iter_per_epoch=100,
                 val_ratio=0.2,
                 max_val_size=10000,
                 warm_start=True,
                 gam_sample_size=5000,
                 mlp_sample_size=1000,
                 heredity=True,
                 loss_threshold=0.0,
                 reg_clarity=0.1,
                 reg_mono=0.1,
                 mono_sample_size=200,
                 mono_increasing_list=None,
                 mono_decreasing_list=None,
                 boundary_clip=True,
                 normalize=True,
                 verbose=False,
                 device="cpu",
                 random_state=0):

        super(GAMINet, self).__init__()

        self.loss_fn = loss_fn
        self.meta_info = meta_info
        self.interact_num = interact_num
        self.subnet_size_main_effect = subnet_size_main_effect
        self.subnet_size_interaction = subnet_size_interaction
        self.activation_func = activation_func

        self.max_epochs = max_epochs
        self.learning_rates = learning_rates
        self.early_stop_thres = early_stop_thres
        self.batch_size = batch_size
        self.batch_size_inference = batch_size_inference
        self.max_iter_per_epoch = max_iter_per_epoch
        self.val_ratio = val_ratio
        self.max_val_size = max_val_size

        self.warm_start = warm_start
        self.gam_sample_size = gam_sample_size
        self.mlp_sample_size = mlp_sample_size

        self.heredity = heredity
        self.reg_clarity = reg_clarity
        self.loss_threshold = loss_threshold

        self.reg_mono = reg_mono
        self.mono_increasing_list = mono_increasing_list if mono_increasing_list is not None else []
        self.mono_decreasing_list = mono_decreasing_list if mono_decreasing_list is not None else []
        self.monotonicity = True if len(self.mono_increasing_list + self.mono_decreasing_list) > 0 else False
        self.mono_sample_size = mono_sample_size

        self.boundary_clip = boundary_clip
        self.normalize = normalize
        self.verbose = verbose
        self.device = device
        self.random_state = random_state

    @abstractmethod
    def build_teacher_main_effect(self):

        pass
    
    @abstractmethod
    def build_teacher_interaction(self):

        pass

    @abstractmethod
    def get_interaction_list(self):

        pass

    @abstractmethod
    def _validate_input(self):

        pass

    def estimate_density(self, x, sample_weight):

        n_samples = x.shape[0]
        self.data_dict_density = {}
        for idx in self.nfeature_index_list:
            feature_name = self.feature_names[idx]
            density, bins = np.histogram(x[:,[idx]], bins=10, weights=sample_weight.reshape(-1, 1), density=True)
            self.data_dict_density.update({feature_name: {"density": {"names": bins,"scores": density}}})
        for idx in self.cfeature_index_list:
            feature_name = self.feature_names[idx]
            unique, counts = np.unique(x[:, idx], return_counts=True)
            density = np.zeros((len(self.dummy_values[feature_name])))
            for val in unique:
                density[val.round().astype(int)] = np.sum((x[:, idx] == val).astype(int) * sample_weight) / sample_weight.sum()
            self.data_dict_density.update({feature_name: {"density": {"names": np.arange(len(self.dummy_values[feature_name])),
                                                   "scores": density}}})

    def get_main_effect_raw_output(self, x):

        pred = []
        self.net.eval()
        train_size = x.shape[0]
        xx = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float, device=self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        for iterations in range(train_size // batch_size):
            offset = (iterations * batch_size) % train_size
            batch_data = xx[offset:(offset + batch_size), :]
            batch_data = torch.max(torch.min(batch_data, self.max_value), self.min_value) if self.boundary_clip else batch_xx
            batch_data = (batch_data - self.mu_list) / self.std_list if self.normalize else batch_data
            with torch.no_grad():
                pred.append(self.net.main_effect_blocks(batch_data).detach())
        if train_size % batch_size > 0:
            batch_data = xx[((iterations + 1) * batch_size):, :]
            batch_data = torch.max(torch.min(batch_data, self.max_value), self.min_value) if self.boundary_clip else batch_xx
            batch_data = (batch_data - self.mu_list) / self.std_list if self.normalize else batch_data
            with torch.no_grad():
                pred.append(self.net.main_effect_blocks(batch_data).detach())
        return torch.vstack(pred)

    def get_interaction_raw_output(self, x):

        pred = []
        self.net.eval()
        train_size = x.shape[0]
        xx = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float, device=self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        for iterations in range(train_size // batch_size):
            offset = (iterations * batch_size) % train_size
            batch_data = xx[offset:(offset + batch_size), :]
            batch_data = torch.max(torch.min(batch_data, self.max_value), self.min_value) if self.boundary_clip else batch_xx
            batch_data = (batch_data - self.mu_list) / self.std_list if self.normalize else batch_data
            with torch.no_grad():
                pred.append(self.net.interaction_blocks(batch_data).detach())
        if train_size % batch_size > 0:
            batch_data = xx[((iterations + 1) * batch_size):, :]
            batch_data = torch.max(torch.min(batch_data, self.max_value), self.min_value) if self.boundary_clip else batch_xx
            batch_data = (batch_data - self.mu_list) / self.std_list if self.normalize else batch_data
            with torch.no_grad():
                pred.append(self.net.interaction_blocks(batch_data).detach())
        return torch.vstack(pred)

    def get_clarity_loss(self, x=None, sample_weight=None):

        clarity_loss = 0
        self.net.eval()
        x = self.tr_x if x is None else x
        xx = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float, device=self.device)
        data_generator = FastTensorDataLoader(xx, batch_size=self.batch_size_inference, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            self.net(batch_xx, sample_weight=sample_weight,
                       main_effect=True, interaction=True, clarity=True, monotonicity=False)
            clarity_loss += len(batch_data[0]) * self.net.clarity_loss.detach().cpu().numpy()
        clarity_loss = clarity_loss / data_generator.dataset_len
        return clarity_loss

    def get_mono_loss(self, x=None):

        mono_loss = 0
        self.net.eval()
        x = self.tr_x if x is None else x
        xx = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float, device=self.device)
        data_generator = FastTensorDataLoader(xx, batch_size=self.batch_size_inference, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            self.net(batch_xx, sample_weight=None,
                       main_effect=True, interaction=True, clarity=False, monotonicity=True)
            mono_loss += len(batch_data[0]) * self.net.mono_loss.detach().cpu().numpy()
        mono_loss = mono_loss / data_generator.dataset_len
        return mono_loss

    def certify_mono(self, n_samples=10000):

        x = np.random.uniform(self.min_value, self.max_value, size=(n_samples, len(self.max_value)))
        mono_loss = self.get_mono_loss(x)
        if mono_loss > 0:
            return False
        else:
            return True

    def decision_function(self, x, main_effect=True, interaction=True):

        pred = []
        self.net.eval()
        xx = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float, device=self.device)
        data_generator = FastTensorDataLoader(xx, batch_size=self.batch_size_inference, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            pred.append(self.net(batch_xx, sample_weight=None,
                       main_effect=main_effect, interaction=interaction, clarity=False, monotonicity=False).detach())
        return torch.vstack(pred)

    def evaluate(self, x, y, sample_weight, main_effect=True, interaction=True):

        pred = self.decision_function(x, main_effect=main_effect, interaction=interaction)
        return torch.mean(self.loss_fn(pred.ravel(),
                     y.ravel()) * sample_weight.ravel()).cpu().detach().numpy()

    def get_main_effect_rank(self):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.n_features)]
        beta = self.net.main_effect_weights.cpu().detach().numpy() ** 2 * self.main_effect_norm.reshape([-1, 1])
        componment_scales = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def get_interaction_rank(self):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.n_interactions)]
        if self.n_interactions > 0:
            gamma = self.net.interaction_weights.cpu().detach().numpy() ** 2 * self.interaction_norm.reshape([-1, 1])
            componment_scales = (np.abs(gamma) / np.sum(np.abs(gamma))).reshape([-1])
            sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def get_all_active_rank(self):

        componment_scales = [0 for i in range(self.n_features + self.n_interactions)]
        beta = (self.net.main_effect_weights.cpu().detach().numpy() ** 2 * np.array([self.main_effect_norm]).reshape([-1, 1])
             * self.net.main_effect_switcher.cpu().detach().numpy())

        gamma = np.empty((0, 1))
        if self.n_interactions > 0:
            gamma = (self.net.interaction_weights.cpu().detach().numpy()[:self.n_interactions] ** 2
                  * np.array([self.interaction_norm]).reshape([-1, 1])
                  * self.net.interaction_switcher.cpu().detach().numpy()[:self.n_interactions])

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def center_main_effects(self):

        output_bias = self.net.output_bias.cpu().detach().numpy()
        main_effect_weights = self.net.main_effect_switcher.cpu().detach().numpy() * self.net.main_effect_weights.cpu().detach().numpy()
        for i, idx in enumerate(self.nfeature_index_list):
            new_bias = self.net.main_effect_blocks.nsubnets.all_biases[-1][i].cpu().detach().numpy() - self.main_effect_mean[idx]
            self.net.main_effect_blocks.nsubnets.all_biases[-1].data[i] = torch.tensor(new_bias, dtype=torch.float, device=self.device)
            output_bias = output_bias + self.main_effect_mean[idx] * main_effect_weights[idx]
        for i, idx in enumerate(self.cfeature_index_list):
            new_bias = self.net.main_effect_blocks.csubnets.global_bias[i].cpu().detach().numpy() - self.main_effect_mean[idx]
            self.net.main_effect_blocks.csubnets.global_bias[i].data = torch.tensor(new_bias, dtype=torch.float, device=self.device)
            output_bias = output_bias + self.main_effect_mean[idx] * main_effect_weights[idx]
        self.net.output_bias.data = torch.tensor(output_bias, dtype=torch.float, device=self.device)

    def center_interactions(self):

        output_bias = self.net.output_bias.cpu().detach().numpy()
        interaction_weights = self.net.interaction_switcher.cpu().detach().numpy() * self.net.interaction_weights.cpu().detach().numpy()
        for idx in range(self.n_interactions):
            new_bias = self.net.interaction_blocks.subnets.all_biases[-1][idx].cpu().detach().numpy() - self.interaction_mean[idx]
            self.net.interaction_blocks.subnets.all_biases[-1].data[idx] = torch.tensor(new_bias, dtype=torch.float, device=self.device)
            output_bias = output_bias + self.interaction_mean[idx] * interaction_weights[idx]
        self.net.output_bias.data = torch.tensor(output_bias, dtype=torch.float, device=self.device)

    def prepare_data(self, x, y, sample_weight=None, stratified=False):

        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        indices = np.arange(self.n_samples)
        if sample_weight is None:
            sample_weight = np.ones(self.n_samples)
        else:
            sample_weight = self.n_samples * sample_weight.ravel() / np.sum(sample_weight)

        if self.meta_info is None:
            self.meta_info = {}
            for idx in range(self.n_features):
                self.meta_info["X" + str(idx + 1)] = {'type':'continuous'}

        self.dummy_values = {}
        self.cfeature_num = 0
        self.nfeature_num = 0
        self.cfeature_names = []
        self.nfeature_names = []
        self.cfeature_index_list = []
        self.nfeature_index_list = []
        self.num_classes_list = []
        
        self.mu_list = []
        self.std_list = []
        self.feature_names = []
        self.feature_types = []
        for idx, (feature_name, feature_info) in enumerate(self.meta_info.items()):
            if feature_info["type"] == "categorical":
                self.cfeature_num += 1
                self.cfeature_names.append(feature_name)
                self.cfeature_index_list.append(idx)
                categories_ = np.unique(x[:, [idx]])
                self.num_classes_list.append(len(categories_))
                self.dummy_values.update({feature_name: categories_})
                self.feature_types.append("categorical")
                self.feature_names.append(feature_name)
                self.mu_list.append(0)
                self.std_list.append(1)
            elif feature_info["type"] == "continuous":
                self.nfeature_num += 1
                self.nfeature_names.append(feature_name)
                self.nfeature_index_list.append(idx)
                self.feature_types.append("continuous")
                self.feature_names.append(feature_name)
                self.mu_list.append(x[:, idx].mean())
                self.std_list.append(x[:, idx].std())

        self.mu_list = torch.tensor(self.mu_list, dtype=torch.float, device=self.device)
        self.std_list = torch.tensor(self.std_list, dtype=torch.float, device=self.device)
        val_size = min(self.max_val_size, int(self.n_samples * self.val_ratio))
        if stratified:
            tr_x, val_x, tr_y, val_y, tr_sw, val_sw, tr_idx, val_idx = train_test_split(x, y, sample_weight,
                        indices, test_size=val_size, stratify=y, random_state=self.random_state)
        else:
            tr_x, val_x, tr_y, val_y, tr_sw, val_sw, tr_idx, val_idx = train_test_split(x, y, sample_weight,
                        indices, test_size=val_size, random_state=self.random_state)

        self.tr_idx = tr_idx
        self.val_idx = val_idx
        self.tr_x = torch.tensor(tr_x, dtype=torch.float, device=self.device)
        self.tr_y = torch.tensor(tr_y, dtype=torch.float, device=self.device)
        self.tr_sw = torch.tensor(tr_sw, dtype=torch.float, device=self.device)
        self.val_x = torch.tensor(val_x, dtype=torch.float, device=self.device)
        self.val_y = torch.tensor(val_y, dtype=torch.float, device=self.device)
        self.val_sw = torch.tensor(val_sw, dtype=torch.float, device=self.device)
        self.min_value = torch.tensor(np.min(x, axis=0), dtype=torch.float, device=self.device)
        self.max_value = torch.tensor(np.max(x, axis=0), dtype=torch.float, device=self.device)
        self.training_generator = FastTensorDataLoader(self.tr_x, self.tr_y, self.tr_sw,
                                        batch_size=self.batch_size, shuffle=True)
        self.max_iter_per_epoch = min(len(self.training_generator), self.max_iter_per_epoch)
        for i, item in enumerate(self.early_stop_thres):
            if item == "auto":
                self.early_stop_thres[i] = min(int(80000 / (self.max_iter_per_epoch * self.batch_size)), 100)
        self.estimate_density(x, sample_weight)

    def build_net(self, x, y, sample_weight):

        self.n_interactions = 0
        self.interaction_list = []
        self.n_features = self.nfeature_num + self.cfeature_num
        self.max_interact_num = int(round(self.n_features * (self.n_features - 1) / 2))
        self.interact_num = min(self.interact_num, self.max_interact_num)
        
        self.net = pyGAMINet(nfeature_index_list=self.nfeature_index_list,
                      cfeature_index_list=self.cfeature_index_list,
                      num_classes_list=self.num_classes_list,
                      subnet_size_main_effect=self.subnet_size_main_effect,
                      subnet_size_interaction=self.subnet_size_interaction,
                      activation_func=self.activation_func,
                      heredity=self.heredity,
                      mono_increasing_list=self.mono_increasing_list,
                      mono_decreasing_list=self.mono_decreasing_list,
                      boundary_clip=self.boundary_clip,
                      normalize=self.normalize,
                      min_value=self.min_value,
                      max_value=self.max_value,
                      mu_list=self.mu_list,
                      std_list=self.std_list,
                      device=self.device)

    def init_fit(self, x, y, sample_weight=None, stratified=False):

        # initialization
        self.active_indice = np.array([0])
        self.effect_names = np.array(["Intercept"])
        self.data_dict_density = {}
        self.err_train_main_effect_training = []
        self.err_val_main_effect_training = []
        self.err_train_interaction_training = []
        self.err_val_interaction_training = []
        self.err_train_tuning = []
        self.err_val_tuning = []

        self.interaction_list = []
        self.active_main_effect_index = []
        self.active_interaction_index = []
        self.main_effect_val_loss = []
        self.interaction_val_loss = []

        # the seed may not work for data loader. this needs to be checked
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self._validate_input(x, y)
        self.prepare_data(x, y, sample_weight, stratified)
        self.build_net(x, y, sample_weight)

    def fit_individual_subnet(self, x, y, subnet, idx, loss_fn, max_epochs=1000, batch_size=200, early_stop_thres=10):

        last_improvement = 0
        best_validation = np.inf
        opt = torch.optim.Adam(list(subnet.parameters()), lr=0.01)
        training_generator = FastTensorDataLoader(torch.tensor(x, dtype=torch.float, device=self.device),
                                    torch.tensor(y, dtype=torch.float, device=self.device),
                                    batch_size=min(200, int(0.2 * x.shape[0])), shuffle=True)
        for epoch in range(max_epochs):
            self.net.train()
            accumulated_size = 0.0
            accumulated_loss = 0.0
            for batch_no, batch_data in enumerate(training_generator):
                opt.zero_grad(set_to_none=True)
                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                pred = subnet.individual_forward(batch_xx, idx).ravel()
                loss = torch.mean(loss_fn(pred, batch_yy))
                loss.backward()
                opt.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
            self.net.eval()
            accumulated_loss = accumulated_loss / accumulated_size
            if accumulated_loss < best_validation:
                    best_validation = accumulated_loss
                    last_improvement = epoch
            if epoch - last_improvement > early_stop_thres:
                break

    def warm_start_main_effect(self):

        if not self.warm_start:
            return 

        if self.verbose:
            print("#" * 15 + "Run Warm Initialization for Main Effect" + "#" * 15)

        surrogate_estimator, intercept = self.build_teacher_main_effect()
        self.net.output_bias.data = self.net.output_bias.data + torch.tensor(intercept, dtype=torch.float, device=self.device)
        for idx in range(self.n_features):
            if idx in self.nfeature_index_list:
                simu_xx = np.zeros((self.mlp_sample_size, self.n_features))
                simu_xx[:, idx] = np.random.uniform(self.min_value[idx], self.max_value[idx], self.mlp_sample_size)
                if self.normalize:
                    simu_xx[:, idx] = (simu_xx[:, idx] - self.mu_list[idx].detach().numpy()) / self.std_list[idx].detach().numpy()
                simu_yy = surrogate_estimator[idx](simu_xx)
                self.fit_individual_subnet(simu_xx[:, [idx]], simu_yy, self.net.main_effect_blocks.nsubnets,
                                  self.nfeature_index_list.index(idx), loss_fn=torch.nn.MSELoss(reduction="none"))
            if idx in self.cfeature_index_list:
                i = self.cfeature_index_list.index(idx)
                simu_xx = np.zeros((self.num_classes_list[i], self.n_features))
                simu_xx[:, idx] = np.linspace(self.min_value[idx], self.max_value[idx], self.num_classes_list[i])
                simu_yy = surrogate_estimator[idx](simu_xx)
                self.net.main_effect_blocks.csubnets.class_bias[i].data = torch.tensor(simu_yy.reshape(-1, 1),
                                                            dtype=torch.float, device=self.device)

    def warm_start_interaction(self):

        if not self.warm_start:
            return 

        if self.verbose:
            print("#" * 15 + "Run Warm Initialization for Interaction" + "#" * 15)

        surrogate_estimator, intercept = self.build_teacher_interaction()
        self.net.output_bias.data = self.net.output_bias.data + torch.tensor(intercept, dtype=torch.float, device=self.device)
        for i, (idx1, idx2) in enumerate(self.interaction_list):
            simu_xx = np.zeros((self.mlp_sample_size, self.n_features))
            if idx1 in self.cfeature_index_list:
                num_classes = self.num_classes_list[self.cfeature_index_list.index(idx1)]
                simu_xx[:, idx1] = np.random.randint(self.min_value[idx1], self.max_value[idx1] + 1, self.mlp_sample_size)
                x1 = torch.nn.functional.one_hot(torch.tensor(simu_xx[:, idx1]).to(torch.int64),
                                      num_classes=num_classes).to(torch.float).detach().cpu().numpy()
            else:
                simu_xx[:, idx1] = np.random.uniform(self.min_value[idx1], self.max_value[idx1], self.mlp_sample_size)
                if self.normalize:
                    simu_xx[:, idx1] = (simu_xx[:, idx1] - self.mu_list[idx1].detach().numpy()) / self.std_list[idx1].detach().numpy()
                x1 = simu_xx[:, [idx1]]
            if idx2 in self.cfeature_index_list:
                num_classes = self.num_classes_list[self.cfeature_index_list.index(idx2)]
                simu_xx[:, idx2] = np.random.randint(self.min_value[idx2], self.max_value[idx2] + 1, self.mlp_sample_size)
                x2 = torch.nn.functional.one_hot(torch.tensor(simu_xx[:, idx2]).to(torch.int64), 
                                      num_classes=num_classes).to(torch.float).detach().cpu().numpy()
            else:
                simu_xx[:, idx2] = np.random.uniform(self.min_value[idx2], self.max_value[idx2], self.mlp_sample_size)
                if self.normalize:
                    simu_xx[:, idx2] = (simu_xx[:, idx2] - self.mu_list[idx2].detach().numpy()) / self.std_list[idx2].detach().numpy()
                x2 = simu_xx[:, [idx2]]

            xx = np.hstack([x1, x2])
            xx = np.hstack([xx, np.zeros((xx.shape[0], self.net.interaction_blocks.max_n_inputs - xx.shape[1]))])
            yy = surrogate_estimator[i](simu_xx)
            self.fit_individual_subnet(xx, yy, self.net.interaction_blocks.subnets,
                              i, loss_fn=torch.nn.MSELoss(reduction="none"))

    def _get_interaction_list(self, x, y, w, scores, feature_names,
                              feature_types, n_jobs, model_type, num_classes):

        active_main_effect_index = self.active_main_effect_index if self.heredity else np.arange(self.n_features)
        if (len(active_main_effect_index) == 0):
            return []

        preprocessor_ = EBMPreprocessor(feature_names=feature_names,
                              feature_types=feature_types)
        preprocessor_.fit(x)
        X_pair = preprocessor_.transform(x)
        features_categorical = np.array([tp == "categorical" for tp in preprocessor_.col_types_], dtype=np.int64)
        features_bin_count = np.array([len(nbin) for nbin in preprocessor_.col_bin_counts_], dtype=np.int64)

        X = np.ascontiguousarray(X_pair.T).astype(np.int64)
        y = y.ravel()
        w = w.astype(np.float64)
        scores = scores.ravel().astype(np.float64) 

        with InteractionDetector(
            model_type, num_classes, features_categorical, features_bin_count, X, y, w, scores, optional_temp_params=None
        ) as interaction_detector:

            def evaluate_parallel(pair):
                score = interaction_detector.get_interaction_score(pair, min_samples_leaf=2)
                return pair, score

            all_pairs = [pair for pair in combinations(range(len(preprocessor_.col_types_)), 2)
               if (pair[0] in active_main_effect_index) or (pair[1] in active_main_effect_index)]
            interaction_scores = Parallel(n_jobs=n_jobs, backend="threading")(delayed(evaluate_parallel)(pair) for pair in all_pairs)

        ranked_scores = list(sorted(interaction_scores, key=lambda item: item[1], reverse=True))
        interaction_list = [ranked_scores[i][0] for i in range(len(ranked_scores))]
        return interaction_list

    def fit_main_effect(self):

        if self.verbose:
            print("#" * 20 + "Stage 1: Main Effect Training" + "#" * 20)

        last_improvement = 0
        best_validation = np.inf
        self.warm_start_main_effect()
        train_size = self.tr_x.shape[0]
        opt = torch.optim.Adam(list(self.net.main_effect_blocks.parameters()) +
                        [self.net.main_effect_weights, self.net.output_bias], lr=self.learning_rates[0])
        for epoch in range(self.max_epochs[0]):
            self.net.train()
            accumulated_size = 0
            accumulated_loss = 0.0
            if self.verbose:
                pbar = tqdm(self.training_generator, total=self.max_iter_per_epoch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            else:
                pbar = self.training_generator
            for batch_no, batch_data in enumerate(pbar):
                if batch_no >= self.max_iter_per_epoch:
                    break
                opt.zero_grad(set_to_none=True)
                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                batch_sw = batch_data[2].to(self.device).ravel()
                pred = self.net(batch_xx, sample_weight=batch_sw,
                           main_effect=True, interaction=False,
                           clarity=False,
                           monotonicity=self.monotonicity and self.reg_mono > 0).ravel()

                if self.monotonicity and self.reg_mono > 0:
                    mono_loss_reg = self.reg_mono * self.net.mono_loss
                    simu_inputs = np.random.uniform(self.min_value, self.max_value, size=(self.mono_sample_size, len(self.max_value)))
                    simu_inputs = torch.tensor(simu_inputs, dtype=torch.float, device=self.device)
                    simu_pred = self.net(simu_inputs,
                               main_effect=True, interaction=False,
                               clarity=False,
                               monotonicity=self.monotonicity and self.reg_mono > 0).ravel()
                    mono_loss_reg = mono_loss_reg + self.reg_mono * self.net.mono_loss
                    mono_loss_reg.backward(retain_graph=True)

                loss = torch.mean(self.loss_fn(pred, batch_yy) * batch_sw)
                loss.backward()
                opt.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
                if self.verbose:
                    pbar.set_description(("Epoch: %" + str(int(np.ceil(np.log10(self.max_epochs[0]))) + 1)
                                  + "d, train loss: %0.5f") % (epoch + 1, accumulated_loss / accumulated_size))
                if batch_no == (len(self.training_generator) - 1) or batch_no == (self.max_iter_per_epoch - 1):
                    self.net.eval()
                    self.err_train_main_effect_training.append(accumulated_loss / accumulated_size)
                    self.err_val_main_effect_training.append(self.evaluate(self.val_x, self.val_y, self.val_sw,
                                                         main_effect=True, interaction=False))
                    if self.verbose:
                        pbar.set_description(("Epoch: %" + str(int(np.ceil(np.log10(self.max_epochs[0]))) + 1)
                              + "d, train loss: %0.5f, validation loss: %0.5f") % 
                              (epoch + 1, self.err_train_main_effect_training[-1], self.err_val_main_effect_training[-1]))

            if self.err_val_main_effect_training[-1] < best_validation:
                best_validation = self.err_val_main_effect_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres[0]:
                if self.monotonicity and self.reg_mono > 0:
                    if self.certify_mono(n_samples=self.mono_sample_size):
                        break
                    else:
                        self.reg_mono = min(self.reg_mono * 1.2, 1e5)
                else:
                    break

        if self.verbose:
            print("Main Effect Training Stop at Epoch: %d, train loss: %0.5f, validation loss: %0.5f" %
                        (epoch + 1, self.err_train_main_effect_training[-1], self.err_val_main_effect_training[-1]))  

        main_effect_output = self.get_main_effect_raw_output(self.tr_x).cpu().numpy()
        self.main_effect_mean = np.average(main_effect_output, axis=0, weights=self.tr_sw.cpu().numpy())
        self.main_effect_norm = np.diag(np.cov(main_effect_output.T,
                       aweights=self.tr_sw.cpu().numpy()).reshape(self.n_features, self.n_features))
        self.center_main_effects()
        torch.cuda.empty_cache()

    def prune_main_effect(self):

        self.main_effect_val_loss = []
        sorted_index, componment_scales = self.get_main_effect_rank()
        self.net.main_effect_switcher.data = torch.tensor(np.zeros((self.n_features, 1)), dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.main_effect_val_loss.append(self.evaluate(self.val_x, self.val_y, self.val_sw,
                                        main_effect=True, interaction=False))
        for idx in range(self.n_features):
            selected_index = sorted_index[:(idx + 1)]
            main_effect_switcher = np.zeros((self.n_features, 1))
            main_effect_switcher[selected_index] = 1
            self.net.main_effect_switcher.data = torch.tensor(main_effect_switcher, dtype=torch.float,
                                              device=self.device, requires_grad=False)
            val_loss = self.evaluate(self.val_x,
                             self.val_y,
                             self.val_sw,
                             main_effect=True, interaction=False)
            self.main_effect_val_loss.append(val_loss)

        best_idx = np.argmin(self.main_effect_val_loss)
        best_loss = np.min(self.main_effect_val_loss)
        if best_loss > 0:
            if np.sum((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold)[0][0]
            
        self.active_main_effect_index = sorted_index[:best_idx]
        main_effect_switcher = np.zeros((self.n_features, 1))
        main_effect_switcher[self.active_main_effect_index] = 1
        self.net.main_effect_switcher.data = torch.tensor(main_effect_switcher, dtype=torch.float, device=self.device)
        self.net.main_effect_switcher.requires_grad = False

    def add_interaction(self):

        x = torch.vstack([self.tr_x, self.val_x])
        y = torch.vstack([self.tr_y, self.val_y])
        w = torch.hstack([self.tr_sw, self.val_sw])
        scores = self.decision_function(x, main_effect=True, interaction=False)
        interaction_list_all = self.get_interaction_list(x.detach().cpu().numpy(),
                                         y.detach().cpu().numpy(),
                                         w.detach().numpy(),
                                         scores.detach().cpu().numpy(),
                                         self.feature_names,
                                         self.feature_types,
                                         n_jobs=1)

        self.interaction_list = interaction_list_all[:self.interact_num]
        self.n_interactions = len(self.interaction_list)
        self.net.init_interaction_blocks(self.interaction_list)

    def fit_interaction(self):

        if not self.net.interaction_status:
            return 

        if self.verbose:
            print("#" * 20 + "Stage 2: Interaction Training" + "#" * 20)

        last_improvement = 0
        best_validation = np.inf
        train_size = self.tr_x.shape[0]
        self.warm_start_interaction()
        opt = torch.optim.Adam(list(self.net.interaction_blocks.parameters()) +
                        [self.net.interaction_weights, self.net.output_bias], lr=self.learning_rates[1])
        
        for epoch in range(self.max_epochs[1]):
            self.net.train()
            accumulated_size = 0
            accumulated_loss = 0.0
            if self.verbose:
                pbar = tqdm(self.training_generator, total=self.max_iter_per_epoch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            else:
                pbar = self.training_generator
            for batch_no, batch_data in enumerate(pbar):
                if batch_no >= self.max_iter_per_epoch:
                    break
                opt.zero_grad(set_to_none=True)
                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                batch_sw = batch_data[2].to(self.device).ravel()
                pred = self.net(batch_xx, sample_weight=batch_sw,
                           main_effect=True, interaction=True,
                           clarity=self.reg_clarity > 0,
                           monotonicity=self.monotonicity and self.reg_mono > 0).ravel()
                clarity_loss_reg = self.reg_clarity * self.net.clarity_loss
                clarity_loss_reg.backward(retain_graph=True)

                if self.monotonicity and self.reg_mono > 0:
                    mono_loss_reg = self.reg_mono * self.net.mono_loss
                    simu_inputs = np.random.uniform(self.min_value, self.max_value, size=(self.mono_sample_size, len(self.max_value)))
                    simu_inputs = torch.tensor(simu_inputs, dtype=torch.float, device=self.device)
                    simu_pred = self.net(simu_inputs,
                               main_effect=True, interaction=True,
                               clarity=self.reg_clarity > 0,
                               monotonicity=self.monotonicity and self.reg_mono > 0).ravel()
                    mono_loss_reg = mono_loss_reg + self.reg_mono * self.net.mono_loss
                    mono_loss_reg.backward(retain_graph=True)

                loss = torch.mean(self.loss_fn(pred, batch_yy) * batch_sw) 
                loss.backward()
                opt.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
                if self.verbose:
                    pbar.set_description(("Epoch: %" + str(int(np.ceil(np.log10(self.max_epochs[1]))) + 1)
                                  + "d, train loss: %0.5f") % (epoch + 1, accumulated_loss / accumulated_size)) 
                if batch_no == (len(self.training_generator) - 1) or batch_no == (self.max_iter_per_epoch - 1):
                    self.net.eval()
                    self.err_train_interaction_training.append(accumulated_loss / accumulated_size)
                    self.err_val_interaction_training.append(self.evaluate(self.val_x, self.val_y, self.val_sw,
                                                         main_effect=True, interaction=True))
                    if self.verbose:
                        pbar.set_description(("Epoch: %" + str(int(np.ceil(np.log10(self.max_epochs[1]))) + 1)
                              + "d, train loss: %0.5f, validation loss: %0.5f") % 
                              (epoch + 1, self.err_train_interaction_training[-1], self.err_val_interaction_training[-1]))

            if self.err_val_interaction_training[-1] < best_validation:
                best_validation = self.err_val_interaction_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres[1]:
                if self.monotonicity and self.reg_mono > 0:
                    if self.certify_mono(n_samples=self.mono_sample_size):
                        break
                    else:
                        self.reg_mono = min(self.reg_mono * 1.2, 1e5)
                else:
                    break

        if self.verbose:
            print("Interaction Training Stop at Epoch: %d, train loss: %0.5f, validation loss: %0.5f" %
              (epoch + 1, self.err_train_interaction_training[-1], self.err_val_interaction_training[-1]))  

        interaction_output = self.get_interaction_raw_output(self.tr_x).cpu().numpy()
        self.interaction_mean = np.average(interaction_output, axis=0, weights=self.tr_sw.cpu().numpy())
        self.interaction_norm = np.diag(np.cov(interaction_output.T,
                               aweights=self.tr_sw.cpu().numpy()).reshape(self.n_interactions, self.n_interactions))
        self.center_interactions()
        torch.cuda.empty_cache()

    def prune_interaction(self):
        
        if self.n_interactions == 0:
            return 

        self.interaction_val_loss = []
        sorted_index, componment_scales = self.get_interaction_rank()
        self.net.interaction_switcher.data = torch.tensor(np.zeros((self.n_interactions, 1)), dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.interaction_val_loss.append(self.evaluate(self.val_x, self.val_y, self.val_sw,
                              main_effect=True, interaction=True))
        for idx in range(self.n_interactions):
            selected_index = sorted_index[:(idx + 1)]
            interaction_switcher = np.zeros((self.n_interactions, 1))
            interaction_switcher[selected_index] = 1
            self.net.interaction_switcher.data = torch.tensor(interaction_switcher, dtype=torch.float,
                                              device=self.device, requires_grad=False)
            val_loss = self.evaluate(self.val_x, self.val_y, self.val_sw,
                             main_effect=True, interaction=True)
            self.interaction_val_loss.append(val_loss)

        best_idx = np.argmin(self.interaction_val_loss)
        best_loss = np.min(self.interaction_val_loss)
        if best_loss > 0:
            if np.sum((self.interaction_val_loss / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.interaction_val_loss / best_loss - 1) < self.loss_threshold)[0][0]
            
        self.active_interaction_index = sorted_index[:best_idx]
        interaction_switcher = np.zeros((self.n_interactions, 1))
        interaction_switcher[self.active_interaction_index] = 1
        self.net.interaction_switcher.data = torch.tensor(interaction_switcher, dtype=torch.float, device=self.device)
        self.net.interaction_switcher.requires_grad = False

    def fine_tune_all(self):

        if self.verbose:
            print("#" * 25 + "Stage 3: Fine Tuning" + "#" * 25)

        last_improvement = 0
        best_validation = np.inf
        train_size = self.tr_x.shape[0]
        opt_main_effect = torch.optim.Adam(list(self.net.main_effect_blocks.parameters()) +
                                [self.net.main_effect_weights, self.net.output_bias],
                                lr=self.learning_rates[2])
        if self.n_interactions > 0:
            opt_interaction = torch.optim.Adam(list(self.net.interaction_blocks.parameters()) +
                                   [self.net.interaction_weights], lr=self.learning_rates[2])
        for epoch in range(self.max_epochs[2]):
            self.net.train()
            accumulated_size = 0
            accumulated_loss = 0.0
            if self.verbose:
                pbar = tqdm(self.training_generator, total=self.max_iter_per_epoch, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            else:
                pbar = self.training_generator
            for batch_no, batch_data in enumerate(pbar):
                if batch_no >= self.max_iter_per_epoch:
                    break

                opt_main_effect.zero_grad(set_to_none=True)
                if self.n_interactions > 0:
                    opt_interaction.zero_grad(set_to_none=True)

                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                batch_sw = batch_data[2].to(self.device).ravel()

                pred = self.net(batch_xx, sample_weight=batch_sw,
                           main_effect=True, interaction=True,
                           clarity=self.net.interaction_status and self.reg_clarity > 0,
                           monotonicity=self.monotonicity and self.reg_mono > 0).ravel()
                clarity_loss_reg = self.reg_clarity * self.net.clarity_loss
                clarity_loss_reg.backward(retain_graph=True)
                opt_main_effect.zero_grad(set_to_none=True)

                if self.monotonicity and self.reg_mono > 0:
                    mono_loss_reg = self.reg_mono * self.net.mono_loss
                    simu_inputs = np.random.uniform(self.min_value, self.max_value, size=(self.mono_sample_size, len(self.max_value)))
                    simu_inputs = torch.tensor(simu_inputs, dtype=torch.float, device=self.device)
                    simu_pred = self.net(simu_inputs,
                               main_effect=True, interaction=True,
                               clarity=self.reg_clarity > 0,
                               monotonicity=self.monotonicity and self.reg_mono > 0).ravel()
                    mono_loss_reg = mono_loss_reg + self.reg_mono * self.net.mono_loss
                    mono_loss_reg.backward(retain_graph=True)

                loss = torch.mean(self.loss_fn(pred, batch_yy) * batch_sw) 
                loss.backward()
                opt_main_effect.step()
                if self.n_interactions > 0:
                    opt_interaction.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
                if self.verbose:
                    pbar.set_description(("Epoch: %" + str(int(np.ceil(np.log10(self.max_epochs[2]))) + 1)
                                  + "d, train loss: %0.5f") % (epoch + 1, accumulated_loss / accumulated_size))  
                if batch_no == (len(self.training_generator) - 1) or batch_no == (self.max_iter_per_epoch - 1):
                    self.net.eval()
                    self.err_train_tuning.append(accumulated_loss / accumulated_size)
                    self.err_val_tuning.append(self.evaluate(self.val_x, self.val_y, self.val_sw,
                                                         main_effect=True, interaction=True))
                    if self.verbose:
                        pbar.set_description(("Epoch: %" + str(int(np.ceil(np.log10(self.max_epochs[2]))) + 1)
                              + "d, train loss: %0.5f, validation loss: %0.5f") % 
                              (epoch + 1, self.err_train_tuning[-1], self.err_val_tuning[-1]))

            if self.err_val_tuning[-1] < best_validation:
                best_validation = self.err_val_tuning[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres[2]:
                if self.monotonicity and self.reg_mono > 0:
                    if self.certify_mono(n_samples=self.mono_sample_size):
                        break
                    else:
                        self.reg_mono = min(self.reg_mono * 1.2, 1e5)
                else:
                    break

        if self.verbose:
            print("Fine Tuning Stop at Epoch: %d, train loss: %0.5f, validation loss: %0.5f" %
                      (epoch + 1, self.err_train_tuning[-1], self.err_val_tuning[-1]))

        main_effect_output = self.get_main_effect_raw_output(self.tr_x).cpu().numpy()
        self.main_effect_mean = np.average(main_effect_output, axis=0, weights=self.tr_sw.cpu().numpy())
        self.main_effect_norm = np.diag(np.cov(main_effect_output.T,
                               aweights=self.tr_sw.cpu().numpy()).reshape(self.n_features, self.n_features))
        self.center_main_effects()

        if self.n_interactions > 0:
            interaction_output = self.get_interaction_raw_output(self.tr_x).cpu().numpy()
            self.interaction_mean = np.average(interaction_output, axis=0, weights=self.tr_sw.cpu().numpy())
            self.interaction_norm = np.diag(np.cov(interaction_output.T,
                               aweights=self.tr_sw.cpu().numpy()).reshape(self.n_interactions, self.n_interactions))
            self.center_interactions()

        torch.cuda.empty_cache()

    def _fit(self):

        self.fit_main_effect()
        self.prune_main_effect()
        self.add_interaction()
        self.fit_interaction()
        self.prune_interaction()
        self.fine_tune_all()

        self.active_indice = 1 + np.hstack([-1, self.active_main_effect_index,
                                    self.n_features + np.array(self.active_interaction_index)]).astype(int)
        self.effect_names = np.hstack(["Intercept", np.array(self.feature_names), [self.feature_names[self.interaction_list[i][0]] + " x "
                          + self.feature_names[self.interaction_list[i][1]] for i in range(len(self.interaction_list))]])
        self.is_fitted_ = True

    def summary_logs(self, save_dict=False, folder="./", name="summary_logs"):

        data_dict_log = {}
        data_dict_log.update({"err_train_main_effect_training": self.err_train_main_effect_training,
                       "err_val_main_effect_training": self.err_val_main_effect_training,
                       "err_train_interaction_training": self.err_train_interaction_training,
                       "err_val_interaction_training": self.err_val_interaction_training,
                       "err_train_tuning": self.err_train_tuning,
                       "err_val_tuning": self.err_val_tuning,
                       "interaction_list": self.interaction_list,
                       "active_main_effect_index": self.active_main_effect_index,
                       "active_interaction_index": self.active_interaction_index,
                       "main_effect_val_loss": self.main_effect_val_loss,
                       "interaction_val_loss": self.interaction_val_loss})
        
        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_log)

        return data_dict_log

    def partial_derivatives(self, feature_idx, n_samples=10000):
        
        np.random.seed(self.random_state)
        inputs = np.random.uniform(self.min_value, self.max_value, size=(n_samples, len(self.max_value)))
        inputs = torch.tensor(inputs, dtype=torch.float, device=self.device)
        outputs = self.net(inputs)
        grad = torch.autograd.grad(outputs=torch.sum(outputs),
                           inputs=inputs, create_graph=True)[0].detach().numpy()
        plt.scatter(inputs.detach().numpy()[:, feature_idx], grad[:, feature_idx])
        plt.axhline(0, linestyle="--", linewidth=0.5, color="red")
        plt.ylabel("First-order Derivatives")
        plt.xlabel(self.feature_names[feature_idx])
        absmax = 1.05 * np.max(np.abs(grad[:, feature_idx]))
        plt.ylim(-absmax, absmax)
        if feature_idx in self.mono_increasing_list:
            plt.title("Violating Size: %0.2f%%" % (np.where(grad[:, feature_idx] < 0)[0].shape[0] / n_samples * 100))
        if feature_idx in self.mono_decreasing_list:
            plt.title("Violating Size: %0.2f%%" % (np.where(grad[:, feature_idx] > 0)[0].shape[0] / n_samples * 100))
        plt.show()

    def global_explain(self, main_grid_size=100, interact_grid_size=100, save_dict=False, folder="./", name="global_explain"):

        # By default, we use the same main_grid_size and interact_grid_size as that of the zero mean constraint
        # Alternatively, we can also specify it manually, e.g., when we want to have the same grid size as EBM (256).        
        data_dict_global = self.data_dict_density
        sorted_index, componment_scales = self.get_all_active_rank()
        for idx in range(self.n_features):
            feature_name = self.feature_names[idx]
            if idx in self.nfeature_index_list:
                main_effect_inputs = np.zeros((main_grid_size, self.n_features))
                main_effect_inputs[:, idx] = np.linspace(self.min_value[idx], self.max_value[idx], main_grid_size)
                main_effect_inputs_original = main_effect_inputs[:, [idx]]
                main_effect_outputs = (self.net.main_effect_weights.cpu().detach().numpy()[idx]
                                * self.net.main_effect_switcher.cpu().detach().numpy()[idx]
                                * self.get_main_effect_raw_output(main_effect_inputs).cpu().numpy()[:, idx])
                data_dict_global[feature_name].update({"type":"continuous",
                                      "importance":componment_scales[idx],
                                      "inputs":main_effect_inputs_original.ravel(),
                                      "outputs":main_effect_outputs.ravel()})
            elif idx in self.cfeature_index_list:
                main_effect_inputs_original = self.dummy_values[feature_name]
                main_effect_inputs = np.zeros((len(main_effect_inputs_original), self.n_features))
                main_effect_inputs[:, idx] = np.arange(len(main_effect_inputs_original))
                main_effect_outputs = (self.net.main_effect_weights.cpu().detach().numpy()[idx]
                        * self.net.main_effect_switcher.cpu().detach().numpy()[idx]
                        * self.get_main_effect_raw_output(main_effect_inputs).cpu().numpy()[:, idx])
                main_effect_input_ticks = (main_effect_inputs.ravel().astype(int) if len(main_effect_inputs_original) <= 6 else
                              np.linspace(0.1 * len(main_effect_inputs_original), len(main_effect_inputs_original) * 0.9, 4).astype(int))
                main_effect_input_labels = [main_effect_inputs_original[i] for i in main_effect_input_ticks]
                if len("".join(list(map(str, main_effect_input_labels)))) > 30:
                    main_effect_input_labels = [str(main_effect_inputs_original[i])[:4] for i in main_effect_input_ticks]
                data_dict_global[feature_name].update({"feature_name": feature_name,
                                      "type": "categorical",
                                      "importance": componment_scales[idx],
                                      "inputs": main_effect_inputs_original,
                                      "outputs": main_effect_outputs.ravel(),
                                      "input_ticks": main_effect_input_ticks,
                                      "input_labels": main_effect_input_labels})

        for idx in range(self.n_interactions):

            idx1 = self.interaction_list[idx][0]
            idx2 = self.interaction_list[idx][1]
            feature_name1 = self.feature_names[idx1]
            feature_name2 = self.feature_names[idx2]
            feature_type1 = "categorical" if feature_name1 in self.cfeature_names else "continuous"
            feature_type2 = "categorical" if feature_name2 in self.cfeature_names else "continuous"
            
            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.cfeature_names:
                interact_input1_original = self.dummy_values[feature_name1]
                interact_input1 = np.arange(len(interact_input1_original), dtype=np.float32)
                interact_input1_ticks = (interact_input1.astype(int) if len(interact_input1) <= 6 else 
                                 np.linspace(0.1 * len(interact_input1), len(interact_input1) * 0.9, 4).astype(int))
                interact_input1_labels = [interact_input1_original[i] for i in interact_input1_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [str(interact_input1_original[i])[:4] for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(interact_input1_original) - 0.5])
            else:
                interact_input1 = np.array(np.linspace(self.min_value[idx1], self.max_value[idx1], interact_grid_size), dtype=np.float32)
                interact_input1_original = interact_input1.reshape(-1, 1)
                interact_input1_ticks = []
                interact_input1_labels = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.cfeature_names:
                interact_input2_original = self.dummy_values[feature_name2]
                interact_input2 = np.arange(len(interact_input2_original), dtype=np.float32)
                interact_input2_ticks = (interact_input2.astype(int) if len(interact_input2) <= 6 else
                                 np.linspace(0.1 * len(interact_input2), len(interact_input2) * 0.9, 4).astype(int))
                interact_input2_labels = [interact_input2_original[i] for i in interact_input2_ticks]
                if len("".join(list(map(str, interact_input2_labels)))) > 30:
                    interact_input2_labels = [str(interact_input2_original[i])[:4] for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(interact_input2_original) - 0.5])
            else:
                interact_input2 = np.array(np.linspace(self.min_value[idx2], self.max_value[idx2], interact_grid_size), dtype=np.float32)
                interact_input2_original = interact_input2.reshape(-1, 1)
                interact_input2_ticks = []
                interact_input2_labels = []
                interact_input_list.append(interact_input2)
                axis_extent.extend([interact_input2_original.min(), interact_input2_original.max()])

            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            interaction_inputs = np.zeros((x1.shape[0] * x1.shape[1], self.n_features))
            interaction_inputs[:, self.interaction_list[idx][0]] = x1.ravel()
            interaction_inputs[:, self.interaction_list[idx][1]] = x2.ravel()
            interact_outputs = (self.net.interaction_weights.cpu().detach().numpy()[idx]
                    * self.net.interaction_switcher.cpu().detach().numpy()[idx]
                    * self.get_interaction_raw_output(interaction_inputs).cpu().numpy())[:, idx].reshape(x1.shape)
            data_dict_global.update({feature_name1 + " x " + feature_name2:{"feature_name1": feature_name1,
                                                       "feature_name2": feature_name2,
                                                       "type": "pairwise",
                                                       "xtype": feature_type1,
                                                       "ytype": feature_type2,
                                                       "importance": componment_scales[self.n_features + idx],
                                                       "input1": interact_input1_original,
                                                       "input2": interact_input2_original,
                                                       "outputs": interact_outputs,
                                                       "input1_ticks": interact_input1_ticks,
                                                       "input2_ticks": interact_input2_ticks,
                                                       "input1_labels": interact_input1_labels,
                                                       "input2_labels": interact_input2_labels,
                                                       "axis_extent": axis_extent}})

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_global)
            
        return data_dict_global

    def local_explain(self, x, y=None, save_dict=False, folder="./", name="local_explain"):

        predicted = self.predict(x)
        intercept = self.net.output_bias.cpu().detach().numpy()

        main_effect_output = self.get_main_effect_raw_output(x).cpu().numpy()
        if self.n_interactions > 0:
            interaction_output = self.get_interaction_raw_output(x).cpu().numpy()
            interaction_weights = ((self.net.interaction_weights.cpu().detach().numpy())
                      * self.net.interaction_switcher.cpu().detach().numpy()).ravel()
        else:
            interaction_output = np.empty(shape=(x.shape[0], 0))
            interaction_weights = np.empty(shape=(0))

        main_effect_weights = ((self.net.main_effect_weights.cpu().detach().numpy()) * self.net.main_effect_switcher.cpu().detach().numpy()).ravel()
        scores = np.hstack([np.repeat(intercept[0], x.shape[0]).reshape(-1, 1), np.hstack([main_effect_weights, interaction_weights])
                                  * np.hstack([main_effect_output, interaction_output])])

        data_dict_local = [{"active_indice": self.active_indice,
                    "scores": scores[i],
                    "effect_names": self.effect_names,
                    "predicted": predicted[i],
                    "actual": y[i]} for i in range(x.shape[0])]

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_local)

        return data_dict_local

    def load(self, folder="./", name="demo"):
        
        save_path_dict = folder + name + "_dict.pickle"
        save_path_model = folder + name + "_model.pickle"
        if not os.path.exists(save_path_dict):
            raise "dict file not found!"
        if not os.path.exists(save_path_model):
            raise "model file not found!"

        with open(save_path_dict, "rb") as input_file:
            model_dict = pickle.load(input_file)
        for key, item in model_dict.items():
            setattr(self, key, item)
        self.net = torch.load(save_path_model)

    def save(self, folder="./", name="demo"):

        model_dict = {}
        model_dict["meta_info"] = self.meta_info
        model_dict["subnet_size_main_effect"] = self.subnet_size_main_effect
        model_dict["subnet_size_interaction"] = self.subnet_size_interaction

        model_dict["learning_rates"] = self.learning_rates
        model_dict["batch_size"] = self.batch_size
        model_dict["batch_size_inference"] = self.batch_size_inference
        
        model_dict["activation_func"] = self.activation_func
        model_dict["max_epochs"] = self.max_epochs
        model_dict["early_stop_thres"] = self.early_stop_thres
        model_dict["max_iter_per_epoch"] = self.max_iter_per_epoch

        model_dict["heredity"] = self.heredity
        model_dict["reg_clarity"] = self.reg_clarity
        model_dict["loss_threshold"] = self.loss_threshold

        
        model_dict["reg_mono"] = self.reg_mono
        
        model_dict["mono_decreasing_list"] = self.mono_increasing_list
        model_dict["mono_increasing_list"] = self.mono_increasing_list
        model_dict["gam_sample_size"] = self.gam_sample_size
        model_dict["mlp_sample_size"] = self.mlp_sample_size
        model_dict["max_val_size"] = self.max_val_size
        model_dict["boundary_clip"] = self.boundary_clip
        model_dict["normalize"] = self.normalize
        model_dict["warm_start"] = self.warm_start

        model_dict["verbose"] = self.verbose
        model_dict["val_ratio"]= self.val_ratio
        model_dict["random_state"] = self.random_state

        model_dict["dummy_values"] = self.dummy_values
        model_dict["cfeature_num"] = self.cfeature_num
        model_dict["nfeature_num"] = self.nfeature_num
        model_dict["feature_names"] = self.feature_names
        model_dict["feature_types"] = self.feature_types
        model_dict["cfeature_names"] = self.cfeature_names
        model_dict["nfeature_names"] = self.nfeature_names
        model_dict["cfeature_index_list"] = self.cfeature_index_list
        model_dict["nfeature_index_list"] = self.nfeature_index_list

        model_dict["interaction_list"] = self.interaction_list
        model_dict["interact_num_added"] = self.n_interactions 
        model_dict["input_num"] = self.n_features
        model_dict["max_interact_num"] = self.max_interact_num
        model_dict["interact_num"] = self.interact_num
        model_dict["loss_fn"] = self.loss_fn
        model_dict["data_dict_density"] = self.data_dict_density

        model_dict["err_train_main_effect_training"] = self.err_train_main_effect_training
        model_dict["err_val_main_effect_training"] = self.err_val_main_effect_training
        model_dict["err_train_interaction_training"] = self.err_train_interaction_training
        model_dict["err_val_interaction_training"] = self.err_val_interaction_training
        model_dict["err_train_tuning"] = self.err_train_tuning
        model_dict["err_val_tuning"] = self.err_val_tuning
        model_dict["interaction_list"] = self.interaction_list
        model_dict["main_effect_val_loss"] = self.main_effect_val_loss
        model_dict["interaction_val_loss"] = self.interaction_val_loss

        model_dict["active_indice"] = self.active_indice
        model_dict["effect_names"] = self.effect_names
        model_dict["active_main_effect_index"] = self.active_main_effect_index
        model_dict["active_interaction_index"] = self.active_interaction_index

        model_dict["tr_idx"] = self.tr_idx
        model_dict["val_idx"] = self.val_idx
        model_dict["max_value"] = self.max_value
        model_dict["min_value"] = self.min_value
        model_dict["device"] = self.device

        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.net, folder + name + "_model.pickle")
        with open(folder + name + "_dict.pickle", 'wb') as handle:
            pickle.dump(model_dict, handle)
