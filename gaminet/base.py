import os
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from itertools import combinations
from joblib import Parallel, delayed, cpu_count

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .layers import pyGAMINet
from .dataloader import FastTensorDataLoader
from .interpret import EBMPreprocessor, InteractionDetector
from .utils import plot_regularization, plot_trajectory
from .utils import feature_importance_visualize, effect_importance_visualize
from .utils import local_visualize, global_visualize_density, global_visualize_wo_density


ACTIVATIONS = ["ReLU", "Sigmoid", "Tanh"]


class GAMINet(BaseEstimator):
    """
        GAMINet Pytorch Version

        Parameters
        ----------
        meta_info : None or dict
            A dict of feature_name: feature_type pairs, by default None.
            If not None, its length should be the same as the number of features.
            E.g., {"X1": "categorical", "X2": "continuous"}
            For categorical features, the input values should be ordinal encoded,
            i.e., 0, 1, ... n_classes - 1.
        interact_num : int
            The max number of interactions to be included in the second
            stage training, by default 10.
        subnet_size_main_effect : tuple of int
            The hidden layer architecture of each subnetwork in the
            main effect block, by default (20).
        subnet_size_interaction : tuple of int
            The hidden layer architecture of each subnetwork in the
            interaction block, by default (20, 20).
        activation_func : str
            The name of the activation function, one of ["ReLU",
            "Sigmoid", "Tanh"], by default "ReLU".
        max_epochs : tuple of int
            The max number of epochs in the first (main effect training),
            second (interaction training), and third (fine tuning) stages,
            respectively, by default (1000, 1000, 1000).
        learning_rates : tuple of float
            The initial learning rates of Adam optimizer in the first
            (main effect training), second (interaction training), and
            third (fine tuning) stages, respectively, by default (1e-3, 1e-3, 1e-4).
        early_stop_thres : tuple of int or "auto"
            The early stopping threshold in the first (main effect training),
            second (interaction training), and third (fine tuning) stages,
            respectively, by default ["auto", "auto", "auto"].
            In auto mode, the value is set to max(5, min(5000 * n_features
            / (max_iter_per_epoch * batch_size), 100)).
        batch_size : int
            The batch size, by default 1000.
            Note that it should not be larger than the training size * (1 - validation ratio).
        batch_size_inference : int
            The batch size used in the inference stage by default 10000.
            It is imposed to avoid out-of-memory issue when dealing very large dataset.
        max_iter_per_epoch : int
            The max number of iterations per epoch, by default 100.
            In the init stage of model fit, its value will be clipped
            by min(max_iter_per_epoch, int(sample_size / batch_size)).
            For each epoch, the data would be reshuffled and only the
            first "max_iter_per_epoch" batches would be used for training.
            It is imposed to make the training scalable for very large dataset.
        val_ratio : float
            The validation ratiom, should be greater than 0 and smaller
            than 1, by default 0.2.
        warm_start : boolen
            Initialize the network by fitting a rough B-spline based
            GAM model with tensor product interactions, by default True.
            The initialization is performed by,
            1) fit B-spline GAM as teacher model,
            2) generate random samples from the teacher model,
            3) fit each subnetwork using the generated samples.
            And it is used for both main effect and interaction subnetwork initialization.
        gam_sample_size : int
            The sub-sample size for GAM fitting as warm_start=True, by default 5000.
        mlp_sample_size : int
            The generated sample size for individual subnetwork fitting
            as warm_start=True, by default 1000.
        heredity : bool
            Whether to perform interaction screening subject to heredity
            constraint, by default True.
        loss_threshold : float
            The loss tolerance threshold for selecting fewer main effects or
            interactions, according to the validation performance, by default 0.0.
            For instance, assume the best validation performance is achived
            when using 10 main effects; if only use the top 5 main effects
            also gives similar validation performance, we could prune the
            last 5 by setting this parameter to be positive.
        reg_clarity : float
            The regularization strength of marginal clarity constraint, by default 0.1.
        reg_mono : float
            The regularization strength of monotonicity constraint, by default 0.1.
        mono_sample_size : int
            As monotonicity constraint is used, we would generate some data points
            uniformly within the feature spacec per epoch, to impose the monotonicity
            regularization in addition to original training samples, by default 1000.
        mono_increasing_list : None or list
            The feature index list with monotonic increasing constraint, by default None.
        mono_decreasing_list : None or list
            The feature index list with monotonic decreasing constraint, by default None.
        boundary_clip : boolen
            In the inference stage, whether to clip the feature values by their
            min and max values in the training data, by default True.
        normalize : boolen
            Whether to to normalize the data before inputing to the network, by default True.
        verbose : boolen
            Whether to output the training logs, by default False.
        n_jobs : int
            The number of cpu cores for parallel computing. -1 means all the
            available cpus will be used, by default 10.
        device : string
            The hard device name used for training, by default "cpu".
        random_state : int
            The random seed, by default 0.

        Attributes
        ----------
        net_ : torch network object
     """

    def __init__(self, loss_fn,
                 meta_info=None,
                 interact_num=10,
                 subnet_size_main_effect=(20),
                 subnet_size_interaction=(20, 20),
                 activation_func="ReLU",
                 max_epochs=(1000, 1000, 1000),
                 learning_rates=(1e-3, 1e-3, 1e-4),
                 early_stop_thres=("auto", "auto", "auto"),
                 batch_size=1000,
                 batch_size_inference=10000,
                 max_iter_per_epoch=100,
                 val_ratio=0.2,
                 warm_start=True,
                 gam_sample_size=5000,
                 mlp_sample_size=1000,
                 heredity=True,
                 loss_threshold=0.0,
                 reg_clarity=0.1,
                 reg_mono=0.1,
                 mono_sample_size=1000,
                 mono_increasing_list=(),
                 mono_decreasing_list=(),
                 boundary_clip=True,
                 normalize=True,
                 verbose=False,
                 n_jobs=10,
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

        self.warm_start = warm_start
        self.gam_sample_size = gam_sample_size
        self.mlp_sample_size = mlp_sample_size

        self.heredity = heredity
        self.reg_clarity = reg_clarity
        self.loss_threshold = loss_threshold

        self.reg_mono = reg_mono
        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list
        self.mono_sample_size = mono_sample_size

        self.boundary_clip = boundary_clip
        self.normalize = normalize
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.device = device
        self.random_state = random_state

    def _estimate_density(self, x, sample_weight):
        """
        Internal function for estimating the marginal density of each feature.

        The fitted density would be saved in self.data_dict_density_.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.
        """

        self.data_dict_density_ = {}
        for idx in self.nfeature_index_list_:
            feature_name = self.feature_names_[idx]
            density, bins = np.histogram(x[:, [idx]], bins=10,
                             weights=sample_weight.reshape(-1, 1), density=True)
            self.data_dict_density_.update({feature_name: {"density":
                               {"names": bins, "scores": density}}})
        for idx in self.cfeature_index_list_:
            feature_name = self.feature_names_[idx]
            unique, counts = np.unique(x[:, idx], return_counts=True)
            density = np.zeros((len(self.dummy_values_[feature_name])))
            for i, val in enumerate(unique):
                density[i] = np.sum((x[:, idx] == val).astype(int) *
                              sample_weight) / sample_weight.sum()
            self.data_dict_density_.update({feature_name: {"density":
                               {"names": np.arange(len(self.dummy_values_[feature_name])),
                                "scores": density}}})

    def _evaluate(self, main_effect=True, interaction=True):
        """
        Internal function for evaluating the validation loss.

        Parameters
        ----------
        main_effect : boolean
            Whether to include all main effects.
        interaction : boolean
            Whether to include all interactions.

        Returns
        -------
        mean_loss : float
            Average validation loss
        """
        pred = self.get_aggregate_output(self.validation_generator_.tensors[0],
                    main_effect=main_effect, interaction=interaction)
        loss = self.loss_fn(pred.ravel(), self.validation_generator_.tensors[1].ravel() *
                    self.validation_generator_.tensors[2].ravel())
        mean_loss = torch.mean(loss).cpu().detach().numpy().ravel()[0]
        return mean_loss

    def _get_main_effect_rank(self):
        """
        Internal function for ranking the main effects.

        Returns
        -------
        sorted_index : np.ndarray
            The sorted index list of main effects.
        componment_scales : np.ndarray
            The contribution list of each main effect.
        """
        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.n_features_)]
        beta = (self.net_.main_effect_weights.cpu().detach().numpy() ** 2 *
                self.main_effect_norm_.reshape([-1, 1]))
        componment_scales = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def _get_interaction_rank(self):
        """
        Internal function for ranking the interactions.

        Returns
        -------
        sorted_index : np.ndarray
            The sorted index list of interactions.
        componment_scales : np.ndarray
            The contribution list of each interaction.
        """
        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.n_interactions_)]
        if self.n_interactions_ > 0:
            gamma = (self.net_.interaction_weights.cpu().detach().numpy() ** 2 *
                    self.interaction_norm_.reshape([-1, 1]))
            componment_scales = (np.abs(gamma) / np.sum(np.abs(gamma))).reshape([-1])
            sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def _get_all_active_rank(self):
        """
        Internal function for ranking all effects.

        Returns
        -------
        sorted_index : np.ndarray
            The sorted index list of all effects.
        componment_scales : np.ndarray
            The contribution list of each effect.
        """
        componment_scales = [0 for i in range(self.n_features_ + self.n_interactions_)]
        beta = (self.net_.main_effect_weights.cpu().detach().numpy() ** 2 *
                np.array([self.main_effect_norm_]).reshape([-1, 1]) *
                self.net_.main_effect_switcher.cpu().detach().numpy())

        gamma = np.empty((0, 1))
        if self.n_interactions_ > 0:
            gamma = (self.net_.interaction_weights.cpu().detach().numpy()[:self.n_interactions_] ** 2 *
                 np.array([self.interaction_norm_]).reshape([-1, 1]) *
                 self.net_.interaction_switcher.cpu().detach().numpy()[:self.n_interactions_])

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def _center_main_effects(self):
        """
        Internal function for centering main effects.

        This operation is used to have zero-mean main effects,
        and it will not change the model performance.

        For each main effect subnetowrk, the last-layer-node bias
        is adjusted by self.main_effect_mean_. Then, the removed
        intercept is added to the output layer node bias, also
        considering the output layer weights.
        """
        output_bias = self.net_.output_bias.cpu().detach().numpy()
        main_effect_weights = (self.net_.main_effect_switcher *
                       self.net_.main_effect_weights).cpu().detach().numpy()
        for i, idx in enumerate(self.nfeature_index_list_):
            new_bias = (self.net_.main_effect_blocks.nsubnets.all_biases[-1][i].cpu().detach().numpy() -
                        self.main_effect_mean_[idx])
            self.net_.main_effect_blocks.nsubnets.all_biases[-1].data[i] = torch.tensor(new_bias,
                                            dtype=torch.float32, device=self.device)
            output_bias = output_bias + self.main_effect_mean_[idx] * main_effect_weights[idx]
        for i, idx in enumerate(self.cfeature_index_list_):
            new_bias = (self.net_.main_effect_blocks.csubnets.global_bias[i].cpu().detach().numpy() -
                        self.main_effect_mean_[idx])
            self.net_.main_effect_blocks.csubnets.global_bias[i].data = torch.tensor(new_bias,
                         dtype=torch.float32, device=self.device)
            output_bias = output_bias + self.main_effect_mean_[idx] * main_effect_weights[idx]
        self.net_.output_bias.data = torch.tensor(output_bias, dtype=torch.float32, device=self.device)

    def _center_interactions(self):
        """
        Internal function for centering interactions.

        This operation is used to have zero-mean interactions,
        and it will not change the model performance.

        For each interaction subnetowrk, the last-layer-node bias
        is adjusted by self.interaction_mean_. Then, the removed
        intercept is added to the output layer node bias, also
        considering the output layer weights.
        """
        output_bias = self.net_.output_bias.cpu().detach().numpy()
        interaction_weights = (self.net_.interaction_switcher *
                       self.net_.interaction_weights).cpu().detach().numpy()
        for idx in range(self.n_interactions_):
            new_bias = (self.net_.interaction_blocks.subnets.all_biases[-1][idx].cpu().detach().numpy()
                    - self.interaction_mean_[idx])
            self.net_.interaction_blocks.subnets.all_biases[-1].data[idx] = torch.tensor(new_bias,
                                             dtype=torch.float32, device=self.device)
            output_bias = output_bias + self.interaction_mean_[idx] * interaction_weights[idx]
        self.net_.output_bias.data = torch.tensor(output_bias, dtype=torch.float32, device=self.device)

    def _prepare_data(self, x, y, sample_weight, stratified=False):
        """
        Internal function for preparing data beforing training.

        In this function,
        1) It will parse the meta_info, and create
        several attributes summarizing the feature information.
        2) Calculate the min and max of each feature; if self.normalize is True,
        the mean and std of continuous features would also be calculated.
        3) Split the data into train and validation sets, using val_ratio.
        4) Put the data into Torch Data Loader.
        5) Estimate the marginal density of each feature.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray  of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.
        stratified : boolean
            Whether to stratified split valiadtion set by target response,
            by default False.
        """
        self.n_samples_ = x.shape[0]
        self.n_features_ = x.shape[1]
        indices = np.arange(self.n_samples_)

        if self.meta_info is None:
            meta_info = {}
            for idx in range(self.n_features_):
                meta_info["X" + str(idx + 1)] = {'type': 'continuous'}
        else:
            meta_info = self.meta_info

        self.dummy_values_ = {}
        self.cfeature_num_ = 0
        self.nfeature_num_ = 0
        self.cfeature_names_ = []
        self.nfeature_names_ = []
        self.cfeature_index_list_ = []
        self.nfeature_index_list_ = []
        self.num_classes_list_ = []

        self.mu_list_ = []
        self.std_list_ = []
        self.feature_names_ = []
        self.feature_types_ = []
        for idx, (feature_name, feature_info) in enumerate(meta_info.items()):
            if feature_info["type"] == "categorical":
                self.cfeature_num_ += 1
                self.cfeature_names_.append(feature_name)
                self.cfeature_index_list_.append(idx)
                categories_ = np.arange(np.max(x[:, [idx]]) + 1)
                self.num_classes_list_.append(len(categories_))
                self.dummy_values_.update({feature_name: categories_})
                self.feature_types_.append("categorical")
                self.feature_names_.append(feature_name)
                self.mu_list_.append(0)
                self.std_list_.append(1)
            elif feature_info["type"] == "continuous":
                self.nfeature_num_ += 1
                self.nfeature_names_.append(feature_name)
                self.nfeature_index_list_.append(idx)
                self.feature_types_.append("continuous")
                self.feature_names_.append(feature_name)
                if self.normalize:
                    self.mu_list_.append(x[:, idx].mean())
                    self.std_list_.append(x[:, idx].std())
                else:
                    self.mu_list_.append(0)
                    self.std_list_.append(1)

        self._estimate_density(x, sample_weight)
        self.mu_list_ = torch.tensor(self.mu_list_, dtype=torch.float32, device=self.device)
        self.std_list_ = torch.tensor(self.std_list_, dtype=torch.float32, device=self.device)
        self.min_value_ = torch.tensor(np.min(x, axis=0), dtype=torch.float32, device=self.device)
        self.max_value_ = torch.tensor(np.max(x, axis=0), dtype=torch.float32, device=self.device)
        val_size = max(int(self.n_samples_ * self.val_ratio), 1)
        if stratified:
            tr_x, val_x, tr_y, val_y, tr_sw, val_sw, tr_idx, val_idx = train_test_split(
                x, y, sample_weight, indices, test_size=val_size, stratify=y, random_state=self.random_state)
        else:
            tr_x, val_x, tr_y, val_y, tr_sw, val_sw, tr_idx, val_idx = train_test_split(
                x, y, sample_weight, indices, test_size=val_size, random_state=self.random_state)

        batch_size = min(self.batch_size, tr_x.shape[0])
        self.training_generator_ = FastTensorDataLoader(
                            torch.tensor(tr_x, dtype=torch.float32, device=self.device),
                            torch.tensor(tr_y, dtype=torch.float32, device=self.device),
                            torch.tensor(tr_sw, dtype=torch.float32, device=self.device),
                            batch_size=batch_size, shuffle=True)
        self.validation_generator_ = FastTensorDataLoader(
                            torch.tensor(val_x, dtype=torch.float32, device=self.device),
                            torch.tensor(val_y, dtype=torch.float32, device=self.device),
                            torch.tensor(val_sw, dtype=torch.float32, device=self.device),
                            batch_size=self.batch_size_inference, shuffle=False)

    def _build_net(self):
        """
        Internal function for building GAMINET architecture.

        The network is created and assigned to self.net_.
        In this stage, only main effects subnetworks are created
        and initialized. The interaction subnetworks would be
        created dynamically later in the self._add_interaction function.
        In the end of this function, the mean and norm of each
        main effect are also initialized, in case of max_epoch=0.
        """
        self.net_ = pyGAMINet(nfeature_index_list=self.nfeature_index_list_,
                      cfeature_index_list=self.cfeature_index_list_,
                      num_classes_list=self.num_classes_list_,
                      subnet_size_main_effect=list(self.subnet_size_main_effect),
                      subnet_size_interaction=list(self.subnet_size_interaction),
                      activation_func=getattr(torch.nn, self.activation_func)(),
                      heredity=self.heredity,
                      mono_increasing_list=self.mono_increasing_list,
                      mono_decreasing_list=self.mono_decreasing_list,
                      boundary_clip=self.boundary_clip,
                      min_value=self.min_value_,
                      max_value=self.max_value_,
                      mu_list=self.mu_list_,
                      std_list=self.std_list_,
                      device=self.device)
        self.net_.to(dtype=torch.float32)

        main_effect_output = self.get_main_effect_raw_output(self.training_generator_.tensors[0])
        self.main_effect_mean_ = np.average(main_effect_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
        self.main_effect_norm_ = np.diag(np.cov(main_effect_output.T,
                        aweights=self.training_generator_.tensors[2].cpu().numpy()
                        ).reshape(self.n_features_, self.n_features_))
        torch.cuda.empty_cache()

    def _init_fit(self, x, y, sample_weight=None, stratified=False):
        """
        Internal function for initializing fit.

        In this function, most of the attributes are initialized.
        Then some valiation checks are performed, and the random
        state is set for reproducible fitting. Finally, it calls
        self._prepare_data and self._build_net.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray  of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.
        stratified : boolean
            Whether to stratified split valiadtion set by target response,
            by default False.
        """
        self.active_indice_ = np.array([0])
        self.effect_names_ = np.array(["Intercept"])
        self.data_dict_density_ = {}
        self.err_train_main_effect_training_ = []
        self.err_val_main_effect_training_ = []
        self.err_train_interaction_training_ = []
        self.err_val_interaction_training_ = []
        self.err_train_tuning_ = []
        self.err_val_tuning_ = []

        self.interaction_list_ = []
        self.active_main_effect_index_ = []
        self.active_interaction_index_ = []
        self.main_effect_val_loss_ = []
        self.interaction_val_loss_ = []

        self.time_cost_ = {"warm_start_main_effect": 0,
                    "fit_main_effect": 0,
                    "prune_main_effect": 0,
                    "get_interaction_list": 0,
                    "add_interaction": 0,
                    "warm_start_interaction": 0,
                    "fit_interaction": 0,
                    "prune_interaction": 0,
                    "fine_tune_all": 0}

        if self.activation_func not in ACTIVATIONS:
            raise ValueError("The activation '%s' is not supported. Supported activations are %s."
                        % (self.activation_func, list(sorted(ACTIVATIONS))))

        x, y, sample_weight = self._validate_input(x, y, sample_weight)
        n_jobs = cpu_count() if self.n_jobs == -1 else min(int(self.n_jobs), cpu_count())
        if self.reg_clarity > 0:
            self.clarity_ = True
        else:
            self.clarity_ = False
        if len(list(self.mono_increasing_list) + list(self.mono_decreasing_list)) > 0 and self.reg_mono > 0:
            self.monotonicity_ = True 
        else:
            self.monotonicity_ = False

        self.is_fitted_ = False
        self.n_interactions_ = 0
        # the seed may not work for data loader. this needs to be checked
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.set_num_threads(n_jobs)

        self._prepare_data(x, y, sample_weight, stratified)
        self._build_net()

    def _fit_individual_subnet(self, x, y, subnet, idx, loss_fn,
                       max_epochs=1000, batch_size=200, early_stop_thres=10):
        """
        Internal function for individually fitting a given subnet.

        In this function, most of the attributes are initialized.
        Then some valiation checks are performed, and the random
        state is set for reproducible fitting. Finally, it calls
        self._prepare_data and self._build_net.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray  of shape (n_samples, )
            Target response.
        subnet : torch object
            The subnetwork to be fitted.
        idx : int
            The index of the subnetwork in the main effect block
            or interaction block.
        loss_fn : torch function
            The loss function to be optimized.
        max_epochs : int
            The max number of training epochs, by default 1000.
        batch_size : int
            The number sample in a mini-batch, by default 200.
        early_stop_thres : int
            The threshold for controling early stopping.
        """
        last_improvement = 0
        best_validation = np.inf
        opt = torch.optim.Adam(list(subnet.parameters()), lr=0.01)
        training_generator = FastTensorDataLoader(torch.tensor(x, dtype=torch.float32, device=self.device),
                                    torch.tensor(y, dtype=torch.float32, device=self.device),
                                    batch_size=min(200, int(0.2 * x.shape[0])), shuffle=True)
        for epoch in range(max_epochs):
            self.net_.train()
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
            self.net_.eval()
            accumulated_loss = accumulated_loss / accumulated_size
            if accumulated_loss < best_validation:
                best_validation = accumulated_loss
                last_improvement = epoch
            if epoch - last_improvement > early_stop_thres:
                break

    def _warm_start_main_effect(self):
        """
        Internal function for warm start main effect subnetworks.

        This function works as follows:
        1) Perform fast GAM fitting by self._build_teacher_main_effect
        2) Generate random samples according to the fitted GAM effect functions.
        3) Fit each main effect subnetwork individually using data in 2).
        4) Update self.main_effect_mean_ and self.main_effect_norm_,
        and center the fitted main effect subnetworks.
        """
        if not self.warm_start:
            return

        if self.verbose:
            print("#" * 15 + "Run Warm Initialization for Main Effect" + "#" * 15)

        start = time.time()
        self.warm_init_main_effect_data_ = {}
        surrogate_estimator, intercept = self._build_teacher_main_effect()
        self.net_.output_bias.data = self.net_.output_bias.data + torch.tensor(intercept,
                               dtype=torch.float32, device=self.device)

        def initmaineffect(idx):
            torch.set_num_threads(1)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            simu_xx = np.zeros((self.mlp_sample_size, self.n_features_))
            simu_xx[:, idx] = np.random.uniform(self.min_value_[idx].cpu().numpy(),
                                self.max_value_[idx].cpu().numpy(), self.mlp_sample_size)
            if self.normalize:
                simu_xx[:, idx] = ((simu_xx[:, idx] - self.mu_list_[idx].detach().cpu().numpy()) /
                            self.std_list_[idx].detach().cpu().numpy())
            simu_yy = surrogate_estimator[idx](simu_xx)
            self._fit_individual_subnet(simu_xx[:, [idx]], simu_yy, self.net_.main_effect_blocks.nsubnets,
                      self.nfeature_index_list_.index(idx), loss_fn=torch.nn.MSELoss(reduction="none"))

            xgrid = np.linspace(self.min_value_[idx].cpu().numpy(), self.max_value_[idx].cpu().numpy(), 100)
            gam_input_grid = np.zeros((100, self.n_features_))
            gam_input_grid[:, idx] = xgrid
            gaminet_input_grid = torch.tensor(xgrid.reshape(-1, 1), dtype=torch.float32, device=self.device)
            info = {"x": xgrid,
                 "gam": surrogate_estimator[idx](gam_input_grid),
                 "gaminet": self.net_.main_effect_blocks.nsubnets.individual_forward(
                   gaminet_input_grid, self.nfeature_index_list_.index(idx)).ravel().detach().cpu().numpy()}
            w = self.net_.main_effect_blocks.nsubnets.all_weights
            b = self.net_.main_effect_blocks.nsubnets.all_biases
            return w, b, info

        n_jobs = cpu_count() if self.n_jobs == -1 else min(int(self.n_jobs), cpu_count())
        delayed_funcs = [delayed(initmaineffect)(idx) for idx in self.nfeature_index_list_]
        res = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(delayed_funcs)
        torch.set_num_threads(n_jobs)
        for i, idx in enumerate(self.nfeature_index_list_):
            self.warm_init_main_effect_data_[idx] = res[i][2]
            for idxlayer in range(len(self.net_.main_effect_blocks.nsubnets.all_weights)):
                self.net_.main_effect_blocks.nsubnets.all_biases[idxlayer].data[i] = res[i][1][idxlayer][i]
                self.net_.main_effect_blocks.nsubnets.all_weights[idxlayer].data[i] = res[i][0][idxlayer][i]

        for idx in self.cfeature_index_list_:
            i = self.cfeature_index_list_.index(idx)
            simu_xx = np.zeros((self.num_classes_list_[i], self.n_features_))
            simu_xx[:, idx] = np.linspace(self.min_value_[idx].cpu().numpy(),
                                self.max_value_[idx].cpu().numpy(), self.num_classes_list_[i])
            simu_yy = surrogate_estimator[idx](simu_xx)
            self.net_.main_effect_blocks.csubnets.class_bias[i].data = torch.tensor(simu_yy.reshape(-1, 1),
                                                dtype=torch.float32, device=self.device)
            self.warm_init_main_effect_data_[idx] = {"x": simu_xx[:, idx],
                                       "gam": simu_yy,
                                       "gaminet": simu_yy}

        main_effect_output = self.get_main_effect_raw_output(self.training_generator_.tensors[0])
        self.main_effect_mean_ = np.average(main_effect_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
        self.main_effect_norm_ = np.diag(np.cov(main_effect_output.T,
                        aweights=self.training_generator_.tensors[2].cpu().numpy()
                        ).reshape(self.n_features_, self.n_features_))
        self._center_main_effects()
        torch.cuda.empty_cache()
        self.time_cost_["warm_start_main_effect"] = round(time.time() - start, 2)

    def _warm_start_interaction(self):
        """
        Internal function for warm start interaction subnetworks.

        This function works as follows:
        1) Perform fast GAM fitting by self._build_teacher_interaction
        2) Generate random samples according to the fitted GAM effect functions.
        3) Fit each interaction subnetwork individually using data in 2).
        4) Update self.interaction_mean_ and self.interaction_norm_,
        and center the fitted interaction subnetworks.
        """
        if not self.net_.interaction_status:
            return

        if not self.warm_start:
            return

        if self.verbose:
            print("#" * 15 + "Run Warm Initialization for Interaction" + "#" * 15)

        start = time.time()
        self.warm_init_interaction_data_ = {}
        surrogate_estimator, intercept = self._build_teacher_interaction()
        self.net_.output_bias.data = self.net_.output_bias.data + torch.tensor(intercept,
                               dtype=torch.float32, device=self.device)

        def initinteraction(i, idx1, idx2):
            torch.set_num_threads(1)
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            simu_xx = np.zeros((self.mlp_sample_size, self.n_features_))
            if idx1 in self.cfeature_index_list_:
                num_classes = self.num_classes_list_[self.cfeature_index_list_.index(idx1)]
                simu_xx[:, idx1] = np.random.randint(self.min_value_[idx1].cpu().numpy(),
                          self.max_value_[idx1].cpu().numpy() + 1, self.mlp_sample_size)
                x1 = torch.nn.functional.one_hot(torch.tensor(simu_xx[:, idx1]).to(torch.int64),
                          num_classes=num_classes).to(torch.float32).detach().cpu().numpy()
                x1grid = np.linspace(self.min_value_[idx1].cpu().numpy(),
                                     self.max_value_[idx1].cpu().numpy(), num_classes).reshape(-1, 1)
            else:
                simu_xx[:, idx1] = np.random.uniform(self.min_value_[idx1].cpu().numpy(),
                                self.max_value_[idx1].cpu().numpy(), self.mlp_sample_size)
                if self.normalize:
                    simu_xx[:, idx1] = ((simu_xx[:, idx1] -
                                self.mu_list_[idx1].detach().cpu().numpy()) /
                                self.std_list_[idx1].detach().cpu().numpy())
                x1 = simu_xx[:, [idx1]]
                x1grid = np.linspace(self.min_value_[idx1].cpu().numpy(),
                                     self.max_value_[idx1].cpu().numpy(), 20).reshape(-1, 1)
            if idx2 in self.cfeature_index_list_:
                num_classes = self.num_classes_list_[self.cfeature_index_list_.index(idx2)]
                simu_xx[:, idx2] = np.random.randint(self.min_value_[idx2].cpu().numpy(),
                          self.max_value_[idx2].cpu().numpy() + 1, self.mlp_sample_size)
                x2 = torch.nn.functional.one_hot(torch.tensor(simu_xx[:, idx2]).to(torch.int64),
                          num_classes=num_classes).to(torch.float32).detach().cpu().numpy()
                x2grid = np.linspace(self.min_value_[idx2].cpu().numpy(),
                                     self.max_value_[idx2].cpu().numpy(), num_classes).reshape(-1, 1)
            else:
                simu_xx[:, idx2] = np.random.uniform(self.min_value_[idx2].cpu().numpy(),
                                self.max_value_[idx2].cpu().numpy(), self.mlp_sample_size)
                if self.normalize:
                    simu_xx[:, idx2] = ((simu_xx[:, idx2] -
                                self.mu_list_[idx2].detach().cpu().numpy()) /
                                self.std_list_[idx2].detach().cpu().numpy())
                x2 = simu_xx[:, [idx2]]
                x2grid = np.linspace(self.min_value_[idx2].cpu().numpy(),
                                     self.max_value_[idx2].cpu().numpy(), 20).reshape(-1, 1)

            xx = np.hstack([x1, x2])
            xx = np.hstack([xx, np.zeros((xx.shape[0],
                      self.net_.interaction_blocks.max_n_inputs - xx.shape[1]))])
            yy = surrogate_estimator[i](simu_xx)
            self._fit_individual_subnet(xx, yy, self.net_.interaction_blocks.subnets,
                              i, loss_fn=torch.nn.MSELoss(reduction="none"))

            xx1grid, xx2grid = np.meshgrid(x1grid, x2grid)
            gam_input_grid = np.zeros((xx1grid.shape[0] * xx1grid.shape[1], self.n_features_))
            gam_input_grid[:, idx1] = xx1grid.ravel()
            gam_input_grid[:, idx2] = xx2grid.ravel()
            xxgrid = gam_input_grid[:, [idx1, idx2]]
            gaminet_input_grid = np.hstack([xxgrid, np.zeros((xxgrid.shape[0],
                          self.net_.interaction_blocks.max_n_inputs - xxgrid.shape[1]))])
            info = {"x": xxgrid,
                 "gam": surrogate_estimator[i](gam_input_grid),
                 "gaminet": self.net_.interaction_blocks.subnets.individual_forward(
                       torch.tensor(gaminet_input_grid, dtype=torch.float32,
                       device=self.device), i).ravel().detach().cpu().numpy()}
            w = self.net_.interaction_blocks.subnets.all_weights
            b = self.net_.interaction_blocks.subnets.all_biases
            return w, b, info

        n_jobs = cpu_count() if self.n_jobs == -1 else min(int(self.n_jobs), cpu_count())
        delayed_funcs = [delayed(initinteraction)(i, idx1, idx2)
                     for i, (idx1, idx2) in enumerate(self.interaction_list_)]
        res = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(delayed_funcs)
        torch.set_num_threads(n_jobs)
        for i in range(self.n_interactions_):
            self.warm_init_interaction_data_[i] = res[i][2]
            for idxlayer in range(len(self.net_.interaction_blocks.subnets.all_weights)):
                self.net_.interaction_blocks.subnets.all_biases[idxlayer].data[i] = res[i][1][idxlayer][i]
                self.net_.interaction_blocks.subnets.all_weights[idxlayer].data[i] = res[i][0][idxlayer][i]

        interaction_output = self.get_interaction_raw_output(self.training_generator_.tensors[0])
        self.interaction_mean_ = np.average(interaction_output, axis=0,
                    weights=self.training_generator_.tensors[2].cpu().numpy())
        self.interaction_norm_ = np.diag(np.cov(interaction_output.T,
                    aweights=self.training_generator_.tensors[2].cpu().numpy()
                    ).reshape(self.n_interactions_, self.n_interactions_))
        self._center_interactions()
        torch.cuda.empty_cache()
        self.time_cost_["warm_start_interaction"] = round(time.time() - start, 2)

    def _interaction_screening(self, x, y, w, scores, feature_names,
                      feature_types, model_type, num_classes):
        """
        Internal function for screening important pairwise interactions.

        If heredity is true, then only run interaction screening over
        interactions that are related to active features. This core of this
        function is the FAST algorithm provided by Microsoft/interpretml.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray  of shape (n_samples, )
            Target response.
        w : np.ndarray of shape (n_samples, )
            Sample weight.
        scores : np.ndarray of shape (n_samples, )
            The prediction of main effect subnetworks.
        feature_names : list of str
            Feature name list
        feature_types : list of str
            Feature type list, including "continuous" and "categorical".
        model_type : str
            task type, including "regression" and "classification".
        num_classes : int
            number of classes in response, would be -1 for regression.

        Returns
        -------
        interaction_list : list of int
            List of paired tuple index, each indicating the feature index.
        """
        active_main_effect_index = self.active_main_effect_index_ if self.heredity else np.arange(self.n_features_)
        if (len(active_main_effect_index) == 0):
            return []

        start = time.time()
        preprocessor_ = EBMPreprocessor(feature_names=feature_names,
                              feature_types=feature_types)
        preprocessor_.fit(x)
        X_pair = preprocessor_.transform(x)
        features_categorical = np.array([tp == "categorical"
                             for tp in preprocessor_.col_types_], dtype=np.int64)
        features_bin_count = np.array([len(nbin)
                       for nbin in preprocessor_.col_bin_counts_], dtype=np.int64)

        X = np.ascontiguousarray(X_pair.T).astype(np.int64)
        y = y.ravel()
        w = w.astype(np.float64)
        scores = scores.ravel().astype(np.float64)

        with InteractionDetector(
            model_type, num_classes, features_categorical, features_bin_count,
            X, y, w, scores, optional_temp_params=None
        ) as interaction_detector:

            def evaluate(pair):
                score = interaction_detector.get_interaction_score(pair, min_samples_leaf=2)
                return pair, score

            all_pairs = [pair for pair in combinations(range(len(preprocessor_.col_types_)), 2)
               if (pair[0] in active_main_effect_index) or (pair[1] in active_main_effect_index)]
            interaction_scores = [evaluate(pair) for pair in all_pairs]

        ranked_scores = list(sorted(interaction_scores, key=lambda item: item[1], reverse=True))
        interaction_list = [ranked_scores[i][0] for i in range(len(ranked_scores))]
        self.time_cost_["get_interaction_list"] = round(time.time() - start, 2)
        return interaction_list

    def _fit_main_effect(self):
        """
        Internal function for fitting main effects.

        Monotonic regularization would be imposed if self.mono_decreasing_list
        or self.mono_increasing_list are not empty.
        After training, the mean and norm of each effect would be updated,
        and the subnetworks are also centered.
        """
        if self.max_epochs[0] <= 0:
            return

        if self.verbose:
            print("#" * 20 + "Stage 1: Main Effect Training" + "#" * 20)

        start = time.time()
        last_improvement = 0
        best_validation = np.inf
        opt = torch.optim.Adam(list(self.net_.main_effect_blocks.parameters()) +
                        [self.net_.main_effect_weights, self.net_.output_bias],
                        lr=self.learning_rates[0])
        max_iter_per_epoch = min(len(self.training_generator_), self.max_iter_per_epoch)
        if self.early_stop_thres[0] == "auto":
            early_stop_thres = max(5, min(int(5000 * self.n_features_ /
                            (max_iter_per_epoch * self.batch_size)), 100))
        else:
            early_stop_thres = self.early_stop_thres[0]
        for epoch in range(self.max_epochs[0]):
            self.net_.train()
            accumulated_size = 0
            accumulated_loss = 0.0
            if self.verbose:
                pbar = tqdm(self.training_generator_, total=max_iter_per_epoch,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            else:
                pbar = self.training_generator_
            for batch_no, batch_data in enumerate(pbar):
                if batch_no >= max_iter_per_epoch:
                    break
                opt.zero_grad(set_to_none=True)
                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                batch_sw = batch_data[2].to(self.device).ravel()
                pred = self.net_(batch_xx, sample_weight=batch_sw,
                           main_effect=True, interaction=False,
                           clarity=False,
                           monotonicity=self.monotonicity_).ravel()

                if self.monotonicity_:
                    mono_loss_reg = self.reg_mono * self.net_.mono_loss
                    simu_inputs = np.random.uniform(self.min_value_.cpu().numpy(),
                            self.max_value_.cpu().numpy(), size=(self.mono_sample_size, len(self.max_value_)))
                    simu_inputs = torch.tensor(simu_inputs, dtype=torch.float32, device=self.device)
                    self.net_(simu_inputs,
                            main_effect=True, interaction=False,
                            clarity=False,
                            monotonicity=self.monotonicity_).ravel()
                    mono_loss_reg = mono_loss_reg + self.reg_mono * self.net_.mono_loss
                    mono_loss_reg.backward(retain_graph=True)

                loss = torch.mean(self.loss_fn(pred, batch_yy) * batch_sw)
                loss.backward()
                opt.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
                if self.verbose:
                    pbar.set_description(("Epoch: %" +
                            str(int(np.ceil(np.log10(self.max_epochs[0]))) + 1)
                            + "d, train loss: %0.5f") %
                            (epoch + 1, accumulated_loss / accumulated_size))
                if batch_no == (len(self.training_generator_) - 1) or batch_no == (max_iter_per_epoch - 1):
                    self.net_.eval()
                    self.err_train_main_effect_training_.append(
                                accumulated_loss / accumulated_size)
                    self.err_val_main_effect_training_.append(self._evaluate(
                                main_effect=True, interaction=False))
                    if self.verbose:
                        pbar.set_description(("Epoch: %" +
                            str(int(np.ceil(np.log10(self.max_epochs[0]))) + 1)
                            + "d, train loss: %0.5f, validation loss: %0.5f") %
                            (epoch + 1, self.err_train_main_effect_training_[-1],
                            self.err_val_main_effect_training_[-1]))

            if self.err_val_main_effect_training_[-1] < best_validation:
                best_validation = self.err_val_main_effect_training_[-1]
                last_improvement = epoch

            if epoch - last_improvement > early_stop_thres:
                if self.monotonicity_:
                    if self.certify_mono():
                        break
                    else:
                        self.reg_mono = min(self.reg_mono * 1.2, 1e5)
                else:
                    break

        if self.verbose:
            print("Main Effect Training Stop at Epoch: %d, train loss: %0.5f, validation loss: %0.5f" %
                (epoch + 1, self.err_train_main_effect_training_[-1],
                self.err_val_main_effect_training_[-1]))

        main_effect_output = self.get_main_effect_raw_output(self.training_generator_.tensors[0])
        self.main_effect_mean_ = np.average(main_effect_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
        self.main_effect_norm_ = np.diag(np.cov(main_effect_output.T,
                        aweights=self.training_generator_.tensors[2].cpu().numpy()
                        ).reshape(self.n_features_, self.n_features_))
        self._center_main_effects()
        torch.cuda.empty_cache()
        self.time_cost_["fit_main_effect"] = round(time.time() - start, 2)

    def _prune_main_effect(self):
        """
        Internal function for prunning main effects.

        Fitted main effect subnetworks are first removed and then
        sequentially added to the model, in the descending order
        of contributions.

        The best number of main effect is determiend by the validation
        performance, subject to a certain tolerance threshold, i.e.,
        self.loss_threshold, to pursue more sparse result.
        """
        start = time.time()
        self.main_effect_val_loss_ = []
        sorted_index, componment_scales = self._get_main_effect_rank()
        self.net_.main_effect_switcher.data = torch.tensor(np.zeros((self.n_features_, 1)),
                               dtype=torch.float32, device=self.device, requires_grad=False)
        self.main_effect_val_loss_.append(self._evaluate(main_effect=True, interaction=False))
        for idx in range(self.n_features_):
            selected_index = sorted_index[:(idx + 1)]
            main_effect_switcher = np.zeros((self.n_features_, 1))
            main_effect_switcher[selected_index] = 1
            self.net_.main_effect_switcher.data = torch.tensor(main_effect_switcher, dtype=torch.float32,
                                              device=self.device, requires_grad=False)
            val_loss = self._evaluate(main_effect=True, interaction=False)
            self.main_effect_val_loss_.append(val_loss)

        best_idx = np.argmin(self.main_effect_val_loss_)
        best_loss = np.min(self.main_effect_val_loss_)
        if best_loss > 0:
            if np.sum((self.main_effect_val_loss_ / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.main_effect_val_loss_ /
                             best_loss - 1) < self.loss_threshold)[0][0]

        self.active_main_effect_index_ = sorted_index[:best_idx]
        main_effect_switcher = np.zeros((self.n_features_, 1))
        main_effect_switcher[self.active_main_effect_index_] = 1
        self.net_.main_effect_switcher.data = torch.tensor(main_effect_switcher,
                                        dtype=torch.float32, device=self.device)
        self.net_.main_effect_switcher.requires_grad = False
        self.time_cost_["prune_main_effect"] = round(time.time() - start, 2)

    def _add_interaction(self):

        start = time.time()
        x = torch.vstack([self.training_generator_.tensors[0],
                    self.validation_generator_.tensors[0]]).cpu().numpy()
        y = torch.hstack([self.training_generator_.tensors[1],
                    self.validation_generator_.tensors[1]]).cpu().numpy()
        w = torch.hstack([self.training_generator_.tensors[2],
                    self.validation_generator_.tensors[2]]).cpu().numpy()
        scores = self.get_aggregate_output(x, main_effect=True,
                    interaction=False).detach().cpu().numpy()
        interaction_list_all = self._get_interaction_list(x, y, w, scores,
                    self.feature_names_, self.feature_types_)

        max_interact_num_ = min(self.interact_num,
                    int(round(self.n_features_ * (self.n_features_ - 1) / 2)))
        self.interaction_list_ = interaction_list_all[:max_interact_num_]
        self.n_interactions_ = len(self.interaction_list_)
        self.net_.init_interaction_blocks(self.interaction_list_)

        if self.net_.interaction_status:
            interaction_output = self.get_interaction_raw_output(self.training_generator_.tensors[0])
            self.interaction_mean_ = np.average(interaction_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
            self.interaction_norm_ = np.diag(np.cov(interaction_output.T,
                        aweights=self.training_generator_.tensors[2].cpu().numpy()
                        ).reshape(self.n_interactions_, self.n_interactions_))
            torch.cuda.empty_cache()
        self.time_cost_["add_interaction"] = round(time.time() - start, 2)

    def _fit_interaction(self):
        """
        Internal function for fitting interactions.

        In this stage, main effect subnetworks are freezed, and only
        interaction subnetworks are fitted.
        Clarity regularization would be triggered and only penalize
        interaction subnetworks.
        Monotonic regularization would be imposed if self.mono_decreasing_list
        or self.mono_increasing_list are not empty.
        After training, the mean and norm of each effect would be updated,
        and the subnetworks are also centered.
        """
        if not self.net_.interaction_status:
            return

        if self.max_epochs[1] <= 0:
            return

        if self.verbose:
            print("#" * 20 + "Stage 2: Interaction Training" + "#" * 20)

        start = time.time()
        last_improvement = 0
        best_validation = np.inf
        opt = torch.optim.Adam(list(self.net_.interaction_blocks.parameters()) +
                        [self.net_.interaction_weights, self.net_.output_bias],
                        lr=self.learning_rates[1])
        max_iter_per_epoch = min(len(self.training_generator_), self.max_iter_per_epoch)
        if self.early_stop_thres[1] == "auto":
            early_stop_thres = max(5, min(int(5000 * self.n_features_ /
                          (max_iter_per_epoch * self.batch_size)), 100))
        else:
            early_stop_thres = self.early_stop_thres[1]
        for epoch in range(self.max_epochs[1]):
            self.net_.train()
            accumulated_size = 0
            accumulated_loss = 0.0
            if self.verbose:
                pbar = tqdm(self.training_generator_, total=max_iter_per_epoch,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            else:
                pbar = self.training_generator_
            for batch_no, batch_data in enumerate(pbar):
                if batch_no >= max_iter_per_epoch:
                    break
                opt.zero_grad(set_to_none=True)
                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                batch_sw = batch_data[2].to(self.device).ravel()
                pred = self.net_(batch_xx, sample_weight=batch_sw,
                           main_effect=True, interaction=True,
                           clarity=self.clarity_,
                           monotonicity=self.monotonicity_).ravel()
                clarity_loss_reg = self.reg_clarity * self.net_.clarity_loss
                clarity_loss_reg.backward(retain_graph=True)

                if self.monotonicity_ > 0:
                    mono_loss_reg = self.reg_mono * self.net_.mono_loss
                    simu_inputs = np.random.uniform(self.min_value_.cpu().numpy(),
                            self.max_value_.cpu().numpy(), size=(self.mono_sample_size, len(self.max_value_)))
                    simu_inputs = torch.tensor(simu_inputs, dtype=torch.float32, device=self.device)
                    self.net_(simu_inputs,
                               main_effect=True, interaction=True,
                               clarity=self.clarity_,
                               monotonicity=self.monotonicity_).ravel()
                    mono_loss_reg = mono_loss_reg + self.reg_mono * self.net_.mono_loss
                    mono_loss_reg.backward(retain_graph=True)

                loss = torch.mean(self.loss_fn(pred, batch_yy) * batch_sw)
                loss.backward()
                opt.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
                if self.verbose:
                    pbar.set_description(("Epoch: %" +
                            str(int(np.ceil(np.log10(self.max_epochs[1]))) + 1)
                            + "d, train loss: %0.5f") %
                             (epoch + 1, accumulated_loss / accumulated_size))
                if batch_no == (len(self.training_generator_) - 1) or batch_no == (max_iter_per_epoch - 1):
                    self.net_.eval()
                    self.err_train_interaction_training_.append(accumulated_loss / accumulated_size)
                    self.err_val_interaction_training_.append(self._evaluate(main_effect=True, interaction=True))
                    if self.verbose:
                        pbar.set_description(("Epoch: %" +
                            str(int(np.ceil(np.log10(self.max_epochs[1]))) + 1)
                            + "d, train loss: %0.5f, validation loss: %0.5f") %
                            (epoch + 1, self.err_train_interaction_training_[-1],
                            self.err_val_interaction_training_[-1]))

            if self.err_val_interaction_training_[-1] < best_validation:
                best_validation = self.err_val_interaction_training_[-1]
                last_improvement = epoch
            if epoch - last_improvement > early_stop_thres:
                if self.monotonicity_:
                    if self.certify_mono():
                        break
                    else:
                        self.reg_mono = min(self.reg_mono * 1.2, 1e5)
                else:
                    break

        if self.verbose:
            print("Interaction Training Stop at Epoch: %d, train loss: %0.5f, validation loss: %0.5f" %
              (epoch + 1, self.err_train_interaction_training_[-1],
               self.err_val_interaction_training_[-1]))

        interaction_output = self.get_interaction_raw_output(self.training_generator_.tensors[0])
        self.interaction_mean_ = np.average(interaction_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
        self.interaction_norm_ = np.diag(np.cov(interaction_output.T,
                        aweights=self.training_generator_.tensors[2].cpu().numpy()
                        ).reshape(self.n_interactions_, self.n_interactions_))
        self._center_interactions()
        torch.cuda.empty_cache()
        self.time_cost_["fit_interaction"] = round(time.time() - start, 2)

    def _prune_interaction(self):
        """
        Internal function for prunning interactions.

        Fitted interaction subnetworks are first removed and then
        sequentially added to the model, in the descending order
        of contributions.

        The best number of main effect is determiend by the validation
        performance, subject to a certain tolerance threshold, i.e.,
        self.loss_threshold, to pursue more sparse result.
        """
        if self.n_interactions_ == 0:
            return

        start = time.time()
        self.interaction_val_loss_ = []
        sorted_index, componment_scales = self._get_interaction_rank()
        self.net_.interaction_switcher.data = torch.tensor(np.zeros((self.n_interactions_, 1)),
                               dtype=torch.float32, device=self.device, requires_grad=False)
        self.interaction_val_loss_.append(self._evaluate(main_effect=True, interaction=True))
        for idx in range(self.n_interactions_):
            selected_index = sorted_index[:(idx + 1)]
            interaction_switcher = np.zeros((self.n_interactions_, 1))
            interaction_switcher[selected_index] = 1
            self.net_.interaction_switcher.data = torch.tensor(interaction_switcher, dtype=torch.float32,
                                              device=self.device, requires_grad=False)
            val_loss = self._evaluate(main_effect=True, interaction=True)
            self.interaction_val_loss_.append(val_loss)

        best_idx = np.argmin(self.interaction_val_loss_)
        best_loss = np.min(self.interaction_val_loss_)
        if best_loss > 0:
            if np.sum((self.interaction_val_loss_ / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.interaction_val_loss_ /
                         best_loss - 1) < self.loss_threshold)[0][0]

        self.active_interaction_index_ = sorted_index[:best_idx]
        interaction_switcher = np.zeros((self.n_interactions_, 1))
        interaction_switcher[self.active_interaction_index_] = 1
        self.net_.interaction_switcher.data = torch.tensor(interaction_switcher,
                                   dtype=torch.float32, device=self.device)
        self.net_.interaction_switcher.requires_grad = False
        self.time_cost_["prune_interaction"] = round(time.time() - start, 2)

    def _fine_tune_all(self):
        """
        Internal function for fine-tuning all subnetworks.

        All the network parameters are updated together.
        Clarity regularization would be triggered and only penalize
        interaction subnetworks.
        Monotonic regularization would be imposed if self.mono_decreasing_list
        or self.mono_increasing_list are not empty.
        After training, the mean and norm of each effect would be updated,
        and the subnetworks are also centered.
        """
        if self.max_epochs[2] <= 0:
            return

        if self.verbose:
            print("#" * 25 + "Stage 3: Fine Tuning" + "#" * 25)

        start = time.time()
        last_improvement = 0
        best_validation = np.inf
        opt_main_effect = torch.optim.Adam(list(self.net_.main_effect_blocks.parameters()) +
                                [self.net_.main_effect_weights, self.net_.output_bias],
                                lr=self.learning_rates[2])
        if self.n_interactions_ > 0:
            opt_interaction = torch.optim.Adam(list(self.net_.interaction_blocks.parameters()) +
                                   [self.net_.interaction_weights],
                                   lr=self.learning_rates[2])
        max_iter_per_epoch = min(len(self.training_generator_), self.max_iter_per_epoch)
        if self.early_stop_thres[2] == "auto":
            early_stop_thres = max(5, min(int(5000 * self.n_features_ /
                          (max_iter_per_epoch * self.batch_size)), 100))
        else:
            early_stop_thres = self.early_stop_thres[2]
        for epoch in range(self.max_epochs[2]):
            self.net_.train()
            accumulated_size = 0
            accumulated_loss = 0.0
            if self.verbose:
                pbar = tqdm(self.training_generator_, total=max_iter_per_epoch,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            else:
                pbar = self.training_generator_
            for batch_no, batch_data in enumerate(pbar):
                if batch_no >= max_iter_per_epoch:
                    break

                opt_main_effect.zero_grad(set_to_none=True)
                if self.n_interactions_ > 0:
                    opt_interaction.zero_grad(set_to_none=True)

                batch_xx = batch_data[0].to(self.device)
                batch_yy = batch_data[1].to(self.device).ravel()
                batch_sw = batch_data[2].to(self.device).ravel()

                pred = self.net_(batch_xx, sample_weight=batch_sw,
                           main_effect=True, interaction=True,
                           clarity=self.net_.interaction_status and self.clarity_,
                           monotonicity=self.monotonicity_).ravel()
                clarity_loss_reg = self.reg_clarity * self.net_.clarity_loss
                clarity_loss_reg.backward(retain_graph=True)
                opt_main_effect.zero_grad(set_to_none=True)

                if self.monotonicity_:
                    mono_loss_reg = self.reg_mono * self.net_.mono_loss
                    simu_inputs = np.random.uniform(self.min_value_.cpu().numpy(),
                            self.max_value_.cpu().numpy(), size=(self.mono_sample_size, len(self.max_value_)))
                    simu_inputs = torch.tensor(simu_inputs, dtype=torch.float32, device=self.device)
                    self.net_(simu_inputs,
                               main_effect=True, interaction=True,
                               clarity=self.clarity_,
                               monotonicity=self.monotonicity_).ravel()
                    mono_loss_reg = mono_loss_reg + self.reg_mono * self.net_.mono_loss
                    mono_loss_reg.backward(retain_graph=True)

                loss = torch.mean(self.loss_fn(pred, batch_yy) * batch_sw)
                loss.backward()
                opt_main_effect.step()
                if self.n_interactions_ > 0:
                    opt_interaction.step()
                accumulated_size += batch_xx.shape[0]
                accumulated_loss += (loss * batch_xx.shape[0]).cpu().detach().numpy()
                if self.verbose:
                    pbar.set_description(("Epoch: %" +
                            str(int(np.ceil(np.log10(self.max_epochs[2]))) + 1)
                            + "d, train loss: %0.5f") %
                            (epoch + 1, accumulated_loss / accumulated_size))
                if batch_no == (len(self.training_generator_) - 1) or batch_no == (max_iter_per_epoch - 1):
                    self.net_.eval()
                    self.err_train_tuning_.append(accumulated_loss / accumulated_size)
                    self.err_val_tuning_.append(self._evaluate(main_effect=True, interaction=True))
                    if self.verbose:
                        pbar.set_description(("Epoch: %" +
                          str(int(np.ceil(np.log10(self.max_epochs[2]))) + 1)
                          + "d, train loss: %0.5f, validation loss: %0.5f") %
                          (epoch + 1, self.err_train_tuning_[-1], self.err_val_tuning_[-1]))

            if self.err_val_tuning_[-1] < best_validation:
                best_validation = self.err_val_tuning_[-1]
                last_improvement = epoch
            if epoch - last_improvement > early_stop_thres:
                if self.monotonicity_:
                    if self.certify_mono():
                        break
                    else:
                        self.reg_mono = min(self.reg_mono * 1.2, 1e5)
                else:
                    break

        if self.verbose:
            print("Fine Tuning Stop at Epoch: %d, train loss: %0.5f, validation loss: %0.5f" %
                      (epoch + 1, self.err_train_tuning_[-1], self.err_val_tuning_[-1]))

        main_effect_output = self.get_main_effect_raw_output(self.training_generator_.tensors[0])
        self.main_effect_mean_ = np.average(main_effect_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
        self.main_effect_norm_ = np.diag(np.cov(main_effect_output.T,
                       aweights=self.training_generator_.tensors[2].cpu().numpy()
                       ).reshape(self.n_features_, self.n_features_))
        self._center_main_effects()

        if self.n_interactions_ > 0:
            interaction_output = self.get_interaction_raw_output(self.training_generator_.tensors[0])
            self.interaction_mean_ = np.average(interaction_output, axis=0,
                        weights=self.training_generator_.tensors[2].cpu().numpy())
            self.interaction_norm_ = np.diag(np.cov(interaction_output.T,
                        aweights=self.training_generator_.tensors[2].cpu().numpy()
                        ).reshape(self.n_interactions_, self.n_interactions_))
            self._center_interactions()
        torch.cuda.empty_cache()
        self.time_cost_["fine_tune_all"] = round(time.time() - start, 2)

    def _fit(self):
        """
        Internal function for controlling the fitting procedures.

        It consists the following steps.
        1) Warm init the main effect subnetworks.
        2) Fit the main effects only, s.t. monotonicity regularization.
        3) Prune the trivial main effects.
        4) Select important interactions, s.t. heredit constraint.
        5) Warm init the interaction subnetworks.
        6) Fit the interactions only, s.t. monotonicity and clarity regularization.
        7) Prune the trivial interactions.
        8) Fine-tune all networks, s.t. monotonicity and clarity regularization.
        """
        self._warm_start_main_effect()
        self._fit_main_effect()
        self._prune_main_effect()
        self._add_interaction()
        self._warm_start_interaction()
        self._fit_interaction()
        self._prune_interaction()
        self._fine_tune_all()

        self.time_cost_ = sorted(self.time_cost_.items(), key=lambda x: x[1], reverse=True)
        self.active_indice_ = 1 + np.hstack([-1, self.active_main_effect_index_,
                        self.n_features_ + np.array(self.active_interaction_index_)]).astype(int)
        self.effect_names_ = np.hstack(["Intercept", np.array(self.feature_names_),
                        [self.feature_names_[self.interaction_list_[i][0]] + " x "
                      + self.feature_names_[self.interaction_list_[i][1]]
                         for i in range(len(self.interaction_list_))]])

        mout = self.get_main_effect_raw_output(self.training_generator_.tensors[0])
        if len(self.interaction_list_) > 0:
            iout = self.get_interaction_raw_output(self.training_generator_.tensors[0])
            shapley_value = np.vstack([mout[:, fidx] +
                       0.5 * iout[:, np.where(np.vstack(self.interaction_list_) == fidx)[0]].sum(1)
                       for fidx in range(self.n_features_)]).T
        else:
            shapley_value = mout

        feature_importance_raw = shapley_value.var(0)
        if np.sum(feature_importance_raw) == 0:
            self.feature_importance_ = np.zeros((self.n_features_))
        else:
            self.feature_importance_ = feature_importance_raw / feature_importance_raw.sum()
        self.is_fitted_ = True
        return self

    def get_main_effect_raw_output(self, x):
        """
        Returns numpy array of main effects' raw prediction.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.

        Returns
        -------
        pred : np.ndarray of shape (n_samples, n_features)
            numpy array of main effects' raw prediction.
        """
        if x.shape[1] != self.n_features_:
            raise ValueError("""The number of features in predict is
                     different from the number of features in fit""")

        pred = []
        self.net_.eval()
        x = x.reshape(-1, self.n_features_)
        xx = x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32)).to(self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        data_generator = FastTensorDataLoader(xx, batch_size=batch_size, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            pred.append(self.net_.forward_main_effect(batch_xx).detach())
        pred = torch.vstack(pred).detach().cpu().numpy()
        return pred

    def get_interaction_raw_output(self, x):
        """
        Returns numpy array of interactions' raw prediction.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.

        Returns
        -------
        pred : np.ndarray of shape (n_samples, n_interactions)
            numpy array of interactions' raw prediction.
        """
        if x.shape[1] != self.n_features_:
            raise ValueError("""The number of features in predict is
                     different from the number of features in fit""")

        pred = []
        self.net_.eval()
        x = x.reshape(-1, self.n_features_)
        xx = x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32)).to(self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        data_generator = FastTensorDataLoader(xx, batch_size=batch_size, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            pred.append(self.net_.forward_interaction(batch_xx).detach())
        pred = torch.vstack(pred).detach().cpu().numpy()
        return pred

    def get_aggregate_output(self, x, main_effect=True, interaction=True):
        """
        Returns numpy array of raw prediction.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        main_effect : boolean
            Whether to include main effects, default to True.
        interaction : boolean
            Whether to include interactions, default to True.

        Returns
        -------
        pred : np.ndarray of shape (n_samples, 1)
            numpy array of raw prediction.
        """
        if x.shape[1] != self.n_features_:
            raise ValueError("""The number of features in predict is
                     different from the number of features in fit""")

        pred = []
        self.net_.eval()
        x = x.reshape(-1, self.n_features_)
        xx = x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32)).to(self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        data_generator = FastTensorDataLoader(xx, batch_size=batch_size, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            pred.append(self.net_(batch_xx, sample_weight=None,
                    main_effect=main_effect, interaction=interaction,
                    clarity=False, monotonicity=False).detach())
        pred = torch.vstack(pred)
        return pred

    def get_clarity_loss(self, x, sample_weight=None):
        """
        Returns clarity loss of given samples.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.

        Returns
        -------
        clarity_loss : float
            clarity loss
        """
        clarity_loss = 0
        self.net_.eval()
        x = np.asarray(x).reshape(-1, self.n_features_)
        xx = x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32)).to(self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        data_generator = FastTensorDataLoader(xx, batch_size=batch_size, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            self.net_(batch_xx, sample_weight=sample_weight,
                  main_effect=True, interaction=True,
                  clarity=True, monotonicity=False)
            clarity_loss += len(batch_data[0]) * self.net_.clarity_loss.detach().cpu().numpy()
        clarity_loss = clarity_loss / data_generator.dataset_len
        return clarity_loss

    def get_mono_loss(self, x, sample_weight=None):
        """
        Returns monotonicity loss of given samples.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight, by default None.

        Returns
        -------
        mono_loss : float
            monotonicity loss
        """
        mono_loss = 0
        self.net_.eval()
        x = np.asarray(x).reshape(-1, self.n_features_)
        xx = x if torch.is_tensor(x) else torch.from_numpy(x.astype(np.float32)).to(self.device)
        batch_size = int(np.minimum(self.batch_size_inference, x.shape[0]))
        data_generator = FastTensorDataLoader(xx, batch_size=batch_size, shuffle=False)
        for batch_no, batch_data in enumerate(data_generator):
            batch_xx = batch_data[0]
            self.net_(batch_xx, sample_weight=sample_weight,
                  main_effect=True, interaction=True,
                  clarity=False, monotonicity=True)
            mono_loss += len(batch_data[0]) * self.net_.mono_loss.detach().cpu().numpy()
        mono_loss = mono_loss / data_generator.dataset_len
        return mono_loss

    def certify_mono(self, n_samples=10000):
        """
        Certify whether monotonicity constraint is satisfied.

        Parameters
        ----------
        n_samples : int
            Size of random samples for certifying
            the monotonicity constraint, by default 10000.

        Returns
        -------
        mono_status : boolean
            True means monotonicity constraint is satisfied.
        """
        x = np.random.uniform(self.min_value_.cpu().numpy(),
                      self.max_value_.cpu().numpy(), size=(n_samples, self.n_features_))
        mono_loss = self.get_mono_loss(x)
        mono_status = mono_loss <= 0
        return mono_status

    def partial_derivatives(self, feature_idx, n_samples=10000):
        """
        Plot the first-order partial derivatives w.r.t. given feature index.

        Parameters
        ----------
        feature_idx : int
            Feature index.
        n_samples : int
            Size of random samples for ploting the derivatives,
            by default 10000.
        """
        np.random.seed(self.random_state)
        inputs = np.random.uniform(self.min_value_.cpu().numpy(),
                       self.max_value_.cpu().numpy(), size=(n_samples, len(self.max_value_)))
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        outputs = self.net_(inputs)
        grad = torch.autograd.grad(outputs=torch.sum(outputs),
                          inputs=inputs, create_graph=True)[0].cpu().detach().numpy()
        plt.scatter(inputs.cpu().detach().numpy()[:, feature_idx], grad[:, feature_idx])
        plt.axhline(0, linestyle="--", linewidth=0.5, color="red")
        plt.ylabel("First-order Derivatives")
        plt.xlabel(self.feature_names_[feature_idx])
        absmax = 1.05 * np.max(np.abs(grad[:, feature_idx]))
        plt.ylim(-absmax, absmax)
        if feature_idx in self.mono_increasing_list:
            plt.title("Violating Size: %0.2f%%" %
              (np.where(grad[:, feature_idx] < 0)[0].shape[0] / n_samples * 100))
        if feature_idx in self.mono_decreasing_list:
            plt.title("Violating Size: %0.2f%%" %
              (np.where(grad[:, feature_idx] > 0)[0].shape[0] / n_samples * 100))
        plt.show()

    def load(self, folder="./", name="demo"):
        """
        Load a model from local disk.

        Parameters
        ----------
        folder : str
            The path of folder, by default "./".
        name : str
            Name of the file, by default "demo".
        """
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
        self.net_ = torch.load(save_path_model)

    def save(self, folder="./", name="demo"):
        """
        Save a model to local disk.

        Parameters
        ----------
        folder : str
            The path of folder, by default "./".
        name : str
            Name of the file, by default "demo".
        """
        model_dict = self.__dict__
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.net_, folder + name + "_model.pickle")
        with open(folder + name + "_dict.pickle", 'wb') as handle:
            pickle.dump(model_dict, handle)

    def global_explain(self, main_grid_size=100, interact_grid_size=20):
        """
        Extract the fitted main effects and interactions.

        Parameters
        ----------
        main_grid_size : int
            The grid size of main effects, by default 100.
        interact_grid_size : int
            The grid size of interactions, by default 20.
        """
        if hasattr(self, "data_dict_global_"):
            return self.data_dict_global_

        data_dict_global = self.data_dict_density_
        sorted_index, componment_scales = self._get_all_active_rank()
        for idx in range(self.n_features_):
            feature_name = self.feature_names_[idx]
            if idx in self.nfeature_index_list_:
                main_effect_inputs = np.zeros((main_grid_size, self.n_features_))
                main_effect_inputs[:, idx] = np.linspace(self.min_value_[idx].cpu().numpy(),
                                         self.max_value_[idx].cpu().numpy(), main_grid_size)
                main_effect_inputs_original = main_effect_inputs[:, [idx]]
                main_effect_outputs = (self.net_.main_effect_weights.cpu().detach().numpy()[idx] *
                               self.net_.main_effect_switcher.cpu().detach().numpy()[idx] *
                               self.get_main_effect_raw_output(main_effect_inputs)[:, idx])
                data_dict_global[feature_name].update({"type": "continuous",
                                      "importance": componment_scales[idx],
                                      "inputs": main_effect_inputs_original.ravel(),
                                      "outputs": main_effect_outputs.ravel()})
            elif idx in self.cfeature_index_list_:
                main_effect_inputs_original = self.dummy_values_[feature_name]
                main_effect_inputs = np.zeros((len(main_effect_inputs_original), self.n_features_))
                main_effect_inputs[:, idx] = np.arange(len(main_effect_inputs_original))
                main_effect_outputs = (self.net_.main_effect_weights.cpu().detach().numpy()[idx] *
                       self.net_.main_effect_switcher.cpu().detach().numpy()[idx] *
                       self.get_main_effect_raw_output(main_effect_inputs)[:, idx])
                main_effect_input_ticks = (main_effect_inputs_original.ravel().astype(int)
                       if len(main_effect_inputs_original) <= 6
                       else np.linspace(0.1 * len(main_effect_inputs_original),
                       len(main_effect_inputs_original) * 0.9, 4).astype(int))
                main_effect_input_labels = [main_effect_inputs_original[i]
                                        for i in main_effect_input_ticks]
                if len("".join(list(map(str, main_effect_input_labels)))) > 30:
                    main_effect_input_labels = [str(main_effect_inputs_original[i])[:4]
                                        for i in main_effect_input_ticks]
                data_dict_global[feature_name].update({"feature_name": feature_name,
                                      "type": "categorical",
                                      "importance": componment_scales[idx],
                                      "inputs": main_effect_inputs_original,
                                      "outputs": main_effect_outputs.ravel(),
                                      "input_ticks": main_effect_input_ticks,
                                      "input_labels": main_effect_input_labels})

        for idx in range(self.n_interactions_):

            idx1 = self.interaction_list_[idx][0]
            idx2 = self.interaction_list_[idx][1]
            feature_name1 = self.feature_names_[idx1]
            feature_name2 = self.feature_names_[idx2]
            feature_type1 = "categorical" if feature_name1 in self.cfeature_names_ else "continuous"
            feature_type2 = "categorical" if feature_name2 in self.cfeature_names_ else "continuous"

            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.cfeature_names_:
                interact_input1_original = self.dummy_values_[feature_name1]
                interact_input1 = np.arange(len(interact_input1_original), dtype=np.float32)
                interact_input1_ticks = (interact_input1.astype(int) if len(interact_input1) <= 6 else
                                 np.linspace(0.1 * len(interact_input1),
                                 len(interact_input1) * 0.9, 4).astype(int))
                interact_input1_labels = [interact_input1_original[i] for i in interact_input1_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [str(interact_input1_original[i])[:4]
                                      for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(interact_input1_original) - 0.5])
            else:
                interact_input1 = np.array(np.linspace(self.min_value_[idx1].cpu().numpy(),
                               self.max_value_[idx1].cpu().numpy(), interact_grid_size), dtype=np.float32)
                interact_input1_original = interact_input1.reshape(-1, 1)
                interact_input1_ticks = []
                interact_input1_labels = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.cfeature_names_:
                interact_input2_original = self.dummy_values_[feature_name2]
                interact_input2 = np.arange(len(interact_input2_original), dtype=np.float32)
                interact_input2_ticks = (interact_input2.astype(int) if len(interact_input2) <= 6 else
                                 np.linspace(0.1 * len(interact_input2),
                                 len(interact_input2) * 0.9, 4).astype(int))
                interact_input2_labels = [interact_input2_original[i] for i in interact_input2_ticks]
                if len("".join(list(map(str, interact_input2_labels)))) > 30:
                    interact_input2_labels = [str(interact_input2_original[i])[:4]
                                      for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(interact_input2_original) - 0.5])
            else:
                interact_input2 = np.array(np.linspace(self.min_value_[idx2].cpu().numpy(),
                                self.max_value_[idx2].cpu().numpy(), interact_grid_size), dtype=np.float32)
                interact_input2_original = interact_input2.reshape(-1, 1)
                interact_input2_ticks = []
                interact_input2_labels = []
                interact_input_list.append(interact_input2)
                axis_extent.extend([interact_input2_original.min(), interact_input2_original.max()])

            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            interaction_inputs = np.zeros((x1.shape[0] * x1.shape[1], self.n_features_))
            interaction_inputs[:, self.interaction_list_[idx][0]] = x1.ravel()
            interaction_inputs[:, self.interaction_list_[idx][1]] = x2.ravel()
            interact_outputs = (self.net_.interaction_weights.cpu().detach().numpy()[idx]
                    * self.net_.interaction_switcher.cpu().detach().numpy()[idx]
                    * self.get_interaction_raw_output(interaction_inputs))[:, idx].reshape(x1.shape)
            data_dict_global.update({feature_name1 + " x " + feature_name2:
                                {"feature_name1": feature_name1,
                                "feature_name2": feature_name2,
                                "type": "pairwise",
                                "xtype": feature_type1,
                                "ytype": feature_type2,
                                "importance": componment_scales[self.n_features_ + idx],
                                "input1": interact_input1_original,
                                "input2": interact_input2_original,
                                "outputs": interact_outputs,
                                "input1_ticks": interact_input1_ticks,
                                "input2_ticks": interact_input2_ticks,
                                "input1_labels": interact_input1_labels,
                                "input2_labels": interact_input2_labels,
                                "axis_extent": axis_extent}})
        self.data_dict_global_ = data_dict_global
        return data_dict_global

    def local_explain(self, x, y=None):
        """
        Extract the main effects and interactions values of a given sample.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        """
        predicted = self.predict(x)
        intercept = self.net_.output_bias.cpu().detach().numpy()

        main_effect_output = self.get_main_effect_raw_output(x)
        if self.n_interactions_ > 0:
            interaction_output = self.get_interaction_raw_output(x)
            interaction_weights = ((self.net_.interaction_weights.cpu().detach().numpy())
                      * self.net_.interaction_switcher.cpu().detach().numpy()).ravel()
        else:
            interaction_output = np.empty(shape=(x.shape[0], 0))
            interaction_weights = np.empty(shape=(0))

        main_effect_weights = ((self.net_.main_effect_weights.cpu().detach().numpy()) *
                        self.net_.main_effect_switcher.cpu().detach().numpy()).ravel()
        scores = np.hstack([np.repeat(intercept[0], x.shape[0]).reshape(-1, 1),
                        np.hstack([main_effect_weights, interaction_weights]) *
                        np.hstack([main_effect_output, interaction_output])])
        data_dict_local = [{"active_indice": self.active_indice_,
                    "scores": scores[i],
                    "effect_names": self.effect_names_,
                    "predicted": predicted[i],
                    "actual": y[i]} for i in range(x.shape[0])]
        return data_dict_local

    def show_global_explain(self, key=None, density=True,
                    main_effect_num=None, interaction_num=None,
                    folder="./", name="global_explain", save_eps=False, save_png=False):
        """
        Visualize the global explanation.

        Parameters
        ----------
        key : str or None
            The name of the effect to be shown, by default None.
            As key=None, all the effects would be visualized.
        density : boolean
            Whether to show the marginal density of each feature.
        main_effect_num : int or None
            The number of top main effects to show, by default None,
            As main_effect_num=None, all main effects would be shown.
        interaction_num : int or None
            The number of top interactions to show, by default None,
            As interaction_num=None, all main effects would be shown.
        folder : str
            The path of folder to save figure, by default "./".
        name : str
            Name of the file, by default "global_explain".
        save_png : boolean
            Whether to save the plot in PNG format, by default False.
        save_eps : boolean
            Whether to save the plot in EPS format, by default False.
        """
        data_dict_global = self.global_explain()
        if key is None:
            exp_dict = data_dict_global
        else:
            exp_dict = {}
            main_effect_num = None
            if "feature_name1" in data_dict_global[key]:
                feature_name1 = data_dict_global[key]["feature_name1"]
                feature_dict1 = data_dict_global[feature_name1]
                exp_dict.update({feature_name1: feature_dict1})
                main_effect_num = 0
            if "feature_name2" in data_dict_global[key]:
                feature_name2 = data_dict_global[key]["feature_name2"]
                feature_dict2 = data_dict_global[feature_name2]
                exp_dict.update({feature_name2: feature_dict2})
                main_effect_num = 0
            exp_dict.update({key: data_dict_global[key]})
        if density:
            global_visualize_density(exp_dict, main_effect_num=main_effect_num, 
                         interaction_num=interaction_num, folder=folder, name=name,
                         save_eps=save_eps, save_png=save_png)
        else:
            global_visualize_wo_density(exp_dict, main_effect_num=main_effect_num, 
                         interaction_num=interaction_num, folder=folder, name=name,
                         save_eps=save_eps, save_png=save_png)

    def show_local_explain(self, x, y=None, folder="./", name="local_explain", save_eps=False, save_png=False):
        """
        Visualize the local explanation.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        folder : str
            The path of folder to save figure, by default "./".
        name : str
            Name of the file, by default "local_explain".
        save_png : boolean
            Whether to save the plot in PNG format, by default False.
        save_eps : boolean
            Whether to save the plot in EPS format, by default False.
        """
        if x.shape[0] == 1:
            data_dict_local = self.local_explain(x, y)
            local_visualize(data_dict_local[0],
                    folder=folder, name=name, save_eps=save_eps, save_png=save_png)
        elif x.shape[0] > 1:
            data_dict_local = self.local_explain(x, y)
            for i in range(x.shape[0]):
                local_visualize(data_dict_local[i],
                    folder=folder, name=name, save_eps=save_eps, save_png=save_png)

    def show_feature_importance(self, folder="./", name="feature_importance", save_eps=False, save_png=False):
        """
        Visualize the feature importance.

        Parameters
        ----------
        folder : str
            The path of folder to save figure, by default "./".
        name : str
            Name of the file, by default "feature_importance".
        save_png : boolean
            Whether to save the plot in PNG format, by default False.
        save_eps : boolean
            Whether to save the plot in EPS format, by default False.
        """
        feature_importance_visualize(self.feature_importance_, self.feature_names_,
                             folder=folder, name=name, save_eps=save_eps, save_png=save_png)

    def show_effect_importance(self, folder="./", name="effect_importance", save_eps=False, save_png=False):
        """
        Visualize the effect importance.

        Parameters
        ----------
        folder : str
            The path of folder to save figure, by default "./".
        name : str
            Name of the file, by default "effect_importance".
        save_png : boolean
            Whether to save the plot in PNG format, by default False.
        save_eps : boolean
            Whether to save the plot in EPS format, by default False.
        """
        data_dict_global_ = self.global_explain()
        effect_importance_visualize(data_dict_global_,
                            folder=folder, name=name, save_eps=save_eps, save_png=save_png)

    def show_loss_trajectory(self, folder="./", name="loss_trajectory", save_eps=False, save_png=False):
        """
        Visualize the loss trajectory.

        Parameters
        ----------
        folder : str
            The path of folder to save figure, by default "./".
        name : str
            Name of the file, by default "loss_trajectory".
        save_png : boolean
            Whether to save the plot in PNG format, by default False.
        save_eps : boolean
            Whether to save the plot in EPS format, by default False.
        """
        data_dict_logs = {"err_train_main_effect_training": self.err_train_main_effect_training_,
                    "err_val_main_effect_training": self.err_val_main_effect_training_,
                    "err_train_interaction_training": self.err_train_interaction_training_,
                    "err_val_interaction_training": self.err_val_interaction_training_,
                    "err_train_tuning": self.err_train_tuning_,
                    "err_val_tuning": self.err_val_tuning_}
        plot_trajectory(data_dict_logs, folder=folder, name=name, save_eps=save_eps, save_png=save_png)

    def show_regularization_path(self, folder="./", name="regularization_path", save_eps=False, save_png=False):
        """
        Visualize the regularization path.

        Parameters
        ----------
        folder : str
            The path of folder to save figure, by default "./".
        name : str
            Name of the file, by default "regularization_path".
        save_png : boolean
            Whether to save the plot in PNG format, by default False.
        save_eps : boolean
            Whether to save the plot in EPS format, by default False.
        """
        data_dict_logs = {"active_main_effect_index": self.active_main_effect_index_,
                    "active_interaction_index": self.active_interaction_index_,
                    "main_effect_val_loss": self.main_effect_val_loss_,
                    "interaction_val_loss": self.interaction_val_loss_}
        plot_regularization(data_dict_logs, folder=folder, name=name, save_eps=save_eps, save_png=save_png)
