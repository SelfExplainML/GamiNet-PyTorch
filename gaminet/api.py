import torch
import numpy as np
from sklearn.utils import column_or_1d
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.base import RegressorMixin, ClassifierMixin

from pygam.terms import TermList
from pygam import LinearGAM, s, te

from .base import GAMINet


class GAMINetRegressor(GAMINet, RegressorMixin):

    def __init__(self, meta_info=None, interact_num=10,
                 subnet_size_main_effect=(20,), subnet_size_interaction=(20, 20), activation_func="ReLU",
                 max_epochs=(1000, 1000, 1000), learning_rates=(1e-3, 1e-3, 1e-4), early_stop_thres=("auto", "auto", "auto"),
                 batch_size=200, batch_size_inference=10000, max_iter_per_epoch=100, val_ratio=0.2,
                 warm_start=True, gam_sample_size=5000, mlp_sample_size=1000,
                 heredity=True, reg_clarity=0.1, loss_threshold=0.0,
                 reg_mono=0.1, mono_increasing_list=(), mono_decreasing_list=(), mono_sample_size=1000,
                 boundary_clip=True, normalize=True, verbose=False, n_jobs=10, device="cpu", random_state=0):

        super(GAMINetRegressor, self).__init__(loss_fn=torch.nn.MSELoss(reduction="none"),
                                   meta_info=meta_info,
                                   interact_num=interact_num,
                                   subnet_size_main_effect=subnet_size_main_effect,
                                   subnet_size_interaction=subnet_size_interaction,
                                   activation_func=activation_func,
                                   max_epochs=max_epochs,
                                   learning_rates=learning_rates,
                                   early_stop_thres=early_stop_thres,
                                   batch_size=batch_size,
                                   batch_size_inference=batch_size_inference,
                                   max_iter_per_epoch=max_iter_per_epoch,
                                   val_ratio=val_ratio,
                                   warm_start=warm_start,
                                   gam_sample_size=gam_sample_size,
                                   mlp_sample_size=mlp_sample_size,
                                   heredity=heredity,
                                   reg_clarity=reg_clarity,
                                   loss_threshold=loss_threshold,
                                   reg_mono=reg_mono,
                                   mono_sample_size=mono_sample_size,
                                   mono_increasing_list=mono_increasing_list,
                                   mono_decreasing_list=mono_decreasing_list,
                                   boundary_clip=boundary_clip,
                                   normalize=normalize,
                                   verbose=verbose,
                                   n_jobs=n_jobs,
                                   device=device,
                                   random_state=random_state)

    def _more_tags(self):
        """
        Internal function for skipping some sklearn estimator checks.
        """
        return {"_xfail_checks": {"check_sample_weights_invariance":
                          ("zero sample_weight is not equivalent to removing samples")}}

    def _validate_input(self, x, y, sample_weight):
        """
        Internal function for validating the inputs of the fit function.

        Samples with zero sample_weight are removed.
        Sample_weight would be normalized, such that the sum equals sample size.
        Will raise an error if only one sample is given.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.

        Returns
        -------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.
        """
        x, y = self._validate_data(x, y, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        if sample_weight is None:
            sample_weight = np.ones(x.shape[0])
        else:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != x.shape[0]:
                raise ValueError("sample_weight shape mismatches the input")
            valid_idx = np.where(sample_weight > 0)[0]
            x, y, sample_weight = x[valid_idx], y[valid_idx], sample_weight[valid_idx]
            if np.sum(sample_weight) > 0:
                sample_weight = x.shape[0] * sample_weight.ravel() / np.sum(sample_weight)
        if x.shape[0] == 1:
            raise ValueError("n_samples=1")
        return x, y.ravel(), sample_weight.ravel()

    def _build_teacher_main_effect(self):
        """
        Internal function for fiting a spline based additive interaction model.

        It works as follows.
        1) Subsample at most self.gam_sample_size data from training set.
        2) Get the residual with respect to the fitted main effect networks,
          for classification case, the residual is y_label - pred_proba.
        3) Fit a tensor-product spline GAM for selected interactions, to make it
          scalable for large number of interactions, the number of knots
          in spline is adaptively adjusted from 10 to 2, according to
          the number of interactions.
        4) Wrap the partial function of each effect and intercept.

        Returns
        -------
        surrogate_estimator : object
            List of wrapped functions, each element is a fitted effect.
        intercept : float
            Fitted intercept.
        """

        x = self.training_generator_.tensors[0].cpu().numpy()
        y = self.training_generator_.tensors[1].cpu().numpy()
        sw = self.training_generator_.tensors[2].cpu().numpy()
        if self.gam_sample_size >= x.shape[0]:
            xx, yy, swsw = x, y, sw
        else:
            _, xx, _, yy, _, swsw = train_test_split(x, y, sw, test_size=self.gam_sample_size, random_state=self.random_state)

        termlist = TermList()
        n_splines = max(11 - np.ceil(self.n_features_ / 100).astype(int), 2)
        for idx in range(self.n_features_):
            termlist += s(idx, n_splines=n_splines, spline_order=1, lam=0.6)

        gam = LinearGAM(termlist)
        gam.fit((xx - self.mu_list_.cpu().numpy()) / self.std_list_.cpu().numpy(), yy, weights=swsw)

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_features_)]
        return surrogate_estimator, intercept

    def _build_teacher_interaction(self):
        """
        Internal function for fiting a spline based additive interaction model.

        It works as follows.
        1) Subsample at most self.gam_sample_size data from training set.
        2) Get the residual with respect to the fitted main effect networks,
          for classification case, the residual is y_label - pred_proba.
        3) Fit a tensor-product spline GAM for selected interactions, to make it
          scalable for large number of interactions, the number of knots
          in spline is adaptively adjusted from 10 to 2, according to
          the number of interactions.
        4) Wrap the partial function of each effect and intercept.

        Returns
        -------
        surrogate_estimator : object
            List of wrapped functions, each element is a fitted effect.
        intercept : float
            Fitted intercept.
        """
        x = self.training_generator_.tensors[0].cpu().numpy()
        y = self.training_generator_.tensors[1].cpu().numpy()
        sw = self.training_generator_.tensors[2].cpu().numpy()
        if self.gam_sample_size >= x.shape[0]:
            xx, yy, swsw = x, y, sw
        else:
            _, xx, _, yy, _, swsw = train_test_split(x, y, sw, test_size=self.gam_sample_size, random_state=self.random_state)
        residual = yy - self.get_aggregate_output(xx, main_effect=True, interaction=False).detach().cpu().numpy().ravel()

        termlist = TermList()
        n_splines = max(11 - np.ceil(self.n_interactions_ / 10).astype(int), 2)
        for i, (idx1, idx2) in enumerate(self.interaction_list_):
            termlist += te(s(idx1, n_splines=n_splines, spline_order=1, lam=0.6),
                      s(idx2, n_splines=n_splines, spline_order=1, lam=0.6))

        gam = LinearGAM(termlist)
        gam.fit((xx - self.mu_list_.cpu().numpy()) / self.std_list_.cpu().numpy(), residual, weights=swsw)

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_interactions_)]
        return surrogate_estimator, intercept

    def _get_interaction_list(self, x, y, w, scores, feature_names, feature_types):
        """
        Internal function for screening interactions in regression setting.

        Returns
        -------
        interaction_list : list of int
            List of paired tuple index, each indicating the feature index.
        """
        num_classes = -1
        model_type = "regression"
        interaction_list = self._interaction_screening(x, y.astype(np.float64), w, scores, feature_names, feature_types,
                                       model_type, num_classes)
        return interaction_list

    def fit(self, x, y, sample_weight=None):
        """
        Fit GAMINetRegressor model.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        self._init_fit(x, y, sample_weight)
        return self._fit()

    def predict(self, x, main_effect=True, interaction=True):
        """
        Returns numpy array of predicted values.

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
        pred: np.ndarray of shape (n_samples, )
            numpy array of predicted values.
        """
        check_is_fitted(self)
        x = self._validate_data(x)
        pred = self.get_aggregate_output(x, main_effect=main_effect, interaction=interaction).detach().cpu().numpy().ravel()
        return pred


class GAMINetClassifier(GAMINet, ClassifierMixin):

    def __init__(self, meta_info=None, interact_num=10,
                 subnet_size_main_effect=(20,), subnet_size_interaction=(20, 20), activation_func="ReLU",
                 max_epochs=(1000, 1000, 1000), learning_rates=(1e-3, 1e-3, 1e-4), early_stop_thres=("auto", "auto", "auto"),
                 batch_size=200, batch_size_inference=10000, max_iter_per_epoch=100, val_ratio=0.2,
                 warm_start=True, gam_sample_size=5000, mlp_sample_size=1000,
                 heredity=True, reg_clarity=0.1, loss_threshold=0.0,
                 reg_mono=0.1, mono_increasing_list=(), mono_decreasing_list=(), mono_sample_size=1000,
                 boundary_clip=True, normalize=True, verbose=False, n_jobs=10, device="cpu", random_state=0):

        super(GAMINetClassifier, self).__init__(loss_fn=torch.nn.BCEWithLogitsLoss(reduction="none"),
                                   meta_info=meta_info,
                                   interact_num=interact_num,
                                   subnet_size_main_effect=subnet_size_main_effect,
                                   subnet_size_interaction=subnet_size_interaction,
                                   activation_func=activation_func,
                                   max_epochs=max_epochs,
                                   learning_rates=learning_rates,
                                   early_stop_thres=early_stop_thres,
                                   batch_size=batch_size,
                                   batch_size_inference=batch_size_inference,
                                   max_iter_per_epoch=max_iter_per_epoch,
                                   val_ratio=val_ratio,
                                   warm_start=warm_start,
                                   gam_sample_size=gam_sample_size,
                                   mlp_sample_size=mlp_sample_size,
                                   heredity=heredity,
                                   reg_clarity=reg_clarity,
                                   loss_threshold=loss_threshold,
                                   reg_mono=reg_mono,
                                   mono_sample_size=mono_sample_size,
                                   mono_increasing_list=mono_increasing_list,
                                   mono_decreasing_list=mono_decreasing_list,
                                   boundary_clip=boundary_clip,
                                   normalize=normalize,
                                   verbose=verbose,
                                   n_jobs=n_jobs,
                                   device=device,
                                   random_state=random_state)

    def _more_tags(self):
        """
        Internal function for skipping some sklearn estimator checks.
        """
        return {"binary_only": True,
              "_xfail_checks": {"check_sample_weights_invariance":
                          ("zero sample_weight is not equivalent to removing samples")}}

    def _validate_input(self, x, y, sample_weight):
        """
        Internal function for validating the inputs of the fit function.

        Samples with zero sample_weight are removed.
        Sample_weight would be normalized, such that the sum equals sample size.
        Will raise an error if only one sample is given.
        The target label would be encoded as 0 and 1.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.

        Returns
        -------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.
        """
        x, y = self._validate_data(x, y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)
        if sample_weight is None:
            sample_weight = np.ones(x.shape[0])
        else:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != x.shape[0]:
                raise ValueError("sample_weight shape mismatches the input")
            valid_idx = np.where(sample_weight > 0)[0]
            x, y, sample_weight = x[valid_idx], y[valid_idx], sample_weight[valid_idx]
            if np.sum(sample_weight) > 0:
                sample_weight = x.shape[0] * sample_weight.ravel() / np.sum(sample_weight)
        if x.shape[0] == 1:
            raise ValueError("n_samples=1")

        self.label_binarizer_ = LabelBinarizer()
        self.label_binarizer_.fit(y)
        self.classes_ = self.label_binarizer_.classes_
        if len(self.classes_) > 2:
            raise ValueError("multi-classification not supported")
        y = self.label_binarizer_.transform(y) * 1.0
        return x, y.ravel(), sample_weight.ravel()

    def _build_teacher_main_effect(self):
        """
        Internal function for fiting a spline based additive model.

        It works as follows.
        1) Subsample at most self.gam_sample_size data from training set.
        2) Fit a B-spline GAM for all input features, to make it
          scalable for large number of interactions, the number of knots
          in spline is adaptively adjusted from 10 to 2, according to
          the number of features.
        3) Wrap the partial function of each effect and intercept.

        Returns
        -------
        surrogate_estimator : object
            List of wrapped functions, each element is a fitted effect.
        intercept : float
            Fitted intercept.
        """
        x = self.training_generator_.tensors[0].cpu().numpy()
        y = self.training_generator_.tensors[1].cpu().numpy() * 4 - 2
        sw = self.training_generator_.tensors[2].cpu().numpy()
        if self.gam_sample_size >= x.shape[0]:
            xx, yy, swsw = x, y, sw
        else:
            _, xx, _, yy, _, swsw = train_test_split(x, y, sw,
             test_size=self.gam_sample_size, stratify=y, random_state=self.random_state)

        termlist = TermList()
        n_splines = max(11 - np.ceil(self.n_features_ / 100).astype(int), 2)
        for idx in range(self.n_features_):
            termlist += s(idx, n_splines=n_splines, spline_order=1, lam=0.6)

        gam = LinearGAM(termlist)
        gam.fit((xx - self.mu_list_.cpu().numpy()) / self.std_list_.cpu().numpy(),
            yy, weights=swsw)

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_features_)]
        return surrogate_estimator, intercept

    def _build_teacher_interaction(self):
        """
        Internal function for fiting a spline based additive interaction model.

        It works as follows.
        1) Subsample at most self.gam_sample_size data from training set.
        2) Get the residual with respect to the fitted main effect networks,
          for classification case, the residual is y_label - pred_proba.
        3) Fit a tensor-product spline GAM for selected interactions, to make it
          scalable for large number of interactions, the number of knots
          in spline is adaptively adjusted from 10 to 2, according to
          the number of interactions.
        4) Wrap the partial function of each effect and intercept.

        Returns
        -------
        surrogate_estimator : object
            List of wrapped functions, each element is a fitted effect.
        intercept : float
            Fitted intercept.
        """
        x = self.training_generator_.tensors[0].cpu().numpy()
        y = self.training_generator_.tensors[1].cpu().numpy()
        sw = self.training_generator_.tensors[2].cpu().numpy()
        if self.gam_sample_size >= x.shape[0]:
            xx, yy, swsw = x, y, sw
        else:
            _, xx, _, yy, _, swsw = train_test_split(x, y, sw,
             test_size=self.gam_sample_size, stratify=y, random_state=self.random_state)

        pred = self.get_aggregate_output(xx, main_effect=True,
             interaction=False).detach().cpu().numpy().ravel()
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        residual = yy - pred_proba

        termlist = TermList()
        n_splines = max(11 - np.ceil(self.n_interactions_ / 10).astype(int), 2)
        for i, (idx1, idx2) in enumerate(self.interaction_list_):
            termlist += te(s(idx1, n_splines=n_splines, spline_order=1, lam=0.6),
                      s(idx2, n_splines=n_splines, spline_order=1, lam=0.6))

        gam = LinearGAM(termlist)
        gam.fit((xx - self.mu_list_.cpu().numpy()) / self.std_list_.cpu().numpy(),
             residual, weights=swsw)

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_interactions_)]
        return surrogate_estimator, intercept

    def _get_interaction_list(self, x, y, w, scores, feature_names, feature_types):
        """
        Internal function for screening interactions in classification setting.

        Returns
        -------
        interaction_list : list of int
            List of paired tuple index, each indicating the feature index.
        """
        num_classes = 2
        model_type = "classification"
        scores = np.minimum(np.maximum(scores, 0.0000001), 0.9999999)
        scores = np.log(scores / (1 - scores))

        interaction_list = self._interaction_screening(x, y.astype(np.int64), w,
                       scores, feature_names, feature_types, model_type, num_classes)
        return interaction_list

    def fit(self, x, y, sample_weight=None):
        """
        Fit GAMINetClassifier model.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features.
        y : np.ndarray of shape (n_samples, )
            Target response.
        sample_weight : np.ndarray of shape (n_samples, )
            Sample weight.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        self._init_fit(x, y, sample_weight, stratified=True)
        return self._fit()

    def decision_function(self, x, main_effect=True, interaction=True):
        """
        Returns numpy array of raw predicted value before softmax.

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
        pred : np.ndarray of shape (n_samples, )
            numpy array of predicted class values.
        """
        check_is_fitted(self)
        x = self._validate_data(x)
        pred = self.get_aggregate_output(x, main_effect=main_effect,
                 interaction=interaction).detach().cpu().numpy().ravel()
        return pred

    def predict_proba(self, x, main_effect=True, interaction=True):
        """
        Returns numpy array of predicted probabilities of each class.

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
        pred_proba : np.ndarray of shape (n_samples, 2)
            numpy array of predicted proba values.
        """
        pred = self.decision_function(x, main_effect=main_effect, interaction=interaction)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)
        return pred_proba

    def predict(self, x, main_effect=True, interaction=True):
        """
        Returns numpy array of predicted class.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Data features
        main_effect : boolean
            Whether to include main effects, default to True.
        interaction : boolean
            Whether to include interactions, default to True.

        Returns
        -------
        pred : np.ndarray of shape (n_samples, )
            numpy array of predicted class values.
        """
        pred_proba = self.predict_proba(x, main_effect=main_effect, interaction=interaction)[:, 1]
        pred = np.array(pred_proba > 0.5, dtype=np.int)
        return self.label_binarizer_.inverse_transform(pred)
