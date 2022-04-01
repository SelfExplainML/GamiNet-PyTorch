import torch
import numpy as np
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator

from pygam.terms import TermList
from pygam import LinearGAM, LogisticGAM, s, te

from .base import GAMINet


class GAMINetRegressor(GAMINet, RegressorMixin):

    def __init__(self, meta_info=None, interact_num=10,
                 subnet_size_main_effect=[100], subnet_size_interaction=[200], activation_func=torch.nn.ReLU(),
                 max_epochs=[1000, 1000, 1000], learning_rates=[1e-3, 1e-3, 1e-3], early_stop_thres=["auto", "auto", "auto"],
                 batch_size=200, batch_size_inference=10000, max_iter_per_epoch=100, val_ratio=0.2, max_val_size=10000, 
                 warm_start=True, gam_sample_size=5000, mlp_sample_size=1000, 
                 heredity=True, reg_clarity=0.1, loss_threshold=0.0, 
                 reg_mono=0.1, mono_increasing_list=None, mono_decreasing_list=None, mono_sample_size=200,
                 boundary_clip=True, normalize=True, verbose=False, device="cpu", random_state=0):

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
                                   max_val_size=max_val_size,
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
                                   device=device,
                                   random_state=random_state)

    def _validate_input(self, x, y):

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def build_teacher_main_effect(self):

        termlist = TermList()
        for idx, (key, item) in enumerate(self.meta_info.items()):
            if (item["type"] == "continuous") or (item["type"] == "categorical"):
                termlist += s(idx, n_splines=10, spline_order=1, lam=0.6)
            else:
                continue

        gam = LinearGAM(termlist)
        allx = torch.vstack([self.tr_x, self.val_x])
        ally = torch.vstack([self.tr_y, self.val_y])

        suffleidx = np.arange(allx.shape[0])
        np.random.shuffle(suffleidx)
        subx = allx[suffleidx][:self.gam_sample_size]
        suby = ally[suffleidx][:self.gam_sample_size]
        gam.fit(((subx - self.mu_list) / self.std_list if self.normalize else subx).detach().cpu().numpy(), suby.detach().cpu().numpy())

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_features)]
        return surrogate_estimator, intercept

    def build_teacher_interaction(self):

        termlist = TermList()
        for i, (idx1, idx2) in enumerate(self.interaction_list):
            termlist += te(s(idx1, n_splines=10, spline_order=1, lam=0.6), 
                      s(idx2, n_splines=10, spline_order=1, lam=0.6))

        gam = LinearGAM(termlist)
        allx = torch.vstack([self.tr_x, self.val_x])
        ally = torch.vstack([self.tr_y, self.val_y])

        suffleidx = np.arange(allx.shape[0])
        np.random.shuffle(suffleidx)
        subx = allx[suffleidx][:self.gam_sample_size]
        suby = ally[suffleidx][:self.gam_sample_size]
        residual = (suby - self.decision_function(subx, main_effect=True, interaction=False))
        gam.fit(((subx - self.mu_list) / self.std_list if self.normalize else subx).detach().cpu().numpy(), residual.detach().cpu().numpy())

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_interactions)]
        return surrogate_estimator, intercept

    def get_interaction_list(self, x, y, w, scores, feature_names, feature_types, n_jobs):

        num_classes = -1
        model_type = "regression"
        interaction_list = self._get_interaction_list(x, y.astype(np.float64), w, scores, feature_names, feature_types,
                                       n_jobs, model_type, num_classes)
        return interaction_list

    def fit(self, x, y, sample_weight=None):

        self.init_fit(x, y, sample_weight)
        self._fit()

    def predict(self, x, main_effect=True, interaction=True):
        """Returns numpy array of predicted values for test data
        Parameters
        ----------
        x : np.ndarray
            Test data features of shape (n_samples, n_features)
        main_effect : boolean
            Whether to include main effects, default to True
        interaction : boolean 
            Whether to include interactions, default to True
        Returns
        -------
        np.ndarray
            numpy array of predicted values
        """
        pred = self.decision_function(x, main_effect=main_effect, interaction=interaction).detach().cpu().numpy()
        return pred


class GAMINetClassifier(GAMINet, ClassifierMixin):

    def __init__(self, meta_info=None, interact_num=10,
                 subnet_size_main_effect=[100], subnet_size_interaction=[200], activation_func=torch.nn.ReLU(),
                 max_epochs=[1000, 1000, 100], learning_rates=[1e-3, 1e-3, 1e-3], early_stop_thres=["auto", "auto", "auto"],
                 batch_size=200, batch_size_inference=10000, max_iter_per_epoch=100, val_ratio=0.2, max_val_size=10000, 
                 warm_start=True, gam_sample_size=5000, mlp_sample_size=1000, 
                 heredity=True, reg_clarity=0.1, loss_threshold=0.0, 
                 reg_mono=0.1, mono_increasing_list=None, mono_decreasing_list=None, mono_sample_size=200,
                 boundary_clip=True, normalize=True, verbose=False, device="cpu", random_state=0):

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
                                   max_val_size=max_val_size,
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
                                   device=device,
                                   random_state=random_state)

    def _validate_input(self, x, y):

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()

    def build_teacher_main_effect(self):

        termlist = TermList()
        for idx, (key, item) in enumerate(self.meta_info.items()):
            if (item["type"] == "continuous") or (item["type"] == "categorical"):
                termlist += s(idx, n_splines=10, spline_order=1, lam=0.6)
            else:
                continue

        gam = LinearGAM(termlist)
        allx = torch.vstack([self.tr_x, self.val_x])
        ally = torch.vstack([self.tr_y, self.val_y]) * 4 - 2

        suffleidx = np.arange(allx.shape[0])
        np.random.shuffle(suffleidx)
        subx = allx[suffleidx][:self.gam_sample_size]
        suby = ally[suffleidx][:self.gam_sample_size]
        gam.fit(((subx - self.mu_list) / self.std_list if self.normalize else subx).detach().cpu().numpy(), suby.detach().cpu().numpy())

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_features)]
        return surrogate_estimator, intercept

    def build_teacher_interaction(self):

        termlist = TermList()
        for i, (idx1, idx2) in enumerate(self.interaction_list):
            termlist += te(s(idx1, n_splines=10, spline_order=1, lam=0.6), 
                      s(idx2, n_splines=10, spline_order=1, lam=0.6))

        gam = LinearGAM(termlist)
        allx = torch.vstack([self.tr_x, self.val_x])
        ally = torch.vstack([self.tr_y, self.val_y])

        suffleidx = np.arange(allx.shape[0])
        np.random.shuffle(suffleidx)
        subx = allx[suffleidx][:self.gam_sample_size]
        suby = ally[suffleidx][:self.gam_sample_size]
        residual = (suby - self.predict_proba(subx, main_effect=True, interaction=False)[:, [1]])
        gam.fit(((subx - self.mu_list) / self.std_list if self.normalize else subx).detach().cpu().numpy(), residual.detach().cpu().numpy())

        def margial_effect(i):
            return lambda x: gam.partial_dependence(i, x)

        intercept = gam.coef_[-1]
        surrogate_estimator = [margial_effect(i) for i in range(self.n_interactions)]
        return surrogate_estimator, intercept

    def get_interaction_list(self, x, y, w, scores, feature_names, feature_types, n_jobs):

        num_classes = 2
        model_type = "classification"
        scores = np.minimum(np.maximum(scores, 0.0000001), 0.9999999)
        scores = np.log(scores / (1 - scores))

        interaction_list = self._get_interaction_list(x, y.astype(np.int64), w, scores, feature_names,
                                       feature_types, n_jobs, model_type, num_classes)
        return interaction_list

    def fit(self, x, y, sample_weight=None):

        self.init_fit(x, y, sample_weight, stratified=True)
        self._fit()
        
    def predict_proba(self, x, main_effect=True, interaction=True):
        """Returns numpy array of predicted probabilities
        Parameters
        ----------
        x : np.ndarray
            Test data features of shape (n_samples, n_features)
        main_effect : boolean
            Whether to include main effects, default to True
        interaction : boolean 
            Whether to include interactions, default to True
        Returns
        -------
        np.ndarray
            numpy array of predicted proba values
        """
        pred = self.decision_function(x, main_effect=main_effect, interaction=interaction).detach().cpu().numpy().ravel()
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)
        return pred_proba

    def predict(self, x, main_effect=True, interaction=True):
        """Returns numpy array of predicted class
        Parameters
        ----------
        x : np.ndarray
            Test data features of shape (n_samples, n_features)
        main_effect : boolean
            Whether to include main effects, default to True
        interaction : boolean 
            Whether to include interactions, default to True
        Returns
        -------
        np.ndarray
            numpy array of predicted class values
        """

        pred_proba = self.predict_proba(x, main_effect=main_effect, interaction=interaction)[:, 1]
        pred = np.array(pred_proba > 0.5, dtype=np.int)
        return pred
