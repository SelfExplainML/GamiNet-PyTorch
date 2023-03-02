import os
import struct
import logging
import numpy as np
import pandas as pd
import ctypes as ct
from sys import platform
from numpy.ctypeslib import ndpointer
from collections import OrderedDict
from pandas.core.generic import NDFrame
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from contextlib import AbstractContextManager
import pkg_resources

try:
    from pandas.api.types import is_numeric_dtype, is_string_dtype
except ImportError:  # pragma: no cover
    from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

log = logging.getLogger(__name__)


# All the codes in this file are from Interpretml by Microsoft.


def autogen_schema(X, ordinal_max_items=2, feature_names=None, feature_types=None):
    """ Generates data schema for a given dataset as JSON representable.
    Args:
        X: Dataframe/ndarray to build schema from.
        ordinal_max_items: If a numeric column's cardinality
            is at most this integer,
            consider it as ordinal instead of continuous.
        feature_names: Feature names
        feature_types: Feature types
    Returns:
        A dictionary - schema that encapsulates column information,
        such as type and domain.
    """
    schema = OrderedDict()
    col_number = 0
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f'feature_{i:04}' for i in range(1, 1 + X.shape[1])]

        # NOTE: Use rolled out infer_objects for old pandas.
        # As used from SO:
        # https://stackoverflow.com/questions/47393134/attributeerror-dataframe-object-has-no-attribute-infer-objects
        X = pd.DataFrame(X, columns=feature_names)
        try:
            X = X.infer_objects()
        except AttributeError:
            for k in list(X):
                X[k] = pd.to_numeric(X[k], errors="ignore")

    if isinstance(X, NDFrame):
        for name, col_dtype in zip(X.dtypes.index, X.dtypes):
            schema[name] = {}
            if is_numeric_dtype(col_dtype):
                if X[name].isin([np.nan, 0, 1]).all():
                    schema[name]["type"] = "categorical"
                else:
                    schema[name]["type"] = "continuous"
            elif is_string_dtype(col_dtype):
                schema[name]["type"] = "categorical"
            else:  # pragma: no cover
                warnings.warn("Unknown column: " + name, RuntimeWarning)
                schema[name]["type"] = "unknown"
            schema[name]["column_number"] = col_number
            col_number += 1

        # Override if feature_types is passed as arg.
        if feature_types is not None:
            for idx, name in enumerate(X.dtypes.index):
                schema[name]["type"] = feature_types[idx]
    else:  # pragma: no cover
        raise TypeError("EBMs only supports numpy arrays or pandas dataframes.")

    return schema

# TODO: More documentation in binning process to be explicit.
# TODO: Consider stripping this down to the bare minimum.
class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """ Transformer that preprocesses data to be ready before EBM. """

    def __init__(
        self, feature_names=None, feature_types=None, max_bins=256, binning="quantile", missing_str=str(np.nan), 
        epsilon=None, delta=None, privacy_schema=None
    ):
        """ Initializes EBM preprocessor.

        Args:
            feature_names: Feature names as list.
            feature_types: Feature types as list, for example "continuous" or "categorical".
            max_bins: Max number of bins to process numeric features.
            binning: Strategy to compute bins: "quantile", "quantile_humanized", "uniform", or "private". 
            missing_str: By default np.nan values are missing for all datatypes. Setting this parameter changes the string representation for missing
            epsilon: Privacy budget parameter. Only applicable when binning is "private".
            delta: Privacy budget parameter. Only applicable when binning is "private".
            privacy_schema: User specified min/maxes for numeric features as dictionary. Only applicable when binning is "private".
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.max_bins = max_bins
        self.binning = binning
        self.missing_str = missing_str
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_schema = privacy_schema

    def fit(self, X):
        """ Fits transformer to provided samples.

        Args:
            X: Numpy array for training samples.

        Returns:
            Itself.
        """

        self.col_bin_edges_ = {}
        self.col_min_ = {}
        self.col_max_ = {}

        self.hist_counts_ = {}
        self.hist_edges_ = {}

        self.col_mapping_ = {}

        self.col_bin_counts_ = []
        self.col_names_ = []
        self.col_types_ = []

        self.has_fitted_ = False

        native = Native.get_native_singleton()
        schema = autogen_schema(
            X, feature_names=self.feature_names, feature_types=self.feature_types
        )

        noise_scale = None # only applicable for private binning
        if "private" in self.binning:
            DPUtils.validate_eps_delta(self.epsilon, self.delta)
            noise_scale = DPUtils.calc_gdp_noise_multi(
                total_queries = X.shape[1], 
                target_epsilon = self.epsilon, 
                delta = self.delta
            )
            if self.privacy_schema is None:
                warn("Possible privacy violation: assuming min/max values per feature are public info."
                     "Pass a privacy schema with known public ranges per feature to avoid this warning.")
                self.privacy_schema = DPUtils.build_privacy_schema(X)
                
        if self.max_bins < 2:
            raise ValueError("max_bins must be 2 or higher.  One bin is required for missing, and annother for non-missing values.")

        for col_idx in range(X.shape[1]):
            col_name = list(schema.keys())[col_idx]
            self.col_names_.append(col_name)

            col_info = schema[col_name]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]

            self.col_types_.append(col_info["type"])
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)
                count_missing = 0

                if self.binning == "private":
                    min_val, max_val = self.privacy_schema[col_idx]
                    cuts, bin_counts = DPUtils.private_numeric_binning(
                        col_data, noise_scale, self.max_bins, min_val, max_val
                    )

                    # Use previously calculated bins for density estimates
                    hist_edges = np.concatenate([[min_val], cuts, [max_val]])
                    hist_counts = bin_counts[1:]
                else:  # Standard binning
                    min_samples_bin = 1 # TODO: Expose
                    is_humanized = 0
                    if self.binning == 'quantile' or self.binning == 'quantile_humanized':
                        if self.binning == 'quantile_humanized':
                            is_humanized = 1

                        (
                            cuts, 
                            count_missing, 
                            min_val, 
                            max_val, 
                        ) = native.cut_quantile(
                            col_data, 
                            min_samples_bin, 
                            is_humanized, 
                            self.max_bins - 2, # one bin for missing, and # of cuts is one less again
                        )
                    elif self.binning == "uniform":
                        (
                            cuts, 
                            count_missing, 
                            min_val, 
                            max_val,
                        ) = native.cut_uniform(
                            col_data, 
                            self.max_bins - 2, # one bin for missing, and # of cuts is one less again
                        )
                    else:
                        raise ValueError(f"Unrecognized bin type: {self.binning}")

                    discretized = native.discretize(col_data, cuts)
                    bin_counts = np.bincount(discretized, minlength=len(cuts) + 2)
                    if count_missing != 0:
                        col_data = col_data[~np.isnan(col_data)]

                    hist_counts, hist_edges = np.histogram(col_data, bins="doane")

                
                self.col_bin_counts_.append(bin_counts)
                self.col_bin_edges_[col_idx] = cuts
                self.col_min_[col_idx] = min_val
                self.col_max_[col_idx] = max_val
                self.hist_edges_[col_idx] = hist_edges
                self.hist_counts_[col_idx] = hist_counts
            elif col_info["type"] == "ordinal":
                mapping = {val: indx + 1 for indx, val in enumerate(col_info["order"])}
                self.col_mapping_[col_idx] = mapping
                self.col_bin_counts_.append(None) # TODO count the values in each bin
            elif col_info["type"] == "categorical":
                col_data = col_data.astype('U')

                if self.binning == "private":
                    uniq_vals, counts = DPUtils.private_categorical_binning(col_data, noise_scale, self.max_bins)
                else: # Standard binning
                    uniq_vals, counts = np.unique(col_data, return_counts=True)

                missings = np.isin(uniq_vals, self.missing_str)

                count_missing = np.sum(counts[missings])
                bin_counts = np.concatenate(([count_missing], counts[~missings]))
                self.col_bin_counts_.append(bin_counts)

                uniq_vals = uniq_vals[~missings]
                mapping = {val: indx + 1 for indx, val in enumerate(uniq_vals)}
                self.col_mapping_[col_idx] = mapping

        self.has_fitted_ = True
        return self

    def transform(self, X):
        """ Transform on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        missing_constant = 0
        unknown_constant = -1

        native = Native.get_native_singleton()

        X_new = np.copy(X)
        if issubclass(X.dtype.type, np.unsignedinteger):
            X_new = X_new.astype(np.int64)

        for col_idx in range(X.shape[1]):
            col_type = self.col_types_[col_idx]
            col_data = X[:, col_idx]

            if col_type == "continuous":
                col_data = col_data.astype(float)
                cuts = self.col_bin_edges_[col_idx]

                discretized = native.discretize(col_data, cuts)
                X_new[:, col_idx] = discretized

            elif col_type == "ordinal":
                mapping = self.col_mapping_[col_idx].copy()
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)
            elif col_type == "categorical":
                mapping = self.col_mapping_[col_idx].copy()

                # Use "DPOther" bin when possible to handle unknown values during DP.
                if "private" in self.binning:
                    for key, val in mapping.items():
                        if key == "DPOther": 
                            unknown_constant = val
                            missing_constant = val
                            break
                    else: # If DPOther keyword doesn't exist, revert to standard encoding scheme
                        missing_constant = 0
                        unknown_constant = -1

                if isinstance(self.missing_str, list):
                    for val in self.missing_str:
                        mapping[val] = missing_constant
                else:
                    mapping[self.missing_str] = missing_constant

                col_data = col_data.astype('U')
                X_new[:, col_idx] = np.fromiter(
                    (mapping.get(x, unknown_constant) for x in col_data), dtype=np.int64, count=X.shape[0]
                )

        return X_new.astype(np.int64)

    def _get_hist_counts(self, feature_index):
        col_type = self.col_types_[feature_index]
        if col_type == "continuous":
            return list(self.hist_counts_[feature_index])
        elif col_type == "categorical":
            return list(self.col_bin_counts_[feature_index][1:])
        else:  # pragma: no cover
            raise Exception("Cannot get counts for type: {0}".format(col_type))

    def _get_hist_edges(self, feature_index):
        col_type = self.col_types_[feature_index]
        if col_type == "continuous":
            return list(self.hist_edges_[feature_index])
        elif col_type == "categorical":
            map = self.col_mapping_[feature_index]
            return list(map.keys())
        else:  # pragma: no cover
            raise Exception("Cannot get counts for type: {0}".format(col_type))


    def _get_bin_labels(self, feature_index):
        """ Returns bin labels for a given feature index.

        Args:
            feature_index: An integer for feature index.

        Returns:
            List of labels for bins.
        """

        col_type = self.col_types_[feature_index]
        if col_type == "continuous":
            min_val = self.col_min_[feature_index]
            cuts = self.col_bin_edges_[feature_index]
            max_val = self.col_max_[feature_index]
            return list(np.concatenate(([min_val], cuts, [max_val])))
        elif col_type == "ordinal":
            map = self.col_mapping_[feature_index]
            return list(map.keys())
        elif col_type == "categorical":
            map = self.col_mapping_[feature_index]
            return list(map.keys())
        else:  # pragma: no cover
            raise Exception("Unknown column type")


class InteractionDetector(AbstractContextManager):
    """Lightweight wrapper for EBM C interaction code.
    """

    def __init__(
        self, 
        model_type, 
        n_classes, 
        features_categorical, 
        features_bin_count, 
        X, 
        y, 
        w, 
        scores, 
        optional_temp_params
    ):

        """ Initializes internal wrapper for EBM C code.

        Args:
            model_type: 'regression'/'classification'.
            n_classes: Specific to classification,
                number of unique classes.
            features_categorical: list of categorical features represented by bools 
            features_bin_count: count of the number of bins for each feature
            X: Training design matrix as 2-D ndarray.
            y: Training response as 1-D ndarray.
            w: Sample weights as 1-D ndarray (must be same shape as y).
            scores: predictions from a prior predictor.  For regression
                there is 1 prediction per sample.  For binary classification
                there is one logit.  For multiclass there are n_classes logits

        """

        self.model_type = model_type
        self.n_classes = n_classes
        self.features_categorical = features_categorical
        self.features_bin_count = features_bin_count
        self.X = X
        self.y = y
        self.w = w
        self.scores = scores
        self.optional_temp_params = optional_temp_params

    def __enter__(self):
        # check inputs for important inputs or things that would segfault in C
        if not isinstance(self.features_categorical, np.ndarray):  # pragma: no cover
            raise ValueError("features_categorical should be an np.ndarray")

        if not isinstance(self.features_bin_count, np.ndarray):  # pragma: no cover
            raise ValueError("features_bin_count should be an np.ndarray")

        if self.X.ndim != 2:  # pragma: no cover
            raise ValueError("X should have exactly 2 dimensions")

        if self.y.ndim != 1:  # pragma: no cover
            raise ValueError("y should have exactly 1 dimension")


        if self.X.shape[0] != len(self.features_categorical):  # pragma: no cover
            raise ValueError(
                "X does not have the same number of items as the features_categorical array"
            )

        if self.X.shape[0] != len(self.features_bin_count):  # pragma: no cover
            raise ValueError(
                "X does not have the same number of items as the features_bin_count array"
            )

        if self.X.shape[1] != len(self.y):  # pragma: no cover
            raise ValueError("X does not have the same number of samples as y")

        native = Native.get_native_singleton()

        log.info("Allocation interaction start")

        n_scores = Native.get_count_scores_c(self.n_classes)
        scores = self.scores
        if scores is None:  # pragma: no cover
            scores = np.zeros(len(self.y) * n_scores, dtype=ct.c_double, order="C")
        else:
            if scores.shape[0] != len(self.y):  # pragma: no cover
                raise ValueError(
                    "scores does not have the same number of samples as y"
                )
            if n_scores == 1:
                if scores.ndim != 1:  # pragma: no cover
                    raise ValueError(
                        "scores should have exactly 1 dimensions for regression or binary classification"
                    )
            else:
                if scores.ndim != 2:  # pragma: no cover
                    raise ValueError(
                        "scores should have exactly 2 dimensions for multiclass"
                    )
                if scores.shape[1] != n_scores:  # pragma: no cover
                    raise ValueError(
                        "scores does not have the same number of logit scores as n_scores"
                    )

        optional_temp_params = self.optional_temp_params
        if optional_temp_params is not None:  # pragma: no cover
            optional_temp_params = (ct.c_double * len(optional_temp_params))(
                *optional_temp_params
            )

        # Allocate external resources
        interaction_handle = ct.c_void_p(0)
        if self.model_type == "classification":
            return_code = native._unsafe.CreateClassificationInteractionDetector(
                self.n_classes,
                len(self.features_bin_count),
                self.features_categorical, 
                self.features_bin_count,
                len(self.y),
                self.X,
                self.y,
                self.w,
                scores,
                optional_temp_params,
                ct.byref(interaction_handle),
            )
            if return_code:  # pragma: no cover
                raise Native._get_native_exception(return_code, "CreateClassificationInteractionDetector")
        elif self.model_type == "regression":
            return_code = native._unsafe.CreateRegressionInteractionDetector(
                len(self.features_bin_count),
                self.features_categorical, 
                self.features_bin_count,
                len(self.y),
                self.X,
                self.y,
                self.w,
                scores,
                optional_temp_params,
                ct.byref(interaction_handle),
            )
            if return_code:  # pragma: no cover
                raise Native._get_native_exception(return_code, "CreateRegressionInteractionDetector")
        else:  # pragma: no cover
            raise AttributeError("Unrecognized model_type")

        self._interaction_handle = interaction_handle.value

        log.info("Allocation interaction end")
        return self

    def __exit__(self, *args):

        self.close()

    def close(self):

        """ Deallocates C objects used to determine interactions in EBM. """
        log.info("Deallocation interaction start")

        interaction_handle = getattr(self, "_interaction_handle", None)
        if interaction_handle:
            native = Native.get_native_singleton()
            self._interaction_handle = None
            native._unsafe.FreeInteractionDetector(interaction_handle)
        
        log.info("Deallocation interaction end")

    def get_interaction_score(self, feature_index_tuple, min_samples_leaf):
        """ Provides score for an feature interaction. Higher is better."""
        log.info("Fast interaction score start")

        native = Native.get_native_singleton()

        score = ct.c_double(0.0)
        return_code = native._unsafe.CalculateInteractionScore(
            self._interaction_handle,
            len(feature_index_tuple),
            np.array(feature_index_tuple, dtype=ct.c_int64),
            min_samples_leaf,
            ct.byref(score),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CalculateInteractionScore")

        log.info("Fast interaction score end")
        return score.value

class Native:

    # GenerateUpdateOptionsType
    GenerateUpdateOptions_Default               = 0x0000000000000000
    GenerateUpdateOptions_DisableNewtonGain     = 0x0000000000000001
    GenerateUpdateOptions_DisableNewtonUpdate   = 0x0000000000000002
    GenerateUpdateOptions_GradientSums          = 0x0000000000000004
    GenerateUpdateOptions_RandomSplits          = 0x0000000000000008

    # TraceLevel
    _TraceLevelOff = 0
    _TraceLevelError = 1
    _TraceLevelWarning = 2
    _TraceLevelInfo = 3
    _TraceLevelVerbose = 4

    _native = None
    _LogFuncType = ct.CFUNCTYPE(None, ct.c_int32, ct.c_char_p)

    def __init__(self):
        # Do not call "Native()".  Call "Native.get_native_singleton()" instead
        pass

    @staticmethod
    def get_native_singleton(is_debug=False):
        if Native._native is None:
            log.info("EBM lib loading.")
            native = Native()
            native._initialize(is_debug=is_debug)
            Native._native = native
        return Native._native

    @staticmethod
    def _get_native_exception(error_code, native_function):  # pragma: no cover
        if error_code == 2:
            return Exception(f'Out of memory in {native_function}')
        elif error_code == 3:
            return Exception(f'Unexpected internal error in {native_function}')
        elif error_code == 4:
            return Exception(f'Illegal native parameter value in {native_function}')
        elif error_code == 5:
            return Exception(f'User native parameter value error in {native_function}')
        elif error_code == 6:
            return Exception(f'Thread start failed in {native_function}')
        elif error_code == 10:
            return Exception(f'Loss constructor native exception in {native_function}')
        elif error_code == 11:
            return Exception(f'Loss parameter unknown')
        elif error_code == 12:
            return Exception(f'Loss parameter value malformed')
        elif error_code == 13:
            return Exception(f'Loss parameter value out of range')
        elif error_code == 14:
            return Exception(f'Loss parameter mismatch')
        elif error_code == 15:
            return Exception(f'Unrecognized loss type')
        elif error_code == 16:
            return Exception(f'Illegal loss registration name')
        elif error_code == 17:
            return Exception(f'Illegal loss parameter name')
        elif error_code == 18:
            return Exception(f'Duplicate loss parameter name')
        else:
            return Exception(f'Unrecognized native return code {error_code} in {native_function}')

    @staticmethod
    def get_count_scores_c(n_classes):
        # this should reflect how the C code represents scores
        return 1 if n_classes <= 2 else n_classes

    def set_logging(self, level=None):
        # NOTE: Not part of code coverage. It runs in tests, but isn't registered for some reason.
        def native_log(trace_level, message):  # pragma: no cover
            try:
                message = message.decode("ascii")

                if trace_level == self._TraceLevelError:
                    log.error(message)
                elif trace_level == self._TraceLevelWarning:
                    log.warning(message)
                elif trace_level == self._TraceLevelInfo:
                    log.info(message)
                elif trace_level == self._TraceLevelVerbose:
                    log.debug(message)
            except:  # pragma: no cover
                # we're being called from C, so we can't raise exceptions
                pass

        if level is None:
            root = logging.getLogger("interpret")
            level = root.getEffectiveLevel()

        level_dict = {
            logging.DEBUG: self._TraceLevelVerbose,
            logging.INFO: self._TraceLevelInfo,
            logging.WARNING: self._TraceLevelWarning,
            logging.ERROR: self._TraceLevelError,
            logging.CRITICAL: self._TraceLevelError,
            logging.NOTSET: self._TraceLevelOff,
            "DEBUG": self._TraceLevelVerbose,
            "INFO": self._TraceLevelInfo,
            "WARNING": self._TraceLevelWarning,
            "ERROR": self._TraceLevelError,
            "CRITICAL": self._TraceLevelError,
            "NOTSET": self._TraceLevelOff,
        }

        trace_level = level_dict[level]
        if self._typed_log_func is None and trace_level != self._TraceLevelOff:
            # it's critical that we put _LogFuncType(native_log) into 
            # self._typed_log_func, otherwise it will be garbage collected
            self._typed_log_func = self._LogFuncType(native_log)
            self._unsafe.SetLogMessageFunction(self._typed_log_func)

        self._unsafe.SetTraceLevel(trace_level)

    def generate_random_number(self, random_seed, stage_randomization_mix):
        return self._unsafe.GenerateRandomNumber(random_seed, stage_randomization_mix)

    def sample_without_replacement(
        self, 
        random_seed, 
        count_training_samples,
        count_validation_samples
    ):
        count_samples = count_training_samples + count_validation_samples
        random_seed = ct.c_int32(random_seed)
        count_training_samples = ct.c_int64(count_training_samples)
        count_validation_samples = ct.c_int64(count_validation_samples)

        sample_counts_out = np.empty(count_samples, dtype=np.int64, order="C")

        self._unsafe.SampleWithoutReplacement(
            random_seed,
            count_training_samples,
            count_validation_samples,
            sample_counts_out
        )

        return sample_counts_out

    def stratified_sampling_without_replacement(
        self, 
        random_seed, 
        count_target_classes,
        count_training_samples,
        count_validation_samples,
        targets
    ):
        count_samples = count_training_samples + count_validation_samples

        if len(targets) != count_samples:
            raise ValueError("count_training_samples + count_validation_samples should be equal to len(targets)")

        random_seed = ct.c_int32(random_seed)
        count_target_classes = ct.c_int64(count_target_classes)
        count_training_samples = ct.c_int64(count_training_samples)
        count_validation_samples = ct.c_int64(count_validation_samples)

        sample_counts_out = np.empty(count_samples, dtype=np.int64, order="C")

        return_code = self._unsafe.StratifiedSamplingWithoutReplacement(
            random_seed,
            count_target_classes,
            count_training_samples,
            count_validation_samples,
            targets,
            sample_counts_out
        )

        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "StratifiedSamplingWithoutReplacement")

        return sample_counts_out

    def cut_quantile(self, col_data, min_samples_bin, is_humanized, max_cuts):
        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        count_missing = ct.c_int64(0)
        min_val = ct.c_double(0)
        count_neg_inf = ct.c_int64(0)
        max_val = ct.c_double(0)
        count_inf = ct.c_int64(0)

        return_code = self._unsafe.CutQuantile(
            col_data.shape[0],
            col_data, 
            min_samples_bin,
            is_humanized,
            ct.byref(count_cuts),
            cuts,
            ct.byref(count_missing),
            ct.byref(min_val),
            ct.byref(count_neg_inf),
            ct.byref(max_val),
            ct.byref(count_inf)
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "CutQuantile")

        cuts = cuts[:count_cuts.value]
        count_missing = count_missing.value
        min_val = min_val.value
        max_val = max_val.value

        return cuts, count_missing, min_val, max_val

    def cut_uniform(self, col_data, max_cuts):
        cuts = np.empty(max_cuts, dtype=np.float64, order="C")
        count_cuts = ct.c_int64(max_cuts)
        count_missing = ct.c_int64(0)
        min_val = ct.c_double(0)
        count_neg_inf = ct.c_int64(0)
        max_val = ct.c_double(0)
        count_inf = ct.c_int64(0)

        self._unsafe.CutUniform(
            col_data.shape[0],
            col_data, 
            ct.byref(count_cuts),
            cuts,
            ct.byref(count_missing),
            ct.byref(min_val),
            ct.byref(count_neg_inf),
            ct.byref(max_val),
            ct.byref(count_inf)
        )

        cuts = cuts[:count_cuts.value]
        count_missing = count_missing.value
        min_val = min_val.value
        max_val = max_val.value

        return cuts, count_missing, min_val, max_val

    def suggest_graph_bounds(self, cuts, min_val=np.nan, max_val=np.nan):
        low_graph_bound = ct.c_double(0)
        high_graph_bound = ct.c_double(0)
        return_code = self._unsafe.SuggestGraphBounds(
            len(cuts),
            cuts[0] if 0 < len(cuts) else np.nan,
            cuts[-1] if 0 < len(cuts) else np.nan,
            min_val,
            max_val,
            ct.byref(low_graph_bound),
            ct.byref(high_graph_bound),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "SuggestGraphBounds")

        return low_graph_bound.value, high_graph_bound.value

    def discretize(self, col_data, cuts):
        discretized = np.empty(col_data.shape[0], dtype=np.int64, order="C")
        return_code = self._unsafe.Discretize(
            col_data.shape[0],
            col_data,
            cuts.shape[0],
            cuts,
            discretized
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "Discretize")

        return discretized


    def size_data_set_header(self, n_features, n_weights, n_targets):
        n_bytes = self._unsafe.SizeDataSetHeader(n_features, n_weights, n_targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(3, "SizeDataSetHeader")
        return n_bytes

    def fill_data_set_header(self, n_features, n_weights, n_targets, n_bytes, shared_data):
        opaque_state = ct.c_int64(0)
        return_code = self._unsafe.FillDataSetHeader(
            n_features, 
            n_weights, 
            n_targets, 
            n_bytes, 
            shared_data, 
            ct.byref(opaque_state),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillDataSetHeader")

        return opaque_state.value

    def size_feature(self, categorical, n_bins, binned_data):
        n_bytes = self._unsafe.SizeFeature(categorical, n_bins, len(binned_data), binned_data)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(3, "SizeFeature")
        return n_bytes

    def fill_feature(self, categorical, n_bins, binned_data, n_bytes, shared_data, opaque_state):
        opaque_state = ct.c_int64(opaque_state)
        return_code = self._unsafe.FillFeature(
            categorical, 
            n_bins, 
            len(binned_data), 
            binned_data, 
            n_bytes, 
            shared_data,
            ct.byref(opaque_state),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillFeature")

        return opaque_state.value

    def size_weight(self, weights):
        n_bytes = self._unsafe.SizeWeight(len(weights), weights)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(3, "SizeWeight")
        return n_bytes

    def fill_weight(self, weights, n_bytes, shared_data, opaque_state):
        opaque_state = ct.c_int64(opaque_state)
        return_code = self._unsafe.FillWeight(
            len(weights), 
            weights, 
            n_bytes, 
            shared_data, 
            ct.byref(opaque_state),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillWeight")

        return opaque_state.value

    def size_classification_target(self, n_classes, targets):
        n_bytes = self._unsafe.SizeClassificationTarget(n_classes, len(targets), targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(3, "SizeClassificationTarget")
        return n_bytes

    def fill_classification_target(self, n_classes, targets, n_bytes, shared_data, opaque_state):
        opaque_state = ct.c_int64(opaque_state)
        return_code = self._unsafe.FillClassificationTarget(
            n_classes, 
            len(targets), 
            targets, 
            n_bytes, 
            shared_data,
            ct.byref(opaque_state),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillClassificationTarget")

        return opaque_state.value

    def size_regression_target(self, targets):
        n_bytes = self._unsafe.SizeRegressionTarget(len(targets), targets)
        if n_bytes < 0:  # pragma: no cover
            raise Native._get_native_exception(3, "SizeRegressionTarget")
        return n_bytes

    def fill_regression_target(self, targets, n_bytes, shared_data, opaque_state):
        opaque_state = ct.c_int64(opaque_state)
        return_code = self._unsafe.FillRegressionTarget(
            len(targets), 
            targets, 
            n_bytes, 
            shared_data, 
            ct.byref(opaque_state),
        )
        if return_code:  # pragma: no cover
            raise Native._get_native_exception(return_code, "FillRegressionTarget")

        return opaque_state.value


    @staticmethod
    def _get_ebm_lib_path(debug=False):
        """ Returns filepath of core EBM library.

        Returns:
            A string representing filepath.
        """
        bitsize = struct.calcsize("P") * 8
        is_64_bit = bitsize == 64

        script_path = os.path.dirname(os.path.abspath(__file__))
        package_path = script_path # os.path.join(script_path, "..", "..")

        debug_str = "" # "_debug" if debug else ""
        log.info("Loading native on {0} | debug = {1}".format(platform, debug))
        if platform == "linux" or platform == "linux2" and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_linux_x64{0}.so".format(debug_str)
            )
        elif platform == "win32" and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_win_x64{0}.dll".format(debug_str)
            )
        elif platform == "darwin" and is_64_bit:  # pragma: no cover
            return os.path.join(
                package_path, "lib", "lib_ebm_native_mac_x64{0}.dylib".format(debug_str)
            )
        else:  # pragma: no cover
            msg = "Platform {0} at {1} bit not supported for EBM".format(
                platform, bitsize
            )
            log.error(msg)
            raise Exception(msg)

    def _initialize(self, is_debug):
        self.is_debug = is_debug

        self._typed_log_func = None
        self._unsafe = ct.cdll.LoadLibrary(Native._get_ebm_lib_path(debug=is_debug))

        self._unsafe.SetLogMessageFunction.argtypes = [
            # void (* fn)(int32 traceLevel, const char * message) logMessageFunction
            self._LogFuncType
        ]
        self._unsafe.SetLogMessageFunction.restype = None

        self._unsafe.SetTraceLevel.argtypes = [
            # int32 traceLevel
            ct.c_int32
        ]
        self._unsafe.SetTraceLevel.restype = None


        self._unsafe.GenerateRandomNumber.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t stageRandomizationMix
            ct.c_int32,
        ]
        self._unsafe.GenerateRandomNumber.restype = ct.c_int32

        self._unsafe.SampleWithoutReplacement.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * sampleCountsOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.SampleWithoutReplacement.restype = None

        self._unsafe.StratifiedSamplingWithoutReplacement.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * targets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * sampleCountsOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.StratifiedSamplingWithoutReplacement.restype = ct.c_int32

        self._unsafe.CutQuantile.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t countSamplesPerBinMin
            ct.c_int64,
            # int64_t isHumanized
            ct.c_int64,
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countMissingValuesOut
            ct.POINTER(ct.c_int64),
            # double * minNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countNegativeInfinityOut
            ct.POINTER(ct.c_int64),
            # double * maxNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countPositiveInfinityOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.CutQuantile.restype = ct.c_int32

        self._unsafe.CutUniform.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countMissingValuesOut
            ct.POINTER(ct.c_int64),
            # double * minNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countNegativeInfinityOut
            ct.POINTER(ct.c_int64),
            # double * maxNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countPositiveInfinityOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.CutUniform.restype = None

        self._unsafe.CutWinsorized.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countCutsInOut
            ct.POINTER(ct.c_int64),
            # double * cutsLowerBoundInclusiveOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * countMissingValuesOut
            ct.POINTER(ct.c_int64),
            # double * minNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countNegativeInfinityOut
            ct.POINTER(ct.c_int64),
            # double * maxNonInfinityValueOut
            ct.POINTER(ct.c_double),
            # int64_t * countPositiveInfinityOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.CutWinsorized.restype = ct.c_int32


        self._unsafe.SuggestGraphBounds.argtypes = [
            # int64_t countCuts
            ct.c_int64,
            # double lowestCut
            ct.c_double,
            # double highestCut
            ct.c_double,
            # double minValue
            ct.c_double,
            # double maxValue
            ct.c_double,
            # double * lowGraphBoundOut
            ct.POINTER(ct.c_double),
            # double * highGraphBoundOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.SuggestGraphBounds.restype = ct.c_int32


        self._unsafe.Discretize.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # double * featureValues
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t countCuts
            ct.c_int64,
            # double * cutsLowerBoundInclusive
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # int64_t * discretizedOut
            ndpointer(dtype=ct.c_int64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.Discretize.restype = ct.c_int32


        self._unsafe.SizeDataSetHeader.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t countWeights
            ct.c_int64,
            # int64_t countTargets
            ct.c_int64,
        ]
        self._unsafe.SizeDataSetHeader.restype = ct.c_int64

        self._unsafe.FillDataSetHeader.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t countWeights
            ct.c_int64,
            # int64_t countTargets
            ct.c_int64,
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
            # int64_t * opaqueStateOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.FillDataSetHeader.restype = ct.c_int32

        self._unsafe.SizeFeature.argtypes = [
            # int64_t categorical
            ct.c_int64,
            # int64_t countBins
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=1),
        ]
        self._unsafe.SizeFeature.restype = ct.c_int64

        self._unsafe.FillFeature.argtypes = [
            # int64_t categorical
            ct.c_int64,
            # int64_t countBins
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
            # int64_t * opaqueStateInOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.FillFeature.restype = ct.c_int32

        self._unsafe.SizeWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * weights
            ndpointer(dtype=ct.c_double, ndim=1),
        ]
        self._unsafe.SizeWeight.restype = ct.c_int64

        self._unsafe.FillWeight.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * weights
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
            # int64_t * opaqueStateInOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.FillWeight.restype = ct.c_int32

        self._unsafe.SizeClassificationTarget.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * targets
            ndpointer(dtype=ct.c_int64, ndim=1),
        ]
        self._unsafe.SizeClassificationTarget.restype = ct.c_int64

        self._unsafe.FillClassificationTarget.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # int64_t * targets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
            # int64_t * opaqueStateInOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.FillClassificationTarget.restype = ct.c_int32

        self._unsafe.SizeRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * targets
            ndpointer(dtype=ct.c_double, ndim=1),
        ]
        self._unsafe.SizeRegressionTarget.restype = ct.c_int64

        self._unsafe.FillRegressionTarget.argtypes = [
            # int64_t countSamples
            ct.c_int64,
            # FloatEbmType * targets
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countBytesAllocated
            ct.c_int64,
            # void * fillMem
            ct.c_void_p,
            # int64_t * opaqueStateInOut
            ct.POINTER(ct.c_int64),
        ]
        self._unsafe.FillRegressionTarget.restype = ct.c_int32


        self._unsafe.Softmax.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countSamples
            ct.c_int64,
            # double * logits
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
            # double * probabilitiesOut
            ndpointer(dtype=ct.c_double, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.Softmax.restype = ct.c_int32


        self._unsafe.CreateClassificationBooster.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countFeatureGroups
            ct.c_int64,
            # int64_t * featureGroupsDimensionCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featureGroupsFeatureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t * trainingBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # int64_t * trainingTargets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * trainingWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * trainingPredictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * validationBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # int64_t * validationTargets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * validationWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * validationPredictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # BoosterHandle * boosterHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateClassificationBooster.restype = ct.c_int32

        self._unsafe.CreateRegressionBooster.argtypes = [
            # int32_t randomSeed
            ct.c_int32,
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countFeatureGroups
            ct.c_int64,
            # int64_t * featureGroupsDimensionCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featureGroupsFeatureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countTrainingSamples
            ct.c_int64,
            # int64_t * trainingBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * trainingTargets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * trainingWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * trainingPredictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countValidationSamples
            ct.c_int64,
            # int64_t * validationBinnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * validationTargets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * validationWeights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * validationPredictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # int64_t countInnerBags
            ct.c_int64,
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # BoosterHandle * boosterHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateRegressionBooster.restype = ct.c_int32

        self._unsafe.GenerateModelUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # GenerateUpdateOptionsType options 
            ct.c_int64,
            # double learningRate
            ct.c_double,
            # int64_t countSamplesRequiredForChildSplitMin
            ct.c_int64,
            # int64_t * leavesMax
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * gainOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.GenerateModelUpdate.restype = ct.c_int32

        self._unsafe.GetModelUpdateSplits.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexDimension
            ct.c_int64,
            # int64_t * countSplitsInOut
            ct.POINTER(ct.c_int64),
            # int64_t * splitIndexesOut
            ndpointer(dtype=ct.c_int64, ndim=1),
        ]
        self._unsafe.GetModelUpdateSplits.restype = ct.c_int32

        self._unsafe.GetModelUpdateExpanded.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * modelFeatureGroupUpdateTensorOut
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.GetModelUpdateExpanded.restype = ct.c_int32

        self._unsafe.SetModelUpdateExpanded.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupUpdateTensor
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.SetModelUpdateExpanded.restype = ct.c_int32

        self._unsafe.ApplyModelUpdate.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # double * validationMetricOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.ApplyModelUpdate.restype = ct.c_int32

        self._unsafe.GetBestModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupTensorOut
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.GetBestModelFeatureGroup.restype = ct.c_int32

        self._unsafe.GetCurrentModelFeatureGroup.argtypes = [
            # void * boosterHandle
            ct.c_void_p,
            # int64_t indexFeatureGroup
            ct.c_int64,
            # double * modelFeatureGroupTensorOut
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
        ]
        self._unsafe.GetCurrentModelFeatureGroup.restype = ct.c_int32

        self._unsafe.FreeBooster.argtypes = [
            # void * boosterHandle
            ct.c_void_p
        ]
        self._unsafe.FreeBooster.restype = None


        self._unsafe.CreateClassificationInteractionDetector.argtypes = [
            # int64_t countTargetClasses
            ct.c_int64,
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # int64_t * targets
            ndpointer(dtype=ct.c_int64, ndim=1),
            # double * weights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * predictorScores
            # scores can either be 1 or 2 dimensional
            ndpointer(dtype=ct.c_double, flags="C_CONTIGUOUS"),
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # InteractionHandle * interactionHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateClassificationInteractionDetector.restype = ct.c_int32

        self._unsafe.CreateRegressionInteractionDetector.argtypes = [
            # int64_t countFeatures
            ct.c_int64,
            # int64_t * featuresCategorical
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t * featuresBinCount
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countSamples
            ct.c_int64,
            # int64_t * binnedData
            ndpointer(dtype=ct.c_int64, ndim=2, flags="C_CONTIGUOUS"),
            # double * targets
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * weights
            ndpointer(dtype=ct.c_double, ndim=1),
            # ct.c_void_p,
            # double * predictorScores
            ndpointer(dtype=ct.c_double, ndim=1),
            # double * optionalTempParams
            ct.POINTER(ct.c_double),
            # InteractionHandle * interactionHandleOut
            ct.POINTER(ct.c_void_p),
        ]
        self._unsafe.CreateRegressionInteractionDetector.restype = ct.c_int32

        self._unsafe.CalculateInteractionScore.argtypes = [
            # void * interactionHandle
            ct.c_void_p,
            # int64_t countDimensions
            ct.c_int64,
            # int64_t * featureIndexes
            ndpointer(dtype=ct.c_int64, ndim=1),
            # int64_t countSamplesRequiredForChildSplitMin
            ct.c_int64,
            # double * interactionScoreOut
            ct.POINTER(ct.c_double),
        ]
        self._unsafe.CalculateInteractionScore.restype = ct.c_int32

        self._unsafe.FreeInteractionDetector.argtypes = [
            # void * interactionHandle
            ct.c_void_p
        ]
        self._unsafe.FreeInteractionDetector.restype = None

    @staticmethod
    def _convert_feature_groups_to_c(feature_groups):
        # Create C form of feature_groups

        feature_groups_feature_count = np.empty(len(feature_groups), dtype=ct.c_int64, order='C')
        feature_groups_feature_indexes = []
        for idx, features_in_group in enumerate(feature_groups):
            feature_groups_feature_count[idx] = len(features_in_group)
            for feature_idx in features_in_group:
                feature_groups_feature_indexes.append(feature_idx)

        feature_groups_feature_indexes = np.array(feature_groups_feature_indexes, dtype=ct.c_int64)

        return feature_groups_feature_count, feature_groups_feature_indexes
