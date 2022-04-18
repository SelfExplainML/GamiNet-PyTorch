import os
import numpy as np

import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_regularization(data_dict_logs, log_scale=True, folder="./results/", name="regularization_path", save_eps=False, save_png=False):
    """
    Helper function for visualizing regularization path.

    Parameters
    ----------
    data_dict_logs : dict
        Dictionary containing regularization path information.
    log_scale : boolean
        Whether to use log scale for y-axis.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "regularization_path".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """

    main_loss = data_dict_logs["main_effect_val_loss"]
    inter_loss = data_dict_logs["interaction_val_loss"]
    active_main_effect_index = data_dict_logs["active_main_effect_index"]
    active_interaction_index = data_dict_logs["active_interaction_index"]

    fig = plt.figure(figsize=(14, 4))
    if len(main_loss) > 0:
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(np.arange(0, len(main_loss), 1), main_loss)
        ax1.axvline(np.argmin(main_loss), linestyle="dotted", color="red")
        ax1.axvline(len(active_main_effect_index), linestyle="dotted", color="red")
        ax1.plot(np.argmin(main_loss), np.min(main_loss), "*", markersize=12, color="red")
        ax1.plot(len(active_main_effect_index), main_loss[len(active_main_effect_index)], "o", markersize=8, color="red")
        ax1.set_xlabel("Number of Main Effects", fontsize=12)
        ax1.set_xlim(-0.5, len(main_loss) - 0.5)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax1.set_yscale("log")
            ax1.set_yticks((10 ** np.linspace(np.log10(np.nanmin(main_loss)), np.log10(np.nanmax(main_loss)), 5)).round(5))
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax1.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        else:
            ax1.set_yticks((np.linspace(np.nanmin(main_loss), np.nanmax(main_loss), 5)).round(5))
            ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax1.set_ylabel("Validation Loss", fontsize=12)

    if len(inter_loss) > 0:
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(0, len(inter_loss), 1), inter_loss)
        ax2.axvline(np.argmin(inter_loss), linestyle="dotted", color="red")
        ax2.axvline(len(active_interaction_index), linestyle="dotted", color="red")
        ax2.plot(np.argmin(inter_loss), np.min(inter_loss), "*", markersize=12, color="red")
        ax2.plot(len(active_interaction_index), inter_loss[len(active_interaction_index)], "o", markersize=8, color="red")
        ax2.set_xlabel("Number of Interactions", fontsize=12)
        ax2.set_xlim(-0.5, len(inter_loss) - 0.5)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        if log_scale:
            ax2.set_yscale("log")
            ax2.set_yticks((10 ** np.linspace(np.log10(np.nanmin(inter_loss)), np.log10(np.nanmax(inter_loss)), 5)).round(5))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
        else:
            ax2.set_yticks((np.linspace(np.nanmin(inter_loss), np.nanmax(inter_loss), 5)).round(5))
            ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_ylabel("Validation Loss", fontsize=12)
    plt.show()

    save_path = folder + name
    if save_eps:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if save_png:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def plot_trajectory(data_dict_logs, log_scale=True, folder="./", name="loss_trajectory", save_eps=False, save_png=False):
    """
    Helper function for visualizing loss trajectory.

    Parameters
    ----------
    data_dict_logs : dict
        Dictionary containing loss trajectory information.
    log_scale : boolean
        Whether to use log scale for y-axis.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "trajectory_plot".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """
    t1, t2, t3 = [data_dict_logs["err_train_main_effect_training"],
              data_dict_logs["err_train_interaction_training"], data_dict_logs["err_train_tuning"]]
    v1, v2, v3 = [data_dict_logs["err_val_main_effect_training"],
              data_dict_logs["err_val_interaction_training"], data_dict_logs["err_val_tuning"]]

    if len(t1) + len(t2) + len(t3) == 0:
        return

    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(1, len(t1) + 1, 1), t1, color="r")
    ax1.plot(np.arange(len(t1) + 1, len(t1 + t2) + 1, 1), t2, color="b")
    ax1.plot(np.arange(len(t1 + t2) + 1, len(t1 + t2 + t3) + 1, 1), t3, color="y")
    if log_scale:
        ax1.set_yscale("log")
        ax1.set_yticks((10 ** np.linspace(np.log10(np.nanmin(t1 + t2 + t3)), np.log10(np.nanmax(t1 + t2 + t3)), 5)).round(5))
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xlabel("Number of Epochs", fontsize=12)
        ax1.set_ylabel("Training Loss (Log Scale)", fontsize=12)
    else:
        ax1.set_yticks((np.linspace(np.nanmin(t1 + t2), np.nanmax(t1 + t2), 5)).round(5))
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xlabel("Number of Epochs", fontsize=12)
        ax1.set_ylabel("Training Loss", fontsize=12)

    ax1.legend(["Stage 1: Training Main Effects", "Stage 2: Training Interactions", "Stage 3: Fine Tuning"])

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(np.arange(1, len(v1) + 1, 1), v1, color="r")
    ax2.plot(np.arange(len(v1) + 1, len(v1 + v2) + 1, 1), v2, color="b")
    ax2.plot(np.arange(len(v1 + v2) + 1, len(v1 + v2 + v3) + 1, 1), v3, color="y")
    if log_scale:
        ax2.set_yscale("log")
        ax2.set_yticks((10 ** np.linspace(np.log10(np.nanmin(v1 + v2 + v3)), np.log10(np.nanmax(v1 + v2 + v3)), 5)).round(5))
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xlabel("Number of Epochs", fontsize=12)
        ax2.set_ylabel("Validation Loss (Log Scale)", fontsize=12)
    else:
        ax2.set_yticks((np.linspace(np.nanmin(v1 + v2 + v3), np.nanmax(v1 + v2 + v3), 5)).round(5))
        ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xlabel("Number of Epochs", fontsize=12)
        ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.legend(["Stage 1: Training Main Effects", "Stage 2: Training Interactions", "Stage 3: Fine Tuning"])
    plt.show()

    save_path = folder + name
    if save_eps:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
    if save_png:
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def feature_importance_visualize(feature_importance, feature_names, folder="./", name="feature_importance", save_png=False, save_eps=False):
    """
    Helper function for visualizing feature importance.

    Parameters
    ----------
    feature_importance : np.ndarray of shape (n_features, )
        Feature importance based on Shapley value.
    feature_names : list of str of shape (n_features, )
        Feature name list.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "feature_importance".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """
    all_ir = []
    all_names = []
    for name, importance in zip(feature_names, feature_importance):
        if importance > 0:
            all_ir.append(importance)
            all_names.append(name)

    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(0.4 + 0.65 * max_ids, 4))
        ax = plt.axes()
        ax.bar(np.arange(len(all_ir)), [ir for ir, _ in sorted(zip(all_ir, all_names))][::-1])
        ax.set_xticks(np.arange(len(all_ir)))
        ax.set_xticklabels([name for _, name in sorted(zip(all_ir, all_names))][::-1], rotation=60)
        plt.ylim(0, np.max(all_ir) + 0.05)
        plt.xlim(-1, len(all_names))
        plt.title("Feature Importance")

        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def effect_importance_visualize(data_dict_global, folder="./", name="effect_importance", save_png=False, save_eps=False):
    """
    Helper function for visualizing effect importance.

    Parameters
    ----------
    data_dict_global : dict
        Dictionary with global explanation information.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "effect_importance".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """
    all_ir = []
    all_names = []
    for key, item in data_dict_global.items():
        if item["importance"] > 0:
            all_ir.append(item["importance"])
            all_names.append(key)

    max_ids = len(all_names)
    if max_ids > 0:
        fig = plt.figure(figsize=(0.4 + 0.65 * max_ids, 4))
        ax = plt.axes()
        ax.bar(np.arange(len(all_ir)), [ir for ir, _ in sorted(zip(all_ir, all_names))][::-1])
        ax.set_xticks(np.arange(len(all_ir)))
        ax.set_xticklabels([name for _, name in sorted(zip(all_ir, all_names))][::-1], rotation=60)
        plt.ylim(0, np.max(all_ir) + 0.05)
        plt.xlim(-1, len(all_names))
        plt.title("Effect Importance")

        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def global_visualize_density(data_dict_global, main_effect_num=None, interaction_num=None, cols_per_row=4,
                        save_png=False, save_eps=False, folder="./", name="global_explain"):
    """
    Helper function for visualizing global explanation with density plots.

    Parameters
    ----------
    data_dict_global : dict
        Dictionary with global explanation information.
    main_effect_num : int or None
        The number of top main effects to show, by default None,
        As main_effect_num=None, all main effects would be shown.
    interaction_num : int or None
        The number of top interactions to show, by default None,
        As interaction_num=None, all main effects would be shown.
    cols_per_row : int
        The number of subfigures each row, by default 4.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "global_explain".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """
    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum() > 0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    if max_ids == 0:
        return

    idx = 0
    fig = plt.figure(figsize=(6 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.25, hspace=0.35)
    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            xint = ((np.array(data_dict_global[feature_name]["density"]["names"][1:])
                            + np.array(data_dict_global[feature_name]["density"]["names"][:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax2.bar(xint, data_dict_global[feature_name]["density"]["scores"], width=xint[1] - xint[0])
            ax2.get_shared_x_axes().join(ax1, ax2)
            ax2.set_yticklabels([])
            ax2.autoscale()
            fig.add_subplot(ax2)

        elif data_dict_global[feature_name]["type"] == "categorical":

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx],
                                        wspace=0.1, hspace=0.1, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                        data_dict_global[feature_name]["outputs"])
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            ax2.bar(np.arange(len(data_dict_global[feature_name]["density"]["names"])),
                    data_dict_global[feature_name]["density"]["scores"])
            ax2.get_shared_x_axes().join(ax1, ax2)
            ax2.autoscale()
            ax2.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax2.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            ax2.set_yticklabels([])
            fig.add_subplot(ax2)

        idx = idx + 1
        if len(str(ax2.get_xticks())) > 60:
            ax2.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        ax_main = plt.Subplot(fig, outer[idx])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                             aspect="auto", extent=axis_extent)

        if data_dict_global[feature_name]["xtype"] == "categorical":
            ax_main.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_main.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            ax_main.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_main.set_yticklabels(data_dict_global[feature_name]["input2_labels"])

        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"])
                                   - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        ax_main.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)
        fig.add_subplot(ax_main)
        fig.colorbar(interact_plot, ax=ax_main, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)
        idx = idx + 1
        if len(str(ax_main.get_xticks())) > 60:
            ax_main.xaxis.set_tick_params(rotation=20)

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def global_visualize_wo_density(data_dict_global, main_effect_num=None, interaction_num=None, cols_per_row=4,
                        save_png=False, save_eps=False, folder="./", name="global_explain"):
    """
    Helper function for visualizing global explanation without density plots.

    Parameters
    ----------
    data_dict_global : dict
        Dictionary with global explanation information.
    main_effect_num : int or None
        The number of top main effects to show, by default None,
        As main_effect_num=None, all main effects would be shown.
    interaction_num : int or None
        The number of top interactions to show, by default None,
        As interaction_num=None, all main effects would be shown.
    cols_per_row : int
        The number of subfigures each row, by default 4.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "global_explain".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """
    maineffect_count = 0
    componment_scales = []
    for key, item in data_dict_global.items():
        componment_scales.append(item["importance"])
        if item["type"] != "pairwise":
            maineffect_count += 1

    componment_scales = np.array(componment_scales)
    sorted_index = np.argsort(componment_scales)
    active_index = sorted_index[componment_scales[sorted_index].cumsum() > 0][::-1]
    active_univariate_index = active_index[active_index < maineffect_count][:main_effect_num]
    active_interaction_index = active_index[active_index >= maineffect_count][:interaction_num]
    max_ids = len(active_univariate_index) + len(active_interaction_index)

    idx = 0
    fig = plt.figure(figsize=(5.2 * cols_per_row, 4 * int(np.ceil(max_ids / cols_per_row))))
    outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.25, hspace=0.35)
    for indice in active_univariate_index:

        feature_name = list(data_dict_global.keys())[indice]
        if data_dict_global[feature_name]["type"] == "continuous":

            ax1 = plt.Subplot(fig, outer[idx])
            ax1.plot(data_dict_global[feature_name]["inputs"], data_dict_global[feature_name]["outputs"])
            ax1.set_title(feature_name, fontsize=12)
            fig.add_subplot(ax1)
            if len(str(ax1.get_xticks())) > 80:
                ax1.xaxis.set_tick_params(rotation=20)

        elif data_dict_global[feature_name]["type"] == "categorical":

            ax1 = plt.Subplot(fig, outer[idx])
            ax1.bar(np.arange(len(data_dict_global[feature_name]["inputs"])),
                        data_dict_global[feature_name]["outputs"])
            ax1.set_title(feature_name, fontsize=12)
            ax1.set_xticks(data_dict_global[feature_name]["input_ticks"])
            ax1.set_xticklabels(data_dict_global[feature_name]["input_labels"])
            fig.add_subplot(ax1)

        idx = idx + 1
        if len(str(ax1.get_xticks())) > 60:
            ax1.xaxis.set_tick_params(rotation=20)
        ax1.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)

    for indice in active_interaction_index:

        feature_name = list(data_dict_global.keys())[indice]
        axis_extent = data_dict_global[feature_name]["axis_extent"]

        ax_main = plt.Subplot(fig, outer[idx])
        interact_plot = ax_main.imshow(data_dict_global[feature_name]["outputs"], interpolation="nearest",
                             aspect="auto", extent=axis_extent)

        if data_dict_global[feature_name]["xtype"] == "categorical":
            ax_main.set_xticks(data_dict_global[feature_name]["input1_ticks"])
            ax_main.set_xticklabels(data_dict_global[feature_name]["input1_labels"])
        if data_dict_global[feature_name]["ytype"] == "categorical":
            ax_main.set_yticks(data_dict_global[feature_name]["input2_ticks"])
            ax_main.set_yticklabels(data_dict_global[feature_name]["input2_labels"])

        response_precision = max(int(- np.log10(np.max(data_dict_global[feature_name]["outputs"])
                                   - np.min(data_dict_global[feature_name]["outputs"]))) + 2, 0)
        ax_main.set_title(feature_name + " (" + str(np.round(100 * data_dict_global[feature_name]["importance"], 1)) + "%)", fontsize=12)
        fig.add_subplot(ax_main)
        fig.colorbar(interact_plot, ax=ax_main, orientation="vertical",
                     format="%0." + str(response_precision) + "f", use_gridspec=True)

        idx = idx + 1
        if len(str(ax_main.get_xticks())) > 60:
            ax_main.xaxis.set_tick_params(rotation=20)

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


def local_visualize(data_dict_local, folder="./", name="local_explain", save_png=False, save_eps=False):
    """
    Helper function for visualizing local explanation.

    Parameters
    ----------
    data_dict_local : dict
        Dictionary with local explanation information.
    folder : str
        The path of folder to save figure, by default "./".
    name : str
        Name of the file, by default "local_explain".
    save_png : boolean
        Whether to save the plot in PNG format, by default False.
    save_eps : boolean
        Whether to save the plot in EPS format, by default False.
    """
    idx = np.argsort(np.abs(data_dict_local["scores"][data_dict_local["active_indice"]]))[::-1]

    max_ids = len(data_dict_local["active_indice"])
    fig = plt.figure(figsize=(round((len(data_dict_local["active_indice"]) + 1) * 0.6), 4))
    plt.bar(np.arange(len(data_dict_local["active_indice"])), data_dict_local["scores"][data_dict_local["active_indice"]][idx])
    plt.xticks(np.arange(len(data_dict_local["active_indice"])),
            data_dict_local["effect_names"][data_dict_local["active_indice"]][idx], rotation=60)

    if "actual" in data_dict_local.keys():
        title = "Predicted: %0.4f | Actual: %0.4f" % (data_dict_local["predicted"], data_dict_local["actual"])
    else:
        title = "Predicted: %0.4f" % (data_dict_local["predicted"])
    plt.title(title, fontsize=12)

    if max_ids > 0:
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
