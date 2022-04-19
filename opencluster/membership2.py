import os
import sys
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from attr import attrs, attrib, validators
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.base import ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    normalized_mutual_info_score,
    pairwise_distances,
)
import pandas as pd
from sklearn.preprocessing import RobustScaler
from astropy.table.table import Table
from astropy.stats.sigma_clipping import sigma_clipped_stats

from itertools import permutations

sys.path.append(os.path.join(os.path.dirname("opencluster"), "."))

from opencluster.hkde import HKDE, PluginSelector, pair_density_plot
from opencluster.synthetic import one_cluster_sample_small, three_clusters_sample
from opencluster.utils import combinations, Colnames2
from opencluster.fetcher import load_file
from opencluster.masker import RangeMasker, DistanceMasker, CrustMasker
from opencluster.plot_gauss_err import plot_kernels

def one_hot_encode(labels: np.ndarray):
    # labels must be np array.
    # Dinstinct labels must be able to be aranged into a list of consecutive int numbers
    # e.g. [-1, 0, 1, 2] is ok, [-1, 1, 3] is not ok
    labels = labels + labels.min() * -1
    one_hot = np.zeros((labels.shape[0], labels.max()+1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


@attrs(auto_attribs=True)
class DensityBasedMembershipEstimator(ClassifierMixin):

    min_cluster_size: int
    n_iters: int = 3
    iteration_atol: float = 0.01
    clustering_metric: str = "euclidean"
    clustering_scaler: TransformerMixin = RobustScaler()
    clusterer: ClusterMixin = None
    pdf_estimator: HKDE = HKDE()
    allow_single_cluster: bool = False
    auto_toggle_allow_single_cluster: bool = True
    min_samples: int = None

    cluster_centers: list = None

    kde_leave1out:bool = True
    
    atol: float = 1e-2
    rtol: float = 1
    labeltol: int = 0
    
    kernel_calculation_mode: str = attrib(validator=validators.in_(['same', 'per_class', 'per_class_per_iter']), default='per_class')
    # internal attrs
    n: int = None
    d: int = None
    n_classes: int = None
    n_obs: float = None

    unique_labels: np.ndarray = None
    labels: np.ndarray = None
    priors: np.ndarray = None
    counts: np.ndarray = None
    posteriors: np.ndarray = None
    data: np.ndarray = None
    neff_iters: int = None
    
    estimators: list = attrib(factory=list)

    iter_priors: list = attrib(factory=list)
    iter_counts: list = attrib(factory=list)
    
    iter_log_likelihood_diff: list = attrib(factory=list)
    iter_log_likelihood_perc: list = attrib(factory=list)
    log_likelihoods: list = attrib(factory=list)
    iter_label_diff: list = attrib(factory=list)
    iter_labels: list = attrib(factory=list)
    """     
    noise_mask: np.ndarray = None
    permanent_noise: int = 0 """

    mixing_error: float = 0

    iter_dists: list = attrib(factory=list)
    dist: np.ndarray = None
    
    diff_parametric: list= attrib(factory=list)

    def get_labels(self, posteriors):
        labels = np.argmax(posteriors, axis=1)-1
        return labels

    def get_log_likelihood(self, densities):
        # calculate likelihood function from posteriors
        # should be L(X) = prod(i=1...N)(p(xi))
        # p(x)=sum(j=1...K)(wj*p(x|j))
        # in practice, the multiplication of small probabilities will underflow the float precision
        # so we calculate log(L(X)) using log multiplication property
        # log(a*b) = log(a)+log(b)
        # TODO: double and triple check, with normalized data also
        # return np.log((densities * (self.counts/self.n_obs)).sum(axis=1)).sum()
        total_density = densities.sum(axis=1, keepdims=True)
        cond_prob = densities / total_density
        return np.sum(np.log(np.sum(cond_prob * self.priors, axis=1)))

    def get_log_likelihood_diff(self):
        if len(self.log_likelihoods) > 1:
            ll_t_minus_1 = self.log_likelihoods[-2]
            ll_t = self.log_likelihoods[-1]
            # TODO: check
            log_diff = ll_t_minus_1-ll_t
            increase_perc = (ll_t_minus_1-ll_t) * 100.0 / ll_t_minus_1
            return log_diff, increase_perc
        return np.inf, np.inf

    def update_dist(self, posteriors):
        if self.dist is None:
            self.dist = pairwise_distances(RobustScaler().fit_transform(self.data))
        dists = list()
        l = self.labels.copy() + 1
        for i in range(1, posteriors.shape[1]):
            center = ((self.data.T * posteriors[:,i].ravel()).T).sum(axis=0) / posteriors[:,i].sum()
            mean_mem_dist_to_center = np.sum(pairwise_distances(self.data, center.reshape(1, -1)) * np.atleast_2d(posteriors[:,i]).T) / posteriors[:,i].sum()
            dists.append(mean_mem_dist_to_center)
        self.iter_dists.append(np.array(dists))

    def update_log_likelihood(self, densities):
        self.log_likelihoods.append(self.get_log_likelihood(densities))
        log_diff, increase_perc = self.get_log_likelihood_diff()
        self.iter_log_likelihood_perc.append(increase_perc)
        self.iter_log_likelihood_diff.append(log_diff)

    def update_diff_parametric(self, parametric):
        self.diff_parametric.append(np.abs(self.posteriors[:,1] - parametric[:,1]).mean())

    def is_stopping_criteria_achieved(self):
        # (ln(LLt) - ln(LLt-1)) < atol
        # or increase_percentage < ptol
        if not self.iter_log_likelihood_diff or not self.iter_log_likelihood_perc:
            return False
        log_diff = self.iter_log_likelihood_diff[-1]
        increase_perc = self.iter_log_likelihood_perc[-1]
        if self.atol is not None and log_diff < self.atol:
            return True
        if self.rtol is not None and increase_perc < self.rtol:
            return True
        if self.labeltol is not None and self.iter_label_diff and self.iter_label_diff[-1] == self.labeltol:
            return True
        return False

    def update_class_mixtures(self, posteriors: np.ndarray):
        # test
        if self.mixing_error >= 0 and len(self.iter_priors) > 1:
            if np.abs((self.iter_priors[0] - self.iter_priors[-1]))[0] < self.mixing_error:
                self.labels = np.argmax(posteriors, axis=1)-1
                self.iter_labels.append(self.labels)
                if len(self.iter_labels) > 1:
                    label_diff = (self.iter_labels[-1] != self.iter_labels[-2]).astype(int).sum()
                    self.iter_label_diff.append(label_diff)
                self.counts = posteriors.sum(axis=0)
                self.priors = self.counts / self.n
                self.iter_counts.append(self.counts)
                self.iter_priors.append(self.priors)
            else:
                self.iter_counts.append(self.counts)
                self.iter_priors.append(self.priors)
                self.labels = np.argmax(posteriors, axis=1)-1
                self.iter_labels.append(self.labels)
                if len(self.iter_labels) > 1:
                    label_diff = (self.iter_labels[-1] != self.iter_labels[-2]).astype(int).sum()
                    self.iter_label_diff.append(label_diff)
        else:
        # end test
            self.labels = np.argmax(posteriors, axis=1)-1
            self.iter_labels.append(self.labels)
            if len(self.iter_labels) > 1:
                label_diff = (self.iter_labels[-1] != self.iter_labels[-2]).astype(int).sum()
                self.iter_label_diff.append(label_diff)
            self.counts = posteriors.sum(axis=0)
            self.priors = self.counts / self.n
            self.iter_counts.append(self.counts)
            self.iter_priors.append(self.priors)

    def cluster(self, data):
        
        # distance_matrix = pairwise_distances(data, metric=self.clustering_metric)

        if self.clusterer is None:
            allow_single_cluster = self.allow_single_cluster if self.cluster_centers is None else False
            min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
            self.clusterer = HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=self.min_cluster_size,
                allow_single_cluster=allow_single_cluster,
                metric=self.clustering_metric,
                prediction_data=True,
            )
        self.clusterer.fit(data)
        
        if np.unique(self.clusterer.labels_).shape[0] == 1 and self.auto_toggle_allow_single_cluster and not allow_single_cluster:
            min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
            self.clusterer = HDBSCAN(
                min_samples=min_samples,
                min_cluster_size=self.min_cluster_size,
                allow_single_cluster=True,
                metric=self.clustering_metric,
                prediction_data=True,
            )
            self.clusterer.fit(data)

        return self.clusterer


    def get_posteriors_from_clustering(self):
        one_hot_code = one_hot_encode(self.clusterer.labels_)
        if self.clusterer.allow_single_cluster:
            noise_proba = np.vstack((
                one_hot_code[:,0], # probably not needed
                self.clusterer.outlier_scores_,
                1 - self.clusterer.probabilities_,
                )).max(0)
            cluster_proba = 1 - noise_proba
        else:
            membership = all_points_membership_vectors(self.clusterer)
            noise_proba = np.vstack((
                1 - membership.sum(axis=1), # probably not needed
                one_hot_code[:,0], # probably not needed
                self.clusterer.outlier_scores_,
                1 - self.clusterer.probabilities_,
                )).max(0)

            cluster_proba = np.zeros((self.n, self.n_classes - 1))

            # way 1: get cl_proba from crude membership vector
            # problem: some points have haigh membership for different clusters (e.g. (c1: .3, cl2: .3, cl3: .3) )
            # those points, instead of being between clusters, are actually inside one of them
            # and make no sense. That significantly affects how the populations are estimated in the next phases
            """ full_cl_proba = (membership * membership.sum(axis=1, keepdims=True) / np.tile((1 - noise_proba), (self.n_classes-1, 1)).T)
            cluster_proba[1-noise_proba > 0] = full_cl_proba[1-noise_proba > 0] """

            # way 2: hard classify among clusters, so probs look like (noise: .8, cluster1: .2, cluster2: 0)
            # problem: when a point is really .5/.5 between 2 clusters, like in the middle point, it would be considererd
            # 1/0
            cluster_proba = one_hot_code[:,1:]*np.tile((1-noise_proba), (self.n_classes-1, 1)).T
            
        if len(cluster_proba.shape) == 1:
            cluster_proba = np.atleast_2d(cluster_proba).T
            
        posteriors = np.zeros((self.n, self.n_classes))
        posteriors[:,0] = noise_proba
        posteriors[:,1:] = cluster_proba
        assert np.allclose(posteriors.sum(axis=1), 1)
        return posteriors

    def center_based_cluster_selection(self, data, labels, input_centers):
        
        # compares input cluster centers with obtained cluster centers
        # if input cluster centers are less than obtained, then select
        # onbtained clusters that match input centers the best
        
        cluster_labels = self.unique_labels[self.unique_labels != -1]
        cluster_centers = np.array([
            [
                sigma_clipped_stats(
                    data[labels == label][:,i],
                    cenfunc="median",
                    stdfunc="mad_std",
                    maxiters=None,
                    sigma=1,
                    )[1] for i in range(self.d)
            ] for label in cluster_labels
        ])

        # there are obtained clusters to label as noise
        # we should select those that match input centers the best

        short = input_centers
        long = cluster_centers
        
        center_distances = pairwise_distances(X=short, Y=long)
        idx_columns = np.array(list(permutations(np.arange(long.shape[0]), short.shape[0])))
        idx_rows = np.arange(short.shape[0])


        if short.shape[0] == 1:
            distance_sum_per_solution = center_distances.ravel()
        else:
            dist_idcs = tuple([tuple(map(tuple, x)) for x in np.stack((np.tile(idx_rows, (idx_columns.shape[0], 1)), idx_columns), axis=1)])
            distance_sum_per_solution = np.array([center_distances[dist_idcs[i]] for i in range(len(dist_idcs))]).sum(axis=1)
            
        best_solution = idx_columns[distance_sum_per_solution == distance_sum_per_solution.min()].ravel()

        # lets delete some clusters
        # shot is self.class_centers
        # long is class_centers
        # labels are in class_centers order
        # i need to keep labels that are in best_solution
        # the rest should be noise
        
        new_labels = np.copy(labels)
        new_labels[~np.isin(labels, best_solution)] = -1
        
        posteriors = self.get_posteriors_from_clustering()

        noise_proba = posteriors[:,tuple([0] + list((cluster_labels + 1)[~np.isin(cluster_labels, best_solution)]))].sum(axis=1)
        cluster_proba = posteriors[:, tuple((cluster_labels + 1)[np.isin(cluster_labels, best_solution)])]

        new_n_classes = short.shape[0] + 1

        # create new posteriors array
        new_posteriors = np.zeros((self.n, new_n_classes))
        new_posteriors[:,0] = noise_proba
        new_posteriors[:,1:] = cluster_proba

        assert np.allclose(new_posteriors.sum(axis=1), 1)

        # reorder kept labels
        for i, label in enumerate(best_solution):
            new_labels[new_labels == label] = i

        return new_labels, new_posteriors

    def get_posteriors(self, densities):
        # probability calculation
        # P(Ci|x) = Di(x) * P(Ci) / sumj(Dj(x) * P(Cj))
        total_density = (densities * self.counts).sum(axis=1, keepdims=True).repeat(self.n_classes, axis=1)
        posteriors = densities*self.counts / total_density
        return posteriors

    def get_densities(self, data: np.ndarray, err, corr, weights: np.ndarray):
        densities = np.zeros((self.n, self.n_classes))

        # estimator(s) fitting
        if not self.estimators or self.kernel_calculation_mode == 'per_class_per_iter':
            if self.kernel_calculation_mode == 'same':
                self.estimators = [ self.pdf_estimator.fit(data, err, corr) ]
            else:
                self.estimators = []
                for i in range(self.n_classes):
                    self.estimators.append(
                        deepcopy(self.pdf_estimator).fit(
                            data=data, err=err, corr=corr, weights=weights[:,i],
                        ),
                    )
        
        # pdf estimation
        for i in range(self.n_classes):
            if self.kernel_calculation_mode == 'same':
                class_estimator = self.estimators[0]
            else:
                class_estimator = self.estimators[i]
            densities[:, i] = class_estimator.set_weights(weights[:,i]).pdf(data, leave1out=self.kde_leave1out)
        
        return densities

    def fit_predict(
        self,
        data: np.ndarray,
        err: np.ndarray = None,
        corr: Union[np.ndarray, str] = None,
    ):

        self.n, self.d = data.shape
        self.data = data

        if self.clustering_scaler is not None:
            cl_data = self.clustering_scaler.fit(data).transform(data)
        else:
            cl_data = data

        self.cluster(cl_data)

        self.labels = self.clusterer.labels_
        self.unique_labels = np.sort(np.unique(self.labels))
        self.n_classes = self.unique_labels.shape[0]

        # case no clusters found
        if self.n_classes == 1:
            self.posteriors = one_hot_encode(self.labels)
            self.update_class_mixtures(self.posteriors)
            # there are no populations to fit
            return self

        # case cluster selection required
        if not self.clusterer.allow_single_cluster and self.cluster_centers is not None and self.cluster_centers.shape[0] < self.n_classes - 1:
            if self.clustering_scaler is not None:
                input_centers = self.clustering_scaler.transform(self.cluster_centers)
            else:
                input_centers = self.cluster_centers
            self.labels, self.posteriors = self.center_based_cluster_selection(cl_data, self.labels, input_centers)
            self.unique_labels = np.sort(np.unique(self.labels))
            self.n_classes = self.unique_labels.shape[0]
        else:
            self.posteriors = self.get_posteriors_from_clustering()

        self.update_class_mixtures(self.posteriors)

        # p_parametric, mix = parametric(data, self.labels, self.priors)
        f = lambda x: mix.predict_proba(x)[:,0]
        # testing
        self.update_dist(self.posteriors)
        # end testing
        
        #self.update_diff_parametric(p_parametric)

        for i in range(self.n_iters):
            # is copy actually needed?
            previous_posteriors = self.posteriors.copy()
            weights = previous_posteriors
            # bad idea
            """ if self.noise_mask is not None:
                weights[self.noise_mask] = np.array([1.] + [0.]*(self.n_classes - 1)) """
            densities = self.get_densities(data, err, corr, weights)
            self.posteriors = self.get_posteriors(densities)
            self.update_class_mixtures(self.posteriors)
            self.update_log_likelihood(densities)
            # testing
            self.update_dist(self.posteriors)
            grid = self.membership_plot()
            lll = self.posteriors[:,1] > .5
            mem_plot_kernels(index=1, dbme=self, ax=grid.axes[1,0], data=data, n=10000, labels=lll)
            print(f'iter {i}')
            plt.show()

            #self.update_diff_parametric(p_parametric)
            """ nn = self.n_obs
            self.n_obs = 50**3
            g = lambda x: self.get_posteriors(x, None, None, weights=weights)[0][:,1]
            total_density = lambda x: np.atleast_2d((self.get_posteriors(x, None, None, weights=weights)[1] * self.priors).sum(axis=1, keepdims=True)).T
            total_density_only_sum = lambda x: np.atleast_2d((self.get_posteriors(x, None, None, weights=weights)[1]).sum(axis=1, keepdims=True)).T
            p_only_sum = lambda x: np.atleast_2d(np.atleast_2d(self.get_posteriors(x, None, None, weights=weights)[1][:,1]).T / total_density_only_sum(x)).T
            # p_check = lambda x: np.atleast_2d(np.atleast_2d(self.get_posteriors(x, None, None, weights=weights)[1][:,1]).T * self.priors[1] / total_density(x)).T
            gfig, gaxes = pair_density_plot(data, g, grid_resolution=50)
            gfig.suptitle('pc')
            tdfig, tdaxes = pair_density_plot(data, total_density, grid_resolution=50)
            tdfig.suptitle('cum_d_per_priors')
            tdosfig, tdosaxes = pair_density_plot(data, total_density_only_sum, grid_resolution=50)
            tdosfig.suptitle('cum_d_only')
            fig, ax = self.estimators[1].density_plot()
            fig.suptitle('dens_cum')
            #fig2, ax2 = pair_density_plot(data, p_check, grid_resolution=50)
            #fig2.suptitle('p_check')
            fig3, ax3 = pair_density_plot(data, p_only_sum, grid_resolution=50)
            fig3.suptitle('p_only_sum')
            plt.show()
            self.n_obs = nn
             """
            # end testing
            self.is_stopping_criteria_achieved()
                # break
            # testing
            # self.membership_3d_plot(marker='o', palette='viridis_r', marker_size=100**(self.posteriors[:,1]))
            # end test
            self.neff_iters = i + 1
        return self

    def iter_plot(self, **kwargs):

        sns.set_style('darkgrid')

        df = pd.DataFrame(np.array(self.iter_counts))
        df.columns = [f'N{str(i)}' for i in range(self.iter_counts[0].shape[0])]
        df['t'] = np.arange(self.neff_iters + 1)

        fig, axes = plt.subplots(7,1, sharex=True)

        axes[0].set_xticks(range(self.neff_iters+1))

        for i in range(len(axes)):
            axes[i].axvline(1, color='black')
            axes[i].axvline(self.neff_iters, color='black')

        sns.lineplot(ax=axes[0], x='t', y='value', hue='variable', data=pd.melt(df, ['t']))
        
        df['log_likelihood'] = np.array([np.nan] + self.log_likelihoods)
        df['ll_%'] = np.array([np.nan] + self.iter_log_likelihood_perc)
        df['ll_diff'] = np.array([np.nan] + self.iter_log_likelihood_diff)
        df['label_diff'] = np.array([np.nan] + self.iter_label_diff)

        sns.lineplot(ax=axes[1], x=df.t, y=df['log_likelihood'])

        sns.lineplot(ax=axes[2], x=df.t, y=df['ll_%'])
        if self.rtol is not None:
            axes[2].axhline(self.rtol, color='red')
        
        sns.lineplot(ax=axes[3], x=df.t, y=df['ll_diff'])
        if self.atol is not None:
            axes[3].axhline(self.atol, color='red')
        
        sns.lineplot(ax=axes[4], x=df.t, y=df['label_diff'])
        if self.labeltol is not None:
            axes[4].axhline(self.labeltol, color='red')

        # test
        dists = np.array(self.iter_dists)
        for i in range(self.n_classes-1):
            df[f'dist{i}'] = dists[:,i]
            sns.lineplot(ax=axes[5], x=df.t, y=df[f'dist{i}'])

        """ df['diff_parametric'] = np.array(self.diff_parametric)
        sns.lineplot(ax=axes[6], x=df.t, y=df.diff_parametric) """
        # end test

        return fig, axes

    def membership_plot(self, label=0, **kwargs):
        return membership_plot(self.data, self.posteriors[:,label+1], **kwargs)

    def clustering_plot(self, label=0, **kwargs):
        return membership_plot(self.data, self.clusterer.labels_ + 1, self.clusterer.labels_ + 1, **kwargs)

    def class_plot(self, **kwargs):
        return membership_plot(self.data, self.labels+1, self.labels, **kwargs)

    def membership_3d_plot(self, label=0, **kwargs):
        return membership_3d_plot(self.data, self.posteriors[:, label+1], **kwargs)

    def clustering_3d_plot(self, label=0, **kwargs):
        return membership_3d_plot(self.data, self.clusterer.labels_ + 1, self.clusterer.labels_ + 1, **kwargs)

def membership_3d_plot(
    data: Union[np.ndarray, pd.DataFrame],
    posteriors,
    colnames:list=None,
    var_index=(1,2,3),
    palette: str='viridis',
    alpha=.5,
    marker='x',
    marker_size=40,
    ):
    fig, ax = plt.subplots(subplot_kw={ 'projection': '3d' })
    cmap = ListedColormap(sns.color_palette(palette, 256).as_hex())
    if len(var_index) != 3:
        raise ValueError()
    var_index = np.array(var_index) - 1
    ax.scatter3D(data[:,var_index[0]], data[:,var_index[1]], data[:,var_index[2]], s=marker_size, marker=marker, cmap=cmap, c=posteriors, alpha=alpha)
    return fig, ax   

    

#TODO: fix colors problem
def membership_plot(
    data:Union[np.ndarray, pd.DataFrame],
    posteriors,
    labels=None,
    colnames:list=None,
    palette:str='viridis',
    corner=True, # breaks colorbar
    marker='x',
    alpha=.5,
    density_intervals=4,
    density_mode='stack',
    size=None,
    ):

    sns.set_style('darkgrid')
    if isinstance(data, np.ndarray):
        obs, dims = data.shape
        data = pd.DataFrame(data)
        if colnames is not None:
            data.columns = colnames
        else:
            data.columns = [f"var {i+1}" for i in range(dims)]
    
    if labels is None and density_intervals is not None:
        if isinstance(density_intervals, int):
            density_intervals = np.linspace(0, 1, density_intervals+1)
        labels = pd.cut(x=posteriors, bins=density_intervals, include_lowest=True)
        ticks = np.array([interval.right for interval in labels.categories.values])
        if density_mode == 'stack':
            # reverse order in which stacked graf will appear
            hue_order = np.flip(labels.categories.astype(str))
            labels = labels.astype(str)
            if palette.endswith('_r'):
                diag_palette = palette.strip('_r')
            else:
                diag_palette = palette + '_r'
            diag_kws = {
                "hue": labels,
                "hue_order": hue_order,
                "multiple": density_mode,
                'palette': sns.color_palette(diag_palette, n_colors=len(hue_order)),
                'linewidth': 0,
                'cut': 0,
                }
    else:
        ticks = np.arange(0, 1, .1)
        diag_kws = { "hue": labels, "multiple": density_mode, 'palette': palette, 'linewidth': 0, 'cut': 0 }

    cmap = ListedColormap(sns.color_palette(palette).as_hex())
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette(palette, as_cmap=True))
    sm.set_array([])

    plot_kws={ "hue": posteriors, "hue_norm": (0, 1), "marker": marker, 'alpha': alpha, 'palette': palette }

    if size is not None:
        plot_kws['size'] = size
    
    grid = sns.pairplot(
        data,
        plot_kws=plot_kws,
        diag_kind="kde",
        diag_kws=diag_kws,
    )
    plt.colorbar(sm, ax=grid.axes, ticks=ticks)
    # its done this way to avoid map_lower error when having different hues for diag and non diag elements
    if corner:
        for i in np.vstack(np.triu_indices(data.shape[1])).T:
            grid.axes[i[0], i[1]].set_visible(False)
    return grid


def parametric(data, labels, priors):
    import pomegranate as pg
    f = pg.IndependentComponentsDistribution([
        pg.UniformDistribution.from_samples(data[:,0][labels == -1]),
        pg.UniformDistribution.from_samples(data[:,1][labels == -1]),
        pg.UniformDistribution.from_samples(data[:,2][labels == -1]),
    ])

    c = pg.IndependentComponentsDistribution([
        pg.NormalDistribution.from_samples(data[:,0][labels==0]),
        pg.NormalDistribution.from_samples(data[:,1][labels==0]),
        pg.NormalDistribution.from_samples(data[:,2][labels==0]),
    ])

    mix = pg.GeneralMixtureModel([c, f])
    mix.fit(data)
    return mix.predict_proba(data), mix

def test_membership():
    np.random.seed(0)
    df = one_cluster_sample_small(cluster_size=50, field_size=int(1e4))
    data = df[["pmra", "pmdec", "parallax"]].to_numpy()

    real_pmp = df["p_pm_cluster1"].to_numpy()
    real_pmlabels = np.zeros_like(real_pmp)
    real_pmlabels[real_pmp > 0.5] = 1

    estimator = DensityBasedMembershipEstimator(
        min_cluster_size=50,
        n_iters=5,
        pdf_estimator=HKDE(bw=PluginSelector(diag=True)),
        iter_pdf_update=False,
        mixing_error=1,
    )
    result = estimator.fit_predict(data)
    
    calculated_pmp = result.p[:, 1]
    calculated_labels = np.zeros_like(calculated_pmp)
    calculated_labels[calculated_pmp > 0.5] = 1

    acc = accuracy_score(real_pmlabels, calculated_labels)
    conf = confusion_matrix(real_pmlabels, calculated_labels)
    minfo = normalized_mutual_info_score(real_pmlabels, calculated_labels)

    print("minfo")
    print(minfo)
    print("acc")
    print(acc)
    print("conf")
    print(conf)
    print("end")

def mem_plot_kernels(dbme, ax, data, index=1, n=10, nstd=3, labels=None):
    n = min(n, data.shape[0])
    e = dbme.estimators[index]
    if labels is None:
        condition = dbme.labels == index - 1
    else:
        condition = labels
    means = data[condition][:n]
    cov = e.covariances[condition][:n]
    return plot_kernels(
        means=means,
        cov_matrices=cov,
        ax=ax, alpha=1, linewidth=.5, edgecolor='k', facecolor='none', nstd=nstd)


# it just does not fucking work:
# log_likelihood decreases instead of increasing
# mixtures do not stabilize
def test_membership_real():
    s1_5 = "tests/data/clusters_phot/ngc2527.xml"
    s2 = "ng2527_x2.xml"
    s2_5 = "ng2527_x2.5.xml"
    s3 = "ng2527_x3.xml"
    s3_5 = "ng2527_x3.5.xml"
    s2_5_phot = "ng2527_phot_x2.5.xml"
    s15_mag = "scripts/data/clusters_phot/ngc2527bright1.csv"
    s7_5 = "ngc2527_select_9_sigmas.xml"
    s2_5_cured = "ng2527_cured_x2.5.xml"
    s7_5_cured = "ng2527_cured_x7.5.xml"

    print('reading')
    df = Table.read(s7_5_cured).to_pandas()
    cnames = Colnames2(df.columns.to_list())
    fiveparameters = ["pmra", "pmdec", "parallax", "ra", "dec"]
    threeparameters = ["pmra", 'pmdec', 'parallax']
    twoparameters = ["pmra", 'pmdec']
    datanames = cnames.get_data_names(fiveparameters)
    errornames, missing_err = cnames.get_error_names(datanames)
    corrnames, missing_corr = cnames.get_corr_names(datanames)
    data = df[datanames].to_numpy()
    err = df[errornames].to_numpy()
    if missing_corr:
        corr=None
    else:
        corr = df[corrnames].to_numpy()
    n, d = data.shape
    w = np.ones(n)
    print('calculating')

    """ scaled = RobustScaler().fit_transform(data)
    mask = DistanceMasker(center='geometric', percentage=10).mask(data=scaled)
    mask2 = CrustMasker(percentage=10).mask(data=scaled)
    sns.scatterplot(data[:,0], data[:,1], hue=mask) """

    normal = 190
    cured = 167
    dbme = DensityBasedMembershipEstimator(
        min_cluster_size=cured,
        n_iters=10,
        pdf_estimator=HKDE(bw=PluginSelector(diag=True)),
        kernel_calculation_mode='per_class',
        mixing_error=1,
        )
    result = dbme.fit_predict(data)#, err=err, corr=corr)
    dbme.iter_plot()
    plt.show()
    dbme.membership_plot(0, palette='viridis', density_intervals=10, colnames=datanames)

    df['p'] = result.p[:,1]
    mems = df[df.p > .5]
    nonmems = df[df.p <= .5]
    sns.scatterplot(mems.bp_rp, mems.phot_g_mean_mag, hue=mems.p, hue_norm=(0,1)).invert_yaxis()
    sns.scatterplot(nonmems.bp_rp, nonmems.phot_g_mean_mag, hue=nonmems.p, hue_norm=(0,1)).invert_yaxis()

    plt.show()
    print('coso')

def test_cluster_selection():
    
    np.random.seed(0)
    df = three_clusters_sample(cluster_size=50, field_size=int(1e3))
    data = df[["pmra", "pmdec", "parallax"]].to_numpy()
    real_pmp = df["p_pm_cluster1"].to_numpy()
    real_pmlabels = np.zeros_like(real_pmp)
    real_pmlabels[real_pmp > .5] = 1

    estimator = DensityBasedMembershipEstimator(
        min_cluster_size=50,
        min_samples=30,
        cluster_centers=np.array([(8,8,5), (5,5,5)]),
        # allow_single_cluster=True,
        n_iters=30,
        kernel_calculation_mode='per_class',
        mixing_error=1,
    )

    estimator.fit_predict(data)
    # estimator.iter_plot()
    estimator.clustering_plot()
    print('coso')



# test_membership()
test_membership_real()
# test_cluster_selection()
# test_simul()