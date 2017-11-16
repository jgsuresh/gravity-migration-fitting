import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import ast
from cluster_and_link_individuals import *


def bin_trip_data(trips_df, cluster_df, tuple_df, pairs_df, Nbins=20,eps=0.2,season=None,age_range=None,gender=None,
                  link_pair_norm_method='None',bin_by_quantiles=False):
    our_df = trips_df.copy()

    def season_cut(df,rd_start,rd_end):
        return np.logical_or(
            np.logical_and(df['rd_start']==rd_start, df['rd_end']==rd_end),
            np.logical_and(df['rd_start']==-1, df['rd_end']==rd_end)
        )

    # Impose restrictions on trips we are looking at:
    if season == "dry":
        cut12 = season_cut(our_df,1,2)
        cut23 = season_cut(our_df,2,3)
        cut34 = season_cut(our_df,3,4)
        cut56 = season_cut(our_df,5,6)
        dry_cut = np.logical_or(
            np.logical_or(cut12,cut23),
            np.logical_or(cut34,cut56))
        our_df = our_df[dry_cut]
    elif season == "wet":
        cut78 = season_cut(our_df,7,8)
        cut910 = season_cut(our_df,9,10)
        wet_cut = np.logical_or(cut78,cut910)
        our_df = our_df[wet_cut]

    if age_range != None:
        age_cut = np.logical_and(our_df['age'] >= age_range[0], our_df['age'] < age_range[1])
        our_df = our_df[age_cut]

    if gender != None:
        our_df = our_df[our_df["gender"] == gender]


    # Set up binning:
    if bin_by_quantiles:
        linkedpair_d_cut = our_df['d'] <= 350
        dbins = np.array(our_df[linkedpair_d_cut].quantile(np.linspace(0., 1., Nbins))['d']) + eps
        Pbins = np.array(our_df[linkedpair_d_cut].quantile(np.linspace(0., 1., Nbins))['home_cluster_pops'])
    else:
        N_Pbins = Nbins
        log_Pbins = np.linspace(np.log10(np.min(cluster_df['pops'])),
                                np.log10(np.max(cluster_df['pops'])), Nbins)
        Pbins = 10. ** log_Pbins

        N_dbins = Nbins
        log_dbins = np.linspace(np.log10(np.min(tuple_df['d_centr'])), np.log10(350), N_dbins)
        dbins = 10.**log_dbins


    Pbins_cent = (Pbins[1:] + Pbins[:-1]) / 2.
    dbins_cent = (dbins[1:] + dbins[:-1]) / 2.

    Pbin_width = Pbins[1:] - Pbins[:-1]
    Pbin_width_norm = Pbin_width / (Pbins[-1] - Pbins[0])
    dbin_width = dbins[1:] - dbins[:-1]
    dbin_width_norm = dbin_width / (dbins[-1] - dbins[0])


    N_all = np.sum(tuple_df['count'])

    foo = np.zeros((Nbins - 1) ** 3)
    ii = 0

    N_allbins = (N_dbins - 1) * (N_Pbins - 1) ** 2
    linked_pair_count = np.zeros(N_allbins)
    full_pair_count = np.zeros(N_allbins)
    possible_round_link_count = np.zeros(N_allbins)
    possible_round_link_count_norm = np.zeros(N_allbins)
    trip_count = np.zeros(N_allbins)

    Ph_arr = np.zeros(N_allbins)
    Pd_arr = np.zeros(N_allbins)
    d_arr = np.zeros(N_allbins)


    for Ph_i in np.arange(N_Pbins - 1):
        print "Ph_i ", Ph_i
        linkedpair_Ph_cut = np.logical_and(tuple_df["home_cluster_pops"] >= Pbins[Ph_i],
                                           tuple_df["home_cluster_pops"] <= Pbins[Ph_i + 1])

        allpairs_Ph_cut = np.logical_and(pairs_df["c1_p"] >= Pbins[Ph_i],
                                         pairs_df["c1_p"] <= Pbins[Ph_i + 1])

        for Pd_i in np.arange(N_Pbins - 1):
            linkedpair_Pd_cut = np.logical_and(tuple_df["dest_cluster_pops"] >= Pbins[Pd_i],
                                               tuple_df["dest_cluster_pops"] <= Pbins[Pd_i + 1])

            h1 = pairs_df[allpairs_Ph_cut]["c2_p"]
            allpairs_Pd_cut = np.logical_and(h1 >= Pbins[Pd_i],
                                             h1 <= Pbins[Pd_i + 1])


            for d_i in np.arange(N_dbins - 1):
                linkedpair_d_cut = np.logical_and(tuple_df["d_centr"] + eps >= dbins[d_i],
                                                  tuple_df["d_centr"] + eps <= dbins[d_i + 1])
                # # BIN ON ACTUAL TRIP DISTANCE INSTEAD OF CENTROID-ED DISTANCE
                # linkedpair_d_cut = np.logical_and(tuple_df["d"] + eps >= dbins[d_i],
                #                                   tuple_df["d"] + eps <= dbins[d_i + 1])

                h2 = pairs_df[allpairs_Ph_cut][allpairs_Pd_cut]["d"] + eps
                allpairs_d_cut = np.logical_and(h2 >= dbins[d_i],
                                                h2 <= dbins[d_i + 1])


                linkedpair_full_cut = np.logical_and(np.logical_and(linkedpair_Ph_cut, linkedpair_Pd_cut), linkedpair_d_cut)
                # foo[ii] = np.sum(full_cut)

                linked_pair_count[ii] = np.float(np.sum(linkedpair_full_cut))
                full_pair_count[ii] = np.float(np.sum(allpairs_d_cut))
                possible_round_link_count[ii] = np.float(
                    np.sum(pairs_df['possible_link_pairs'][allpairs_Ph_cut][allpairs_Pd_cut][allpairs_d_cut]))
                if link_pair_norm_method == 'None':
                    possible_round_link_count_norm[ii] = possible_round_link_count[ii]
                elif link_pair_norm_method == 'clpop':
                    possible_round_link_count_norm[ii] = np.float(
                        np.sum(pairs_df['possible_link_pairs_normed_by_clpop'][allpairs_Ph_cut][allpairs_Pd_cut][allpairs_d_cut]))
                elif link_pair_norm_method == 'maxrd':
                    possible_round_link_count_norm[ii] = np.float(
                        np.sum(pairs_df['possible_link_pairs_normed_by_maxrd'][allpairs_Ph_cut][allpairs_Pd_cut][allpairs_d_cut]))

                trip_count[ii] = np.sum(tuple_df['count'][linkedpair_full_cut])

                Ph_arr[ii] = Pbins_cent[Ph_i]
                Pd_arr[ii] = Pbins_cent[Pd_i]
                d_arr[ii] = dbins_cent[d_i]

                ii += 1


    # Create dataframe from binning results:
    bins_df = pd.DataFrame({
        "Ph": Ph_arr,
        "Pd": Pd_arr,
        "d": d_arr,
        "linked_pair_count": linked_pair_count,
        "full_pair_count": full_pair_count,
        "possible_round_link_count": possible_round_link_count,
        "possible_round_link_count_norm": possible_round_link_count_norm,
        "trip_count": trip_count
    })

    possible_links_exist = bins_df['possible_round_link_count'] > 0
    bins_df = bins_df[possible_links_exist]
    logbins_df = np.log10(bins_df)
    logebins_df = np.log(bins_df)

    return bins_df

def fit_gravity_parameters(bins_df,pop_weighted=False, verbose=True):
    first_guess = [1e-5, 1., 1., -1.]
    bounds = [(1e-9, 1), (0, 3), (0, 3), (-4., 0.)]

    df = bins_df.copy()

    def likelihood_N(params, func_scale=1, pop_weight=0):
        k, a1, a2, g = params

        model_counts = df['possible_round_link_count_norm'] * k * (df['Ph'] ** a1 * df['Pd'] ** a2 * df['d'] ** g)
        measured_counts = df['trip_count']

        if pop_weight == 1:
            w = (df['Ph'] + df['Pd'])
        else:
            w = np.ones_like(df['Ph'])

        return func_scale * np.sum(w * (-model_counts + measured_counts * np.log(model_counts)))


    x=minimize(likelihood_N,
               first_guess,
               args=(-1,int(pop_weighted)),
               bounds=bounds,
               options = {'disp': verbose})
    x_fit = x.x
    return x_fit

def fit_gravity_model_to_links(verbose=True,save_result=True):
    #fixme Does not have capability to save to file yet.
    cluster_df = pd.read_csv("../data/intermediate/cluster_info.csv")
    trips_df = pd.read_csv("../data/intermediate/trips_with_clusters.csv")
    pairs_df = pd.read_csv("../data/intermediate/all_pairs.csv")

    # Create new dataframe which has one row per [pop_home, pop_dest, d_centr] combination:
    by_tuple = pd.DataFrame(
        {'count': trips_df.groupby(["home_cluster_pops", "dest_cluster_pops", "d_centr"]).size()}).reset_index()
    by_tuple = by_tuple[by_tuple['d_centr'] > 0.]  # limit to inter-cluster trips and exclude label=-1, which have d_centr = -1

    # Bin data, with different normalizations
    bins_norm_none = bin_trip_data(trips_df, cluster_df, by_tuple, pairs_df, link_pair_norm_method='None')
    bins_norm_clpop = bin_trip_data(trips_df, cluster_df, by_tuple, pairs_df, link_pair_norm_method='clpop')
    bins_norm_maxrd = bin_trip_data(trips_df, cluster_df, by_tuple, pairs_df, link_pair_norm_method='maxrd')

    # Fit gravity models to each:
    fit_dict = {}
    fit_dict["norm_none"] = fit_gravity_parameters(bins_norm_none,verbose=verbose)
    fit_dict["norm_clpop"] = fit_gravity_parameters(bins_norm_clpop, verbose=verbose)
    fit_dict["norm_maxrd"] = fit_gravity_parameters(bins_norm_maxrd, verbose=verbose)

    fit_dict_popweighted = {}
    fit_dict_popweighted["norm_none"] = fit_gravity_parameters(bins_norm_none,verbose=verbose,pop_weighted=True)
    fit_dict_popweighted["norm_clpop"] = fit_gravity_parameters(bins_norm_clpop, verbose=verbose,pop_weighted=True)
    fit_dict_popweighted["norm_maxrd"] = fit_gravity_parameters(bins_norm_maxrd, verbose=verbose,pop_weighted=True)

    fit_dict_5km = {}
    fit_dict_5km["norm_none"] = fit_gravity_parameters(bins_norm_none[bins_norm_none['d']<=5.],verbose=verbose)
    fit_dict_5km["norm_clpop"] = fit_gravity_parameters(bins_norm_clpop[bins_norm_clpop['d']<=5.], verbose=verbose)
    fit_dict_5km["norm_maxrd"] = fit_gravity_parameters(bins_norm_maxrd[bins_norm_maxrd['d']<=5.], verbose=verbose)

    fit_dict_350km = {}
    fit_dict_350km["norm_none"] = fit_gravity_parameters(bins_norm_none[bins_norm_none['d']>5.],verbose=verbose)
    fit_dict_350km["norm_clpop"] = fit_gravity_parameters(bins_norm_clpop[bins_norm_clpop['d']>5.], verbose=verbose)
    fit_dict_350km["norm_maxrd"] = fit_gravity_parameters(bins_norm_maxrd[bins_norm_maxrd['d']>5.], verbose=verbose)

    if save_result:
        print fit_dict
        print fit_dict_popweighted
        print "Saving this to file is not implemented yet"

    return [fit_dict, fit_dict_popweighted]


def return_fit_params(fit_dict, d, d_edges=None):
    # Return fit parameters for fit which has N distance-dependent regimes:
    params = np.zeros([4,np.size(d)])

    if d_edges == None:
        d_edges = np.array([-1])

    n_bins = d_edges.size
    for jj in xrange(4):
        for i in np.arange(n_bins):
            if i == 0:
                cut = d < d_edges[0]
            elif i == n_bins-1:
                cut = d >= d_edges[-1]
            else:
                cut = np.logical_and(d >= d_edges[i], d < d_edges[i+1])

            params[jj,:][cut] = np.repeat(fit_dict[i],np.sum(cut))

    return params

def get_cluster_tuples(trips_df,cluster_df):
    ok_cl_ids = cluster_df['ids']
    c1 = np.in1d(trips_df['home_cluster_labels'],cluster_df['ids'])
    c2 = np.in1d(trips_df['dest_cluster_labels'],cluster_df['ids'])
    trips_df = trips_df[np.logical_and(c1,c2)]

    by_tuple2 = pd.DataFrame(
        {'count': trips_df.groupby(["home_cluster_labels", "dest_cluster_labels"]).size()}).reset_index()
    # by_tuple2 = by_tuple2[by_tuple2['d_centr'] > 0.]  # inter-cluster trips and exclude label=-1, which have d_centr = -1
    by_tuple2 = by_tuple2[by_tuple2['home_cluster_labels'] != by_tuple2['dest_cluster_labels']]
    by_tuple2 = by_tuple2[by_tuple2['home_cluster_labels'] != -1]
    by_tuple2 = by_tuple2[by_tuple2['dest_cluster_labels'] != -1]

    # Get lat/long for each pair:
    by_tuple3 = by_tuple2.merge(cluster_df, how='left', left_on='home_cluster_labels', right_on='ids',copy=False)
    by_tuple3["home_cluster_pops"] = by_tuple3["pops"]
    by_tuple3["home_cluster_lat"] = by_tuple3["lat"]
    by_tuple3["home_cluster_long"] = by_tuple3["long"]
    by_tuple3["home_cluster_rounds_seen"] = by_tuple3["rounds_seen"]
    by_tuple3 = by_tuple3.drop(["pops", "ids", "lat", "long","total_rounds_seen","rounds_seen"], axis=1)

    by_tuple3 = by_tuple3.merge(cluster_df, how='left', left_on='dest_cluster_labels', right_on='ids',copy=False)
    by_tuple3["dest_cluster_pops"] = by_tuple3["pops"]
    by_tuple3["dest_cluster_lat"] = by_tuple3["lat"]
    by_tuple3["dest_cluster_long"] = by_tuple3["long"]
    by_tuple3["dest_cluster_rounds_seen"] = by_tuple3["rounds_seen"]
    by_tuple3 = by_tuple3.drop(["pops", "ids", "lat", "long", "total_rounds_seen", "rounds_seen"], axis=1)

    # Get distance for each pair:
    func = lambda x: vincenty(x[0:2], x[2:4]).km

    dat_columns = np.column_stack((
        by_tuple3['home_cluster_lat'], by_tuple3['home_cluster_long'],
        by_tuple3['dest_cluster_lat'], by_tuple3['dest_cluster_long']
    ))
    d_allpairs = map(func, dat_columns)
    by_tuple3["d_centr"] = pd.Series(d_allpairs,index=by_tuple3.index)


    # Get possible link count for each pair:
    def possible_link_pairs(rd_list1, rd_list2):
        rd_list1 = ast.literal_eval(rd_list1)
        rd_list2 = ast.literal_eval(rd_list2)
        n = 0
        for rdi in rd_list1:
            if rdi in rd_list2:
                n += len(rd_list2)-1
            else:
                n += len(rd_list2)
        return n
        # Could vectorize this with np.in1d...

    by_tuple3['possible_link_pairs'] = by_tuple3.apply(
        lambda x: possible_link_pairs(x['home_cluster_rounds_seen'], x['dest_cluster_rounds_seen']), axis=1)

    return by_tuple3

def add_prediction_to_tuple_df(tuple_df, fit_dict, d_edges=None,label="count_model"):
    xf = return_fit_params(fit_dict,tuple_df['d_centr'],d_edges=d_edges)

    cnts_model = tuple_df['possible_link_pairs'] * \
                 xf[0,:] * (
                 tuple_df['home_cluster_pops'] ** xf[1,:] * tuple_df['dest_cluster_pops'] ** xf[2,:] *
                 tuple_df['d_centr'] ** xf[3,:])
    tuple_df[label] = pd.Series(cnts_model, index=tuple_df.index)
    return tuple_df

def kariba_cluster_scatter(ax, cluster_df, lat_range=None, long_range=None):
    # Add Kariba scatter plot of clusters to given ax
    if lat_range != None:
        cluster_df = cluster_df.copy()
        long_cut = np.logical_and(cluster_df['long'] >= long_range[0],cluster_df['long'] <= long_range[1])
        lat_cut = np.logical_and(cluster_df['lat'] >= lat_range[0], cluster_df['lat'] <= lat_range[1])
        cluster_df = cluster_df[np.logical_and(long_cut,lat_cut)]

    ax.scatter(
        np.array(cluster_df['long']),
        np.array(cluster_df['lat']),
        s=(cluster_df['pops']) ** 0.6,cmap=plt.cm.Greys,c=(cluster_df['pops']) ** 0.2, alpha=0.55,edgecolors='black')
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")

def plot_map_of_count_difference(tuple_df,cluster_df,c1_label="count",c2_label="count_model",
                                 c1_vs_c2_range=None,c1_cnt_edges=None,c2_cnt_edges=None,lat_range=None,long_range=None,alpha=0.7):
    plt.close('all')
    plt.figure()
    ax = plt.subplot(111)
    kariba_cluster_scatter(ax, cluster_df, lat_range=lat_range, long_range=long_range)

    if np.size(c1_vs_c2_range)==1 and c1_vs_c2_range > 1:
        cut = tuple_df['count_model'] >= c1_vs_c2_range * tuple_df['count']
        title = "{} > {} x {}".format(c1_label,c1_vs_c2_range,c2_label)
    elif np.size(c1_vs_c2_range)==1 and c1_vs_c2_range < 1:
        cut = tuple_df[c1_label] <= c1_vs_c2_range * tuple_df[c2_label]
        title = "{} < {} x {}".format(c1_label,c1_vs_c2_range,c2_label)
    else:
        cut = np.logical_and(tuple_df['count_model'] >= c1_vs_c2_range[0] * tuple_df['count'],
                             tuple_df['count_model'] <= c1_vs_c2_range[1] * tuple_df['count'])
        title = "{} = ({} to {}) x {}".format(c1_label,c1_vs_c2_range[0],c1_vs_c2_range[1],c2_label)


    if c1_cnt_edges != None:
        loopby = c1_label
        loopedges = c1_cnt_edges
    elif c2_cnt_edges != None:
        loopby = c2_label
        loopedges = c2_cnt_edges
    else:
        loopby = "count"
        loopedges = np.array([1,10,100])

    n_loop_bins = np.size(loopedges) + 1
    loop_list = []
    loop_labels = []
    for i in np.arange(n_loop_bins):
        if i == 0:
            ccut = tuple_df[loopby][cut] <= loopedges[0]
            ll = "{} <= {}".format(loopby, loopedges[0])
        elif i == n_loop_bins-1:
            ccut = tuple_df[loopby][cut] > loopedges[-1]
            ll = "{} > {}".format(loopby, loopedges[-1])
        else:
            ccut = np.logical_and(tuple_df[loopby][cut] >= loopedges[i], tuple_df[loopby][cut] < loopedges[i])
            ll = "{} between {} to {}".format(loopby, loopedges[i], loopedges[i+1])
        # loop_list.append(np.copy(ccut))
        # loop_labels.append(ll)

        ax.quiver(tuple_df['home_cluster_long'][cut][ccut],
                   tuple_df['home_cluster_lat'][cut][ccut],
                   (tuple_df['dest_cluster_long']-tuple_df['home_cluster_long'])[cut][ccut],
                   (tuple_df['dest_cluster_lat'] - tuple_df['home_cluster_lat'])[cut][ccut],
                   angles='xy',scale_units='xy',scale=1,
                   width=0.003,headwidth=4*(ii+1)/2.,alpha=alpha,label=ll,
                   color='C{}'.format(i))

    ax.legend()
    ax.set_title(title)
    plt.show()

def get_cluster_pop_bins(tuple_df):
    small_small = np.logical_and(tuple_df['home_cluster_pops'] <= 100.,tuple_df['dest_cluster_pops'] <= 100.)
    
    small_med = np.logical_or(
        np.logical_and(tuple_df['home_cluster_pops'] <= 100., np.logical_and(tuple_df['dest_cluster_pops'] > 100., tuple_df['dest_cluster_pops'] <= 1000.)),
        np.logical_and(tuple_df['dest_cluster_pops'] <= 100., np.logical_and(tuple_df['home_cluster_pops'] > 100., tuple_df['home_cluster_pops'] <= 1000.)),
    )
    small_big = np.logical_or(
        np.logical_and(tuple_df['home_cluster_pops'] <= 100., tuple_df['dest_cluster_pops'] > 1000.),
        np.logical_and(tuple_df['dest_cluster_pops'] <= 100., tuple_df['home_cluster_pops'] > 1000.),
    )
    
    med_med = np.logical_and(
        np.logical_and(tuple_df['home_cluster_pops'] > 100.,tuple_df['home_cluster_pops'] <= 1000.),
        np.logical_and(tuple_df['dest_cluster_pops'] > 100.,tuple_df['dest_cluster_pops'] <= 1000.))
    
    med_big = np.logical_or(
        np.logical_and(tuple_df['home_cluster_pops'] > 1000., np.logical_and(tuple_df['dest_cluster_pops'] > 100., tuple_df['dest_cluster_pops'] <= 1000.)),
        np.logical_and(tuple_df['dest_cluster_pops'] > 1000., np.logical_and(tuple_df['home_cluster_pops'] > 100., tuple_df['home_cluster_pops'] <= 1000.)),
    )
    
    big_big = np.logical_and(tuple_df['home_cluster_pops'] > 1000.,tuple_df['dest_cluster_pops'] > 1000.)
    
    slist = [small_small, small_med, small_big, med_med, med_big, big_big]
    slist_names = ['small-small', 'small-med', 'small-big', 'med-med', 'med-big', 'big-big']

    return [slist,slist_names]

# def bin_and_fit_migration_model():
