import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.svm import SVC
from sklearn import linear_model
from scipy.misc import comb
from geopy.distance import vincenty



# Need to correct for the fact that some rounds covered a smaller area than others, many times over.



def possible_link_pairs(rd_list1, rd_list2):
    # Compute how many round-link-pairs are possible for two lists of rounds (data cannot capture links between two clusters on the same round)
    # rd_list1, rd_list2 = rd_meta_list
    n = 0
    for rdi in rd_list1:
        if rdi in rd_list2:
            n += len(rd_list2) - 1
        else:
            n += len(rd_list2)
    return n
    #fixme Could vectorize this with np.in1d...

def possible_link_pairs_normed(cl1_p, cl2_p, rd_pop_list1,rd_pop_list2,method='cl_pop'): #,respect_time_arrow=False
    # Compute how many round-link-pairs are possible for two lists of rounds
    # Normalization method captures the fact that even if a round is seen, its coverage may be poor, so it should be given less weight.
    # The 'cl_pop' normalizes each round's weight by the pop seen in that round versus the total pop ever seen in that cluster
    # The 'max_rd' normalizes each round's weight by the pop seen in that round versus the max pop seen in any round
    #fixme Note: this approach double-counts, not respecting the arrow of time.  For example someone can travel from round 8 to round 2.
    #fixme This can be fixed in post-processing by multiplying the normalization of the fit by 2.

    rd_pop_list1 = np.array(rd_pop_list1)
    rd_pop_list2 = np.array(rd_pop_list2)

    if method == 'cl_pop':
        cov1 = rd_pop_list1/cl1_p
        cov2 = rd_pop_list2/cl2_p
    elif method == 'max_rd':
        cov1 = rd_pop_list1/np.max(rd_pop_list1)
        cov2 = rd_pop_list2/np.max(rd_pop_list2)

    n = 0
    for i in xrange(len(rd_pop_list1)):
        if i == 0:
            n += cov1[i] * np.sum(cov2[1:])
        elif i == len(rd_pop_list1)-1:
            n += cov1[i] * np.sum(cov2[:-1])
        else:
            n += cov1[i]*(np.sum(cov2[:i])+np.sum(cov2[i+1:]))
        # for j in xrange(10):
        #     if respect_time_arrow and i < j:
        #         n += cov1[i] * cov2[j]
        #     elif not respect_time_arrow and i != j:
        #         n += cov1[i] * cov2[j]
    return n

def load_and_clean_data(survey_fn,links_fn,trips_fn):
    survey_df = pd.read_csv(survey_fn)
    # Get rid of any nan's and zeros in the data:
    not_nan = np.logical_not(np.logical_or(np.isnan(survey_df['latitude']), np.isnan(survey_df['longitude'])))
    not_zero = np.logical_not(np.logical_or(survey_df['latitude'] == 0, survey_df['longitude'] == 0))
    survey_df = survey_df[np.logical_and(not_nan, not_zero)]
    # Get rid of duplicates
    survey_df = survey_df.drop_duplicates(subset="person_id", keep="first")

    links_df = pd.read_csv(links_fn)
    trips_df = pd.read_csv(trips_fn)

    return [survey_df,links_df,trips_df,cluster_df]

def scatter_plot_individuals_by_round(survey_df):
    plt.close('all')
    plt.figure()
    for rd_n in np.arange(10) + 1:
        this_round = survey_df['round'] == rd_n

        ax = plt.subplot(3, 4, rd_n)
        ax.scatter(survey_df['longitude'][this_round], survey_df['latitude'][this_round],
                   marker='.', color="C{}".format(np.mod(rd_n, 5)), alpha=0.4)

        ax.set_xlabel("Long")
        ax.set_ylabel("Lat")
        ax.set_title("Round {}".format(rd_n))
        ax.set_xlim(25.5, 29.5)
        ax.set_ylim(-18.5, -15.5)

    plt.tight_layout()
    plt.show()

def get_latlong_of_unique_individuals(survey_df,links_df):
    # Do a [DBSCAN] clustering on union of (People you only see once ever) and (Homes of people you see more than once)
    singletons = links_df[links_df['num.seen'] == 1]
    singletons = singletons.merge(survey_df[["person_id", "latitude", "longitude"]], how='left', left_on='id.i',
                                  right_on='person_id')
    # Can have nan's now because might have been nans or zeros in original survey:
    not_nan = np.logical_not(np.logical_or(np.isnan(singletons['latitude']), np.isnan(singletons['longitude'])))
    singletons = singletons[not_nan]

    # ids_full = np.append()
    lat_full = np.append(singletons['latitude'], trips_df['lat_start'])
    long_full = np.append(singletons['longitude'], trips_df['long_start'])

    return [lat_full,long_full]

def cluster_individuals(lat,long,method='DBSCAN',min_samples=10,eps=0.005):
    clusterer = DBSCAN(min_samples=10, eps=0.005)
    clusterer.fit(np.column_stack((lat, long)))
    labels = np.copy(clusterer.labels_)
    return labels

def create_cluster_df(labels, lat_individuals, long_individuals, save_intermediate_file=False):
    # Save cluster properties into a dataframe/file:
    cluster_ids, cluster_pops = np.unique(labels, return_counts=True)
    cluster_pops[0] = 1  # "cluster population" for cluster id -1 is zero.

    # Get cluster lat/long, to determine centroid-ed distances:
    n_cl = np.size(cluster_ids)
    cluster_lat = np.zeros(n_cl)
    cluster_long = np.zeros(n_cl)
    for ii in np.arange(n_cl):
        cl_id = cluster_ids[ii]

        if cl_id == -1:
            cluster_lat[ii] = np.nan
            cluster_long[ii] = np.nan
        else:
            this_cl = labels == cl_id
            cluster_lat[ii] = np.mean(lat_individuals[this_cl])
            cluster_long[ii] = np.mean(long_individuals[this_cl])

    cluster_df = pd.DataFrame({"ids": cluster_ids,
                               "pops": cluster_pops,
                               "lat": cluster_lat,
                               "long": cluster_long})

    if save_intermediate_file:
        cluster_df.to_csv("../data/intermediate/cluster_info.csv")

    return cluster_df

def attempt_recluster_large_clusters(labels,lat_individuals,long_individuals,recluster_pop_thresh=1000,save_intermediate_file=True):
    print "Attempting reclustering of largest clusters..."

    labels_recl = np.copy(labels)
    biggest_cl_ids = np.array(cluster_df['ids'][cluster_df['pops'] >= recluster_pop_thresh])

    n_recl = 0
    for ii in biggest_cl_ids:
        labels_recl, changed = attempt_recluster_individual_cluster(labels_recl, lat_individuals, long_individuals, ii)
        n_recl += np.int(changed)

    labels_old = np.copy(labels)
    labels = labels_recl

    # Save these because they take forever to generate:
    individual_df = pd.DataFrame({
        "lat_individuals": lat_individuals,
        "long_individuals": long_individuals,
        "labels_old": labels_old,
        "labels_recl": labels_recl
    })

    if save_intermediate_file:
        individual_df.to_csv("labels.csv")

    return [labels,individual_df]

def attempt_recluster_individual_cluster(labels, lat, long, cluster_id, min_samples=50, eps=0.002):
    in_cl = labels == cluster_id

    labels_hold = np.copy(labels)

    original_cl_pop = np.sum(in_cl)
    mc_lat = lat[in_cl]
    mc_long = long[in_cl]

    clusterer = DBSCAN(min_samples=min_samples, eps=eps)
    clusterer.fit(np.column_stack((mc_lat, mc_long)))
    labels2 = clusterer.labels_

    if np.sum(labels2 == -1) >= (1. / 3.) * original_cl_pop:
        # Re-clustering fragmentation is too high, most likely
        return [labels, False]
    elif np.sum(labels2 == -1) == 0 and np.size(np.unique(labels2)) == 1:
        # Got the same cluster back
        return [labels, False]
    else:
        # print labels2
        min_new_label = np.max(labels) + 1
        temp = np.copy(labels[in_cl])
        temp[labels2 == -1] = -1
        temp[labels2 != -1] = labels2[labels2 != -1] + min_new_label
        labels[in_cl] = temp
        return [labels, True]

def compare_old_new_clusterings(cluster_df):
    #fixme Broken I think.
    old_cluster_df = pd.read_csv("../cluster_on_linked_individuals/cluster_info.csv")

    # plt.figure()
    # plt.hist(np.log10(cluster_pops[cluster_pops>1]),bins=100,label="New (all individuals)")
    # plt.hist(np.log10(old_cluster_df['pops'][old_cluster_df['pops']>1]),bins=100,histtype='step',linestyle='dashed',color='black',lw=2,label="Old (only linked individuals)")
    # plt.xlabel("Log10[Cluster Pops]")
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(5, 10))
    ax = plt.subplot(211)
    ax.scatter(cluster_df['long'], cluster_df['lat'], s=(cluster_df['pops']) ** 0.6, cmap=plt.cm.viridis,
               c=(cluster_df['pops']) ** 0.2, alpha=0.55)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    ax = plt.subplot(212)
    ax.scatter(old_cluster_df['long'], old_cluster_df['lat'], s=(old_cluster_df['pops']) ** 0.6, cmap=plt.cm.viridis,
               c=(old_cluster_df['pops']) ** 0.2, alpha=0.55)
    ax.set_xlabel("Long")
    ax.set_ylabel("Lat")
    plt.title("Cluster locations")
    plt.show()

def get_cluster_round_coverage(survey_df,trips_df,cluster_df,labels_individuals,lat_individuals,long_individuals,clean_clusters=True,save_intermediate_file=True):
    # Get list and number of rounds that each cluster was seen in, with appropriate normalization (and add this to cluster_df)
    # This happens by:
    #   1. Associate each person in survey_df with a cluster.
    #   2. Then, since survey_df includes which round the person was seen in, we can figure out in which rounds the given cluster was seen.

    cluster_labels_of_individuals, classifier = associate_survey_individuals_with_clusters(survey_df,cluster_df,labels_individuals,lat_individuals,long_individuals)

    survey_df['cluster_labels'] = pd.Series(cluster_labels_of_individuals, index=survey_df.index)
    cluster_round_coverage = survey_df.groupby('cluster_labels')['round'].nunique()
    cluster_rounds = survey_df.groupby('cluster_labels')['round'].unique()
    cluster_rounds = cluster_rounds.reset_index(drop=True)

    # Sort cluster rounds:
    func = lambda x: sorted(x)
    cluster_rounds = map(func, cluster_rounds)
    cluster_df['rounds_seen'] = pd.Series(cluster_rounds)
    cluster_df['total_rounds_seen'] = pd.Series(np.array(cluster_round_coverage), index=cluster_df.index)

    # Previously made assumption: if a given cluster is seen at all in a given round, it is seen equally "well" (same coverage) as any other
    # cluster/round.
    # New assumption: weight "round seen" by (# of people seen in that round)/(max # of people seen in any round)

    # Get dataframe of people-seen counts in each round for each cluster:
    foo = survey_df.groupby(['cluster_labels', 'round'])['name'].nunique()
    foo = foo.unstack(level=1)
    foo = foo.reset_index()
    foo = foo.fillna(0)

    # INSTEAD of the following: divide by cluster population directly.  This is generally larger than the max seen in any round.
    # # Now divide by max-seen-in-any-round to get weights:
    # foo['max'] = foo.apply(lambda x: np.max([x[i] for i in xrange(1,11)]), axis=1)
    # foo_norm = foo.copy()
    # for i in xrange(1,11):
    #     foo_norm[i] = foo.apply(lambda x: x[i]/x['max'],axis=1)

    # plt.figure()
    # for i in xrange(1,11):
    #     plt.hist(foo_norm[i],label="Round {}".format(i),bins=20,histtype='step',color=plt.cm.tab10(np.float(i)/11.),lw=2) #[foo_norm[i]>0.]
    # plt.legend(loc=1)
    # plt.xlabel("Round Coverage (compared to max[all rounds])")
    # plt.ylabel("# of clusters")
    # plt.show()

    cluster_df = cluster_df.merge(foo, how='left', left_on='ids', right_on='cluster_labels')
    cluster_df = cluster_df.drop('cluster_labels', axis=1)

    if clean_clusters:
        # Get rid of weird spurious clusters far away:
        cluster_df = cluster_df[cluster_df['long'] < 30]

    if save_intermediate_file:
        cluster_df.to_csv("cluster_info.csv")

    return [survey_df,cluster_df, classifier]

def plot_cluster_round_coverage(cluster_df):
    # Scatter plot clusters colored by how many rounds they were seen in:
    plt.figure(figsize=(5, 5))
    for ii in xrange(1,11):
        cut = cluster_df['total_rounds_seen'] == ii
        plt.scatter(cluster_df['long'][cut], cluster_df['lat'][cut], s=(cluster_df['pops'][cut]) ** 0.6,
                    c=plt.cm.magma(ii/10.), alpha=0.55,label="Seen in {} rounds".format(ii),edgecolors='black')
    plt.xlabel("Long")
    plt.ylabel("Lat")
    plt.legend()
    plt.title("Cluster Round Coverage")
    plt.show()

def associate_survey_individuals_with_clusters(survey_df,cluster_df,labels,lat_full,long_full):
    # Dumb/brute-force: Associate each person in survey_df with a cluster, by doing a KNN search.
    classifier = neighbors.KNeighborsClassifier(7, weights='distance')

    classifier.fit(np.column_stack((lat_full, long_full)), labels)
    print "Done training KKN.  Starting KKN prediction..."
    KNN_labels = classifier.predict(np.array(survey_df[['latitude', 'longitude']]))

    # Use same linking length in DBSCAN to exclude points that are more than this distance away from any cluster.
    dist, ind = classifier.kneighbors(X=np.array(survey_df[['latitude', 'longitude']]), n_neighbors=1,
                                      return_distance=True)
    d2 = np.ravel(dist)
    KNN_labels[d2 > 0.005] = -1

    return KNN_labels, classifier

def generate_all_possible_link_pairs(cluster_df,plot_hist=False,save_intermediate_file=True):
    n_cl = len(cluster_df)

    func = lambda x: vincenty(x[0:2], x[2:4]).km

    dat_columns = np.column_stack((
        np.repeat(cluster_df['lat'], n_cl), np.repeat(cluster_df['long'], n_cl),
        np.tile(cluster_df['lat'], n_cl), np.tile(cluster_df['long'], n_cl)
    ))
    d_allpairs = map(func, dat_columns)

    hold = pd.DataFrame({
        "rds_seen1": np.repeat(cluster_df['rounds_seen'], n_cl),
        "rds_seen2": np.tile(cluster_df['rounds_seen'], n_cl)
    })
    plp = hold.apply(lambda x: possible_link_pairs(x['rds_seen1'], x['rds_seen2']), axis=1)

    if plot_hist:
        plt.figure()
        plt.hist(hold['possible_link_pairs'],bins=90)
        plt.xlabel("Number of possible round-links between cluster-pair")
        plt.ylabel("Number of cluster-pairs")
        plt.show()


    # Do almost same thing, but now normalized by max cluster pop, or by max pop seen in any round
    hold = pd.DataFrame({
        "p1": np.repeat(cluster_df['pops'],n_cl),
        "p2": np.tile(cluster_df['pops'],n_cl)
    })

    for i in xrange(1,11):
        if i in cluster_df.columns:
            hold["cl1_rd{}".format(i)] = np.repeat(cluster_df[i],n_cl)
            hold["cl2_rd{}".format(i)] = np.tile(cluster_df[i], n_cl)
        else:
            hold["cl1_rd{}".format(i)] = np.repeat(cluster_df[str(i)], n_cl)
            hold["cl2_rd{}".format(i)] = np.tile(cluster_df[str(i)], n_cl)

    plp_norm_by_cl = \
        hold.apply(lambda x: possible_link_pairs_normed(x['p1'],x['p2'],
                                                        [x["cl1_rd{}".format(i)] for i in xrange(1,11)],
                                                        [x["cl2_rd{}".format(i)] for i in xrange(1, 11)],method='cl_pop'),
                   axis=1)
    plp_norm_by_maxrd = \
        hold.apply(lambda x: possible_link_pairs_normed(x['p1'],x['p2'],
                                                        [x["cl1_rd{}".format(i)] for i in xrange(1,11)],
                                                        [x["cl2_rd{}".format(i)] for i in xrange(1, 11)],method='max_rd'),
                   axis=1)

    del hold # Free up some memory

    pairs_df = pd.DataFrame({
        "c1_p": np.repeat(cluster_df['pops'],n_cl),
        "c2_p": np.tile(cluster_df['pops'],n_cl),
        "d": d_allpairs,
        "possible_link_pairs": plp,
        "possible_link_pairs_normed_by_clpop": plp_norm_by_cl,
        "possible_link_pairs_normed_by_maxrd": plp_norm_by_maxrd
    })

    if save_intermediate_file:
        pairs_df.to_csv("../data/intermediate/all_pairs.csv")

    return pairs_df

def associate_trip_endpoints_with_clusters(trips_df,cluster_df,labels,classifier,save_intermediate_file=True):
    # We already have the home labels from the DBSCAN clustering.
    # The last len(trips_df) of these are the trips, the entries before this are all singletons, by construction.
    home_labels = labels[-1 * len(trips_df):]
    trips_df['home_cluster_labels'] = pd.Series(home_labels, index=trips_df.index)

    # classifier.fit(np.array(trips_df[home_in_cluster][['lat_start','long_start']]),
    #                np.array(trips_df[home_in_cluster]['home_cluster_labels']))
    # print "Done training KKN.  Starting KKN prediction..."
    # Classifier already trained with full singleton/trip dataset from earlier, so no need to redo the training part.

    dest_coords = np.array(trips_df[['lat_end', 'long_end']])
    dest_labels = classifier.predict(dest_coords)

    # Use same linking length in DBSCAN to exclude points that are more than this distance away from any cluster.
    dist, ind = classifier.kneighbors(X=dest_coords, n_neighbors=1, return_distance=True)
    d2 = np.ravel(dist)
    dest_labels[d2 > 0.005] = -1
    print "Done with KKN."

    trips_df['dest_cluster_labels'] = pd.Series(dest_labels, index=trips_df.index)

    # Merge trips dataframe with cluster properties dataframe
    # Note: THIS WILL NOT WORK FOR CLUSTER -1
    hold = trips_df.copy(deep=True)
    trips_df = trips_df.merge(cluster_df, how='left', left_on='home_cluster_labels', right_on='ids', copy=False)
    trips_df["home_cluster_pops"] = trips_df["pops"]
    trips_df["home_cluster_lat"] = trips_df["lat"]
    trips_df["home_cluster_long"] = trips_df["long"]
    trips_df = trips_df.drop(["pops", "ids", "lat", "long", "rounds_seen", "total_rounds_seen"], axis=1)

    trips_df = trips_df.merge(cluster_df, how='left', left_on='dest_cluster_labels', right_on='ids', copy=False)
    trips_df["dest_cluster_pops"] = trips_df["pops"]
    trips_df["dest_cluster_lat"] = trips_df["lat"]
    trips_df["dest_cluster_long"] = trips_df["long"]
    trips_df = trips_df.drop(["pops", "ids", "lat", "long", "rounds_seen", "total_rounds_seen"], axis=1)

    # Calculate centroid-ed distance:
    func = lambda x: -1 if (np.isnan(x[0]) or np.isnan(x[1]) or np.isnan(x[2]) or np.isnan(x[3])) else vincenty(x[0:2],
                                                                                                                x[2:4]).km

    dat_columns = np.column_stack((trips_df['home_cluster_lat'], trips_df['home_cluster_long'],
                                   trips_df['dest_cluster_lat'], trips_df['dest_cluster_long']))
    d_centr = map(func, dat_columns)

    trips_df['d_centr'] = pd.Series(d_centr, index=trips_df.index)

    # Clean and save trip data to new CSV:
    trips_df = trips_df.drop('Unnamed: 0', axis=1)

    if save_intermediate_file:
        trips_df.to_csv("../data/intermediate/trips_with_clusters.csv")

    return trips_df

def save_clusters_to_kml(cluster_df):
    # Save cluster locations to KML file (google earth, etc.)
    import simplekml
    kml = simplekml.Kml()

    p = np.array(cluster_df['pops'])
    lats = np.array(cluster_df['lat'])
    longs = np.array(cluster_df['long'])

    for i in np.arange(np.size(lats)):
        if p[i] > 5000:
            kml.newpoint(name="{}".format(p[i]), coords=[(longs[i], lats[i])])
    kml.save("../data/intermediate/clusters_big.kml")

def plot_reclustering():
    # fixme ugly and broken
    lat_bnds = [-16.35, -15.5]  # [-18.054466666666698,-15.849351882899999]
    long_bnds = [28.7, 29.0]  # [26.4771866666667,27.7398967743]

    # Get people in corner
    in_corner = np.logical_and(
        np.logical_and(lat_full >= lat_bnds[0], lat_full <= lat_bnds[1]),
        np.logical_and(long_full >= long_bnds[0], long_full <= long_bnds[1]))
    # Get their cluster labels
    corner_labels = np.unique(labels[in_corner])

    # for cl in corner_labels:
    #     in_cl = labels[in_corner] == cl
    #     p = np.sum(in_cl)
    #     if p > 10000:
    #         label='Cluster {}: p={}'.format(cl, p)
    #     else:
    #         label=None
    #     plt.scatter(long_full[in_corner][in_cl], lat_full[in_corner][in_cl], alpha=0.6, marker='.', label=label)
    # plt.legend()
    # plt.show()


    biggest_cl_ids = np.array(cluster_df['ids'][cluster_df['pops'] >= 1000])

    for ii in biggest_cl_ids:
        # Megacluster re-clustering plots
        in_cl = labels == ii  # 3 #16 biggest
        # plt.scatter(long_full[in_corner][in_cl], lat_full[in_corner][in_cl], alpha=0.1, marker='.')
        # # sns.jointplot(long_full[in_corner],lat_full[in_corner],kind="kde")
        # plt.show()

        print "DBSCAN clustering..."
        mc_lat = lat_full[in_cl]
        mc_long = long_full[in_cl]
        clusterer2 = DBSCAN(min_samples=50, eps=0.002)
        clusterer2.fit(np.column_stack((mc_lat, mc_long)))
        labels2 = clusterer2.labels_
        print "DBSCAN clustering finished."

        plt.close('all')
        plt.figure()
        for l in np.unique(labels2):
            inl = labels2 == l
            if l == -1:
                c = 'black'
            else:
                c = None
            p = np.sum(inl)
            plt.scatter(mc_long[inl], mc_lat[inl], marker='.', c=c, label='pop={}'.format(p))
        plt.legend()
        plt.title("Reclustering cl {}, pop={}".format(ii, np.sum(in_cl)))
        plt.savefig("clid_{}_recluster.png".format(ii))

def cluster_and_link_individuals():
    survey_fn = '../data/raw/masterDatasetAllRounds2012-2016.csv'
    links_fn = '../data/raw/linksForJoshRounds1-10.csv'
    trips_fn = '../data/intermediate/all_trips_rds1-10.csv'
    survey_df,links_df,trips_df = load_and_clean_data(survey_fn,links_fn,trips_fn)

    # Get individuals and cluster them
    lat_full,long_full = get_latlong_of_unique_individuals(survey_df,links_df)
    DBSCAN_labels = cluster_individuals(lat_full,long_full)
    DBSCAN_labels,individual_df = attempt_recluster_large_clusters(DBSCAN_labels,lat_full,long_full)
    cluster_df = create_cluster_df(DBSCAN_labels)

    # Find coverage of each cluster by round
    survey_df,cluster_df,survey_to_cluster_classifier = get_cluster_round_coverage(survey_df,trips_df,cluster_df,DBSCAN_labels,lat_full,long_full)
    # plot_cluster_round_coverage(cluster_df)
    # compare_old_new_clusterings(cluster_df)

    # Given by-round cluster coverage, compute possible link pairs across rounds for all pairs:
    pairs_df = generate_all_possible_link_pairs(cluster_df)

    # Associate trip endpoints with clusters
    trips_df = associate_trip_endpoints_with_clusters(trips_df,cluster_df,DBSCAN_labels,survey_to_cluster_classifier)
