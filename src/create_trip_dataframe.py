# Scripts to create a dataframe of trips, with home and destination locations.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from geopy.distance import vincenty

import time

from scipy.spatial.distance import cdist, euclidean



def merge_links_and_survey(f_links_file,f_surv_file):
    # Clean and merge links and survey data:

    # First, narrow by people you've actually seen more than once.
    f_links = f_links_file[f_links_file['num.seen'] > 1]

    # Prepare databases for merge:
    # f_surv_file['person_id'] = f_surv_file['person_id'].apply(lambda x: (x[5:]).upper()) # strip "uuid:", and change to upper case
    col_keep_list = ['person_id', 'household_id', 'round', 'latitude', 'longitude', 'gender', 'age', 'catch', 'date1']
    f_surv_keep = f_surv_file[col_keep_list]

    # Remove duplicates:
    f_surv_keep = f_surv_keep.drop_duplicates(subset="person_id", keep="first")

    # Merge this database with the other one, to make searches faster:
    for rd_n in range(1, 11):
        f_links = f_links.merge(f_surv_keep, how='left', left_on='id.{}'.format(rd_n), right_on='person_id')
        for col_name in col_keep_list:
            f_links["{}.{}".format(col_name, rd_n)] = f_links[col_name]
            f_links = f_links.drop(col_name, axis=1)

    return f_links

def narrow_by_gps_validity(f_links):
    # Future: If lat/long is missing for a given person, fill it in with the centroid of the catchment that they are in:
    # FOR NOW: going to simply throw away iterations where the person's GPS coordinates are missing.

    # Add new field, num.seen.gps, which incorporates not only how many times person is seen, but how many times we also have their GPS coords.
    # Eliminate any where we have num.seen.gps <= 1.

    num_seen_gps = np.ones(len(f_links))
    first_seen_gps = np.ones(len(f_links))
    last_seen_gps = np.ones(len(f_links))
    ii = 0
    for index, person in f_links.iterrows():
        coords_OK_count = 0
        fs = -1
        ls = -1
        for j in range(10):
            rd_num = j + 1
            not_nan = np.logical_not(np.logical_or(np.isnan(person['latitude.{}'.format(rd_num)]),
                                                   np.isnan(person['longitude.{}'.format(rd_num)])))
            not_zero = np.logical_not(
                np.logical_or(person['latitude.{}'.format(rd_num)] == 0, person['longitude.{}'.format(rd_num)] == 0))
            coords_OK_count += np.logical_and(not_nan, not_zero)
            if not_nan and not_zero:
                if fs == -1: fs = rd_num
                ls = rd_num

        num_seen_gps[ii] = coords_OK_count
        first_seen_gps[ii] = fs
        last_seen_gps[ii] = ls
        ii += 1

    f_links['num.seen.gps'] = pd.Series(num_seen_gps, index=f_links.index)
    f_links['first.seen.gps'] = pd.Series(first_seen_gps, index=f_links.index)
    f_links['last.seen.gps'] = pd.Series(last_seen_gps, index=f_links.index)

    f_links_gps = f_links[f_links['num.seen.gps'] > 1.]

    return f_links_gps

def get_trips_for_one_trippers(f_links_gps):
    # Create new "trips" dataframe:
    my_data = []

    # If someone is seen only twice, this case is easy.  Just upload their trip details.
    print "Compiling one-trip people..."
    one_trips = f_links_gps[f_links_gps['num.seen.gps'] == 2]

    for index,trip in one_trips.iterrows():
        rd_start = int(trip['first.seen.gps'])
        rd_end = int(trip['last.seen.gps'])
        # Get lat/long difference
        coords_start = [trip['latitude.{}'.format(rd_start)], trip['longitude.{}'.format(rd_start)]]
        coords_end = [trip['latitude.{}'.format(rd_end)], trip['longitude.{}'.format(rd_end)]]

        d = vincenty((coords_start[0], coords_start[1]), (coords_end[0], coords_end[1])).km

        endpoint_daynum = pd.to_datetime(trip['date1.{}'.format(rd_end)])
        endpoint_daynum = endpoint_daynum.dayofyear

        if d > 300:
            print "WARNING!  d={}. coords_start={}.  coords_end={}".format(d,coords_start,coords_end)

        my_data.append(
            {"gender": trip['gender.{}'.format(rd_end)],
             "age": trip['age.{}'.format(rd_end)],
             "rd_start": rd_start,
             "rd_end": rd_end,
             "endpoint_daynum": endpoint_daynum,
             "d":d,
             "lat_start": coords_start[0],
             "long_start": coords_start[1],
             "lat_end": coords_end[0],
             "long_end": coords_end[1],
             "trip_direction_uncertainty": 1
             })

        return my_data

def compile_multi_trippers(f_links_gps):
    # If someone is seen more than twice, it gets more complicated.
    # Try to find "mode", within 200 meters (or note people who have no mode)
    # if you can find a mode, this is their home.  calculate distances with respect to the mode.

    # Find home for everyone who is found more than twice.
    print "Compiling multi-trip people..."

    multi_trips = f_links_gps[f_links_gps['num.seen.gps'] > 2]

    # home_flag:
    # 0 - no home
    # 1 - home exists

    # home_lat, home_long

    home_flag = np.ones(len(multi_trips))
    home_lat = np.zeros(len(multi_trips))
    home_long = np.zeros(len(multi_trips))

    ii = 0
    for index,person in multi_trips.iterrows():
        rd_lats = np.array([])
        rd_longs = np.array([])
        rd_isnan = np.array([],dtype=bool)
        rd_iszero = np.array([], dtype=bool)
        for rd_n in range(1, 11):
            a=person['latitude.{}'.format(rd_n)]
            b=person['longitude.{}'.format(rd_n)]
            rd_lats = np.append(rd_lats,a)
            rd_longs = np.append(rd_longs, b)
            rd_isnan = np.append(rd_isnan, np.logical_or(np.isnan(a),np.isnan(b)))
            rd_iszero = np.append(rd_iszero, np.logical_or(a == 0, b == 0))
        rd_OK = np.logical_and(np.logical_not(rd_isnan), np.logical_not(rd_iszero))

        bucket_list = []
        for j in range(10):
            rd_j = j+1
            if rd_OK[j]:
                for k in range(j+1,10):
                    rd_k = k+1
                    if rd_OK[k]:
                        d = vincenty((rd_lats[j],rd_longs[j]),(rd_lats[k],rd_longs[k])).km

                        if d <= 0.2:
                            # Search if this pair could go into an already-existing bucket:
                            found_bucket = 0
                            if len(bucket_list) > 0:
                                for bi in range(len(bucket_list)):
                                    if (rd_j in bucket_list[bi]["rd_nums"]) and not (rd_k in bucket_list[bi]["rd_nums"]):
                                        bucket = bucket_list[bi]
                                        bucket["lat_centr"] = (bucket["lat_centr"]*bucket["n_pts"] + rd_lats[k])/(bucket["n_pts"] + 1.)
                                        bucket["long_centr"] = (bucket["long_centr"]*bucket["n_pts"] + rd_longs[k]) / (bucket["n_pts"] + 1.)
                                        bucket["n_pts"] += 1
                                        bucket["rd_nums"].append(rd_k)

                                        found_bucket = 1
                                    elif (rd_k in bucket_list[bi]["rd_nums"]) and not (rd_j in bucket_list[bi]["rd_nums"]):
                                        bucket = bucket_list[bi]
                                        bucket["lat_centr"] = (bucket["lat_centr"] * bucket["n_pts"] + rd_lats[j]) / (bucket["n_pts"] + 1.)
                                        bucket["long_centr"] = (bucket["long_centr"] * bucket["n_pts"] + rd_longs[j]) / (bucket["n_pts"] + 1.)
                                        bucket["n_pts"] += 1
                                        bucket["rd_nums"].append(rd_j)

                                        found_bucket = 1
                                    elif (rd_j in bucket_list[bi]["rd_nums"]) and (rd_k in bucket_list[bi]["rd_nums"]):
                                        found_bucket = 1

                            if found_bucket == 0:
                                new_bucket = {"n_pts":2,
                                              "rd_nums":[rd_j,rd_k],
                                              "lat_centr": (rd_lats[j]+rd_lats[k])/2.,
                                              "long_centr": (rd_longs[j]+rd_longs[k])/2.}
                                bucket_list.append(new_bucket)

        if len(bucket_list) == 0:
            home_flag[ii] = 0
            home_lat[ii] = np.nan
            home_long[ii] = np.nan

        elif len(bucket_list) > 0:
            home_flag[ii] = 1
            if len(bucket_list) == 1:
                home_lat[ii] = bucket_list[0]["lat_centr"]
                home_long[ii] = bucket_list[0]["long_centr"]
            elif len(bucket_list) > 1:
                print "More than 1 bucket!!"
                print bucket_list

                # Take largest (or first largest, in case of tie) bucket:
                largest_bucket_size = 0
                for bi in range(len(bucket_list)):
                    bucket = bucket_list[bi]
                    if bucket["n_pts"] > largest_bucket_size:
                        home_lat[ii] = bucket["lat_centr"]
                        home_long[ii] = bucket["long_centr"]

        ii += 1


    multi_trips['home_flag'] = pd.Series(home_flag, index=multi_trips.index)
    multi_trips['home_lat'] = pd.Series(home_lat, index=multi_trips.index)
    multi_trips['home_long'] = pd.Series(home_long, index=multi_trips.index)

    return f_links_gps

def get_trips_for_multi_trippers_with_no_home(f_links_gps,trip_data):
    # For people who have no home, take each adjacent trip:
    print "Compiling multi-trip people WITH NO HOME..."
    multi_trips = f_links_gps[f_links_gps['num.seen.gps'] > 2]

    multi_trips_wo_home = multi_trips[multi_trips['home_flag']==0]
    if True:
        for index,person in multi_trips_wo_home.iterrows():
            rd_lats = np.array([])
            rd_longs = np.array([])
            rd_isnan = np.array([], dtype=bool)
            rd_iszero = np.array([], dtype=bool)
            for rd_n in range(1, 11):
                a = person['latitude.{}'.format(rd_n)]
                b = person['longitude.{}'.format(rd_n)]
                rd_lats = np.append(rd_lats, a)
                rd_longs = np.append(rd_longs, b)
                rd_isnan = np.append(rd_isnan, np.logical_or(np.isnan(a), np.isnan(b)))
                rd_iszero = np.append(rd_iszero, np.logical_or(a == 0, b == 0))
            rd_OK = np.logical_and(np.logical_not(rd_isnan), np.logical_not(rd_iszero))

            rd_nums = np.arange(10)+1
            rd_nums = rd_nums[rd_OK]

            for trip_i in range(int(person['num.seen.gps']-1)):
                # print "trip_i ",trip_i
                rd_start = int(rd_nums[trip_i])
                rd_end = int(rd_nums[trip_i + 1])

                coords_start = [person['latitude.{}'.format(rd_start)], person['longitude.{}'.format(rd_start)]]
                coords_end = [person['latitude.{}'.format(rd_end)], person['longitude.{}'.format(rd_end)]]

                d = vincenty((coords_start[0], coords_start[1]), (coords_end[0], coords_end[1])).km

                endpoint_daynum = pd.to_datetime(person['date1.{}'.format(rd_end)])
                endpoint_daynum = endpoint_daynum.dayofyear

                if d > 300:
                    print "WARNING!  d={}. coords_start={}.  coords_end={}".format(d, coords_start, coords_end)

                trip_data.append(
                    {"gender": person['gender.{}'.format(rd_end)],
                     "age": person['age.{}'.format(rd_end)],
                     "rd_start": rd_start,
                     "rd_end": rd_end,
                     "endpoint_daynum": endpoint_daynum,
                     "d":d,
                     "lat_start": coords_start[0],
                     "long_start": coords_start[1],
                     "lat_end": coords_end[0],
                     "long_end": coords_end[1],
                     "trip_direction_uncertainty": 1
                     })

def get_trips_for_multi_trippers_with_home(f_links_gps,trip_data):
    # For people who have a home, take each round as a trip from this home:
    print "Compiling multi-trip people WITH HOME..."
    multi_trips = f_links_gps[f_links_gps['num.seen.gps'] > 2]
    multi_trips_w_home = multi_trips[multi_trips['home_flag'] == 1]
    if True:
        for index, person in multi_trips_w_home.iterrows():
            rd_lats = np.array([])
            rd_longs = np.array([])
            rd_isnan = np.array([], dtype=bool)
            rd_iszero = np.array([], dtype=bool)
            for rd_n in range(1, 11):
                a = person['latitude.{}'.format(rd_n)]
                b = person['longitude.{}'.format(rd_n)]
                rd_lats = np.append(rd_lats, a)
                rd_longs = np.append(rd_longs, b)
                rd_isnan = np.append(rd_isnan, np.logical_or(np.isnan(a), np.isnan(b)))
                rd_iszero = np.append(rd_iszero, np.logical_or(a == 0, b == 0))
            rd_OK = np.logical_and(np.logical_not(rd_isnan), np.logical_not(rd_iszero))

            rd_nums = np.arange(10) + 1
            rd_nums = rd_nums[rd_OK]

            for trip_i in range(int(person['num.seen.gps'])):
                rd_end = int(rd_nums[trip_i])

                coords_start = [person['home_lat'], person['home_long']]
                coords_end = [person['latitude.{}'.format(rd_end)], person['longitude.{}'.format(rd_end)]]

                d = vincenty((coords_start[0], coords_start[1]), (coords_end[0], coords_end[1])).km

                endpoint_daynum = pd.to_datetime(person['date1.{}'.format(rd_end)])
                endpoint_daynum = endpoint_daynum.dayofyear

                if d > 300:
                    print "WARNING!  d={}. coords_start={}.  coords_end={}".format(d, coords_start, coords_end)

                trip_data.append(
                    {"gender": person['gender.{}'.format(rd_end)],
                     "age": person['age.{}'.format(rd_end)],
                     "rd_start": -1,
                     "rd_end": rd_end,
                     "endpoint_daynum": endpoint_daynum,
                     "d": d,
                     "lat_start": coords_start[0],
                     "long_start": coords_start[1],
                     "lat_end": coords_end[0],
                     "long_end": coords_end[1],
                     "trip_direction_uncertainty": 0
                     })

def trip_exploratory_plots(trips_df):
        # trips_df.plot.hist(by='d',bins=100)
        # plt.show()

        # d = np.array(trips_df['d'])

        # plt.close('all')
        n, bins, patches = plt.hist(trips_df['d'], bins=50, log=True, range=[0, 350.], color='gray', alpha=0.3,
                                    label='All')
        # plt.hist(trips_df[trips_df['gender']==1.0]['d'], bins=50, log=True, range=[0, 350.],color='blue',histtype='step',label='Male',lw=1.2)
        # plt.hist(trips_df[trips_df['gender'] == 2.0]['d'], bins=50, log=True, range=[0, 350.], color='red',histtype='step',label='Female',lw=1.2)

        # s1 = np.logical_or(trips_df['endpoint_daynum'] < 60.,trips_df['endpoint_daynum'] >= 330.)
        # s2 = np.logical_and(trips_df['endpoint_daynum'] < 150., trips_df['endpoint_daynum'] >= 60.)
        # s3 = np.logical_and(trips_df['endpoint_daynum'] < 240., trips_df['endpoint_daynum'] >= 150.)
        # s4 = np.logical_and(trips_df['endpoint_daynum'] < 330., trips_df['endpoint_daynum'] >= 240.)
        #
        # plt.hist(trips_df[s1]['d'], bins=50, log=True, range=[0, 350.],color='C0',histtype='step',label='Winter',lw=1.2)
        # plt.hist(trips_df[s2]['d'], bins=50, log=True, range=[0, 350.], color='C1', histtype='step', label='Spring',lw=1.2)
        # plt.hist(trips_df[s3]['d'], bins=50, log=True, range=[0, 350.], color='C2', histtype='step', label='Summer',lw=1.2)
        # plt.hist(trips_df[s4]['d'], bins=50, log=True, range=[0, 350.], color='C3', histtype='step', label='Fall',lw=1.2)


        # a1 = trips_df['age'] < 5.
        # a2 = np.logical_and(trips_df['age'] >= 5., trips_df['age'] < 18.)
        # a3 = np.logical_and(trips_df['age'] >= 18., trips_df['age'] < 50.)
        # a4 = trips_df['age'] >= 50.
        #
        # plt.hist(trips_df[a1]['d'], bins=50, log=True, range=[0, 350.],color='C0',histtype='step',label='Age bin 1',lw=1.2)
        # plt.hist(trips_df[a2]['d'], bins=50, log=True, range=[0, 350.], color='C1', histtype='step', label='Age bin 2',lw=1.2)
        # plt.hist(trips_df[a3]['d'], bins=50, log=True, range=[0, 350.], color='C2', histtype='step', label='Age bin 3',lw=1.2)
        # plt.hist(trips_df[a4]['d'], bins=50, log=True, range=[0, 350.], color='C3', histtype='step', label='Age bin 4',lw=1.2)

        # plt.legend()
        # plt.ylabel("Number of Trips")
        # plt.xlabel("Distance travelled (km)")
        # plt.show()


        # plt.close('all')
        # age_bin_min = [0,5,12,26] #[0,5,18,50]
        # age_bin_max = [5,12,26,1000]
        # dat_list = []
        # for ai in range(len(age_bin_max)):
        #     age_cut = np.logical_and(trips_df['age'] >= age_bin_min[ai], trips_df['age'] < age_bin_max[ai])
        #     n2, bins2, patches2 = plt.hist(trips_df[age_cut]['d'], bins=50, log=True, range=[0, 350.])
        #     dat = n2/n
        #     dat_list.append(dat)
        #
        # plt.close('all')
        # plt.plot(bins[:-1],dat_list[0],label='age=0-5')
        # plt.plot(bins[:-1], dat_list[1],label='age=5-12')
        # plt.plot(bins[:-1], dat_list[2],label='age=12-26')
        # plt.plot(bins[:-1], dat_list[3],label='age=26+')
        # plt.legend()
        # plt.xlabel("Fraction of trips in given age bin")
        # plt.ylabel("Distance (km)")
        # plt.show()


        n_women = np.float(np.sum(trips_df['gender'] == 2.0))
        n_men = np.float(np.sum(trips_df['gender'] == 1.0))
        n_total = np.float(n_men + n_women)

        plt.close('all')
        n2, bins2, patches2 = plt.hist(trips_df[trips_df['gender'] == 1.0]['d'], bins=50, log=True, range=[0, 350.])
        n3, bins3, patches3 = plt.hist(trips_df[trips_df['gender'] == 2.0]['d'], bins=50, log=True, range=[0, 350.])

        plt.close('all')
        plt.plot(bins[:-1], n2 / n / (n_men / n_total), label='Male', color='blue')
        plt.plot(bins[:-1], n3 / n / (n_women / n_total), label='Female', color='red')
        plt.legend()
        plt.ylabel("Fraction of trips in given age bin \n (normalized by total male/female counts)")
        plt.xlabel("Distance (km)")
        plt.show()

def create_trips_df(save_intermediate_file=True):
    f_links_file = pd.read_csv("../data/raw/linksForJoshRounds1-10.csv")
    f_surv_file = pd.read_csv("../data/raw/masterDatasetAllRounds2012-2016.csv")

    f_links = merge_links_and_survey(f_links_file,f_surv_file)

    f_links_gps = narrow_by_gps_validity(f_links)

    trip_data = get_trips_for_one_trippers(f_links_gps)

    f_links_gps = compile_multi_trippers(f_links_gps)

    trip_data = get_trips_for_multi_trippers_with_no_home(f_links_gps,trip_data)
    trip_data = get_trips_for_multi_trippers_with_home(f_links_gps,trip_data)

    trips_df = pd.DataFrame(trip_data)
    if save_intermediate_file:
        trips_df.to_csv("../data/intermediate/all_trips_rds1-10.csv")

    return trips_df
