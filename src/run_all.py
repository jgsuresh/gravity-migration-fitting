# Run full analysis and generate gravity fit dataframe.

from create_trip_dataframe import create_trips_df
from cluster_and_link_individuals import cluster_and_link_individuals
from gravity_fit import fit_gravity_model_to_links

create_trips_df()
cluster_and_link_individuals()
fit_gravity_model_to_links()