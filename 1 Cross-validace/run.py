import pandas as pd
import cv_read, cv_folds, cv_sets, cv_vis
import matplotlib.pyplot as plt

df = cv_read.read_excel("./infos/Annotation figure IDs.xlsx")
df = cv_read.replace_filenames(df, "./infos/common_database_double_checked_creation.txt")
df = cv_read.drop_samples(df, ["59-79-1_10-20230807T081832.pcd", "51-97-1_8-20220705T054818.pcd"])


#%% Random
for i in range(4):
    folds = cv_folds.create_folds(df, col_weights=None, col_combinations=None, random_seed=i)
    # cv_vis.plot_folds(folds, f"Random seed {i}", ["species", "age_category"], "ones")
    cvs = cv_sets.create_nested_cv_sets(folds, method_name=f"rnd_{i}", col_filename="file_name")
    cv_sets.save_cvs(cvs, "./cvs_2025-08/")

#%% per pre
df = cv_read.add_days_since_seeding(df)
df = cv_read.add_age_category(df)
df = cv_read.add_count_category(df)
iterations = 100

#%% Per Cloud
categories = ["species", "age_category"]
df = cv_read.add_combinations(df, categories)
folds = cv_folds.search_for_folds(df, col_weights="ones", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="pcl", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-08/")

#%% Per Cloud with Num Groups
categories = ["species", "age_category", "num_category"]
df = cv_read.add_combinations(df, categories)
folds = cv_folds.search_for_folds(df, col_weights="ones", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="pcl_num", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-08/")

#%% Per Plant
categories = ["species", "age_category"]
df = cv_read.add_combinations(df, categories)
folds = cv_folds.search_for_folds(df, col_weights="num_plants", iterations=iterations, categories=categories)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="ppl", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-08/")

#%% SPOI

categories = ["SPOI_binned"]
df_spoi = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_spoi = cv_read.add_quantile_bins(df_spoi, "SPOI", bins=3)
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "SPOI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="oi-3b", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-24/")

categories = ["SPOI_binned"]
df_spoi = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_spoi = cv_read.add_quantile_bins(df_spoi, "SPOI", quantiles=[0.25, 0.5, 1.0])
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "SPOI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="oi-3b-qrt", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-24/")

categories = ["SPOI_binned", "species"]
df_spoi = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_spoi = cv_read.add_quantile_bins(df_spoi, "SPOI", bins=3)
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "SPOI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="oi-3b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-24/")

categories = ["SPOI_binned", "species"]
df_spoi = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_spoi = cv_read.add_quantile_bins(df_spoi, "SPOI", quantiles=[0.25, 0.5, 1.0])
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "SPOI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="oi-3b-qrt_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-24/")


#%%
categories = ["LAI_binned"]
df_lai = cv_read.add_file(df, "./infos/lai.csv", "file_name", "filename", ["LAI"])
df_lai = cv_read.add_quantile_bins(df_lai, "LAI", bins=3)
df_lai = cv_read.add_combinations(df_lai, categories)
folds = cv_folds.search_for_folds(df_lai, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "LAI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="lai-3b", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-10-20/")

categories = ["LAI_binned"]
df_lai = cv_read.add_file(df, "./infos/lai.csv", "file_name", "filename", ["LAI"])
df_lai = cv_read.add_quantile_bins(df_lai, "LAI", bins=5)
df_lai = cv_read.add_combinations(df_lai, categories)
folds = cv_folds.search_for_folds(df_lai, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "LAI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="lai-5b", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-10-20/")

#%%

categories = ["LAI_binned", "species"]
df_lai = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_lai = cv_read.add_file(df_lai, "./infos/lai.csv", "file_name", "filename", ["LAI"])
df_lai = cv_read.add_quantile_bins(df_lai, "LAI", bins=3)
df_lai = cv_read.add_combinations(df_lai, categories)
folds = cv_folds.search_for_folds(df_lai, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "LAI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="lai-3b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-10-20/")

#%%

categories = ["proj_to_real_binned"]
df_pv = cv_read.add_file(df, "./infos/lai.csv", "file_name", "filename", ["proj_to_real"])
df_pv = cv_read.add_quantile_bins(df_pv, "proj_to_real", bins=3)
df_pv = cv_read.add_combinations(df_pv, categories)
folds = cv_folds.search_for_folds(df_pv, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "P/V", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="pv-3b", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-05/")

categories = ["proj_to_real_binned"]
df_pv = cv_read.add_file(df, "./infos/lai.csv", "file_name", "filename", ["proj_to_real"])
df_pv = cv_read.add_quantile_bins(df_pv, "proj_to_real", bins=5)
df_pv = cv_read.add_combinations(df_pv, categories)
folds = cv_folds.search_for_folds(df_pv, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "P/V", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="pv-5b", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-05/")

#%%

categories = ["proj_to_real_binned", "species"]
df_pv = cv_read.add_file(df, "./infos/lai.csv", "file_name", "filename", ["proj_to_real"])
df_pv = cv_read.add_quantile_bins(df_pv, "proj_to_real", bins=3)
df_pv = cv_read.add_combinations(df_pv, categories)
folds = cv_folds.search_for_folds(df_pv, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "P/V", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="pv-3b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-11/")

categories = ["proj_to_real_binned", "species"]
df_pv = cv_read.add_file(df, "./infos/lai.csv", "file_name", "filename", ["proj_to_real"])
df_pv = cv_read.add_quantile_bins(df_pv, "proj_to_real", bins=5)
df_pv = cv_read.add_combinations(df_pv, categories)
folds = cv_folds.search_for_folds(df_pv, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "P/V", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="pv-5b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-11/")

categories = ["LAI_binned", "species"]
df_lai = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_lai = cv_read.add_file(df_lai, "./infos/lai.csv", "file_name", "filename", ["LAI"])
df_lai = cv_read.add_quantile_bins(df_lai, "LAI", bins=5)
df_lai = cv_read.add_combinations(df_lai, categories)
folds = cv_folds.search_for_folds(df_lai, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "LAI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="lai-5b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-11/")

categories = ["SPOI_binned"]
df_spoi = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_spoi = cv_read.add_quantile_bins(df_spoi, "SPOI", bins=5)
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "SPOI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="oi-5b", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-11/")

categories = ["SPOI_binned", "species"]
df_spoi = cv_read.add_file(df, "./infos/SPOI_fixed.csv", "file_name", "pcd_file_name", ["SPOI"])
df_spoi = cv_read.add_quantile_bins(df_spoi, "SPOI", bins=5)
df_spoi = cv_read.add_combinations(df_spoi, categories)
folds = cv_folds.search_for_folds(df_spoi, col_weights="ones", iterations=iterations, categories=categories)
cv_vis.plot_folds(folds, "SPOI", categories, "ones")
cvs = cv_sets.create_nested_cv_sets(folds, method_name="oi-5b_sp", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_2025-11-11/")

#%%

import scipy.stats as st
rsq = st.pearsonr(df_lai.LAI, df_lai.SPOI).statistic ** 2

fig, ax = plt.subplots(figsize=[5,5], dpi=144)
ax.scatter(df_lai.LAI, df_lai.SPOI)
ax.set_xlabel("Leaf Area Index")
ax.set_ylabel("Overlap Index")
fig.suptitle(f"Correlation between LAI and OI (R^2 = {rsq*100:.2f}%)")

#%%

fig, axes = plt.subplots(ncols=2, figsize=[10,5], dpi=144)
for a, m, l in zip(axes, ["LAI", "SPOI"], ["Leaf Area Index", "Overlap Index"]):
    a.hist(df_lai[m])
    a.set_xlabel("Metric")
    a.set_ylabel("Count")
    a.set_title(l)
fig.suptitle("Histograms for sample variables")




