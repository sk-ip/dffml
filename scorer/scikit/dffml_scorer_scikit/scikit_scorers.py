from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    jaccard_score,
    roc_auc_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    v_measure_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_absolute_percentage_error,
)

# Classification
for entrypoint_name, name, cls in (
    ("", "accuracy_score", accuracy_score,),
    ("", "balanced_accuracy_score", balanced_accuracy_score,),
    ("", "top_k_accuracy_score", top_k_accuracy_score,),
    ("", "average_precision_score", average_precision_score,),
    ("", "brier_score_loss", brier_score_loss,),
    ("", "f1_score", f1_score,),
    ("", "log_loss", log_loss,),
    ("", "precision_score", precision_score,),
    ("", "recall_score", recall_score,),
    ("", "jaccard_score", jaccard_score,),
    ("", "roc_auc_score", roc_auc_score,),
):
    pass

# Clustering
for entrypoint_name, name, cls in (
    ("", "adjusted_mutual_info_score", adjusted_mutual_info_score,),
    ("", "adjusted_rand_score", adjusted_rand_score,),
    ("", "completeness_score", completeness_score,),
    ("", "fowlkes_mallows_score", fowlkes_mallows_score,),
    ("", "homogeneity_score", homogeneity_score,),
    ("", "mutual_info_score", mutual_info_score,),
    ("", "normalized_mutual_info_score", normalized_mutual_info_score,),
    ("", "rand_score", rand_score,),
    ("", "v_measure_score", v_measure_score,),
):
    pass

# Regression
for entrypoint_name, name, cls in (
    ("", "explained_variance_score", explained_variance_score,),
    ("", "max_error", max_error,),
    ("", "mean_absolute_error", mean_absolute_error,),
    ("", "mean_squared_error", mean_squared_error,),
    ("", "mean_squared_log_error", mean_squared_log_error,),
    ("", "median_absolute_error", median_absolute_error,),
    ("", "r2_score", r2_score,),
    ("", "mean_poisson_deviance", mean_poisson_deviance,),
    ("", "mean_gamma_deviance", mean_gamma_deviance,),
    ("", "mean_absolute_percentage_error", mean_absolute_percentage_error,),
):
    config_fields = dict()
