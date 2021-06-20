import sys

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

from dffml.util.entrypoint import entrypoint

from .scikit_base import ScikitScorerContext, ScikitScorer

# Classification
for entrypoint_name, name, cls in (
    ("acscore", "AccuracyScore", accuracy_score,),
    ("bacscore", "BalancedAccuracyScore", balanced_accuracy_score,),
    ("topkscore", "TopKAccuracyScore", top_k_accuracy_score,),
    ("avgprescore", "AveragePrecisionScore", average_precision_score,),
    ("brierscore", "BrierScoreLoss", brier_score_loss,),
    ("f1score", "F1Score", f1_score,),
    ("logloss", "LogLoss", log_loss,),
    ("prescore", "PrecisionScore", precision_score,),
    ("recallscore", "RecallScore", recall_score,),
    ("Jacscore", "JaccardScore", jaccard_score,),
    ("rocaucscore", "RocAucScore", roc_auc_score,),
):
    pass

# Clustering
for entrypoint_name, name, cls in (
    (
        "adjmutinfoscore",
        "AdjustedMutualInfoScore",
        adjusted_mutual_info_score,
    ),
    ("adjrandscore", "AdjustedRandScore", adjusted_rand_score,),
    ("complscore", "CompletenessScore", completeness_score,),
    ("fowlmalscore", "FowlkesMallowsScore", fowlkes_mallows_score,),
    ("homoscore", "HomogeneityScore", homogeneity_score,),
    ("mutinfoscore", "MutualInfoScore", mutual_info_score,),
    (
        "normmutinfoscore",
        "NormalizedMutualInfoScore",
        normalized_mutual_info_score,
    ),
    ("randscore", "RandScore", rand_score,),
    ("vmscore", "VMeasureScore", v_measure_score,),
):
    pass

# Regression
for entrypoint_name, name, cls in (
    ("exvscore", "ExplainedVarianceScore", explained_variance_score,),
    ("maxerr", "MaxError", max_error,),
    ("meanabserr", "MeanAbsoluteError", mean_absolute_error,),
    ("meansqrerr", "MeanSquaredError", mean_squared_error,),
    ("meansqrlogerr", "MeanSquaredLogError", mean_squared_log_error,),
    ("medabserr", "MedianAbsoluteError", median_absolute_error,),
    ("r2score", "R2Score", r2_score,),
    ("meanpoidev", "MeanPoissonDeviance", mean_poisson_deviance,),
    ("meangammadev", "MeanGammaDeviance", mean_gamma_deviance,),
    (
        "meanabspererr",
        "MeanAbsolutePercentageError",
        mean_absolute_percentage_error,
    ),
):
    parentContext = ScikitScorerContext
    parentScorer = ScikitScorer

    dffml_config = make_config_numpy(name + "ScorerConfig", cls, properties={})

    dffml_cls_ctx = type(name + "ScorerContext", (parentContext,), {},)

    dffml_cls = type(
        name + "Scorer",
        (parentScorer,),
        {
            "CONFIG": dffml_config,
            "CONTEXT": dffml_cls_ctx,
            "SCIKIT_SCORER": cls,
        },
    )
    # Add the ENTRY_POINT_ORIG_LABEL
    dffml_cls = entrypoint(entrypoint_name)(dffml_cls)

    setattr(sys.modules[__name__], dffml_config.__qualname__, dffml_config)
    setattr(sys.modules[__name__], dffml_cls_ctx.__qualname__, dffml_cls_ctx)
    setattr(sys.modules[__name__], dffml_cls.__qualname__, dffml_cls)
