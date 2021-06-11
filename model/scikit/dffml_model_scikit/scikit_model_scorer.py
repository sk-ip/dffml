from dffml.base import config
from dffml.util.entrypoint import entrypoint
from dffml.source.source import SourcesContext
from dffml.model import ModelContext, ModelNotTrained
from dffml.accuracy import (
    AccuracyScorer,
    AccuracyContext,
    InvalidNumberOfFeaturesError,
)

from sklearn.metrics import silhouette_score, mutual_info_score


@config
class SklearnModelAccuracyConfig:
    pass


class SklearnModelAccuracyContext(AccuracyContext):
    """
    Default scorer for Sklearn Models.
    """

    async def score(self, mctx: ModelContext, sctx: SourcesContext):
        if not mctx._filepath.is_file():
            raise ModelNotTrained("Train model before assessing for accuracy.")
        xdata = []
        ydata = []
        target = []
        estimator_type = mctx.clf._estimator_type
        
        if estimator_type in ("classifier", "regressor"):
            async for record in sctx.with_features(
                mctx.features + [mctx.parent.config.predict.name]
            ):
                record_data = []
                for feature in record.features(mctx.features).values():
                    record_data.extend(
                        [feature] if mctx.np.isscalar(feature) else feature
                    )
                xdata.append(record_data)
                ydata.append(record.feature(mctx.parent.config.predict.name))
            xdata = mctx.np.array(xdata)
            ydata = mctx.np.array(ydata)
            mctx.logger.debug("Number of input records: {}".format(len(xdata)))
            mctx.confidence = mctx.clf.score(xdata, ydata)
        else:
            if estimator_type == "clusterer":
                target = (
                    []
                    if mctx.parent.config.tcluster is None
                    else [mctx.parent.config.tcluster.name]
                )
            async for record in sctx.with_features(mctx.features):
                feature_data = record.features(mctx.features)
                xdata.append(list(feature_data.values()))
                ydata.append(list(record.features(target).values()))
            xdata = mctx.np.array(xdata)
            mctx.logger.debug("Number of input records: {}".format(len(xdata)))
            if target:
                ydata = mctx.np.array(ydata).flatten()
                if hasattr(mctx.clf, "predict"):
                    # xdata can be training data or unseen data
                    # inductive clusterer with ground truth
                    y_pred = mctx.clf.predict(xdata)
                    mctx.confidence = mutual_info_score(ydata, y_pred)
                else:
                    # requires xdata = training data
                    # transductive clusterer with ground truth
                    mctx.logger.critical(
                        "Accuracy found transductive clusterer, ensure data being passed is training data"
                    )
                    mctx.confidence = mutual_info_score(ydata, mctx.clf.labels_)
            else:
                if hasattr(mctx.clf, "predict"):
                    # xdata can be training data or unseen data
                    # inductive clusterer without ground truth
                    y_pred = mctx.clf.predict(xdata)
                    mctx.confidence = silhouette_score(xdata, y_pred)
                else:
                    # requires xdata = training data
                    # transductive clusterer without ground truth
                    mctx.logger.critical(
                        "Accuracy found transductive clusterer, ensure data being passed is training data"
                    )
                    mctx.confidence = silhouette_score(xdata, mctx.clf.labels_)
        mctx.logger.debug("Model Accuracy: {}".format(mctx.confidence))
        return mctx.confidence


@entrypoint("skmodelscore")
class SklearnModelAccuracy(AccuracyScorer):
    CONFIG = SklearnModelAccuracyConfig
    CONTEXT = SklearnModelAccuracyContext
