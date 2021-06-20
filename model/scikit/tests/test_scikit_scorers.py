import sys
import tempfile
import numpy as np

from dffml.record import Record
from dffml.high_level import accuracy
from dffml.source.source import Sources
from dffml.source.memory import MemorySource, MemorySourceConfig
from dffml.feature import Feature, Features
from dffml.util.asynctestcase import AsyncTestCase

import dffml_model_scikit.scikit_models
from sklearn.datasets import make_blobs
from model.scikit.dffml_model_scikit import AccuracyScoreScorer


class TestScikitModel:
    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.TemporaryDirectory()
        cls.features = Features()
        if cls.MODEL_TYPE is "REGRESSION":
            cls.features.append(Feature("A", float, 1))
            cls.features.append(Feature("B", float, 1))
            cls.features.append(Feature("C", float, 1))
            A, B, C, X = list(zip(*FEATURE_DATA_REGRESSION))
            cls.records = [
                Record(
                    str(i),
                    data={
                        "features": {
                            "A": A[i],
                            "B": B[i],
                            "C": C[i],
                            "X": X[i],
                        }
                    },
                )
                for i in range(0, len(A))
            ]

        cls.sources = Sources(
            MemorySource(MemorySourceConfig(records=cls.records))
        )
        cls.scorer = AccuracyScoreScorer()
        properties = {
            "directory": cls.model_dir.name,
            "features": cls.features,
        }
        config_fields = dict()
        estimator_type = cls.MODEL.SCIKIT_MODEL._estimator_type
        if estimator_type in supervised_estimators:
            config_fields["predict"] = Feature("X", float, 1)
        cls.model = cls.MODEL(
            cls.MODEL_CONFIG(**{**properties, **config_fields})
        )

    @classmethod
    def tearDownClass(cls):
        cls.model_dir.cleanup()

    async def test_00_train(self):
        async with self.sources as sources, self.model as model:
            async with sources() as sctx, model() as mctx:
                await mctx.train(sctx)

    async def test_01_accuracy(self):
        res = await accuracy(self.model, self.scorer, self.sources)
        self.assertTrue(0 <= res <= float("inf"))


FEATURE_DATA_REGRESSION = [
    [12.39999962, 11.19999981, 1.1, 42393.0],
    [14.30000019, 12.5, 1.3, 49255.0],
    [14.5, 12.69999981, 1.5, 40781.0],
    [14.89999962, 13.10000038, 2.0, 46575.0],
    [16.10000038, 14.10000038, 2.2, 42941.0],
    [16.89999962, 14.80000019, 2.9, 59692.0],
    [16.5, 14.39999962, 3.0, 63200.0],
    [15.39999962, 13.39999962, 3.2, 57495.0],
    [17, 14.89999962, 3.7, 60239.0],
    [17.89999962, 15.60000038, 3.9, 66268.0],
    [18.79999924, 16.39999962, 4.0, 58844.0],
    [20.29999924, 17.70000076, 4.1, 60131.0],
    [22.39999962, 19.60000038, 4.5, 64161.0],
    [19.39999962, 16.89999962, 4.9, 70988.0],
    [15.5, 14, 5.1, 69079.0],
    [16.70000076, 14.60000038, 5.3, 86138.0],
    [17.29999924, 15.10000038, 5.9, 84413.0],
    [18.39999962, 16.10000038, 6.0, 96990.0],
    [19.20000076, 16.79999924, 6.8, 94788.0],
    [17.39999962, 15.19999981, 7.1, 101323.0],
    [19.5, 17, 7.9, 104352.0],
    [19.70000076, 17.20000076, 8.2, 116862.0],
    [21.20000076, 18.60000038, 8.7, 112481.0],
]

"""
FEATURE_DATA_CLUSTERING = [
    [-9.01904123, 6.44409816, 5.95914173, 6.30718146],
    [ 7.10630876, -2.07342124, -0.72564101,  3.81251745],
    ...
    ]
"""
data, labels = make_blobs(
    n_samples=80, centers=8, n_features=4, random_state=2020
)
FEATURE_DATA_CLUSTERING = np.concatenate((data, labels[:, None]), axis=1)

REGRESSORS = [
    "LinearRegression",
    "ElasticNet",
    "BayesianRidge",
    "Lasso",
    "ARDRegression",
    "RANSACRegressor",
    "DecisionTreeRegressor",
    "GaussianProcessRegressor",
    "OrthogonalMatchingPursuit",
    "Lars",
    "Ridge",
]

supervised_estimators = ["classifier", "regressor"]
unsupervised_estimators = ["clusterer"]
valid_estimators = supervised_estimators + unsupervised_estimators

for reg in REGRESSORS:
    test_cls = type(
        f"Test{reg}Model",
        (TestScikitModel, AsyncTestCase),
        {
            "MODEL_TYPE": "REGRESSION",
            "MODEL": getattr(dffml_model_scikit.scikit_models, reg + "Model"),
            "MODEL_CONFIG": getattr(
                dffml_model_scikit.scikit_models, reg + "ModelConfig"
            ),
            "SCORER": AccuracyScoreScorer(),
        },
    )
    setattr(sys.modules[__name__], test_cls.__qualname__, test_cls)
