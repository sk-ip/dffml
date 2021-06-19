# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Intel Corporation
"""
Base class for Scikit scorers
"""
import importlib

from dffml.model.model import ModelContext
from dffml.source.source import SourcesContext
from dffml.accuracy import AccuracyScorer, AccuracyContext


class ScikitScorerConfig(NamedTuple):
    pass


class ScikitScorerContext(AccuracyContext):
    def __init__(self):
        super().__init__()
        self.np = importlib.import_module("numpy")

    async def aenter(self):
        # initialize the actual scorer here.
        pass

    async def aexit(self):
        pass

    async def score(self, mctx: ModelContext, sctx: SourcesContext):
        y_true = []
        y_pred = []
        async for record in sctx.with_features(
            mctx.features + [mctx.parent.config.predict.name]
        ):
            y_true.append(record.feature(mctx.parent.config.predict.name))
            y_pred.append(
                record.prediction(mctx.parent.config.predict.name).value
            )
        y_true = self.np.asarray(y_true)
        y_pred = self.np.asarray(y_pred)
        return self.scorer(y_true, y_pred)


class ScikitScorer(AccuracyScorer):
    async def aenter(self):
        pass

    async def aexit(self):
        pass
