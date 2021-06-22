# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Intel Corporation
"""
Base class for Scikit scorers
"""
import importlib
import functools

from dffml.base import config
from dffml.model.model import ModelContext
from dffml.source.source import SourcesContext
from dffml.accuracy import AccuracyScorer, AccuracyContext


@config
class ScikitScorerConfig:
    pass


class ScikitScorerContext(AccuracyContext):
    def __init__(self, parent):
        super().__init__(parent)
        # avoid
        self.np = importlib.import_module("numpy")
        self.scorer = None

    async def __aenter__(self):
        self.scorer = self.parent.SCIKIT_SCORER
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    async def score(self, mctx: ModelContext, sctx: SourcesContext):
        y_true = []
        y_pred = []
        async for record in mctx.predict(sctx):
            y_true.append(record.feature(mctx.parent.config.predict.name))
            y_pred.append(
                record.prediction(mctx.parent.config.predict.name).value
            )
        y_true = self.np.asarray(y_true)
        y_pred = self.np.asarray(y_pred)
        return self.scorer(y_true, y_pred)


class ScikitScorer(AccuracyScorer):

    CONFIG = ScikitScorerConfig

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass
