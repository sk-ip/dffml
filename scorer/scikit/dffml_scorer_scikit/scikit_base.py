# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Intel Corporation
"""
Base class for Scikit scorers
"""


class ScikitScorerConfig(NamedTuple):
    pass


class ScikitScorerContext(AccuracyContext):
    async def aenter(self):
        pass

    async def aexit(self):
        pass

    async def score(self):
        pass


class ScikitScorer(AccuracyScorer):
    async def aenter(self):
        pass

    async def aexit(self):
        pass
