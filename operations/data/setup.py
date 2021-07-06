import os
import sys
import site
import importlib.util
from setuptools import setup

# See https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Boilerplate to load commonalities
spec = importlib.util.spec_from_file_location(
    "setup_common", os.path.join(os.path.dirname(__file__), "setup_common.py")
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

common.KWARGS["entry_points"] = {
    "dffml.operation": [
        f"principal_componenet_analysis = {common.IMPORT_NAME}.operations:principal_componenet_analysis",
        f"singular_value_decomposition = {common.IMPORT_NAME}.operations:singular_value_decomposition",
        f"simple_imputer = {common.IMPORT_NAME}.operations:simple_imputer",
        f"one_hot_encoder = {common.IMPORT_NAME}.operations:one_hot_encoder",
        f"standard_scaler = {common.IMPORT_NAME}.operations:log_transformation",
        f"remove_whitespaces = {common.IMPORT_NAME}.operations:remove_whitespaces",
    ]
}

setup(**common.KWARGS)
