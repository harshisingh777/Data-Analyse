"""DataLens: an analytics and AutoML workbench for tabular data.

The package is organised as a small pipeline:

    io         -> load and clean raw files into a DataFrame
    profiling  -> exploratory data analysis and data-quality scoring
    stats      -> hypothesis tests between features and a target
    target     -> target-column and task-type inference
    modeling   -> multi-model AutoML with cross-validation and tuning
    insights   -> plain-language findings for non-technical readers
    sql        -> sandboxed, read-only SQL over the dataset
    report     -> standalone HTML report rendering
    api        -> Flask application exposing all of the above
"""

import os

# joblib probes physical cores through `wmic`, which recent Windows builds no longer ship, and
# prints a noisy traceback. A user CPU limit below the logical count skips that probe; leaving one
# core free also keeps the web server responsive while models train.
if os.name == "nt":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

__version__ = "2.0.0"
__author__ = "Harshi Singh"
