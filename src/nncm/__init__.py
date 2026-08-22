"""NNCM — Neural Network Consequence Modelling.

End-to-end pipeline around a consequence-modelling package (Phast/Safeti):

1. :mod:`nncm.sampling` builds a space-filling design of vessel/leak scenarios.
2. :mod:`nncm.phast.input_writer` writes them into a Phast-importable workbook.
3. :mod:`nncm.phast.output_reader` turns Phast's result workbook into a dataset.
4. :mod:`nncm.training` fits a multi-output surrogate model.
5. :mod:`nncm.predict` / :mod:`nncm.gui` serve predictions.

Only the Phast import/export clicks stay manual.
"""

__version__ = "0.2.0"

from .config import NncmConfig, Project  # noqa: F401

__all__ = ["NncmConfig", "Project", "__version__"]
