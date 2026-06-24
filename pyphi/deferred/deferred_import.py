# deferred_import.py

from pyphi.exceptions import MissingOptionalDependenciesError


# TODO(optional-deps) convert to factory pattern
class DeferredPlotly:
    _plotly = None

    def __init__(self):
        pass

    @classmethod
    def plotly(cls):
        if cls._plotly is None:
            try:
                import plotly

                cls._plotly = plotly
            except ModuleNotFoundError as exc:
                raise MissingOptionalDependenciesError(
                    MissingOptionalDependenciesError.MSG.format(dependencies="visualize")
                ) from exc
        return cls._plotly

    def __getattr__(self, attr):
        return getattr(self.plotly(), attr)


plotly = DeferredPlotly()


class DeferredNetworkX:
    _networkx = None

    @classmethod
    def networkx(cls):
        if cls._networkx is None:
            try:
                import networkx

                cls._networkx = networkx
            except ModuleNotFoundError as exc:
                raise MissingOptionalDependenciesError(
                    MissingOptionalDependenciesError.MSG.format(dependencies="visualize")
                ) from exc
        return cls._networkx

    def __getattr__(self, attr):
        return getattr(self.networkx(), attr)


networkx = DeferredNetworkX()
