from TSB_AD.models.base import BaseDetector  # BaseDetector from the first Repository
from pysad.core.base_model import BaseModel  # BaseModel from the second Repository
from outlier import OutlierDetector

class AdapterTSBAD(BaseDetector):
    """
    Adapter to make PySAD models compatible with the TSB-AD BaseDetector interface.
    """
    def __init__(self, pysad_model, contamination=0.1):
        super().__init__(contamination=contamination)
        self.pysad_model = pysad_model

    def fit(self, X, y=None):
        # Uses the PySAD fit method (which calls fit_partial in a loop)
        self.pysad_model.fit(X, y)
        return self

    def decision_function(self, X):
        # Maps to PySAD's fit_score_partial (learns and scores the current point/window)
        return self.pysad_model.fit_score_partial(X)
    

class AdapterDSalmon(OutlierDetector):
    """
    Generic adapter for dSalmon models (SWKNN, xStream, etc.).
    """
    def __init__(self, model, contamination=0.1):
        self.model = model  # Instance of a dSalmon model (e.g., SWKNN, xStream)
        self.contamination = contamination
        self.scores_ = []  # To store scores if necessary
    
    def __getattr__(self, name):
        """
        Delegates any attribute or method not defined in the adapter 
        to the underlying dSalmon model, if present.
        """
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(
                f"Both '{type(self).__name__}' and the underlying model "
                f"lack the attribute/method '{name}'."
            )

    def fit(self, X, y=None):
        """
        Initializes the model with training data.
        For dSalmon, this typically involves processing initial data points.
        """
        self.model.fit(X)
        return self

    def decision_function(self, X):
        """
        Calculates anomaly scores for X and updates the model.
        X can be a single point (1D) or a 2D array.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Convert to 2D if necessary
        
        # Use fit_predict to update the model and obtain scores simultaneously
        scores = self.model.fit_predict(X)
        return scores.flatten()  # Return a 1D array of scores

class PostProcessedModel(BaseModel):
    """
    Wraps a PySAD model and applies a monotonic transformation to the output scores.
    """
    def __init__(self, base_model: BaseModel, transform=None):
        self.base_model = base_model
        # transform: a callable that takes a float or np.array and returns the transformed score
        self.transform = transform if transform is not None else (lambda s: s)

    def fit(self, X, y=None):
        return self.base_model.fit(X, y)

    def fit_partial(self, X, y=None):
        return self.base_model.fit_partial(X, y)

    def score_partial(self, X):
        s = self.base_model.score_partial(X)
        return self.transform(s)

    def fit_score_partial(self, X):
        # Follows PySAD semantics: fits the current point/window and then scores it
        self.base_model.fit_partial(X)
        s = self.base_model.score_partial(X)
        return self.transform(s)
