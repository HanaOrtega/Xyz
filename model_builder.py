
# wrapper model_builder.py -> delegates to new modular builders
from modeling.builders.factory import build_model
from modeling.nas.builder import build_nas_trial_model
__all__ = ['build_model', 'build_nas_trial_model']
