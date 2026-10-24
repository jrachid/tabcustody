import os

from hypothesis import HealthCheck, settings

_SUPPRESS = [HealthCheck.too_slow, HealthCheck.data_too_large]
settings.register_profile(
    "default", max_examples=300, deadline=1000, suppress_health_check=_SUPPRESS
)
settings.register_profile(
    "nightly", max_examples=20000, deadline=None, suppress_health_check=_SUPPRESS
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
