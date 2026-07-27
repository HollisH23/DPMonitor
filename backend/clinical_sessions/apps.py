from django.apps import AppConfig


class ClinicalSessionsConfig(AppConfig):
    """Exercise-session records (the implementation plan's `sessions` app).

    Renamed to `clinical_sessions` at the module level to avoid colliding
    with `django.contrib.sessions` while keeping the conceptual name
    intact in the schema and the URL routes.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "clinical_sessions"
    label = "clinical_sessions"
    verbose_name = "Exercise Sessions"
