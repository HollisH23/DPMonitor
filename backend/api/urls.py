from django.urls import path

from . import auth_views, views

urlpatterns = [
    # Public
    path("health/", views.health, name="health"),

    # Auth
    path("auth/register/", auth_views.register, name="auth-register"),
    path("auth/login/",    auth_views.login,    name="auth-login"),
    path("auth/logout/",   auth_views.logout,   name="auth-logout"),
    path("auth/me/",       auth_views.me,       name="auth-me"),

    # Owner-scoped session endpoints (Phase 2.1)
    path("sessions/",                 views.SessionList.as_view(),   name="session-list"),
    path("sessions/<int:pk>/",        views.SessionDetail.as_view(), name="session-detail"),
    path("sessions/ingest/",          views.session_ingest,          name="session-ingest"),

    # Dashboard support
    path("trend/", views.trend_last_seven, name="trend-last-seven"),
]
