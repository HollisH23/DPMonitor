from django.contrib import admin

from .models import Session, TrajectoryData


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "exercise_type",
        "started_at",
        "rep_count",
        "overall_stability_score",
        "quality_score",
    )
    list_filter = ("exercise_type",)
    search_fields = ("user__username", "user__email")
    date_hierarchy = "started_at"


@admin.register(TrajectoryData)
class TrajectoryDataAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "sample_rate_hz", "frame_count", "created_at")
    search_fields = ("session__id",)
