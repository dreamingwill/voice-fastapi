from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models import SystemSettings


@dataclass
class SystemSettingsSnapshot:
    enable_speaker_recognition: bool = True


def _to_snapshot(settings: SystemSettings) -> SystemSettingsSnapshot:
    return SystemSettingsSnapshot(enable_speaker_recognition=bool(settings.enable_speaker_recognition))


def get_or_create_system_settings(db: Session) -> SystemSettings:
    settings = db.query(SystemSettings).order_by(SystemSettings.id).first()
    if settings is None:
        settings = SystemSettings(enable_speaker_recognition=True)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


def load_system_settings_snapshot(db: Session) -> SystemSettingsSnapshot:
    settings = get_or_create_system_settings(db)
    return _to_snapshot(settings)


def update_system_settings_snapshot(
    db: Session,
    *,
    enable_speaker_recognition: bool,
) -> SystemSettingsSnapshot:
    settings = get_or_create_system_settings(db)
    settings.enable_speaker_recognition = bool(enable_speaker_recognition)
    db.commit()
    db.refresh(settings)
    return _to_snapshot(settings)


__all__ = [
    "SystemSettingsSnapshot",
    "get_or_create_system_settings",
    "load_system_settings_snapshot",
    "update_system_settings_snapshot",
]
