from PySide6.QtCore import QObject, Signal

from core.states.center_template_state import CenterTemplateState


class CenterTemplateModel(QObject):
    """Qt adapter for center-template matching state."""

    template_changed = Signal()
    match_changed = Signal()

    def __init__(self, state: CenterTemplateState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

    def set_template(
        self,
        template_file: str,
        template_path: str,
        anchor_x: int,
        anchor_y: int,
        crop_shape: str,
        crop_width: int,
        crop_height: int,
        center_x: float,
        center_y: float,
    ) -> None:
        self._state.enabled = True
        self._state.template_file = template_file
        self._state.template_path = template_path
        self._state.anchor_x = int(anchor_x)
        self._state.anchor_y = int(anchor_y)
        self._state.crop_shape = crop_shape
        self._state.crop_width = int(crop_width)
        self._state.crop_height = int(crop_height)
        self._state.center_x = float(center_x)
        self._state.center_y = float(center_y)
        self._state.last_score = None
        self.template_changed.emit()

    def clear_template(self) -> None:
        self._state.clear()
        self.template_changed.emit()

    def set_enabled(self, enabled: bool) -> None:
        self._state.enabled = bool(enabled)
        self.template_changed.emit()

    def set_match(self, center_x: float, center_y: float, score: float) -> None:
        self._state.last_score = float(score)
        self.match_changed.emit()

    def enabled(self) -> bool:
        return self._state.enabled

    def has_template(self) -> bool:
        return self._state.has_template()

    def template_path(self) -> str:
        return self._state.template_path

    def template_file(self) -> str:
        return self._state.template_file

    def anchor(self) -> tuple[int, int]:
        return self._state.anchor_x, self._state.anchor_y

    def crop_settings(self) -> dict:
        return {
            "shape": self._state.crop_shape,
            "width": self._state.crop_width,
            "height": self._state.crop_height,
            "center_x": self._state.center_x,
            "center_y": self._state.center_y,
        }

    def last_score(self) -> float | None:
        return self._state.last_score
