import os
import logging

import cv2
from PySide6.QtCore import QObject, Signal

from core.logic.center_template import extract_template, locate_center


_TEMPLATE_FILENAME = "center_template.png"
logger = logging.getLogger("AnnoMate.CenterTemplateController")


class CenterTemplateController(QObject):
    """Headless service for saving and matching center templates."""

    template_saved = Signal(str)
    template_cleared = Signal()
    match_ready = Signal(float, float, float)
    match_failed = Signal(str)

    def __init__(self, center_template_model, parent=None) -> None:
        super().__init__(parent)
        self._model = center_template_model

    def save_template(
        self,
        project_dir: str,
        image_bgr,
        center_x: float,
        center_y: float,
        crop_shape: str,
        crop_width: int,
        crop_height: int,
    ) -> str:
        if not project_dir:
            raise ValueError("Save the project before accepting center calibration.")
        logger.info(
            "Saving center template: project_dir=%s center=(%.1f, %.1f) "
            "shape=%s size=%dx%d",
            project_dir,
            center_x,
            center_y,
            crop_shape,
            crop_width,
            crop_height,
        )
        os.makedirs(project_dir, exist_ok=True)
        template, anchor_x, anchor_y = extract_template(
            image_bgr,
            center_x,
            center_y,
            crop_width,
            crop_height,
        )
        template_path = os.path.join(project_dir, _TEMPLATE_FILENAME)
        if not cv2.imwrite(template_path, template):
            logger.error("Could not write center template: %s", template_path)
            raise OSError(f"Could not write template: {template_path}")
        tpl_h, tpl_w = template.shape[:2]
        logger.info(
            "Center template saved: path=%s template_size=%dx%d anchor=(%d, %d)",
            template_path,
            tpl_w,
            tpl_h,
            anchor_x,
            anchor_y,
        )
        self._model.set_template(
            _TEMPLATE_FILENAME,
            template_path,
            anchor_x,
            anchor_y,
            crop_shape,
            crop_width,
            crop_height,
            center_x,
            center_y,
        )
        self.template_saved.emit(template_path)
        return template_path

    def clear_template(self) -> None:
        logger.info("Clearing center template state.")
        self._model.clear_template()
        self.template_cleared.emit()

    def match_image(self, image_bgr) -> tuple[float, float, float] | None:
        if not self._model.enabled() or not self._model.has_template():
            logger.debug(
                "Skipping center template match: enabled=%s has_template=%s",
                self._model.enabled(),
                self._model.has_template(),
            )
            return None
        template_path = self._model.template_path()
        logger.debug("Loading center template for match: %s", template_path)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            msg = f"Template not found: {template_path}"
            logger.warning(msg)
            self.match_failed.emit(msg)
            return None
        img_h, img_w = image_bgr.shape[:2]
        tpl_h, tpl_w = template.shape[:2]
        if tpl_w > img_w or tpl_h > img_h:
            msg = "Template is larger than the loaded image."
            logger.warning(
                "%s template_size=%dx%d image_size=%dx%d",
                msg,
                tpl_w,
                tpl_h,
                img_w,
                img_h,
            )
            self.match_failed.emit(msg)
            return None
        anchor_x, anchor_y = self._model.anchor()
        logger.debug(
            "Running center template match: image_size=%dx%d template_size=%dx%d "
            "anchor=(%d, %d)",
            img_w,
            img_h,
            tpl_w,
            tpl_h,
            anchor_x,
            anchor_y,
        )
        cx, cy, score = locate_center(image_bgr, template, anchor_x, anchor_y)
        logger.info(
            "Center template matched: center=(%.1f, %.1f) score=%.4f",
            cx,
            cy,
            score,
        )
        self._model.set_match(cx, cy, score)
        self.match_ready.emit(float(cx), float(cy), float(score))
        return float(cx), float(cy), float(score)
