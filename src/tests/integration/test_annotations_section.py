import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox

from core.states.dataset_state import DatasetState
from models.dataset_model import DatasetTableModel
from views.annomate.sections.annotations import AnnotationsSection


@pytest.fixture
def annotations_section(qtbot):
    model = DatasetTableModel(DatasetState())
    model.add_class("Crack", (255, 0, 0))
    model.add_class("Scratch", (0, 255, 0))
    model.load_folder("/fake", ["img.jpg"])
    model.add_annotation(0, "Crack", [(0, 0), (1, 0), (1, 1)])

    widget = AnnotationsSection(model)
    qtbot.addWidget(widget)
    widget.set_current_row(0)
    widget.show()
    qtbot.wait(50)
    return widget, model


def test_annotation_class_combo_ignores_wheel_when_closed(annotations_section):
    widget, model = annotations_section
    combo = widget.findChild(QComboBox)
    assert combo is not None
    assert combo.currentText() == "Crack"

    event = QWheelEvent(
        QPointF(combo.rect().center()),
        QPointF(combo.mapToGlobal(combo.rect().center())),
        QPoint(0, 120),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )

    combo.wheelEvent(event)

    assert combo.currentText() == "Crack"
    assert model.get_annotations(0)[0]["category_name"] == "Crack"
