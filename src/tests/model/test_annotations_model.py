from PySide6.QtCore import Qt

from core.states.dataset_state import DatasetState
from models.annotations_model import (
    ANNOTATION_INDEX_ROLE,
    AnnotationColumns,
    AnnotationSortProxyModel,
    AnnotationTableModel,
)
from models.dataset_model import DatasetTableModel


def _make_model():
    dataset_model = DatasetTableModel(DatasetState())
    dataset_model.add_class("Beta", (20, 20, 20))
    dataset_model.add_class("alpha", (10, 10, 10))
    dataset_model.add_class("Gamma", (30, 30, 30))
    dataset_model.load_folder("/fake", ["one.jpg"])
    dataset_model.add_annotation(0, "Beta", [(0, 0), (1, 0), (1, 1)])
    dataset_model.add_annotation(0, "alpha", [(0, 0), (2, 0), (2, 2), (0, 2)])
    dataset_model.add_annotation(
        0, "Gamma", [(0, 0), (3, 0), (3, 3), (1, 4), (0, 3)]
    )
    return dataset_model


def _proxy_indices(proxy):
    return [
        proxy.index(row, AnnotationColumns.CLASS).data(ANNOTATION_INDEX_ROLE)
        for row in range(proxy.rowCount())
    ]


def test_annotation_rows_reflect_current_image_annotations():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)

    assert table_model.rowCount() == 3
    assert table_model.index(0, AnnotationColumns.CLASS).data() == "Beta"
    assert table_model.index(1, AnnotationColumns.VERTICES).data() == "4"


def test_class_name_sort_is_case_insensitive():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.CLASS, Qt.AscendingOrder)

    assert _proxy_indices(proxy) == [1, 0, 2]


def test_vertex_count_sorts_numerically():
    table_model = AnnotationTableModel(_make_model())
    table_model.set_current_row(0)
    proxy = AnnotationSortProxyModel()
    proxy.setSourceModel(table_model)

    proxy.sort(AnnotationColumns.VERTICES, Qt.DescendingOrder)

    assert _proxy_indices(proxy) == [2, 1, 0]


def test_class_column_updates_source_annotation():
    dataset_model = _make_model()
    table_model = AnnotationTableModel(dataset_model)
    table_model.set_current_row(0)

    assert table_model.setData(
        table_model.index(0, AnnotationColumns.CLASS), "Gamma", Qt.EditRole
    )

    assert dataset_model.get_annotations(0)[0]["category_name"] == "Gamma"
