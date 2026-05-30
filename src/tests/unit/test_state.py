class TestAnnotationCRUD:
    def test_add_annotation(self, state):
        state.image_files = ["img.jpg"]
        state.add_annotation("img.jpg", "Defect", [(0, 0), (1, 0), (1, 1)])
        assert len(state.annotations["img.jpg"]) == 1
        assert state.annotations["img.jpg"][0]["category_name"] == "defect"
        assert state.annotations["img.jpg"][0]["visible"] is True

    def test_delete_annotation(self, state):
        state.annotations["img.jpg"] = [
            {"category_name": "Defect", "polygon": [(0, 0)]}
        ]
        state.delete_annotation("img.jpg", 0)
        assert state.annotations["img.jpg"] == []

    def test_delete_out_of_bounds_is_safe(self, state):
        state.annotations["img.jpg"] = []
        state.delete_annotation("img.jpg", 99)

    def test_update_annotation_points(self, state):
        state.annotations["img.jpg"] = [
            {"category_name": "Defect", "polygon": [(0, 0), (1, 0)]}
        ]
        new_pts = [(5, 5), (6, 5), (6, 6)]
        state.update_annotation_points("img.jpg", 0, new_pts)
        assert state.annotations["img.jpg"][0]["polygon"] == new_pts

    def test_set_annotation_visible(self, state):
        state.add_annotation("img.jpg", "Defect", [(0, 0), (1, 0), (1, 1)])

        state.set_annotation_visible("img.jpg", 0, False)

        assert state.is_annotation_visible("img.jpg", 0) is False


class TestClassRegistry:
    def test_class_names_empty_on_init(self, state):
        assert state.class_names == []
        assert state.class_colors == {}
        assert state.class_visibility == {}

    def test_add_class(self, state):
        state.add_class("Crack", (200, 100, 0))
        assert "crack" in state.class_names
        assert state.class_colors["crack"] == (200, 100, 0)
        assert state.is_class_visible("crack") is True

    def test_add_duplicate_class_is_idempotent(self, state):
        state.add_class("Defect", (255, 0, 0))
        state.add_class("Defect", (0, 0, 0))
        assert state.class_names.count("defect") == 1

    def test_delete_class_removes_annotations(self, state):
        state.add_class("Crack", (200, 100, 0))
        state.set_class_visible("crack", False)
        state.annotations["img.jpg"] = [
            {"category_name": "crack", "polygon": [(0, 0)]},
            {"category_name": "defect", "polygon": [(1, 1)]},
        ]
        state.delete_class("crack")
        remaining = state.annotations["img.jpg"]
        assert "crack" not in state.class_visibility
        assert all(a["category_name"] != "crack" for a in remaining)
        assert any(a["category_name"] == "defect" for a in remaining)

    def test_set_class_visible(self, state):
        state.add_class("Crack", (200, 100, 0))

        state.set_class_visible("crack", False)

        assert state.is_class_visible("crack") is False


class TestClear:
    def test_clear_resets_folder_data(self, state):
        state.image_dir = "/some/path"
        state.image_files = ["a.jpg"]
        state.annotations["a.jpg"] = [{"category_name": "X", "polygon": []}]
        state.clear()
        assert state.image_dir == ""
        assert state.image_files == []
        assert state.annotations == {}

    def test_clear_preserves_class_registry(self, state):
        """Classes and colors survive a folder reload — this is intentional."""
        state.add_class("Custom", (10, 20, 30))
        state.clear()
        assert "custom" in state.class_names
        assert state.class_colors["custom"] == (10, 20, 30)


class TestIsReviewed:
    def test_not_reviewed_when_empty(self, state):
        assert not state.is_reviewed("img.jpg")

    def test_reviewed_when_annotation_exists(self, state):
        state.add_annotation("img.jpg", "Defect", [(0, 0)])
        assert state.is_reviewed("img.jpg")

    def test_reviewed_when_inspector_set(self, state):
        state.set_inspector("img.jpg", "Alice")
        assert state.is_reviewed("img.jpg")

    def test_reviewed_when_note_set(self, state):
        state.set_note("img.jpg", "Looks bad")
        assert state.is_reviewed("img.jpg")
