from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QRadioButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)


class ViewportActionsBar(QFrame):
    """Floating bottom-center actions for canvas view and grid tools."""

    tool_selected = Signal(str)

    _MARGIN = 12
    _BTN_SIZE = 32

    def __init__(self, canvas, calibration_model=None, parent: QWidget = None) -> None:
        super().__init__(parent or canvas)
        self._canvas = canvas
        self._model = None
        self._active_tool = ""
        self._has_image = False
        self._refreshing = False

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setAutoFillBackground(True)

        font = QFont()
        font.setPointSize(16)
        font.setBold(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self._btn_zoom_in = self._make_button("+", "Zoom In")
        self._btn_zoom_in.setFont(font)
        self._btn_zoom_in.clicked.connect(canvas.zoom_in)
        layout.addWidget(self._btn_zoom_in)

        self._btn_zoom_out = self._make_button("-", "Zoom Out")
        self._btn_zoom_out.setFont(font)
        self._btn_zoom_out.clicked.connect(canvas.zoom_out)
        layout.addWidget(self._btn_zoom_out)

        self._btn_reset = self._make_button("⊙", "Reset View")
        self._btn_reset.setFont(font)
        self._btn_reset.clicked.connect(canvas.reset_view)
        layout.addWidget(self._btn_reset)

        self._add_divider(layout)

        self._btn_calibrate = self._make_button("⊕", "Set Calibration Points (C)")
        self._btn_calibrate.setCheckable(True)
        self._btn_calibrate.setFont(font)
        self._btn_calibrate.clicked.connect(
            lambda checked: self._on_tool_clicked("calibrate", checked)
        )
        layout.addWidget(self._btn_calibrate)

        self._btn_measure = self._make_button("⇔", "Measure Distance (M)")
        self._btn_measure.setCheckable(True)
        self._btn_measure.setFont(font)
        self._btn_measure.clicked.connect(
            lambda checked: self._on_tool_clicked("measure", checked)
        )
        layout.addWidget(self._btn_measure)

        self._btn_settings = self._make_button("⚙", "Grid Settings")
        self._btn_settings.setFont(font)
        self._btn_settings.setPopupMode(QToolButton.InstantPopup)
        self._btn_settings.setMenu(self._build_settings_menu())
        layout.addWidget(self._btn_settings)

        self.adjustSize()
        self.set_image_loaded(False)
        if calibration_model is not None:
            self.set_calibration_model(calibration_model)
        else:
            self._refresh_controls()

    def _make_button(self, text: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(self._BTN_SIZE, self._BTN_SIZE)
        return btn

    def _add_divider(self, layout: QHBoxLayout) -> None:
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

    def _build_settings_menu(self) -> QMenu:
        menu = QMenu(self)
        action = QWidgetAction(self)
        panel = QWidget()
        panel.setMinimumWidth(260)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(6)

        self._status_lbl = QLabel("Not calibrated")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setMinimumHeight(44)
        self._status_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._status_lbl.setStyleSheet("color: grey; font-style: italic;")
        panel_layout.addWidget(self._status_lbl)

        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)
        self._grid_chk = QCheckBox("Show Grid")
        self._grid_chk.toggled.connect(self._on_grid_toggled)
        grid_row.addWidget(self._grid_chk)
        grid_row.addStretch()
        grid_row.addWidget(QLabel("Opacity"))
        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(50)
        self._opacity_slider.setFixedWidth(84)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        grid_row.addWidget(self._opacity_slider)
        self._opacity_lbl = QLabel("50%")
        self._opacity_lbl.setFixedWidth(34)
        grid_row.addWidget(self._opacity_lbl)
        panel_layout.addLayout(grid_row)

        spacing_row = QHBoxLayout()
        spacing_row.setSpacing(6)
        spacing_row.addWidget(QLabel("Spacing"))
        self._radio_auto = QRadioButton("Auto")
        self._radio_fixed = QRadioButton("Fixed:")
        self._radio_auto.setChecked(True)
        spacing_group = QButtonGroup(self)
        spacing_group.addButton(self._radio_auto)
        spacing_group.addButton(self._radio_fixed)
        self._radio_auto.toggled.connect(self._on_spacing_mode_changed)
        spacing_row.addWidget(self._radio_auto)
        spacing_row.addWidget(self._radio_fixed)
        self._spacing_edit = QLineEdit()
        self._spacing_edit.setPlaceholderText("1.0")
        self._spacing_edit.setFixedWidth(64)
        self._spacing_edit.setEnabled(False)
        self._spacing_edit.editingFinished.connect(self._on_spacing_edited)
        spacing_row.addWidget(self._spacing_edit)
        self._unit_lbl = QLabel("mm")
        spacing_row.addWidget(self._unit_lbl)
        panel_layout.addLayout(spacing_row)

        details_row = QHBoxLayout()
        details_row.setSpacing(6)
        details_row.addWidget(QLabel("Color"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(32, 20)
        self._color_btn.setToolTip("Change grid color")
        self._color_btn.clicked.connect(self._on_color_clicked)
        details_row.addWidget(self._color_btn)
        details_row.addStretch()
        self._meas_lbl = QLabel("Distance: -")
        self._meas_lbl.setStyleSheet("font-weight: bold;")
        details_row.addWidget(self._meas_lbl)
        panel_layout.addLayout(details_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._btn_clear_measurement = QPushButton("Clear Measurement")
        self._btn_clear_measurement.clicked.connect(self._on_clear_measurement_clicked)
        action_row.addWidget(self._btn_clear_measurement)

        self._btn_reset_calibration = QPushButton("Reset Calibration")
        self._btn_reset_calibration.clicked.connect(self._on_reset_calibration_clicked)
        action_row.addWidget(self._btn_reset_calibration)
        panel_layout.addLayout(action_row)

        self._update_color_swatch((58, 90, 122))
        action.setDefaultWidget(panel)
        menu.addAction(action)
        return menu

    def set_calibration_model(self, model) -> None:
        self._model = model
        model.calibration_changed.connect(self._refresh_all)
        model.grid_changed.connect(self._refresh_all)
        model.measurement_updated.connect(self._refresh_measurement)
        self._refresh_all()

    def set_image_loaded(self, loaded: bool) -> None:
        self._has_image = loaded
        self._refresh_action_availability()

    def set_active_tool(self, tool_name: str) -> None:
        self._active_tool = tool_name if tool_name in ("calibrate", "measure") else ""
        self._refreshing = True
        self._btn_calibrate.setChecked(self._active_tool == "calibrate")
        self._btn_measure.setChecked(self._active_tool == "measure")
        self._refreshing = False

    def toggle_calibrate(self) -> None:
        if self._btn_calibrate.isEnabled():
            self._on_tool_clicked("calibrate", self._active_tool != "calibrate")

    def toggle_measure(self) -> None:
        if self._btn_measure.isEnabled():
            self._on_tool_clicked("measure", self._active_tool != "measure")

    def reposition(self, canvas_size) -> None:
        self.adjustSize()
        x = (canvas_size.width() - self.width()) // 2
        y = canvas_size.height() - self.height() - self._MARGIN
        self.move(max(0, x), max(0, y))

    def _on_tool_clicked(self, tool_name: str, checked: bool) -> None:
        if self._refreshing:
            return
        self._active_tool = tool_name if checked else ""
        self.set_active_tool(self._active_tool)
        self.tool_selected.emit(self._active_tool)

    def _on_grid_toggled(self, checked: bool) -> None:
        if self._model is not None and not self._refreshing:
            self._model.set_grid_visible(checked)

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_lbl.setText(f"{value}%")
        if self._model is not None and not self._refreshing:
            self._model.set_grid_opacity(value / 100.0)

    def _on_color_clicked(self) -> None:
        if self._model is None:
            return
        r, g, b = self._model.grid_color()
        color = QColorDialog.getColor(QColor(r, g, b), self, "Grid Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._model.set_grid_color(rgb)
            self._update_color_swatch(rgb)

    def _on_spacing_mode_changed(self, auto_checked: bool) -> None:
        self._spacing_edit.setEnabled(not auto_checked)
        if self._model is None or self._refreshing:
            return
        if auto_checked:
            self._model.set_grid_spacing_auto()
        else:
            self._try_apply_spacing()

    def _on_spacing_edited(self) -> None:
        if (
            self._model is not None
            and not self._refreshing
            and self._radio_fixed.isChecked()
        ):
            self._try_apply_spacing()

    def _on_clear_measurement_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_measurement()

    def _on_reset_calibration_clicked(self) -> None:
        if self._model is not None:
            self._model.clear_calibration()

    def _try_apply_spacing(self) -> None:
        try:
            value = float(self._spacing_edit.text())
        except ValueError:
            return
        if value > 0:
            self._model.set_grid_spacing(value)

    def _refresh_all(self) -> None:
        self._refresh_status()
        self._refresh_controls()
        self._refresh_measurement()
        self._refresh_action_availability()

    def _refresh_status(self) -> None:
        if self._model is None or not self._model.is_calibrated():
            self._status_lbl.setText("Not calibrated")
            self._status_lbl.setStyleSheet("color: grey; font-style: italic;")
            return
        scale = self._model.scale()
        unit = self._model.unit()
        step = self._model.grid_spacing_world()
        mode = "Auto" if self._model.grid_spacing_auto() else "Fixed"
        self._status_lbl.setText(
            f"Scale: {scale:.4g} {unit}/px\nGrid: {step:g} {unit} ({mode})"
        )
        self._status_lbl.setStyleSheet("color: black; font-style: normal;")

    def _refresh_controls(self) -> None:
        if self._model is None:
            self._refresh_action_availability()
            return
        self._refreshing = True
        grid_visible = self._model.grid_visible() and self._model.is_calibrated()
        self._grid_chk.setChecked(grid_visible)
        opacity_pct = int(self._model.grid_opacity() * 100)
        self._opacity_slider.setValue(opacity_pct)
        self._opacity_lbl.setText(f"{opacity_pct}%")
        self._update_color_swatch(self._model.grid_color())
        auto_spacing = self._model.grid_spacing_auto()
        self._radio_auto.setChecked(auto_spacing)
        self._radio_fixed.setChecked(not auto_spacing)
        self._spacing_edit.setEnabled(not auto_spacing)
        self._spacing_edit.setText(f"{self._model.grid_spacing_world():g}")
        self._unit_lbl.setText(self._model.unit())
        self._refreshing = False
        self._refresh_action_availability()

    def _refresh_measurement(self) -> None:
        if self._model is None:
            self._meas_lbl.setText("Distance: -")
            return
        dist = self._model.measured_distance()
        if dist is None:
            p1, p2 = self._model.meas_points()
            if p1 is not None and p2 is None:
                self._meas_lbl.setText("Distance: click point B...")
            else:
                self._meas_lbl.setText("Distance: -")
            return
        self._meas_lbl.setText(f"Distance: {dist:.4g} {self._model.unit()}")

    def _refresh_action_availability(self) -> None:
        calibrated = self._model is not None and self._model.is_calibrated()
        self._btn_calibrate.setEnabled(self._has_image)
        self._btn_measure.setEnabled(calibrated and self._has_image)
        self._grid_chk.setEnabled(calibrated)
        self._opacity_slider.setEnabled(calibrated)
        self._color_btn.setEnabled(calibrated)
        self._radio_auto.setEnabled(calibrated)
        self._radio_fixed.setEnabled(calibrated)
        self._spacing_edit.setEnabled(calibrated and self._radio_fixed.isChecked())
        self._btn_clear_measurement.setEnabled(calibrated)
        self._btn_reset_calibration.setEnabled(calibrated)
        if not calibrated and self._active_tool == "measure":
            self.set_active_tool("")
            self.tool_selected.emit("")

    def _update_color_swatch(self, rgb: tuple) -> None:
        r, g, b = rgb
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
        )
