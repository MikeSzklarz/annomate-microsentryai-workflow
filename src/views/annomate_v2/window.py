"""
AnnoMate V2 — Experimental Photoshop-style wireframe.

This is a non-functional mockup tab. No existing functionality is altered.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QSizePolicy, QButtonGroup, QGridLayout, QToolButton,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPixmap, QPainter, QPen

from views.annomate.widgets import CustomSplitter
from views.annomate.styles import SPLITTER_STYLE

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

_STATUS_BADGE_COLORS = {
    "In Review": "#E6A817",
    "Reviewed":  "#4A90E2",
    "Completed": "#5CB85C",
}


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

class _Thumbnail(QWidget):
    """Mock thumbnail card for the dataset navigator grid."""

    def __init__(self, idx: int, status: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(104, 92)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 2)
        layout.setSpacing(2)

        img_lbl = QLabel()
        img_lbl.setFixedHeight(62)
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        img_lbl.setPixmap(self._make_thumb(idx))
        layout.addWidget(img_lbl)

        badge = QLabel(status)
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedHeight(16)
        color = _STATUS_BADGE_COLORS.get(status, "#888888")
        badge.setStyleSheet(
            f"font-size:9px; color:#FFFFFF; background:{color};"
            f"border-radius:6px; padding:0 4px;"
        )
        layout.addWidget(badge)

    @staticmethod
    def _make_thumb(idx: int) -> QPixmap:
        px = QPixmap(88, 58)
        hue = (idx * 47) % 360
        px.fill(QColor.fromHsv(hue, 55, 180))
        p = QPainter(px)
        p.setPen(QPen(QColor.fromHsv(hue, 80, 80), 1))
        p.setFont(QFont("Monospace", 7))
        p.drawText(px.rect(), Qt.AlignCenter, f"img_{idx:03d}.png")
        p.end()
        return px


class _CollapsibleSection(QWidget):
    """Panel section with a toggle-able header and collapsible body."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toggle = QPushButton(f"▾  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setStyleSheet("text-align: left; font-weight: bold; padding: 5px 10px;")
        self._toggle.clicked.connect(self._on_toggle)
        root.addWidget(self._toggle)

        # Thin separator line under header
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 6, 8, 8)
        self._body_layout.setSpacing(4)
        root.addWidget(self._body)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def _on_toggle(self, checked: bool):
        self._body.setVisible(checked)
        text = self._toggle.text()[3:]  # strip "▾  " or "▸  "
        self._toggle.setText(f"{'▾' if checked else '▸'}  {text}")


# ---------------------------------------------------------------------------
# Main V2 window
# ---------------------------------------------------------------------------

class AnnoMateV2Window(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_tool = "Polygon"
        self._zoom = 100

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar — spans full width
        root.addWidget(self._build_top_bar())

        # Middle row: left bar + (canvas | right panel via splitter)
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        mid.addWidget(self._build_left_bar())
        mid.addWidget(self._build_center_splitter(), stretch=1)
        root.addLayout(mid, stretch=1)

        # Status bar — spans full width
        root.addWidget(self._make_hline())
        root.addWidget(self._build_status_bar())

        self._res_timer = QTimer(self)
        self._res_timer.timeout.connect(self._update_resources)
        self._res_timer.start(2000)

    # ------------------------------------------------------------------
    # Top bar
    # ------------------------------------------------------------------

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(32)

        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 0, 8, 0)
        h.setSpacing(5)

        for label, tip in (("📂 Open", "Open folder"),
                           ("💾 Save", "Save annotations"),
                           ("↗ Export", "Export dataset")):
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setToolTip(tip)
            h.addWidget(btn)

        h.addStretch()

        self._file_label = QLabel("No folder loaded")
        self._file_label.setStyleSheet("color: gray;")
        h.addWidget(self._file_label)

        return bar

    # ------------------------------------------------------------------
    # Left tool bar (middle-section only, aligned top, fixed width)
    # ------------------------------------------------------------------

    def _build_left_bar(self) -> QWidget:
        # Outer frame gives a visual panel look
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        frame.setFixedWidth(46)

        v = QVBoxLayout(frame)
        v.setContentsMargins(4, 6, 4, 6)
        v.setSpacing(3)
        v.setAlignment(Qt.AlignTop)

        group = QButtonGroup(self)
        group.setExclusive(True)

        tools = [
            ("➤", "Select / Pan"),
            ("⬠", "Polygon"),
            ("▭", "Rectangle"),
            ("○", "Ellipse"),
        ]
        for symbol, name in tools:
            btn = QToolButton()
            btn.setText(symbol)
            btn.setToolTip(name)
            btn.setFixedSize(32, 30)
            btn.setCheckable(True)
            btn.setFont(QFont("Noto Sans", 13))
            if name == "Polygon":
                btn.setChecked(True)
            btn.toggled.connect(lambda on, n=name: self._set_tool(n) if on else None)
            group.addButton(btn)
            v.addWidget(btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        v.addWidget(sep)

        for symbol, tip, handler in (
            ("＋", "Zoom In",  lambda: self._handle_zoom("in")),
            ("－", "Zoom Out", lambda: self._handle_zoom("out")),
            ("⊡", "Fit",      lambda: self._handle_zoom("fit")),
        ):
            btn = QToolButton()
            btn.setText(symbol)
            btn.setToolTip(tip)
            btn.setFixedSize(32, 28)
            btn.setFont(QFont("Noto Sans", 12))
            btn.clicked.connect(handler)
            v.addWidget(btn)

        return frame

    # ------------------------------------------------------------------
    # Canvas + right panel via splitter
    # ------------------------------------------------------------------

    def _build_center_splitter(self) -> CustomSplitter:
        splitter = CustomSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)

        # Canvas area
        canvas_frame = QFrame()
        canvas_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        placeholder = QLabel("Open a folder to load images")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: gray; font-size: 14px;")
        canvas_layout.addWidget(placeholder)

        splitter.addWidget(canvas_frame)
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([900, 300])
        return splitter

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _build_right_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(240)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        cl.addWidget(self._build_dataset_navigator())
        cl.addWidget(self._build_class_panel())
        cl.addWidget(self._build_image_info_panel())
        cl.addStretch()

        scroll.setWidget(content)
        return scroll

    def _build_dataset_navigator(self) -> _CollapsibleSection:
        sec = _CollapsibleSection("Dataset Navigator")

        # Filter chips
        chip_row = QHBoxLayout()
        chip_row.setSpacing(4)
        chip_group = QButtonGroup(self)
        chip_group.setExclusive(True)

        for i, label in enumerate(("All", "In Review", "Reviewed", "Completed")):
            chip = QPushButton(label)
            chip.setCheckable(True)
            chip.setFixedHeight(20)
            chip.setStyleSheet("font-size: 10px; padding: 1px 6px;")
            if i == 0:
                chip.setChecked(True)
            chip_group.addButton(chip)
            chip_row.addWidget(chip)

        chip_row.addStretch()
        sec.body_layout().addLayout(chip_row)

        # Thumbnail grid (2 columns)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setSpacing(6)

        statuses = list(_STATUS_BADGE_COLORS.keys())
        for i in range(12):
            thumb = _Thumbnail(i + 1, statuses[i % len(statuses)])
            grid.addWidget(thumb, i // 2, i % 2)

        sec.body_layout().addWidget(grid_widget)
        return sec

    def _build_class_panel(self) -> _CollapsibleSection:
        sec = _CollapsibleSection("Annotation Classes")

        classes = [
            ("Defect",     "#E05252"),
            ("Anomaly",    "#E6A817"),
            ("Suspect",    "#4A90E2"),
            ("Background", "#5CB85C"),
        ]
        for name, color in classes:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            dot.setFixedWidth(18)
            lbl = QLabel(name)
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            cnt = QLabel("0")
            cnt.setStyleSheet("color: gray; font-size: 10px;")
            row.addWidget(cnt)
            sec.body_layout().addLayout(row)

        return sec

    def _build_image_info_panel(self) -> _CollapsibleSection:
        sec = _CollapsibleSection("Image Info")

        for key, val in (("File", "—"), ("Resolution", "—"),
                         ("Format", "—"), ("Annotations", "0")):
            row = QHBoxLayout()
            k = QLabel(key)
            k.setStyleSheet("color: gray; font-size: 10px;")
            v = QLabel(val)
            v.setStyleSheet("font-size: 11px;")
            row.addWidget(k)
            row.addStretch()
            row.addWidget(v)
            sec.body_layout().addLayout(row)

        return sec

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(22)

        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 0, 8, 0)
        h.setSpacing(0)

        dim = "color: gray; font-size: 10px; padding: 0 6px;"

        self._tool_label = QLabel(f"Tool: {self._active_tool}")
        self._tool_label.setStyleSheet(dim)
        h.addWidget(self._tool_label)

        self._xy_label = QLabel("XY: 0, 0")
        self._xy_label.setStyleSheet(dim)
        h.addWidget(self._xy_label)

        self._res_label = QLabel("Res: —")
        self._res_label.setStyleSheet(dim)
        h.addWidget(self._res_label)

        self._zoom_label = QLabel(f"Zoom: {self._zoom}%")
        self._zoom_label.setStyleSheet(dim)
        h.addWidget(self._zoom_label)

        h.addStretch()

        self._save_label = QLabel("Autosaved: —")
        self._save_label.setStyleSheet(dim)
        h.addWidget(self._save_label)

        self._cpu_label = QLabel("CPU: —")
        self._cpu_label.setStyleSheet(dim)
        h.addWidget(self._cpu_label)

        self._ram_label = QLabel("RAM: —")
        self._ram_label.setStyleSheet(dim)
        h.addWidget(self._ram_label)

        self._gpu_label = QLabel("GPU: —")
        self._gpu_label.setStyleSheet(dim)
        h.addWidget(self._gpu_label)

        return bar

    # ------------------------------------------------------------------
    # Shared separator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    @staticmethod
    def _make_vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedWidth(6)
        return line

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _set_tool(self, name: str):
        self._active_tool = name
        self._tool_label.setText(f"Tool: {name}")

    def _handle_zoom(self, direction: str):
        if direction == "in":
            self._zoom = min(self._zoom + 10, 400)
        elif direction == "out":
            self._zoom = max(self._zoom - 10, 10)
        else:
            self._zoom = 100
        self._zoom_label.setText(f"Zoom: {self._zoom}%")

    def _update_resources(self):
        if HAS_PSUTIL:
            self._cpu_label.setText(f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
            self._ram_label.setText(f"RAM: {psutil.virtual_memory().percent:.0f}%")
