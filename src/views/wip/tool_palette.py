"""
ToolPalette — left tool column for the WIP tab.

Single-column panel with checkable QToolButtons aligned to the top.
Only the polygon tool is active; additional tools can be appended to _TOOLS
in later phases without touching window.py.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QToolButton, QButtonGroup, QSizePolicy, QMenu, QWidgetAction, QSlider, QLabel, QHBoxLayout, QWidget
)

_TOOLS = [
    ("⬠", "polygon", "Polygon"),
]

_SETTINGS = [
    ("◢", "thickness", "Brush Thickness"),
]

class ToolPalette(QFrame):
    """Single-column tool panel.

    Uses QFrame.StyledPanel for a visual border consistent with the rest of
    the workspace. Buttons use the system palette — no custom colors.

    Signals:
        tool_selected (str): Emitted with the tool name when a button is pressed.
        thickess_changed (float): Emitted when the brush thickness slider moves.
    """

    tool_selected = Signal(str)
    thickness_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setFixedWidth(46)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._btn_tool: dict = {}
        self._active_tool: str = ""
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)

        font = QFont()
        font.setPointSize(13)

        for icon, tool_name, tooltip in _TOOLS:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setFixedSize(32, 30)
            btn.setCheckable(True)
            btn.setFont(font)
            self._btn_tool[btn] = tool_name
            self._btn_group.addButton(btn)
            layout.addWidget(btn)

        for icon, setting_name, tooltip in _SETTINGS:
            if setting_name == "thickness":
                btn_thickness = QToolButton()
                btn_thickness.setText(icon)
                btn_thickness.setToolTip(tooltip)
                btn_thickness.setFixedSize(32, 30)
                btn_thickness.setFont(font)
                btn_thickness.setPopupMode(QToolButton.InstantPopup)
                
                thickness_menu = QMenu(self)
                action = QWidgetAction(self)
                
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(8, 4, 8, 4)
                
                # 1-40 step range ensures exact 0.25 intervals up to 10.00
                self.slider_thickness = QSlider(Qt.Horizontal)
                self.slider_thickness.setRange(1, 40) 
                self.slider_thickness.setValue(8)      # 8 * 0.25 = 2.00px Default
                self.slider_thickness.setTickPosition(QSlider.TickPosition.TicksBelow)
                self.slider_thickness.setTickInterval(4) # Tick every 1.0px
                self.slider_thickness.setMinimumWidth(150)
                
                self.lbl_thickness = QLabel("2.00 px")
                self.lbl_thickness.setFixedWidth(55)
                
                h_layout.addWidget(self.slider_thickness)
                h_layout.addWidget(self.lbl_thickness)
                
                action.setDefaultWidget(container)
                thickness_menu.addAction(action)
                btn_thickness.setMenu(thickness_menu)
                
                layout.addWidget(btn_thickness)

                self.slider_thickness.valueChanged.connect(self._on_slider_changed)

        self._btn_group.buttonClicked.connect(self._on_btn_clicked)

    def _on_btn_clicked(self, btn: QToolButton) -> None:
        tool_name = self._btn_tool.get(btn, "")
        if tool_name == self._active_tool:
            self.deselect_all()
            self.tool_selected.emit("")
        else:
            self._active_tool = tool_name
            self.tool_selected.emit(tool_name)
    
    def _on_slider_changed(self, value: int) -> None:
        """Translate the 1-40 step integer to a strict 0.25 interval float and emit."""
        thickness = value * 0.25
        self.lbl_thickness.setText(f"{thickness:.2f} px")
        self.thickness_changed.emit(thickness)

    def deselect_all(self) -> None:
        """Uncheck all tool buttons (called when the canvas cancels a tool)."""
        self._active_tool = ""
        self._btn_group.setExclusive(False)
        for btn in self._btn_tool:
            btn.setChecked(False)
        self._btn_group.setExclusive(True)
