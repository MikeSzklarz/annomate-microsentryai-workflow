class CenterTemplateState:
    """Pure Python state for center template matching."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.template_file: str = ""
        self.template_path: str = ""
        self.anchor_x: int = 0
        self.anchor_y: int = 0
        self.crop_shape: str = "circle"
        self.crop_width: int = 1210
        self.crop_height: int = 1210
        self.center_x: float | None = None
        self.center_y: float | None = None
        self.last_score: float | None = None

    def has_template(self) -> bool:
        return bool(self.template_path or self.template_file)

    def clear(self) -> None:
        self.enabled = False
        self.template_file = ""
        self.template_path = ""
        self.anchor_x = 0
        self.anchor_y = 0
        self.crop_shape = "circle"
        self.crop_width = 1210
        self.crop_height = 1210
        self.center_x = None
        self.center_y = None
        self.last_score = None
