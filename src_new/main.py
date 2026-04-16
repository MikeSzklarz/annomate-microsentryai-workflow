import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from core.logger import setup_logging
setup_logging()
    
from core.state import DatasetState
from models.dataset_model import DatasetTableModel

from controllers.io_controller import IOController
# from controllers.inference_controller import InferenceController
# from controllers.validation_controller import ValidationController

from views.annomate.window import ImageAnnotator
# from views.microsentry.window import MicroSentryWindow
# from views.validation.window import ValidationTab

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnnoMate & MicroSentryAI (MVC)")
        self.resize(1400, 900)

        self.app_state = DatasetState()
        self.dataset_model = DatasetTableModel(self.app_state)

        self.io_controller = IOController(self.dataset_model)
        # self.inference_controller = InferenceController(self.dataset_model)
        # self.validation_controller = ValidationController(self.dataset_model)

        self.annomate_view = ImageAnnotator(self.dataset_model, self.io_controller)
        # self.sentry_view = MicroSentryWindow(self.dataset_model, self.inference_controller)
        # self.validation_view = ValidationTab(self.validation_controller)

        # shared_selection_model = self.annomate_view.table_view.selectionModel()
        # self.sentry_view.table_view.setSelectionModel(shared_selection_model)

        # # Connect view signals (Canvas View syncing, etc.)
        # self._setup_view_syncing()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.annomate_view, "AnnoMate")
        # self.tabs.addTab(self.sentry_view, "MicroSentry AI")
        # self.tabs.addTab(self.validation_view, "Validation")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def _setup_view_syncing(self):
        """
        Sync canvas zooming/panning between AnnoMate and MicroSentry.
        (This remains View-to-View communication because Zoom/Pan is pure UI state, 
        not domain data, so it doesn't belong in the DatasetModel).
        """
        # Example: AnnoMate canvas zooms -> Sentry canvas matches
        self.annomate_view.viewChanged.connect(self.sentry_view.set_view_state)
        
        # Sentry canvas zooms -> AnnoMate canvas matches
        self.sentry_view.viewChanged.connect(self.annomate_view.set_view_state)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = AppWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()