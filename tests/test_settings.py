import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setOrganizationName("Gahan AI Test"); app.setApplicationName("Validation Viewer Test")
    from ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.settings.clear(); w.settings.sync()
    w.close()


def test_window_uses_the_application_settings_name(window):
    # test / smoke runs must never write into the real "Gahan AI / Validation Viewer" settings
    assert window.settings.organizationName() == "Gahan AI Test"
    assert window.settings.applicationName() == "Validation Viewer Test"


def test_single_recent_model_is_one_path(window):
    window.settings.setValue("model/recent", ["/models/a.onnx"]); window.settings.sync()
    assert window._recent_models() == ["/models/a.onnx"]
    window.settings.setValue("model/recent", "/models/b.onnx")
    assert window._recent_models() == ["/models/b.onnx"]
