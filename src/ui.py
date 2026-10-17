import string

from pathlib import Path

import moderngl_window
import moderngl_window.context.pyqt5.window as qtw

from PyQt5 import QtOpenGL, QtWidgets
from PyQt5.QtCore import QSize, Qt, QTimer, QRect
from PyQt5.QtGui import QScreen, QColor, QFontMetrics
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot, QEvent

from config import Config, ConfigVal

from const import WINDOW_SIZE_X, WINDOW_SIZE_Y

_g_ui_widget = None

_g_scaling_factor = 1
def update_scaling_factor(app: QtWidgets.QApplication):
    global _g_scaling_factor

    # Make a test label
    alphabet = string.ascii_lowercase[:14]
    test_label = QtWidgets.QLabel(alphabet)
    test_label.setStyleSheet(app.styleSheet())
    test_label.ensurePolished()

    font_height = test_label.fontMetrics().height()

    _g_scaling_factor = font_height / 13

    print("Scaling factor updated to", _g_scaling_factor)

def get_scaling_factor():
    return _g_scaling_factor

def set_target_size(widget: QtWidgets.QWidget):
    base_size = QSize(*widget.SIZE)
    base_size.setWidth(round(base_size.width() * get_scaling_factor()))
    base_size.setHeight(round(base_size.height() * get_scaling_factor()))
    min_size = widget.sizeHint()

    #if widget.layout() is not None:
    #    min_size = widget.layout().sizeHint()
    #    min_size += QSize(widget.layout().spacing(), widget.layout().spacing()) * 2

    size = QSize(max(base_size.width(), min_size.width()), max(base_size.height(), min_size.height()))
    widget.setFixedSize(size)
    widget.resize(size)

class QConfigVal(QWidget):
    FLOAT_SLIDER_PREC = 100

    def __init__(self, name: str, config_val: ConfigVal, on_change=None):
        QWidget.__init__(self)

        self.name = name
        self.config_val = config_val
        self.on_change = None           # set after the initial label fill (no save while building)

        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        self.label = QtWidgets.QLabel("...")

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setFocusPolicy(Qt.NoFocus)          # keys keep going to the vis
        self.slider.setFixedHeight(round(10 * get_scaling_factor()))

        self.float_mode = (config_val.max - config_val.min) < 10

        if self.float_mode:
            self.slider.setRange(0, self.FLOAT_SLIDER_PREC)
            val_frac = (config_val.val - config_val.min) / (config_val.max - config_val.min)
            self.slider.setValue(round(val_frac * self.FLOAT_SLIDER_PREC))
        else:
            self.slider.setRange(round(config_val.min), round(config_val.max))
            self.slider.setValue(round(config_val.val))

        self.slider.valueChanged.connect(self.on_val_changed)

        self.on_val_changed()
        self.on_change = on_change

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

    def get_beautified_name(self):
        if self.name in Config.SOUND_MIX:
            return Config.SOUND_MIX[self.name]
        n = self.name[len("camera_"):] if self.name.startswith("camera_") else self.name
        return n.replace("_", " ").capitalize().replace("Fov", "FOV")

    @pyqtSlot()
    def on_val_changed(self):
        if self.float_mode:
            slider_frac = self.slider.value() / self.FLOAT_SLIDER_PREC
            self.config_val.val = self.config_val.min + (self.config_val.max - self.config_val.min) * slider_frac
        else:
            self.config_val.val = self.slider.value()
        dec = self.config_val.decimals
        shown = ("{:.%df}" % dec).format(self.config_val.val) if dec is not None else str(self.config_val.val)
        self.label.setText(self.get_beautified_name() + ": " + shown)
        if self.on_change is not None:
            self.on_change()

class QConfigChoice(QWidget):
    """A labelled drop-down for one Config.GRAPHICS entry (stored as a plain attribute on Config)."""

    def __init__(self, config: Config, name: str, on_change=None):
        QWidget.__init__(self)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)
        self.config, self.name = config, name
        label, self.choices, _d = Config.GRAPHICS[name]
        lay = QtWidgets.QVBoxLayout(self)
        lay.setAlignment(Qt.AlignTop)
        lay.addWidget(QtWidgets.QLabel(label))
        self.combo = QtWidgets.QComboBox(self)
        self.combo.setFocusPolicy(Qt.NoFocus)          # keys keep going to the vis
        for _v, text in self.choices:
            self.combo.addItem(text)
        cur = getattr(config, name)
        self.combo.setCurrentIndex([c[0] for c in self.choices].index(cur))
        self.combo.currentIndexChanged.connect(self.on_changed)
        self.on_change = on_change
        lay.addWidget(self.combo)

    @pyqtSlot(int)
    def on_changed(self, idx):
        setattr(self.config, self.name, self.choices[idx][0])
        if self.on_change is not None:
            self.on_change()

    def refresh(self):
        """Show the config's current value (after a preset changed it) without re-triggering on_change."""
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex([c[0] for c in self.choices].index(getattr(self.config, self.name)))
        self.combo.blockSignals(False)


class QConfigLevel(QWidget):
    """A labelled Low / Medium / High (...) slider for one Config.QUALITY entry (a level index on Config)."""

    def __init__(self, config: Config, name: str, on_change=None):
        QWidget.__init__(self)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)
        self.config, self.name = config, name
        self.title, self.levels, _d = Config.QUALITY[name]
        lay = QtWidgets.QVBoxLayout(self)
        lay.setAlignment(Qt.AlignTop)
        self.label = QtWidgets.QLabel()
        lay.addWidget(self.label)
        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setFocusPolicy(Qt.NoFocus)          # keys keep going to the vis
        self.slider.setRange(0, len(self.levels) - 1)
        self.slider.setPageStep(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        lay.addWidget(self.slider)
        self.on_change = on_change
        self.refresh()
        self.slider.valueChanged.connect(self.on_changed)

    def refresh(self):
        v = int(getattr(self.config, self.name))
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.label.setText("%s: %s" % (self.title, self.levels[v]))

    @pyqtSlot(int)
    def on_changed(self, v):
        setattr(self.config, self.name, int(v))
        self.label.setText("%s: %s" % (self.title, self.levels[int(v)]))
        if self.on_change is not None:
            self.on_change()


class QEditConfigWidget(QWidget):
    SIZE = (300, 640)

    def __init__(self, config: Config):
        QWidget.__init__(self)

        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)

        # everything lives in a scroll area: the panel is capped to the window height and scrolls
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setFocusPolicy(Qt.NoFocus)                  # arrow keys switch maps, never scroll the panel
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setAttribute(Qt.WA_StyledBackground)
        self.body = QtWidgets.QVBoxLayout(inner)
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll)

        self.text_label = QtWidgets.QLabel("Settings:\n")
        self.body.addWidget(self.text_label)

        self.config = config

        self.camera_group = QtWidgets.QGroupBox("Camera")
        self.camera_group_layout = QtWidgets.QVBoxLayout()
        self.camera_group.setLayout(self.camera_group_layout)
        self.body.addWidget(self.camera_group)

        for name, obj in self.config.__dict__.items():
            if isinstance(obj, ConfigVal):
                config_val = obj # type: ConfigVal

                if not name.startswith("camera_"):
                    continue
                widget = QConfigVal(name, config_val, on_change=self.config.save)
                self.camera_group_layout.addWidget(widget)

        # Audio (persisted with the camera settings in rsv_settings.json; M toggles mute)
        self.audio_group = QtWidgets.QGroupBox("Audio")
        self.audio_group_layout = QtWidgets.QVBoxLayout()
        self.audio_group.setLayout(self.audio_group_layout)
        self.volume_widget = QConfigVal("volume", self.config.volume, on_change=self.config.save)
        self.volume_widget.slider.setFocusPolicy(Qt.NoFocus)   # keys keep going to the vis
        self.audio_group_layout.addWidget(self.volume_widget)
        for name in Config.SOUND_MIX:
            w = QConfigVal(name, getattr(self.config, name), on_change=self.config.save)
            w.slider.setFocusPolicy(Qt.NoFocus)
            self.audio_group_layout.addWidget(w)
        import os as _os
        from const import DATA_DIR_PATH as _DATA
        if _os.path.exists(_os.path.join(_DATA, "sounds", "manifest.json")):
            self.body.addWidget(self.audio_group)
        else:
            self.audio_group.hide()                  # build without sounds: nothing to adjust

        # Graphics (applied live; persisted with the rest)
        self.gfx_group = QtWidgets.QGroupBox("Graphics")
        self.gfx_group_layout = QtWidgets.QVBoxLayout()
        self.gfx_group.setLayout(self.gfx_group_layout)
        # a quality preset sets every slider below (+ anti-aliasing); moving any of them afterwards shows "Custom"
        self._gfx_widgets = {}

        def on_preset():
            self.config.apply_preset(self.config.gfx_preset)
            for w in self._gfx_widgets.values():
                w.refresh()
            self.config.save()

        def on_setting():
            self.config.gfx_preset = self.config.matching_preset()
            self._gfx_widgets["gfx_preset"].refresh()
            self.config.save()

        for name in Config.GRAPHICS:
            w = QConfigChoice(self.config, name, on_change=on_preset if name == "gfx_preset" else on_setting)
            self._gfx_widgets[name] = w
            self.gfx_group_layout.addWidget(w)
        vq = QtWidgets.QLabel("Visual quality")
        vq.setStyleSheet("font-weight: bold; margin-top: 6px")
        self.gfx_group_layout.addWidget(vq)
        for name in Config.QUALITY:
            w = QConfigLevel(self.config, name, on_change=on_setting)
            self._gfx_widgets[name] = w
            self.gfx_group_layout.addWidget(w)
        self.body.addWidget(self.gfx_group)

        self.footer_label = QtWidgets.QLabel("\n(Same meaning as Rocket League's camera settings;\n"
                                             "saved automatically. Click outside to close.)")
        # TODO: Kinda hacky, ideally use setDisabled(True) and add disabled color to stylesheet?
        self.footer_label.setStyleSheet("color: gray")
        self.body.addWidget(self.footer_label)

        set_target_size(self)
        # the scroll area's size hint is huge; pin the designed size (height is capped to the window on open)
        self.setFixedSize(round(self.SIZE[0] * get_scaling_factor()), round(self.SIZE[1] * get_scaling_factor()))

    def update(self):
        super().update()

class QUIBarWidget(QWidget):
    SIZE = (190, 100)

    def __init__(self, parent_window):
        QWidget.__init__(self)

        self.config_edit_popup = None

        self.parent_window = parent_window

        self.setAttribute(Qt.WA_StyledBackground)
        self.setAutoFillBackground(True)

        vbox = QtWidgets.QFormLayout()

        self.text_label = QtWidgets.QLabel("...")
        vbox.addWidget(self.text_label)

        self.edit_config_button = QtWidgets.QPushButton("Edit Settings")
        self.edit_config_button.clicked.connect(self.on_edit_config)
        # NoFocus so pressing Space (a gameplay bind: ball cam) never activates this button and pops
        # the camera-settings menu. It's still clickable.
        self.edit_config_button.setFocusPolicy(Qt.NoFocus)
        vbox.addWidget(self.edit_config_button)

        self.setLayout(vbox)

        set_target_size(self)

        global _g_ui_widget
        _g_ui_widget = self

    def update(self):
        super().update()

    @pyqtSlot()
    def on_edit_config(self):
        self.parent_window.toggle_edit_config()

    def set_text(self, text: str):
        self.text_label.setText(text)

def get_ui() -> QUIBarWidget:
    return _g_ui_widget

class QRSVWindow(QtWidgets.QMainWindow):
    def __init__(self, gl_widget):
        super().__init__()

        self.setWindowTitle("RocketSimVis")

        path = Path(__file__).parent.resolve() / "qt_style_sheet.css"
        self.setStyleSheet(path.read_text())

        # Set the central widget of the Window.
        self.gl_widget = gl_widget
        self.setCentralWidget(self.gl_widget)

        self.base_layout = QtWidgets.QVBoxLayout(self)

        self.bar_widget = QUIBarWidget(self)
        self.layout().addWidget(self.bar_widget)

        self.edit_config_widget = QEditConfigWidget(self.gl_widget.config)
        self.layout().addWidget(self.edit_config_widget)
        self.edit_config_widget.hide()

        self.resize(WINDOW_SIZE_X, WINDOW_SIZE_Y)
        # No window-size cap: the window can be dragged to any size. The 1080p cap is on the actual
        # GL RENDER resolution instead (see QRSVGLWidget._ensure_render_target in main.py) -- a bigger
        # window just gets that capped image upscaled, so cost stays flat past 1080p either way.
        self._norm_geo = None   # last on-screen rect (used to tell auto- from willing-minimize)

        self.installEventFilter(self)
        self.centralWidget().installEventFilter(self)
        # every key pressed anywhere in this window goes to the vis (arrow keys = maps, Space, ...): the settings
        # panel's scroll area / sliders used to swallow them (the arrow keys scrolled the panel instead)
        QtWidgets.QApplication.instance().installEventFilter(self)

    def moveEvent(self, e):
        if not self.isMinimized():
            self._norm_geo = self.frameGeometry()
        super().moveEvent(e)

    def resizeEvent(self, e):
        if not self.isMinimized():
            self._norm_geo = self.frameGeometry()
        super().resizeEvent(e)

    def changeEvent(self, event):
        # Windows auto-minimizes a near-fullscreen OpenGL window when it loses focus to another window.
        # Undo that: if we minimized while the cursor was OUTSIDE our window (you clicked elsewhere),
        # restore us WITHOUT stealing focus. If the cursor was over us (you hit the minimize button),
        # let it minimize as intended.
        if event.type() == QEvent.WindowStateChange and (self.windowState() & Qt.WindowMinimized):
            from PyQt5.QtGui import QCursor
            geo = self._norm_geo
            if geo is None or not geo.contains(QCursor.pos()):
                QTimer.singleShot(0, self._restore_no_activate)
        super().changeEvent(event)

    def _restore_no_activate(self):
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(int(self.winId()), 4)  # SW_SHOWNOACTIVATE: restore, no focus steal
        except Exception:
            self.setWindowState(self.windowState() & ~Qt.WindowMinimized)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and obj in (self, self.centralWidget()):
            if event.button() == Qt.LeftButton:
                press_pos = event.pos()

                # Close config window if we click outside of it
                if self.edit_config_widget.isVisible():
                    if not (press_pos in self.edit_config_widget.geometry()):
                        self.toggle_edit_config()
        elif event.type() == QEvent.KeyPress:
            w = obj.window() if isinstance(obj, QWidget) else None
            if w is self and not isinstance(obj, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
                self.gl_widget.keyPressEvent(event)
                return True                      # handled once (it used to reach the vis twice when focused)

        return super().eventFilter(obj, event)

    def toggle_panel(self):
        """H: show/hide the whole top-left panel (stats + Edit Settings)."""
        if self.bar_widget.isVisible():
            if self.edit_config_widget.isVisible():
                self.toggle_edit_config()
            self.bar_widget.hide()
            self.gl_widget.renderer.show_phys_hud = False
        else:
            self.bar_widget.show()
            self.gl_widget.renderer.show_phys_hud = True

    def toggle_edit_config(self):
        if not self.edit_config_widget.isVisible():
            vw = self.edit_config_widget.volume_widget          # [ / ] keys may have moved it
            vw.slider.blockSignals(True)
            vw.slider.setValue(round(vw.config_val.val))
            vw.slider.blockSignals(False)
            vw.on_val_changed()
            self.edit_config_widget.show()

            sf = get_scaling_factor()
            size = QSize(round(QEditConfigWidget.SIZE[0] * sf), round(QEditConfigWidget.SIZE[1] * sf))

            # Don't exceed the window (the panel opens below the stats bar); the rest scrolls
            top = self.bar_widget.height() + 20
            size.setWidth(min(size.width(), self.width()))
            size.setHeight(max(120, min(size.height(), self.height() - top - 10)))

            self.edit_config_widget.setFixedSize(size)

            self.edit_config_widget.setGeometry(
                0, self.bar_widget.height() + 20,
                size.width(), size.height()
            )
        else:
            self.edit_config_widget.hide()