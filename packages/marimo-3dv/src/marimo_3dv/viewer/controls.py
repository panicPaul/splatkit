"""Runtime-neutral viewer controls helpers."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, get_args, get_origin

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

ConfigT = TypeVar("ConfigT", bound=BaseModel)

_DESKTOP_PANEL_WIDTH = 360


def _is_model_type(annotation: Any) -> bool:
    """Return whether an annotation is a Pydantic model type."""
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _is_literal_type(annotation: Any) -> bool:
    """Return whether an annotation is a ``Literal``."""
    return get_origin(annotation) is Literal


class DesktopPydanticControls(Generic[ConfigT]):
    """Desktop-side controls rendered with Qt widgets in a dock."""

    def __init__(
        self,
        model_cls: type[ConfigT],
        *,
        value: ConfigT | dict[str, Any] | None = None,
        label: str = "",
        panel_width: int = _DESKTOP_PANEL_WIDTH,
    ) -> None:
        self._model_cls = model_cls
        self._label = label or "Controls"
        self._panel_width = panel_width
        self._value = self._resolve_initial_value(value)
        self._payload = self._value.model_dump(mode="python")
        self._dock_widget: QDockWidget | None = None
        self._error_label: QLabel | None = None

    @property
    def value(self) -> ConfigT:
        """Return the latest valid model value."""
        return self._value

    @property
    def panel_width(self) -> int:
        """Return the preferred dock width in pixels."""
        return self._panel_width

    @property
    def dock_widget(self) -> QDockWidget | None:
        """Return the attached dock widget, if any."""
        return self._dock_widget

    def attach(self, window: QMainWindow) -> None:
        """Attach the controls panel to a Qt main window."""
        dock = QDockWidget(self._label, window)
        dock.setObjectName("viewer-controls-dock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        dock.setMinimumWidth(self._panel_width)
        dock.setWidget(self._build_root_widget())
        dock.setStyleSheet(_controls_stylesheet())
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._dock_widget = dock

    def shutdown(self) -> None:
        """Release UI resources."""
        if self._dock_widget is not None:
            self._dock_widget.close()
            self._dock_widget.deleteLater()
        self._dock_widget = None
        self._error_label = None

    def _resolve_initial_value(
        self,
        value: ConfigT | dict[str, Any] | None,
    ) -> ConfigT:
        if value is None:
            return self._model_cls()
        if isinstance(value, self._model_cls):
            return value
        return self._model_cls.model_validate(value)

    def _build_root_widget(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        body = self._build_model_widget(self._model_cls, self._payload)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(body)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(scroll_area, 1)

        error_label = QLabel("")
        error_label.setObjectName("controls-error")
        error_label.setWordWrap(True)
        error_label.hide()
        layout.addWidget(error_label)
        self._error_label = error_label
        self._validate_payload()
        return container

    def _build_model_widget(
        self,
        model_cls: type[BaseModel],
        payload: dict[str, Any],
    ) -> QWidget:
        widget = QWidget()
        scalar_fields: list[tuple[str, FieldInfo]] = []
        nested_fields: list[tuple[str, FieldInfo]] = []
        for name, info in model_cls.model_fields.items():
            if _is_model_type(info.annotation):
                nested_fields.append((name, info))
            else:
                scalar_fields.append((name, info))

        if nested_fields and not scalar_fields:
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._build_tabs(nested_fields, payload))
            return widget

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(10)
        for name, info in scalar_fields:
            label = info.title or name.replace("_", " ").capitalize()
            field_widget = self._build_field_widget(
                info.annotation,
                info,
                payload,
                name,
            )
            form_layout.addRow(label, field_widget)
        layout.addLayout(form_layout)

        if nested_fields:
            tabs_box = QGroupBox("Sections")
            tabs_layout = QVBoxLayout(tabs_box)
            tabs_layout.setContentsMargins(8, 12, 8, 8)
            tabs_layout.addWidget(self._build_tabs(nested_fields, payload))
            layout.addWidget(tabs_box)
        layout.addStretch(1)
        return widget

    def _build_tabs(
        self,
        fields: Sequence[tuple[str, FieldInfo]],
        payload: dict[str, Any],
    ) -> QTabWidget:
        tabs = QTabWidget()
        for name, info in fields:
            label = info.title or name.replace("_", " ").capitalize()
            nested_payload = payload.setdefault(name, {})
            assert isinstance(info.annotation, type)
            tabs.addTab(
                self._build_model_widget(info.annotation, nested_payload),
                label,
            )
        return tabs

    def _build_field_widget(
        self,
        annotation: Any,
        info: FieldInfo,
        payload: dict[str, Any],
        name: str,
    ) -> QWidget:
        value = payload.get(name)
        if annotation is bool:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.stateChanged.connect(
                lambda state, *, field_name=name: self._set_payload_value(
                    payload,
                    field_name,
                    state == Qt.CheckState.Checked.value,
                )
            )
            return widget
        if annotation is int:
            widget = QSpinBox()
            lower, upper = _numeric_bounds(info)
            widget.setRange(
                int(lower) if lower is not None else -1_000_000_000,
                int(upper) if upper is not None else 1_000_000_000,
            )
            widget.setValue(int(value))
            widget.valueChanged.connect(
                lambda new_value, *, field_name=name: self._set_payload_value(
                    payload,
                    field_name,
                    int(new_value),
                )
            )
            return widget
        if annotation is float:
            widget = QDoubleSpinBox()
            widget.setDecimals(6)
            lower, upper = _numeric_bounds(info)
            widget.setRange(
                float(lower) if lower is not None else -1e12,
                float(upper) if upper is not None else 1e12,
            )
            widget.setValue(float(value))
            widget.valueChanged.connect(
                lambda new_value, *, field_name=name: self._set_payload_value(
                    payload,
                    field_name,
                    float(new_value),
                )
            )
            return widget
        if annotation is Path:
            line_edit = QLineEdit(str(value))
            line_edit.textChanged.connect(
                lambda text, *, field_name=name: self._set_payload_value(
                    payload,
                    field_name,
                    Path(text),
                )
            )
            return line_edit
        if annotation is str:
            line_edit = QLineEdit(str(value))
            line_edit.textChanged.connect(
                lambda text, *, field_name=name: self._set_payload_value(
                    payload,
                    field_name,
                    text,
                )
            )
            return line_edit
        if _is_literal_type(annotation):
            return self._build_choice_widget(
                payload,
                name,
                list(get_args(annotation)),
                value,
            )
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return self._build_choice_widget(
                payload,
                name,
                list(annotation),
                value,
            )
        fallback = QLineEdit(str(value))
        fallback.textChanged.connect(
            lambda text, *, field_name=name: self._set_payload_value(
                payload,
                field_name,
                text,
            )
        )
        return fallback

    def _build_choice_widget(
        self,
        payload: dict[str, Any],
        name: str,
        options: Sequence[Any],
        value: Any,
    ) -> QWidget:
        combo = QComboBox()
        for option in options:
            label = str(option.value if hasattr(option, "value") else option)
            combo.addItem(label, option)
        current_index = next(
            (index for index, option in enumerate(options) if option == value),
            0,
        )
        combo.setCurrentIndex(current_index)
        combo.currentIndexChanged.connect(
            lambda index, *, field_name=name: self._set_payload_value(
                payload,
                field_name,
                combo.itemData(index),
            )
        )
        return combo

    def _set_payload_value(
        self,
        payload: dict[str, Any],
        name: str,
        value: Any,
    ) -> None:
        payload[name] = value
        self._validate_payload()

    def _validate_payload(self) -> None:
        try:
            self._value = self._model_cls.model_validate(self._payload)
        except Exception as exception:
            if self._error_label is not None:
                self._error_label.setText(str(exception))
                self._error_label.show()
        else:
            if self._error_label is not None:
                self._error_label.clear()
                self._error_label.hide()


def _numeric_bounds(
    info: FieldInfo,
) -> tuple[int | float | None, int | float | None]:
    """Extract inclusive numeric bounds from Pydantic metadata."""
    lower: int | float | None = None
    upper: int | float | None = None
    for item in info.metadata:
        if isinstance(item, annotated_types.Ge):
            lower = item.ge
        elif isinstance(item, annotated_types.Gt):
            lower = item.gt
        elif isinstance(item, annotated_types.Le):
            upper = item.le
        elif isinstance(item, annotated_types.Lt):
            upper = item.lt
        elif isinstance(item, annotated_types.Interval):
            if item.ge is not None:
                lower = item.ge
            elif item.gt is not None:
                lower = item.gt
            if item.le is not None:
                upper = item.le
            elif item.lt is not None:
                upper = item.lt
    return lower, upper


def _controls_stylesheet() -> str:
    """Return a minimal stylesheet for the docked desktop controls."""
    return """
    QDockWidget {
        font-size: 13px;
    }
    QDockWidget::title {
        background: #f5f7fb;
        border-bottom: 1px solid #d9e0ea;
        padding: 8px 10px;
        text-align: left;
    }
    QWidget {
        background: #fbfcfe;
        color: #132032;
    }
    QGroupBox {
        border: 1px solid #d9e0ea;
        border-radius: 10px;
        margin-top: 10px;
        padding-top: 8px;
        font-weight: 600;
    }
    QTabWidget::pane {
        border: 1px solid #d9e0ea;
        border-radius: 8px;
        background: #ffffff;
    }
    QTabBar::tab {
        padding: 8px 12px;
        background: #eef3f9;
        border: 1px solid #d9e0ea;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        margin-right: 4px;
    }
    QTabBar::tab:selected {
        background: #ffffff;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background: #ffffff;
        border: 1px solid #c6d0dd;
        border-radius: 6px;
        padding: 6px 8px;
        min-height: 18px;
    }
    QCheckBox {
        padding: 4px 0;
    }
    QLabel#controls-error {
        color: #b42318;
        background: #fef3f2;
        border: 1px solid #fecdca;
        border-radius: 8px;
        padding: 8px 10px;
    }
    """
