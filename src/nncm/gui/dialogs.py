"""Dialogs — where the model explains itself.

A table can only fit an abbreviation, so nothing cryptic is edited in place:
the row carries one button, and this is what it opens. Every setting here gets
its full name, its unit and a sentence saying what changing it does.

Validation runs on Save and reports the *first* problem inline, above the
buttons. Bad input never closes the dialog and never opens a second window to
complain about the first.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLineEdit,
    QScrollArea,
    QWidget,
)

from .. import theme as T
from ..config import Material, Range, format_composition, parse_composition
from . import layout as ly
from .theme import restyle
from .widgets import Card, Form, RangeField, choice_field, decimal_field

PROPERTY_FIELDS = [
    ("molecular_weight", 0.0, 1000.0, 3),
    ("normal_boiling_point_K", 0.0, 2000.0, 2),
    ("critical_temperature_K", 0.0, 3000.0, 2),
    ("critical_pressure_bara", 0.0, 1000.0, 2),
    ("lfl_vol_frac", 0.0, 1.0, 4),
]

class Section(Card):
    """A titled card in a dialog's scrolling column; the blurb is its heading's hover tip."""

    def __init__(self, title: str, blurb: str = "", parent: QWidget | None = None):
        super().__init__(title, parent=parent, tip=blurb)
        self.form = Form()
        self.add(self.form)

class MaterialDialog(QDialog):
    """Everything about one material, named in full."""

    def __init__(self, material: Material, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Material — {material.name}" if material.name else "Material")
        # Derived from the content: the Composition field holds a component
        # list ("METHANE 90, ETHANE 7, PROPANE 3") that must stay readable, and
        # the four Conditions rows put two value fields and a checkbox on one
        # line. Narrower than this and one of them starts clipping.
        self.setMinimumWidth(560)
        self.resize(620, 760)
        self._material = material

        outer = ly.window_layout(self)
        outer.setSpacing(T.SPACING_ROW)

        column = QWidget()
        column.setObjectName("Body")
        stack = ly.vbox(column, spacing=T.SPACING_GROUP)
        stack.setContentsMargins(0, 0, T.SPACING_ROW, 0)

        # -- identity -----------------------------------------------------
        identity = Section(
            "Identity",
            "The name is written into the Phast study and has to resolve there. "
            "Spell it as the property system does, e.g. NITROGEN (ASPHYXIATING).",
        )
        self.name = QLineEdit(material.name)
        self.phase = _phase_box(material.phase_hint)
        self.weight = decimal_field(0.0, 1000.0, material.weight, 2)
        identity.form.add("name", self.name, span=True)
        identity.form.add("weight", self.weight)
        identity.form.add("material_phase_hint", self.phase, span=True)
        stack.addWidget(identity)

        # -- composition --------------------------------------------------
        composition = Section(
            "Composition",
            "A material with no components is a pure component and is declared "
            "on the COMPONENT sheet. Give components and it becomes a mixture, "
            "declared on the MIXTURE sheet with one row each.",
        )
        self.composition = QLineEdit(format_composition(material))
        self.composition.setPlaceholderText("METHANE 90, ETHANE 7, PROPANE 3")
        self.basis = _basis_box(material.composition_basis)
        composition.form.add("composition", self.composition, span=True)
        composition.form.add("composition_basis", self.basis, span=True)
        stack.addWidget(composition)

        # -- conditions ---------------------------------------------------
        conditions = Section(
            "Conditions",
            "Per-material ranges override the design ranges. Keeping each fluid "
            "inside a sensible envelope is the single biggest lever on how "
            "physically valid the generated study is.",
        )
        self.temperature_on = QCheckBox("Give this material its own temperature range")
        self.temperature = RangeField(-273.0, 2000.0)
        self.pressure_on = QCheckBox("Give this material its own pressure range")
        self.pressure = RangeField(0.0, 5000.0)
        conditions.form.add_widget(self.temperature_on)
        conditions.form.add("temperature", self.temperature, span=True)
        conditions.form.add_widget(self.pressure_on)
        conditions.form.add("pressure", self.pressure, span=True)
        self.temperature_on.toggled.connect(self.temperature.setEnabled)
        self.pressure_on.toggled.connect(self.pressure.setEnabled)
        self._load_range(self.temperature_on, self.temperature, material.temperature, -50.0, 50.0)
        self._load_range(self.pressure_on, self.pressure, material.pressure, 1.0, 100.0)
        stack.addWidget(conditions)

        # -- properties ---------------------------------------------------
        properties = Section(
            "Properties",
            "Numeric descriptors become model inputs. They are what lets one "
            "network generalise across materials instead of memorising their "
            "names — a material with none falls back to one-hot identity, which "
            "cannot generalise at all. Leave a field blank if it is unknown.",
        )
        self.properties: dict[str, object] = {}
        for key, low, high, decimals in PROPERTY_FIELDS:
            field = decimal_field(low, high, material.properties.get(key, 0.0), decimals)
            field.setSpecialValueText("unknown")  # zero is not a molecular weight
            properties.form.add(key, field)
            self.properties[key] = field
        stack.addWidget(properties)
        stack.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(column)
        outer.addWidget(scroll, 1)

        # -- footer -------------------------------------------------------
        # The first problem is reported here, above the buttons: bad input
        # never closes the dialog and never opens a second window to complain
        # about the first.
        self._problem = ly.status_label()
        outer.addWidget(self._problem)

        # A button box, so the buttons arrive in the host platform's own order.
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        save = self._buttons.button(QDialogButtonBox.Save)
        save.setText("Save this material")  # the verb, not "OK"
        save.setProperty("variant", "primary")
        save.setDefault(True)
        restyle(save)
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _load_range(box: QCheckBox, field: RangeField, value: Range | None, low: float, high: float) -> None:
        box.setChecked(value is not None)
        field.setEnabled(value is not None)
        if value is None:
            field.set_values(low, high, False)
        else:
            field.set_values(value.min, value.max, value.log)

    def _fail(self, message: str) -> None:
        ly.set_status(self._problem, message, "error")

    # -- result ------------------------------------------------------------
    def _save(self) -> None:
        ly.set_status(self._problem, "")
        name = self.name.text().strip()
        if not name:
            self._fail("Material needs a name — it is how Phast and the case table refer to it.")
            return
        try:
            components = parse_composition(self.composition.text())
        except ValueError as exc:
            self._fail(f"Composition: {exc}")
            return

        properties: dict[str, float] = {}
        for key, field in self.properties.items():
            value = field.value()
            if value > field.minimum():
                properties[key] = value

        material = Material(
            name=name,
            phase_hint=self.phase.currentText(),
            components=components,
            composition_basis=self.basis.currentText(),
            temperature=_range_or_none(self.temperature_on, self.temperature),
            pressure=_range_or_none(self.pressure_on, self.pressure),
            weight=self.weight.value(),
            properties=properties,
        )
        try:
            material.validate()
        except ValueError as exc:
            self._fail(str(exc))
            return
        self._material = material
        self.accept()

    def material(self) -> Material:
        return self._material

def _phase_box(current: str):
    return choice_field(["any", "vapour", "liquid"], current or "any")

def _basis_box(current: str):
    return choice_field(["mole", "mass"], current or "mole")

def _range_or_none(box: QCheckBox, field: RangeField) -> Range | None:
    if not box.isChecked():
        return None
    minimum, maximum, log = field.values()
    return Range(minimum, maximum, log and minimum > 0)
