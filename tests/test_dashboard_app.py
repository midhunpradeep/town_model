from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_explorer_loads_shared_dataset_and_renders_annotations(prepared_dataset: Path) -> None:
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception

    app.sidebar.radio[0].set_value("Image explorer")
    app.sidebar.text_input[0].set_value(str(prepared_dataset))
    app.run()
    assert not app.exception
    assert any(item.value == "Image and annotation explorer" for item in app.subheader)
    assert any(item.label == "Pizza image" for item in app.selectbox)
    assert any("Source-inferred labels:" in item.value for item in app.markdown)
    assert len(app.dataframe) == 1


def test_dashboard_generate_page_is_available_without_triggering_generation() -> None:
    app_path = Path(__file__).parents[1] / "src" / "town_model" / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Generate dataset")
    app.run()
    assert not app.exception
    assert any(item.value == "Explicit dataset generation" for item in app.subheader)
    assert any(item.label == "Validate settings" for item in app.button)
    # The UI offers generation as a separate explicit control; no output is
    # created merely by loading or navigating to the page.
