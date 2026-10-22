from cap_tools.widgets import FinalizationWidget


def test_overall_sort_key_handles_numbers_text_and_missing_values():
    assert FinalizationWidget._sort_key("2.5 (3.1)") < FinalizationWidget._sort_key("10.0")
    assert FinalizationWidget._sort_key("alpha") < FinalizationWidget._sort_key("Beta")
    assert FinalizationWidget._sort_key("—") is None
    assert FinalizationWidget._sort_key("") is None
