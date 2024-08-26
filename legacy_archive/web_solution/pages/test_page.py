from taipy.gui import Markdown


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9


fahrenheit = 100
celsius = fahrenheit_to_celsius(fahrenheit)

test_page = Markdown("""
# Local Callbacks
## Fahrenheit:
<|{fahrenheit}|number|on_change=update_celsius|>

## Celsius:
<|{celsius}|number|active=False|>
""")


def update_celsius(state):
    state.celsius = fahrenheit_to_celsius(state.fahrenheit)
