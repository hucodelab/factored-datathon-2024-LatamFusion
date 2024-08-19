# presentation_page = "# About us"

import os

# Get path of the current file
path = os.path.abspath(__file__)

# Read file presentation_page.md from the same directory
with open(os.path.join(os.path.dirname(path), "presentation_page.md")) as f:
    presentation_page = f.read()
