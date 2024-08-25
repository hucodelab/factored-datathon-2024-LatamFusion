import os

from taipy.gui import Markdown

# Get path of the current file
path = os.path.abspath(__file__)

presentation_page = Markdown("pages/presentation_page.md")
