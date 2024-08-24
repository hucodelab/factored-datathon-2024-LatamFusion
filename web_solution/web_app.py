from dotenv import load_dotenv
from pages.dashboard_page import dashboard_page
from pages.presentation_page import presentation_page
from taipy.gui import Gui

# load_dotenv(".env")

root_md = "<|navbar|>"


pages = {"/": root_md, "Presentation": presentation_page, "Dashboard": dashboard_page}


Gui(pages=pages).run(host="0.0.0.0", port=5000, use_reloader=True)
