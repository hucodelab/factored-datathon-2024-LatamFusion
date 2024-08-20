from library.dashboard_page import dashboard_page
from library.presentation_page import presentation_page
from taipy.gui import Gui

root_md = "<|navbar|>"


pages = {"/": root_md, "Presentation": presentation_page, "Dashboard": dashboard_page}


Gui(pages=pages).run(host="0.0.0.0", port=5000, use_reloader=True)
