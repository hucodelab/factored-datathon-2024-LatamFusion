from pages.dashboard_page import dashboard_page
from pages.presentation_page import presentation_page
from pages.root import root_md
from pages.test_page import test_page
from taipy.gui import Gui

pages = {
    "/": root_md,
    "Presentation": presentation_page,
    "Dashboard": dashboard_page,
    "Test": test_page,
}


Gui(pages=pages).run(host="0.0.0.0", port=5000, debug=True, title="GDELT Dashboard")
