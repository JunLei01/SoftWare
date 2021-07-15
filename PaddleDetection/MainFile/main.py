import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from SWTargetTracking import *


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    myMenuWindow = menu_window()
    myMainWindow = TrackingMainWindow()
    myMenuWindow.show()

    # ui.Main(args)
    sys.exit(app.exec_())
