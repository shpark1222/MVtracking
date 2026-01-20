import os
import sys
from typing import Optional, List

from PySide6 import QtWidgets
import pyqtgraph as pg

from mvpack_io import find_mvpack_in_folder, load_mvpack_h5
from tracker_ui import ValveTracker
from tracking_state import find_mvtrack_files


def _select_work_folder() -> Optional[str]:
    dlg = QtWidgets.QFileDialog()
    dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
    dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
    if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        folders = dlg.selectedFiles()
        if folders:
            return folders[0]
    return None


def _find_mvpack_candidates(folder: str) -> List[str]:
    candidates = []
    direct = os.path.join(folder, "mvpack.h5")
    if os.path.exists(direct):
        candidates.append(direct)

    for root, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith(".h5"):
                continue
            if "mvpack" not in name.lower():
                continue
            candidates.append(os.path.join(root, name))

    unique = sorted(set(candidates), key=lambda p: (len(p), p))
    return unique


def _select_mvpack_path(folder: str) -> str:
    candidates = _find_mvpack_candidates(folder)
    if not candidates:
        return find_mvpack_in_folder(folder)
    if len(candidates) == 1:
        return candidates[0]

    choices = [os.path.relpath(p, folder) for p in candidates]
    choice, ok = QtWidgets.QInputDialog.getItem(
        None,
        "Select mvpack",
        "Select an mvpack file to load:",
        choices,
        0,
        False,
    )
    if not ok or not choice:
        raise FileNotFoundError("No mvpack selected.")
    idx = choices.index(choice)
    return candidates[idx]


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major")

    folder = _select_work_folder()
    if folder is None:
        QtWidgets.QMessageBox.warning(None, "MV tracker", "No folder selected. Exiting.")
        return

    try:
        mvpack_path = _select_mvpack_path(folder)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "MV tracker", f"mvpack.h5 not found:\n{exc}")
        return

    try:
        pack = load_mvpack_h5(mvpack_path)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "MV tracker", f"Failed to load mvpack:\n{exc}")
        return

    mvtrack_candidates = find_mvtrack_files(folder)
    tracking_path = None
    restore_state = False

    if mvtrack_candidates:
        mvtrack_candidates = sorted(
            mvtrack_candidates,
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )

        choices = ["(new tracking)"] + [os.path.basename(p) for p in mvtrack_candidates]
        default_index = 1

        choice, ok = QtWidgets.QInputDialog.getItem(
            None,
            "Select MVtrack",
            "Select an existing MVtrack file to load:",
            choices,
            default_index,
            False,
        )

        if ok and choice and choice != "(new tracking)":
            for p in mvtrack_candidates:
                if os.path.basename(p) == choice:
                    tracking_path = p
                    restore_state = True
                    break
    else:
        restore_state = False

    w = ValveTracker(pack, work_folder=folder, tracking_path=tracking_path, restore_state=restore_state)
    w.resize(1400, 900)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
