# main.py —— 直接整份取代
import sys, json, tomllib
from pathlib import Path
from PyQt6 import QtWidgets
import pyqtgraph as pg

from ui.ui_ecg import Ui_MainWindow
from controllers.ecg_controller import ECGController


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # ① 載入 UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # ② 在 chartWidget 裡嵌入一個 pyqtgraph.PlotWidget
        layout = QtWidgets.QVBoxLayout(self.ui.chartWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)

        # ③ 讀取設定（config.toml）
        cfg_path = Path("config.toml").resolve()
        cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        # 如需除錯可暫時打開：
        # print("CONFIG PATH =", cfg_path)
        # print("ADDRESS =", cfg.get("address"))
        # print("ANALOG_CHANNELS =", cfg.get("analog_channels"))

        # ④ 個資預設與「記住上次輸入」
        LAST = Path("data/last_subject.json")
        LAST.parent.mkdir(parents=True, exist_ok=True)

        # 4-1 先帶入 config.toml 的預設
        sub = cfg.get("subject_defaults", {})
        self.ui.nameEdit.setText(str(sub.get("name", "")))
        age = sub.get("age", "")
        self.ui.ageEdit.setText("" if age == "" else str(age))
        sex = sub.get("sex", "男")
        if sex == "男":
            self.ui.maleRadio.setChecked(True)
        elif sex == "女":
            self.ui.femaleRadio.setChecked(True)

        # 4-2 若有上次使用者資料就覆蓋
        if LAST.exists():
            try:
                last = json.loads(LAST.read_text(encoding="utf-8"))
                self.ui.nameEdit.setText(last.get("name", self.ui.nameEdit.text()))
                if str(last.get("age", "")).isdigit():
                    self.ui.ageEdit.setText(str(last.get("age")))
                if last.get("sex", sex) == "男":
                    self.ui.maleRadio.setChecked(True)
                elif last.get("sex", sex) == "女":
                    self.ui.femaleRadio.setChecked(True)
            except Exception:
                pass

        # 4-3 定義：把目前輸入寫回 last_subject.json（供下次自動帶入）
        def _save_subject():
            data = {
                "name": self.ui.nameEdit.text().strip(),
                "age": int(self.ui.ageEdit.text().strip())
                       if self.ui.ageEdit.text().strip().isdigit() else "",
                "sex": "男" if self.ui.maleRadio.isChecked()
                       else ("女" if self.ui.femaleRadio.isChecked() else ""),
            }
            LAST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 在兩顆按鈕上順便「記住輸入」（不影響原有功能）
        self.ui.saveRRButton.clicked.connect(_save_subject)
        self.ui.analyzeHRVButton.clicked.connect(_save_subject)

        # ⑤ 建立 Controller（把 UI 元件丟進去）
        self.controller = ECGController(
            plot_widget=self.plot,
            lbl_rt_hr=self.ui.RealTimeHR,          # 即時心跳 QLabel
            lbl_stable_hr=self.ui.StableHR,        # 穩定心跳 QLabel
            status_bar=self.statusBar(),           # QStatusBar
            btn_save_rr=self.ui.saveRRButton,      # 儲存 RR 按鈕
            btn_analyze=self.ui.analyzeHRVButton,  # HRV 分析按鈕
            cfg=cfg,
        )

        # ⑥ ToolBar 綁定
        tb = self.ui.toolBar
        tb.clear()
        actConnect    = tb.addAction("連線")
        actDisconnect = tb.addAction("斷線")
        actStart      = tb.addAction("開始")
        actStop       = tb.addAction("停止")

        actConnect.triggered.connect(self.controller.connect_device)
        actDisconnect.triggered.connect(self.controller.disconnect_device)
        actStart.triggered.connect(self.controller.start_stream)
        actStop.triggered.connect(self.controller.stop_stream)

        # ⑦ 開程式就直接連線＋開始（若要手動，註解掉這兩行）
        self.controller.connect_device()
        self.controller.start_stream()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec())
