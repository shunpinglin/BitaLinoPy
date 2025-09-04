# main.py
"""
App entrypoint.
- Loads Qt UI (ui/ui_ecg.py), builds a PlotWidget into chart area
- Wires toolbar actions (Connect/Disconnect/Start/Stop/Auto-connect)
- Instantiates ECGController with config + UI widgets
- On launch: auto-resolve device address (COM/MAC) and try to connect (no streaming yet)
"""
import sys
import json
import tomllib
from pathlib import Path
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

from ui.ui_ecg import Ui_MainWindow
from controllers.ecg_controller import ECGController

# 導入自動連接功能（與 main.py 同層要有 autoconnect_bitalino.py）
try:
    from autoconnect_bitalino import resolve_address, enumerate_candidates
    _HAS_AUTOCONNECT = True
except ModuleNotFoundError:
    _HAS_AUTOCONNECT = False
    print("警告: 找不到 autoconnect_bitalino.py（請確認檔案與 main.py 同層或套件路徑正確）")
except ImportError as e:
    _HAS_AUTOCONNECT = False
    print(f"警告: 載入 autoconnect_bitalino 失敗：{e}")


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # ① 載入 UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("MainWindow")

        # ② 在 chartWidget 裡嵌入一個 pyqtgraph.PlotWidget
        layout = QtWidgets.QVBoxLayout(self.ui.chartWidget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)

        # ✅ 讓 Y 軸自動範圍並留 10% 邊距（上下不會貼邊）
        vb = self.plot.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.setDefaultPadding(0.1)

        # ③ 讀取設定（config.toml）
        cfg_path = Path("config.toml").resolve()
        if not cfg_path.exists():
            QtWidgets.QMessageBox.warning(self, "設定檔缺失",
                                          f"找不到 {cfg_path.name}，將使用內建預設值。")
            cfg = {}
        else:
            try:
                cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as e:
                QtWidgets.QMessageBox.critical(
                    self, "設定檔格式錯誤",
                    f"{cfg_path.name} 解析失敗：{e}\n請檢查是否有重複鍵或少逗號。"
                )
                cfg = {}

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
                self.ui.nameEdit.setText(
                    last.get("name", self.ui.nameEdit.text()))
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
            LAST.write_text(json.dumps(
                data, ensure_ascii=False, indent=2), encoding="utf-8")

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
        actConnect = tb.addAction("連線")
        actDisconnect = tb.addAction("斷線")
        actStart = tb.addAction("開始")
        actStop = tb.addAction("停止")
        actAutoConnect = tb.addAction("自動連線")  # 新增自動連線按鈕

        actConnect.triggered.connect(self.controller.connect_device)
        actDisconnect.triggered.connect(self.controller.disconnect_device)
        actStart.triggered.connect(self.controller.start_stream)
        actStop.triggered.connect(self.controller.stop_stream)
        actAutoConnect.triggered.connect(self.auto_connect_device)  # 綁定自動連線

        # 保存配置參數供自動連接使用
        self.sampling_rate = cfg.get("sampling_rate", 1000)
        self.analog_channels = cfg.get("analog_channels", [1])
        self.preferred_mac = cfg.get("preferred_mac", None)
        self.cfg_address = cfg.get("address", None)

        # ✅ 啟動後 500ms 自動嘗試連線（只連線，不開始串流）
        if _HAS_AUTOCONNECT:
            QtCore.QTimer.singleShot(500, self._auto_connect_on_launch)

    # ---- 啟動時自動連線（只連線，不開始串流）----
    def _auto_connect_on_launch(self):
        addr = self.cfg_address  # 若 config.toml 有寫死 address（如 "COM16"），優先用它
        if _HAS_AUTOCONNECT and not addr:
            try:
                addr = resolve_address(
                    preferred_mac=self.preferred_mac,
                    preferred_com=None,
                    name_hints=["BITalino", "bitalino"],
                )
            except Exception as e:
                self.statusBar().showMessage(f"自動連線模組錯誤：{e}", 6000)
                addr = None

        if not addr:
            self.statusBar().showMessage("未找到 BITalino，可按『自動連線』或手動連線。", 6000)
            return

        # 把位址交給 controller 的同一顆 client 來連線（不啟動串流）
        ok = False
        try:
            # 你的 ECGController.connect_device 若支援 (address=...) 會回傳 bool
            ret = self.controller.connect_device(address=addr)
            ok = bool(ret) if ret is not None else True
        except TypeError:
            # 萬一你尚未更新簽章，改為手動配置後再呼叫
            try:
                if hasattr(self.controller, "client"):
                    self.controller.address = addr
                    self.controller.client.configure(
                        address=addr,
                        sampling_rate=self.sampling_rate,
                        analog_channels=self.analog_channels,
                    )
                self.controller.connect_device()
                ok = True
            except Exception as e:
                self.statusBar().showMessage(f"自動連線失敗：{e}", 6000)
                ok = False
        except Exception as e:
            self.statusBar().showMessage(f"自動連線失敗：{e}", 6000)
            ok = False

        if ok:
            self.statusBar().showMessage(f"✓ 已自動連線（{addr}），請按『開始』開始串流。", 6000)

    # ---- 工具列「自動連線」：掃描→選擇→連線（不自動開始串流）----
    def auto_connect_device(self):
        """掃描裝置 → 讓使用者選擇 → 交給 controller 連線（不自動開始串流）"""
        if not _HAS_AUTOCONNECT:
            self.statusBar().showMessage("未安裝自動連接模組；請手動連線或在 config.toml 填 address。", 6000)
            return

        try:
            # 若 config 已指定 address（例如 COM16），就直接用
            addr = resolve_address(
                preferred_mac=self.preferred_mac,
                preferred_com=self.cfg_address,
                name_hints=["BITalino", "bitalino"],
            )
        except Exception as e:
            self.statusBar().showMessage(f"自動連線模組錯誤：{e}", 6000)
            return

        from PyQt6 import QtWidgets
        if not addr:
            # 掃描並給使用者選
            try:
                cands = enumerate_candidates(
                    name_hints=["BITalino", "bitalino"], timeout_s=6)
            except Exception as e:
                self.statusBar().showMessage(f"掃描錯誤：{e}", 6000)
                return

            if not cands:
                self.statusBar().showMessage("找不到 BITalino（請確認開機/配對/驅動）", 6000)
                return

            labels = [c["label"] for c in cands]
            item, ok = QtWidgets.QInputDialog.getItem(
                self, "選擇裝置", "找到以下裝置：", labels, 0, False
            )
            if not ok:
                self.statusBar().showMessage("已取消自動連線", 3000)
                return
            idx = labels.index(item)
            addr = cands[idx]["address"]

        # 用 controller 內的 BitalinoClient 連線（不開始串流）
        try:
            ret = self.controller.connect_device(address=addr)
            ok = bool(ret) if ret is not None else True
            if ok:
                self.statusBar().showMessage(f"✓ 已連線（{addr}）。按『開始』開始串流。", 5000)
            else:
                self.statusBar().showMessage(f"自動連線失敗（{addr}）。", 6000)
        except TypeError:
            # 萬一簽章未更新，fallback
            try:
                if hasattr(self.controller, "client"):
                    self.controller.address = addr
                    self.controller.client.configure(
                        address=addr,
                        sampling_rate=self.sampling_rate,
                        analog_channels=self.analog_channels,
                    )
                self.controller.connect_device()
                self.statusBar().showMessage(f"✓ 已連線（{addr}）。按『開始』開始串流。", 5000)
            except Exception as e:
                self.statusBar().showMessage(f"自動連線失敗：{e}", 6000)
        except Exception as e:
            self.statusBar().showMessage(f"自動連線失敗：{e}", 6000)

    def closeEvent(self, event):
        """重寫關閉事件，確保正確關閉設備"""
        try:
            if hasattr(self, 'controller'):
                self.controller.stop_stream()
                self.controller.disconnect_device()
        except Exception as e:
            print(f"關閉設備時出錯: {e}")
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec())
