# main.py
"""
App entrypoint.
- Loads Qt UI (ui/ui_ecg.py), builds a PlotWidget into chart area
- Wires toolbar actions (Connect/Disconnect/Start/Stop/Auto-connect/+ Pick Device/COM)
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

# ---- Auto-connect helper (optional) ----
try:
    from autoconnect_bitalino import resolve_address, enumerate_candidates
    _HAS_AUTOCONNECT = True
except ModuleNotFoundError:
    _HAS_AUTOCONNECT = False
    print("警告: 找不到 autoconnect_bitalino.py（請確認檔案與 main.py 同層或套件路徑正確）")
except ImportError as e:
    _HAS_AUTOCONNECT = False
    print(f"警告: 載入 autoconnect_bitalino 失敗：{e}")

# ---- Optional fallback: list local COM ports when no candidate found ----
try:
    from serial.tools import list_ports
    _HAS_PYSERIAL = True
except Exception:
    list_ports = None
    _HAS_PYSERIAL = False

# --# -- 記錄檔路徑（存上次成功的位址） --
_LAST_DEV_FILE = Path("data/last_device.json")


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
                    f"{cfg_path.name} 解析失敗：\n{e}\n請檢查是否有重複鍵或少逗號。"
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

        # ⑥ 工具列（清單 -> 連線 -> 開始 -> 停止 -> 斷線）
        tb = self.ui.toolBar
        tb.clear()

        # 先放「可用 COM 清單」
        tb.addWidget(QtWidgets.QLabel(" 裝置："))
        self.portCombo = QtWidgets.QComboBox()
        self.portCombo.setMinimumWidth(260)
        self.portCombo.setPlaceholderText("掃描中…")
        tb.addWidget(self.portCombo)

        # 連線 / 開始 / 停止 / 斷線（依你的指定順序）
        actConnect = tb.addAction("連線")
        actStart = tb.addAction("開始")
        actStop = tb.addAction("停止")
        actDisconnect = tb.addAction("斷線")

        # 綁定事件（「連線」用我們的自訂 slot，其他直接走 controller）
        actConnect.triggered.connect(self._on_connect_clicked)
        actStart.triggered.connect(self.controller.start_stream)
        actStop.triggered.connect(self.controller.stop_stream)
        actDisconnect.triggered.connect(self.controller.disconnect_device)

        # 下拉改變時僅提示，不自動連線（避免誤觸）
        self.portCombo.activated.connect(self._on_combo_changed)

        # 啟動：先掃描清單，再嘗試用「上次成功的位址」自動連線
        self._populate_port_combo()           # 先把清單顯示出來
        self._autoconnect_last_then_sync_ui()  # 若上次有記錄，嘗試直接連線

        # 保存配置參數供自動連接使用
        self.sampling_rate = cfg.get("sampling_rate", 1000)
        self.analog_channels = cfg.get("analog_channels", [1])
        self.preferred_mac = cfg.get("preferred_mac", None)
        self.cfg_address = cfg.get("address", None)

        # ✅ 啟動後 500ms 自動嘗試連線（只「連線」，不開始串流）
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
            self.statusBar().showMessage("未找到 BITalino，可按『自動連線』或『選擇裝置/COM』。", 6000)
            return

        try:
            ret = self.controller.connect_device(address=addr)
            ok = bool(ret) if ret is not None else True
        except TypeError:
            # fallback for older signature
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
        if not _HAS_AUTOCONNECT:
            self.statusBar().showMessage("未安裝自動連接模組；請改按『選擇裝置/COM』或在 config.toml 填 address。", 7000)
            return

        try:
            addr = resolve_address(
                preferred_mac=self.preferred_mac,
                preferred_com=self.cfg_address,
                name_hints=["BITalino", "bitalino"],
            )
        except Exception as e:
            self.statusBar().showMessage(f"自動連線模組錯誤：{e}", 6000)
            return

        if not addr:
            # 掃描候選清單，交由 pick 對話框處理
            self.pick_device_or_com()
            return

        try:
            ret = self.controller.connect_device(address=addr)
            ok = bool(ret) if ret is not None else True
            if ok:
                self.statusBar().showMessage(f"✓ 已連線（{addr}）。按『開始』開始串流。", 5000)
            else:
                self.statusBar().showMessage(f"自動連線失敗（{addr}）。", 6000)
        except Exception as e:
            self.statusBar().showMessage(f"自動連線失敗：{e}", 6000)

    def _load_last_addr(self) -> str | None:
        try:
            if _LAST_DEV_FILE.exists():
                obj = json.loads(_LAST_DEV_FILE.read_text(encoding="utf-8"))
                addr = str(obj.get("address", "")).strip()
                return addr or None
        except Exception:
            pass
        return None

    def _save_last_addr(self, addr: str) -> None:
        try:
            _LAST_DEV_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LAST_DEV_FILE.write_text(json.dumps({"address": addr}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        except Exception:
            pass

    def _scan_com_ports(self) -> list[tuple[str, str]]:
        """
        回傳 [(address, label), ...]，依 COM 編號排序。
        沒有 pyserial 也不會爆；只是清單會是空的。
        """
        out: list[tuple[str, str]] = []
        if _HAS_PYSERIAL and list_ports:
            for p in list_ports.comports():
                out.append((p.device, f"{p.device} – {p.description}"))
            # 依 COM 編號排序
            def _key(t):
                name = t[0]
                try:
                    return int(name.replace("COM", ""))
                except Exception:
                    return 9999
            out.sort(key=_key)
        return out

    def _populate_port_combo(self, selected: str | None = None) -> None:
        """把掃描到的 COM 寫入下拉；若有 selected 就把它選上。"""
        self.portCombo.blockSignals(True)
        self.portCombo.clear()
        ports = self._scan_com_ports()
        if not ports:
            self.portCombo.addItem("（找不到裝置，請檢查電源/配對/驅動）", "")
        else:
            for addr, label in ports:
                self.portCombo.addItem(label, addr)
            # 如果給了 selected，就設回選取
            if selected:
                idx = self.portCombo.findData(selected)
                if idx >= 0:
                    self.portCombo.setCurrentIndex(idx)
        self.portCombo.blockSignals(False)

    def _current_addr(self) -> str | None:
        """從下拉取得目前選擇的 address（例 'COM16'），取不到回 None。"""
        v = self.portCombo.currentData()
        return v.strip() if isinstance(v, str) and v.strip() else None

    def _on_combo_changed(self, index: int):
        addr = self._current_addr()
        if addr:
            self.statusBar().showMessage(f"已選擇：{addr}", 1500)

    def _on_connect_clicked(self):
        """按『連線』：用清單目前的埠連線；成功後記錄該埠。"""
        addr = self._current_addr()
        ok = self.controller.connect_device(address=addr)
        if ok:
            self.statusBar().showMessage(f"✓ 已連線（{addr or '自動'}）", 2500)
            if addr:
                self._save_last_addr(addr)
        else:
            self.statusBar().showMessage("連線失敗，請更換埠或檢查裝置。", 4000)

    def _autoconnect_last_then_sync_ui(self):
        """
        啟動時：若有『上次成功』的位址先試著連；不成功就維持清單可手選。
        成功時也把清單選到該位址。
        """
        last_addr = self._load_last_addr()
        if not last_addr:
            return
        self.statusBar().showMessage(f"嘗試自動連線：{last_addr} …", 2000)
        if self.controller.connect_device(address=last_addr):
            self._populate_port_combo(selected=last_addr)
            self.statusBar().showMessage(f"✓ 已自動連線（{last_addr}）", 3000)
        else:
            # 自動連不上也沒關係，清單已在，使用者可手選
            self.statusBar().showMessage("自動連線未成功，請從清單選擇再按『連線』。", 4000)

    # ---- 新增：手動挑選候選裝置；若沒有候選則列出本機 COM ports ----
    def pick_device_or_com(self):
        from PyQt6 import QtWidgets

        labels = []
        addresses = []

        # 先用 autoconnect 的 enumerate_candidates（若可用）
        if _HAS_AUTOCONNECT:
            try:
                cands = enumerate_candidates(
                    name_hints=["BITalino", "bitalino"], timeout_s=6)
            except Exception as e:
                cands = []
                self.statusBar().showMessage(f"掃描裝置時發生錯誤：{e}", 6000)
            for c in cands or []:
                labels.append(c.get("label", c.get("address", "未知裝置")))
                addresses.append(c.get("address"))

        # 若一個也沒有，就退回列出本機 COM Port（需 pyserial）
        if not labels and _HAS_PYSERIAL:
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                labels.append(f"{p.device} ({p.description})")
                addresses.append(p.device)

        if not labels:
            QtWidgets.QMessageBox.information(self, "沒有找到裝置",
                                              "未偵測到 BITalino 或任何可用的 COM Port。")
            return

        item, ok = QtWidgets.QInputDialog.getItem(
            self, "選擇裝置/COM", "可用清單：", labels, 0, False
        )
        if not ok:
            self.statusBar().showMessage("已取消選擇。", 3000)
            return

        addr = addresses[labels.index(item)]
        try:
            ret = self.controller.connect_device(address=addr)
            ok = bool(ret) if ret is not None else True
            if ok:
                self.statusBar().showMessage(f"✓ 已連線（{addr}）。按『開始』開始串流。", 5000)
            else:
                self.statusBar().showMessage(f"連線失敗（{addr}）。", 6000)
        except Exception as e:
            self.statusBar().showMessage(f"連線失敗：{e}", 6000)

    # ---- 關閉事件 ----
    def closeEvent(self, event):
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
