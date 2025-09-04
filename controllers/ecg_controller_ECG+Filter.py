# controllers/ecg_controller.py
"""
ECGController —— BITalino 串流 → 濾波 → R 峰 → HR/HRV → 繪圖（掃描式就緒版）

顯示邏輯（符合你的要求）：
- X 軸固定 0~seconds（預設 10s），每個樣本依時間落在固定 x（1kHz 時 0.001, 0.002…）。
- 第 1 輪：從 x=0 開始往右畫，右側保持空白。
- 第 2 輪起：從 x=0 重新畫，逐點覆蓋舊圖（不整頁清空），在接縫處斷線避免跳線。

若想回到舊的「滑動窗」顯示，可在 config.toml 設 [plot].mode="sliding"。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QLabel, QMessageBox, QFileDialog

from bitalino_helpers import BitalinoClient

# ─────────────────────────────────────────────────────────────
# 濾波：優先 ECGFilterRT；沒有就用 fallback
# ─────────────────────────────────────────────────────────────
_HAS_ECGFILTER = False
try:
    from processing.filters import ECGFilterRT, EnhancedBandpass
    _HAS_ECGFILTER = True
except Exception:
    try:
        import scipy.signal as _sig
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False

    class _OnePoleLPF:
        def __init__(self, fs: int, fc: float):
            a = float(np.exp(-2.0 * np.pi * fc / fs)); self.a = a; self.s = 0.0
        def filt_vec(self, x: np.ndarray) -> np.ndarray:
            y = np.empty_like(x, dtype=float); s=self.s; a=self.a
            xv = x.astype(float)
            for i,v in enumerate(xv):
                s = a*s + (1-a)*v; y[i]=s
            self.s = s; return y

    class EnhancedBandpass:
        def __init__(self, fs: int, hp=0.67, lp=30.0, notch_freq: float = 0.0):
            self.fs=int(fs); self.hp_lp=_OnePoleLPF(fs,hp); self.lp_lp=_OnePoleLPF(fs,lp)
            self.use_notch = False
            if notch_freq and _HAS_SCIPY:
                b,a = _sig.iirnotch(w0=notch_freq/(fs/2.0), Q=30.0)
                self.sos_notch = _sig.tf2sos(b,a); self.zi_notch = _sig.sosfilt_zi(self.sos_notch)
                self.use_notch = True
        def process(self, x: np.ndarray) -> np.ndarray:
            base = self.hp_lp.filt_vec(x); y = x - base
            if self.use_notch:
                y, self.zi_notch = _sig.sosfilt(self.sos_notch, y, zi=self.zi_notch)
            y = self.lp_lp.filt_vec(y)
            if ' _HAS_SCIPY' in globals() and _HAS_SCIPY and len(y) >= 5:
                y = _sig.medfilt(y, kernel_size=5)
            return y
        def set_notch_enabled(self, enabled: bool): self.use_notch = bool(enabled)
        def reset_state(self): self.hp_lp.s = 0.0; self.lp_lp.s = 0.0


# ─────────────────────────────────────────────────────────────
# R 峰偵測 & HRV（時域）
# ─────────────────────────────────────────────────────────────
class ECGDetector:
    def __init__(self, fs: int):
        self.fs=int(fs)
        self._integ_buf=np.zeros(max(1,int(0.15*fs)),dtype=float); self._integ_idx=0
        self._thr=0.0; self._last_peak_i=-10_000; self._global_i=0
        self.r_indices: List[int]=[]; self.rr_ms: List[float]=[]
    def process(self, x: np.ndarray)->None:
        if x.size==0: return
        if x.size==1: diff=np.array([x[0]],dtype=float)
        else:
            diff=np.empty_like(x,dtype=float); diff[0]=x[1]-x[0]; diff[1:]=x[1:]-x[:-1]
        sq=diff*diff
        w=self._integ_buf.size; integ=np.empty_like(sq,dtype=float)
        for i,v in enumerate(sq):
            self._integ_buf[self._integ_idx]=v; self._integ_idx=(self._integ_idx+1)%w
            integ[i]=float(self._integ_buf.mean())
        self.r_indices.clear(); self.rr_ms.clear()
        med=float(np.median(integ)); std=float(np.std(integ))
        target_thr=med+0.8*std; self._thr=0.9*self._thr+0.1*target_thr
        refr=int(0.25*self.fs)
        for i in range(1,len(integ)-1):
            gi=self._global_i+i
            if gi-self._last_peak_i<refr: continue
            if integ[i-1]<integ[i]>=integ[i+1] and integ[i]>self._thr:
                self.r_indices.append(gi)
                if self._last_peak_i>=0:
                    rr=(gi-self._last_peak_i)*1000.0/self.fs
                    if 300.0<=rr<=2000.0: self.rr_ms.append(rr)
                self._last_peak_i=gi
        self._global_i+=len(integ)


@dataclass
class TimeDomainHRV:
    count:int; mean_rr:float; sdnn:float; rmssd:float; mean_hr:float

def compute_time_domain(rr_ms: List[float])->Optional[TimeDomainHRV]:
    if len(rr_ms)<2: return None
    rr=np.asarray(rr_ms,dtype=float)
    mean_rr=float(rr.mean()); sdnn=float(rr.std(ddof=1)) if rr.size>1 else 0.0
    diffs=np.diff(rr); rmssd=float(np.sqrt(np.mean(diffs*diffs))) if diffs.size>0 else 0.0
    mean_hr=60000.0/mean_rr if mean_rr>0 else 0.0
    return TimeDomainHRV(len(rr_ms),mean_rr,sdnn,rmssd,mean_hr)


# ─────────────────────────────────────────────────────────────
# Qt 資料橋
# ─────────────────────────────────────────────────────────────
class DataBridge(QObject):
    arrived = pyqtSignal(object)  # numpy.ndarray


# ─────────────────────────────────────────────────────────────
# 主控制器
# ─────────────────────────────────────────────────────────────
class ECGController:
    def __init__(
        self, plot_widget: pg.PlotWidget,
        lbl_rt_hr, lbl_stable_hr, status_bar,
        btn_save_rr, btn_analyze, cfg: Dict[str, Any],
    ):
        # UI
        self.plot=plot_widget; self.lbl_rt=lbl_rt_hr; self.lbl_stable=lbl_stable_hr
        self.status_bar=status_bar; self.btn_save_rr=btn_save_rr; self.btn_analyze=btn_analyze

        # 基本/裝置
        self.fs=int(cfg.get("sampling_rate",1000))
        ch=cfg.get("analog_channels",[1])
        if not isinstance(ch,list): ch=[int(ch)]
        self.analog_channels:List[int]=ch
        self.address:Optional[str]=cfg.get("address") or None

        # 濾波
        f_cfg=cfg.get("filter",{})
        if f_cfg.get("enable",True) and _HAS_ECGFILTER:
            self.ecg_filter=ECGFilterRT(
                fs=self.fs,
                band=tuple(f_cfg.get("band",(0.5,25.0))),
                notch=float(f_cfg.get("notch",60.0)),
                order=int(f_cfg.get("order",4)),
                q=float(f_cfg.get("q",30.0)),
            ); self._use_fallback=False
        elif f_cfg.get("enable",True):
            self.ecg_filter=EnhancedBandpass(self.fs,hp=0.67,lp=30.0,notch_freq=float(f_cfg.get("notch",0.0)))
            self._use_fallback=True
        else:
            self.ecg_filter=None; self._use_fallback=False
        self.filter_enabled=bool(f_cfg.get("enable",True))
        self.notch_enabled=bool(f_cfg.get("notch",60.0) and not self._use_fallback)
        if self.ecg_filter is not None and hasattr(self.ecg_filter,"set_notch_enabled"):
            try: self.ecg_filter.set_notch_enabled(self.notch_enabled)
            except Exception: pass

        # 繪圖 / 緩衝
        plot_cfg=cfg.get("plot",{})
        self.seconds_window=int(plot_cfg.get("seconds",10))
        self.buf_len=max(10,int(self.seconds_window*self.fs))
        self.gain=float(plot_cfg.get("gain",1.5))
        self.ecg_col=int(plot_cfg.get("ecg_col",-1))
        self.chunk=int(plot_cfg.get("chunk",100))

        # 顯示模式：sweep（預設）或 sliding
        self.mode=str(plot_cfg.get("mode","sweep")).strip().lower()  # "sweep" | "sliding"
        self.direction=str(plot_cfg.get("direction","ltr")).strip().lower()  # 只在 sliding 用

        # ring buffer 與座標
        self._ybuf=np.zeros(self.buf_len,dtype=float)
        self._tbase=np.linspace(0.0,float(self.seconds_window),self.buf_len,endpoint=False)

        # 掃描狀態：筆位置（以樣本 index 計）、是否已繞回
        self._pos=0
        self._wrapped_once=False

        # 曲線
        self.curve=self.plot.plot(self._tbase,self._ybuf,pen=pg.mkPen(width=2))
        self.curve.setDownsampling(auto=True); self.curve.setClipToView(True)

        # 座標軸
        self.plot.setLabel("left","Amplitude"); self.plot.setLabel("bottom","Time","s")
        self.plot.showGrid(x=True,y=True)
        self._apply_direction()

        # 偵測/顯示狀態
        self.det=ECGDetector(self.fs)
        self.warmup_left=int(0.8*self.fs)
        self.alpha=0.12; self.hr_stable:Optional[float]=None; self._rr_accum:List[float]=[]

        # 橋接/Client/UI
        self.bridge=DataBridge(); self.bridge.arrived.connect(self._on_arrived_mainthread)
        self.client=BitalinoClient()
        self.client.configure(address=self.address,sampling_rate=self.fs,analog_channels=self.analog_channels)
        self.btn_save_rr.clicked.connect(self._on_save_rr_clicked)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        self._set_hr("--","--")

        # 電池
        b_cfg=cfg.get("battery",{})
        self.batt_enabled=bool(b_cfg.get("enable",True))
        self.batt_poll_ms=int(b_cfg.get("poll_s",15))*1000
        self.batt_raw_min=float(b_cfg.get("raw_min",511)); self.batt_raw_max=float(b_cfg.get("raw_max",645))
        self.batt_low_pct=float(b_cfg.get("low_pct",20)); self.batt_crit_pct=float(b_cfg.get("critical_pct",10))
        self.batt_set_dev_pct=int(b_cfg.get("set_device_threshold_pct",10))
        self.batt_label=QLabel("🔋 --%")
        if self.batt_enabled: self.status_bar.addPermanentWidget(self.batt_label)
        self._batt_timer=QTimer(self.bridge); self._batt_timer.setInterval(self.batt_poll_ms)
        self._batt_timer.timeout.connect(self._on_batt_tick)

        # 吞吐監看
        self.rx_label=QLabel("📡 0/s"); self.status_bar.addPermanentWidget(self.rx_label)
        self._rx_counter=0; self._rx_last=0; self._rx_zero_ticks=0
        self._rx_timer=QTimer(self.bridge); self._rx_timer.setInterval(1000)
        self._rx_timer.timeout.connect(self._on_rx_tick)

        self._is_streaming=False

    # ── 濾波/Notch 切換 ─────────────────────────────────────────
    def set_filter_enabled(self, enabled: bool):
        self.filter_enabled=bool(enabled)
        self.status_bar.showMessage(f"濾波：{'開' if self.filter_enabled else '關'}",1500)
    def set_notch_enabled(self, enabled: bool):
        self.notch_enabled=bool(enabled)
        if self.ecg_filter is not None and hasattr(self.ecg_filter,"set_notch_enabled"):
            try:
                self.ecg_filter.set_notch_enabled(self.notch_enabled)
                self.status_bar.showMessage(f"Notch 60Hz：{'開' if self.notch_enabled else '關'}",1500)
            except Exception:
                self.status_bar.showMessage("Notch 切換失敗（濾波器不支援）",3000)
        else:
            self.status_bar.showMessage("目前濾波器不支援 Notch（未安裝 scipy 或 notch=0）",3000)

    # ── 方向/座標套用（sweep 一律 LTR；sliding 可 rtl/ltr） ───────
    def _apply_direction(self):
        try:
            vb=self.plot.getPlotItem().vb
            if self.mode=="sweep":
                vb.invertX(False)  # 掃描式固定左→右
            else:
                vb.invertX(self.direction=="rtl")
            vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            vb.setXRange(0, self.seconds_window, padding=0)
        except Exception:
            pass

    # ── 連線/斷線 ────────────────────────────────────────────────
    def connect_device(self, address: Optional[str]=None, retries:int=3)->bool:
        try:
            self.status_bar.showMessage("連線中…")
            if address:
                self.address=address
                try: self.client.configure(address=address)
                except Exception: pass
            self.client.connect(retries=retries)
            try:
                self._set_batt_label(self._query_battery_percent())
                if self.batt_enabled and getattr(self.client,"device",None):
                    self.client.device.battery(int(self.batt_set_dev_pct))
            except Exception: pass
            self._start_battery_monitor()
            self.status_bar.showMessage("BITalino 連線成功",2500)
            return True
        except Exception as e:
            self.status_bar.showMessage(f"連線失敗：{e}",6000); return False

    def disconnect_device(self):
        try:
            self.stop_stream(); self.client.close(); self._stop_battery_monitor()
            self.status_bar.showMessage("已斷線",2500)
        except Exception as e:
            self.status_bar.showMessage(f"斷線錯誤：{e}",5000)

    # ── 開始/停止 串流 ───────────────────────────────────────────
    def start_stream(self)->bool:
        if self._is_streaming:
            self.status_bar.showMessage("已在擷取中",1500); return True
        try:
            def _on_data(arr):
                try: self.bridge.arrived.emit(np.asarray(arr))
                except Exception as ex: print("UI dispatch error:", ex)
            self.client.data_callback=_on_data

            if hasattr(self.client,"on_error"):
                def _on_err(e):
                    self._is_streaming=False
                    try: self.client.stop_acquisition()
                    except Exception: pass
                    self._stop_battery_monitor(); self._rx_timer.stop()
                    self.status_bar.showMessage(f"資料擷取中斷：{e}",8000)
                self.client.on_error=_on_err

            if not getattr(self.client,"is_connected",False):
                self.client.connect(retries=3)

            # 復位濾波/偵測
            self.warmup_left=int(0.8*self.fs)
            if getattr(self,"ecg_filter",None) is not None and hasattr(self.ecg_filter,"reset_state"):
                try: self.ecg_filter.reset_state()
                except Exception: pass
            self.det=ECGDetector(self.fs)

            # 清畫面 + 掃描狀態歸零
            self._ybuf[:]=0.0
            self.curve.setData(self._tbase,self._ybuf)
            self._pos=0; self._wrapped_once=False
            self._apply_direction()

            # 啟動擷取
            chunk=max(10,min(self.buf_len//2,int(self.chunk)))
            self.client.start_acquisition(chunk_size=chunk)
            self._is_streaming=True

            # 吞吐監看
            self._rx_counter=0; self._rx_last=0; self._rx_zero_ticks=0
            self._rx_timer.start()

            self.status_bar.showMessage(
                f"擷取中：fs={self.fs}, ch={self.analog_channels}, chunk={chunk}, mode={self.mode}", 3500
            )
            self._start_battery_monitor()
            return True
        except Exception as e:
            self._is_streaming=False
            try: self.client.stop_acquisition()
            except Exception: pass
            self._rx_timer.stop()
            try: QMessageBox.critical(None,"開始擷取失敗",str(e))
            except Exception: pass
            self.status_bar.showMessage(f"開始擷取失敗：{e}",6000)
            return False

    def stop_stream(self):
        try:
            self._is_streaming=False; self._rx_timer.stop()
            self.client.stop_acquisition()
            self._on_batt_tick()
            self.status_bar.showMessage("已停止擷取",1800)
        except Exception as e:
            self.status_bar.showMessage(f"停止擷取失敗：{e}",4000)

    # ── 主執行緒：接資料 → 濾波/偵測/繪圖 ─────────────────────────
    def _on_arrived_mainthread(self, arr_obj: object):
        # 保險：每幀套一次方向
        self._apply_direction()

        data=np.asarray(arr_obj)
        if data.ndim!=2 or data.shape[0]==0: return

        # 吞吐統計
        self._rx_counter += data.shape[0]

        # 取 ECG 欄位
        ecg = data[:, -1].astype(float) if self.ecg_col == -1 else data[:, int(self.ecg_col)].astype(float)

        # 濾波 + 增益
        if self.filter_enabled and self.ecg_filter is not None:
            try: y=self.ecg_filter.process(ecg)*self.gain
            except Exception: y=ecg*self.gain
        else: y=ecg*self.gain

        # 繪圖
        if self.mode=="sweep":
            self._plot_sweep(y)
        else:
            self._plot_sliding(y)

        # 暖機
        n=len(y)
        if self.warmup_left>0:
            self.warmup_left-=n; return

        # R 峰 → RR → HR
        self.det.process(y)
        if self.det.rr_ms:
            self._rr_accum.extend(self.det.rr_ms)
            rr_arr=np.asarray(self.det.rr_ms[-5:],dtype=float)
            hr_inst=60_000.0/float(rr_arr.mean())
            if self.hr_stable is None: self.hr_stable=hr_inst
            else: self.hr_stable=(1-self.alpha)*self.hr_stable+self.alpha*hr_inst
            self._set_hr(f"{hr_inst:.0f}", f"{self.hr_stable:.0f}")

    # ── 掃描式（符合你的需求） ───────────────────────────────────
    def _plot_sweep(self, y: np.ndarray):
        """
        X 固定 0~seconds：
        - 依 _pos 把新樣本寫進固定座標（fs=1000 時每點 0.001s）。
        - 第 1 輪：右側空白（NaN，不連線）。
        - 第 2 輪起：逐點覆蓋舊圖，且在接縫處放 NaN 斷線避免跳線。
        """
        L=self._ybuf.size; n=int(y.size)
        if n<=0 or L==0: return

        pos=self._pos
        if n >= L:
            # 太長：僅保留最後 L 點並視為剛好「寫到尾」
            self._ybuf[:] = y[-L:]
            pos = 0
            self._wrapped_once = True
        else:
            end = pos + n
            if end <= L:
                self._ybuf[pos:end] = y
                pos = end
                if pos == L:
                    pos = 0
                    self._wrapped_once = True
            else:
                k = L - pos
                self._ybuf[pos:] = y[:k]
                self._ybuf[:n-k] = y[k:]
                pos = (n - k)
                self._wrapped_once = True

        self._pos = pos

        # 準備顯示：第 1 輪右側留白；第 2 輪起只在接縫放 NaN 斷線
        y_vis = self._ybuf.copy()
        if not self._wrapped_once:
            if self._pos < L:
                y_vis[self._pos:] = np.nan    # 右側空白
        else:
            y_vis[self._pos % L] = np.nan    # 接縫斷線

        self.curve.setData(self._tbase, y_vis, connect='finite')

    # ── 滑動窗（舊樣式，保留做選項） ─────────────────────────────
    def _plot_sliding(self, y: np.ndarray):
        L=self._ybuf.size; n=int(y.size)
        if n<=0: return
        if n>=L: self._ybuf[:]=y[-L:]
        else:
            self._ybuf = np.roll(self._ybuf, -n)
            self._ybuf[-n:] = y
        self.curve.setData(self._tbase, self._ybuf)

    # ── RR 存檔 / HRV（時域） ───────────────────────────────────
    def _on_save_rr_clicked(self):
        if len(self._rr_accum)==0:
            QMessageBox.information(None,"儲存 RR","目前沒有可儲存的 RR。"); return
        ts=time.strftime("%Y%m%d_%H%M%S")
        fn,_=QFileDialog.getSaveFileName(None,"儲存 RR",f"RR{ts}.txt","Text Files (*.txt)")
        if not fn: return
        Path(fn).write_text("\n".join(f"{v:.1f}" for v in self._rr_accum),encoding="utf-8")
        QMessageBox.information(None,"儲存 RR",f"已儲存：{fn}")

    def _on_analyze_clicked(self):
        res=compute_time_domain(self._rr_accum)
        if res is None:
            QMessageBox.information(None,"HRV 分析","RR 數量不足，請先擷取 RR。"); return
        msg=(f"RR 數量：{res.count}\n"
             f"Mean RR：{res.mean_rr:.1f} ms\n"
             f"SDNN：{res.sdnn:.1f} ms\n"
             f"RMSSD：{res.rmssd:.1f} ms\n"
             f"Mean HR：{res.mean_hr:.1f} bpm\n")
        QMessageBox.information(None,"HRV（時域）",msg)

    # ── UI 小工具 ───────────────────────────────────────────────
    def _set_hr(self, rt:str, stable:str):
        try:
            self.lbl_rt.setText(f"即時心跳：{rt} bpm")
            self.lbl_stable.setText(f"穩定心跳：{stable} bpm")
        except Exception: pass

    # ── 吞吐監看 ───────────────────────────────────────────────
    def _on_rx_tick(self):
        now=self._rx_counter; rate=max(0,now-self._rx_last); self._rx_last=now
        self.rx_label.setText(f"📡 {rate}/s")
        if self._is_streaming:
            if rate==0:
                self._rx_zero_ticks+=1
                if self._rx_zero_ticks>=2:
                    self.status_bar.showMessage("⚠ 未收到資料：請確認 COM/電源/取樣與接線",4000)
            else:
                self._rx_zero_ticks=0

    # ── 電池（idle 才讀） ───────────────────────────────────────
    def _battery_percent_from_raw(self, raw: float)->float:
        r0,r1=self.batt_raw_min,self.batt_raw_max
        if r1<=r0: return 0.0
        pct=1.0+(float(raw)-r0)*(98.0/(r1-r0))
        return max(0.0,min(100.0,pct))

    def _query_battery_percent(self):
        dev=getattr(self.client,"device",None)
        if dev is None: return None
        try:
            st=dev.state(); raw=st.get("battery") if isinstance(st,dict) else None
            if raw is None: return None
            return self._battery_percent_from_raw(raw)
        except Exception:
            return None

    def _set_batt_label(self, pct: Optional[float]):
        if not self.batt_enabled: return
        if pct is None:
            self.batt_label.setText("🔋 --%"); self.batt_label.setStyleSheet(""); return
        self.batt_label.setText(f"🔋 {pct:0.0f}%")
        if pct<=self.batt_crit_pct: self.batt_label.setStyleSheet("color:#e53935;")
        elif pct<=self.batt_low_pct: self.batt_label.setStyleSheet("color:#fb8c00;")
        else: self.batt_label.setStyleSheet("")

    def _on_batt_tick(self):
        if getattr(self.client,"is_acquiring",False): return
        pct=self._query_battery_percent()
        self._set_batt_label(pct)
        if pct is not None and pct<=self.batt_crit_pct:
            self.status_bar.showMessage("電量過低：請儘快充電",2500)

    def _start_battery_monitor(self):
        if self.batt_enabled:
            self._batt_timer.start(); self._on_batt_tick()
    def _stop_battery_monitor(self):
        try: self._batt_timer.stop(); self._set_batt_label(None)
        except Exception: pass
