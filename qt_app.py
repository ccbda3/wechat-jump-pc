import os
import queue
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter,
                           QPalette, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget, QStyleFactory)

import win_backend as wb
import ml.worker as workermod
from ml.worker import Runner

C_BG = "#c0c0c0"
C_PANEL = "#c0c0c0"
C_CARD = "#ffffff"
C_STROKE = "#808080"
C_FG = "#000000"
C_MUTED = "#404040"
C_ACCENT = "#000080"

QSS = f"""
#Root, QMainWindow {{ background: {C_BG}; }}
QFrame#Panel {{ background: {C_PANEL}; border: 2px solid;
    border-color: #ffffff {C_STROKE} {C_STROKE} #ffffff; }}
QFrame#Card {{ background: {C_CARD}; border: 2px solid;
    border-color: {C_STROKE} #ffffff #ffffff {C_STROKE}; }}
QLabel {{ color: {C_FG}; font-size: 12px; background: transparent; }}
#Title {{ color: {C_FG}; font-weight: bold; font-size: 12px; }}
#Muted {{ color: {C_MUTED}; font-size: 10px; }}
#BigNum {{ font-size: 21px; font-weight: bold; color: {C_ACCENT}; }}
QListWidget {{ background: #ffffff; color: {C_FG}; border: 2px solid;
    border-color: {C_STROKE} #ffffff #ffffff {C_STROKE}; font-size: 12px; outline: none; }}
QListWidget::item {{ padding: 3px 6px; }}
QListWidget::item:selected {{ background: {C_ACCENT}; color: #ffffff; }}
QPushButton {{ background: {C_BG}; color: {C_FG}; border: 2px solid;
    border-color: #ffffff {C_STROKE} {C_STROKE} #ffffff;
    padding: 7px 6px; font-size: 12px; font-weight: bold; }}
QPushButton:hover {{ background: #cacaca; }}
QPushButton:pressed {{ border-color: {C_STROKE} #ffffff #ffffff {C_STROKE};
    padding: 8px 5px 6px 7px; }}
QPushButton:disabled {{ color: {C_STROKE}; }}
QCheckBox {{ color: {C_FG}; font-size: 12px; spacing: 6px; background: transparent; }}
QPlainTextEdit {{ background: #ffffff; color: {C_FG}; border: 2px solid;
    border-color: {C_STROKE} #ffffff #ffffff {C_STROKE};
    font-family: Consolas; font-size: 10px; }}
QSlider::groove:horizontal {{ height: 4px; background: {C_STROKE}; border: 1px solid;
    border-color: {C_STROKE} #ffffff #ffffff {C_STROKE}; }}
QSlider::handle:horizontal {{ width: 11px; height: 16px; margin: -7px 0;
    background: {C_BG}; border: 2px solid;
    border-color: #ffffff {C_STROKE} {C_STROKE} #ffffff; }}
#StatusLabel {{ color: {C_FG}; font-size: 11px; }}
#PreviewText {{ color: #ffffff; font-size: 13px; }}
"""


class EngineState:
    def __init__(self):
        self.manual = False
        self.running = False
        self.paused = False
        self.busy = False
        self.got = False
        self.tap_x = 0
        self.tap_y = 0
        self.swallow_up = False
        self.preview = False
        self.greedy = False
        self.base_coef = 1.0
        self.target_hwnd = None
        self.main_hwnd = None


class Engine(QThread):
    sigLog = Signal(str)
    sigStats = Signal(dict)
    sigDet = Signal(dict)
    sigReward = Signal(float)
    sigState = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.daemon = True

    def stop(self):
        self.state.running = False

    def _drain(self, q):
        while True:
            obj = q.get(timeout=90)
            if obj.get("t") == "log":
                self.sigLog.emit(str(obj.get("msg", "")))
                continue
            return obj

    def _wait_stable(self, hwnd):
        time.sleep(0.32)
        prev = wb.grab_frame(hwnd)
        if prev is None:
            return 0
        stable = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.2:
            if not self.state.running:
                return 0
            time.sleep(0.07)
            cur = wb.grab_frame(hwnd)
            if cur is None:
                return 0
            d = np.abs(prev[::4, ::4].astype(np.int16)
                       - cur[::4, ::4].astype(np.int16)).sum(axis=2)
            if float(d.mean()) / 3.0 < 2.2:
                stable += 1
                if stable >= 2:
                    return 1
            else:
                stable = 0
            prev = cur
        return 0

    def run(self):
        state = self.state
        q = queue.Queue()
        workermod.EMIT_HOOK = q.put
        try:
            runner = Runner()
            runner.handle_init({"t": "init"})
            ready = self._drain(q)
            self.sigStats.emit(ready.get("stats", {}))
        except Exception as e:
            self.sigLog.emit("引擎初始化失败：%s" % e)
            state.running = False
            self.sigState.emit("已停止")
            return
        state.running = True
        self.sigState.emit("运行中")
        rid = 0
        try:
            while state.running:
                if state.paused:
                    self.sigState.emit("已暂停")
                    time.sleep(0.12)
                    continue
                self.sigState.emit("运行中")
                hwnd = state.target_hwnd
                if not hwnd or not wb.user32.IsWindow(hwnd):
                    self.sigState.emit("等待窗口")
                    time.sleep(0.6)
                    continue
                title = wb.window_title(hwnd)
                if ("跳一跳" not in title and "小程序" not in title
                        and "微信" not in title) or not wb.is_wechat_process(hwnd):
                    self.sigLog.emit("目标窗口不是微信小程序（%s），请重新选择" % title)
                    self.sigState.emit("等待窗口")
                    time.sleep(0.7)
                    continue
                wb.bring_front(hwnd)
                frame = wb.grab_frame(hwnd)
                if frame is None:
                    time.sleep(0.35)
                    continue
                manual = bool(state.manual)
                has_tap = False
                tx = ty = 0
                if manual and state.got:
                    has_tap = True
                    tx, ty = state.tap_x, state.tap_y
                req = {"t": "frame", "id": rid, "w": frame.shape[1],
                       "h": frame.shape[0], "phase": "before",
                       "base": state.base_coef, "greedy": int(state.greedy),
                       "settled": 1, "manual": int(manual),
                       "has_tap": int(has_tap), "tx": tx, "ty": ty}
                try:
                    runner.handle_frame(req, frame)
                except Exception as e:
                    self.sigLog.emit("推理异常：%s" % e)
                    time.sleep(0.4)
                    continue
                res = self._drain(q)
                self.sigStats.emit(res.get("stats", {}))
                self.sigDet.emit(res.get("det", {}))
                if res.get("reward") is not None:
                    self.sigReward.emit(float(res["reward"]))
                if manual and res.get("kind") == "wait":
                    if has_tap:
                        state.got = False
                    elif state.preview and res.get("det", {}).get("has_board"):
                        bx = res["det"].get("bx")
                        by = res["det"].get("by")
                        if bx is not None and by is not None:
                            ox, oy = wb.client_origin(hwnd)
                            wb.set_cursor(ox + bx, oy + by)
                            state.preview = False
                if res.get("kind") == "press" and "ms" in res:
                    ox, oy = wb.client_origin(hwnd)
                    sx = ox + int(res.get("x", ox + frame.shape[1] * 0.5))
                    sy = oy + int(res.get("y", oy + frame.shape[0] * 0.68))
                    state.busy = True
                    wb.press_hold(sx, sy, res["ms"])
                    state.got = False
                    stable = self._wait_stable(hwnd)
                    frame2 = wb.grab_frame(hwnd)
                    if frame2 is None:
                        break
                    req2 = dict(req, phase="after", settled=stable)
                    try:
                        runner.handle_frame(req2, frame2)
                    except Exception as e:
                        self.sigLog.emit("反馈异常：%s" % e)
                    res2 = self._drain(q)
                    self.sigStats.emit(res2.get("stats", {}))
                    self.sigDet.emit(res2.get("det", {}))
                    if res2.get("reward") is not None:
                        self.sigReward.emit(float(res2["reward"]))
                    if res2.get("kind") == "restart" and "x" in res2:
                        wb.click_at(ox + res2["x"], oy + res2["y"])
                        self.sigLog.emit("已点击 (%.0f, %.0f) 继续" % (res2["x"], res2["y"]))
                        time.sleep(1.5)
                    state.busy = False
                    if manual:
                        state.preview = True
                    rid += 1
                elif res.get("kind") == "restart" and "x" in res:
                    ox, oy = wb.client_origin(hwnd)
                    wb.click_at(ox + res["x"], oy + res["y"])
                    self.sigLog.emit("已点击 (%.0f, %.0f)" % (res["x"], res["y"]))
                    time.sleep(1.5)
                    if manual:
                        state.preview = True
                else:
                    time.sleep(0.38)
        except queue.Empty:
            self.sigLog.emit("推理进程无响应，引擎停止")
        finally:
            state.running = False
            state.busy = False
            workermod.EMIT_HOOK = None
            self.sigState.emit("已停止")


class PreviewThread(QThread):
    sigFrame = Signal(QImage)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.alive = True
        self.daemon = True

    def stop(self):
        self.alive = False

    def run(self):
        while self.alive:
            hwnd = self.state.target_hwnd
            if self.state.running and hwnd and wb.user32.IsWindow(hwnd):
                img = wb.grab_frame(hwnd)
                if img is not None:
                    h, w = img.shape[:2]
                    qi = QImage(img.data, w, h, w * 3, QImage.Format.Format_BGR888)
                    self.sigFrame.emit(qi.copy())
                    self.msleep(33)
                else:
                    self.msleep(120)
            else:
                self.msleep(150)


class CurveWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.data = []
        self.setMinimumHeight(44)

    def set_data(self, d):
        self.data = list(d)[-48:]
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C_CARD))
        if len(self.data) >= 2:
            p.setPen(QPen(QColor(C_ACCENT), 1))
            lo, hi = 0.0, 1.0
            n = len(self.data)
            pts = []
            for i, v in enumerate(self.data):
                x = w * i / (n - 1)
                y = h - 4 - (h - 8) * max(lo, min(hi, v))
                pts.append((x, y))
            for i in range(1, len(pts)):
                p.drawLine(int(pts[i - 1][0]), int(pts[i - 1][1]),
                           int(pts[i][0]), int(pts[i][1]))
        else:
            p.setPen(QColor(C_STROKE))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "暂无数据")
        p.end()


class MainWindow(QMainWindow):
    sigPause = Signal()

    def __init__(self):
        super().__init__()
        self.state = EngineState()
        self.setWindowTitle("微信跳一跳 · 强化学习训练台")
        self.setMinimumSize(900, 640)
        self.resize(1240, 840)
        self.engine = None
        self.det = {}

        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        self.panel = QFrame(objectName="Panel")
        self.panel.setFixedWidth(262)
        pl = QVBoxLayout(self.panel)
        pl.setContentsMargins(14, 14, 14, 14)
        pl.setSpacing(8)
        t1 = QLabel("游戏窗口")
        t1.setObjectName("Title")
        pl.addWidget(t1)
        self.win_list = QListWidget()
        self.win_list.setMinimumHeight(120)
        self.win_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.win_list.setWordWrap(False)
        self.win_list.itemClicked.connect(self.on_pick)
        pl.addWidget(self.win_list, 1)
        self.btn_refresh = QPushButton("刷新窗口")
        self.btn_refresh.clicked.connect(self.refresh_windows)
        pl.addWidget(self.btn_refresh)
        t2 = QLabel("训练参数")
        t2.setObjectName("Title")
        pl.addWidget(t2)
        self.coef_label = QLabel("基础系数  1.000")
        pl.addWidget(self.coef_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(50, 400)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.on_slider)
        pl.addWidget(self.slider)
        rng = QLabel("0.5")
        rng.setObjectName("Muted")
        pl.addWidget(rng, 0, Qt.AlignmentFlag.AlignRight)
        self.chk_greedy = QCheckBox("贪心模式（不再随机探索）")
        self.chk_greedy.toggled.connect(
            lambda v: setattr(self.state, "greedy", v))
        pl.addWidget(self.chk_greedy)
        self.chk_manual = QCheckBox("手动模式（我点目标，程序自己跳）")
        self.chk_manual.toggled.connect(self.on_manual)
        pl.addWidget(self.chk_manual)
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setObjectName("BtnStart")
        self.btn_start.clicked.connect(self.on_start)
        pl.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("BtnStop")
        self.btn_stop.clicked.connect(self.on_stop)
        pl.addWidget(self.btn_stop)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        pl.addWidget(self.btn_pause)
        self.status = QLabel("未选择窗口")
        self.status.setObjectName("StatusLabel")
        self.status.setWordWrap(True)
        pl.addWidget(self.status)

        right = QVBoxLayout()
        right.setSpacing(10)
        cards = QHBoxLayout()
        cards.setSpacing(8)
        self.card_vals = {}
        for name in ("分数", "最高分", "局数", "跳跃数", "平均奖励", "熵"):
            f = QFrame(objectName="Card")
            f.setFixedWidth(104)
            v = QVBoxLayout(f)
            v.setContentsMargins(10, 8, 10, 8)
            v.setSpacing(2)
            lab = QLabel(name)
            lab.setObjectName("Muted")
            val = QLabel("0")
            val.setObjectName("BigNum")
            v.addWidget(lab)
            v.addWidget(val)
            v.addStretch(1)
            cards.addWidget(f)
            self.card_vals[name] = val
        curvef = QFrame(objectName="Card")
        cl = QVBoxLayout(curvef)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(2)
        clab = QLabel("奖励曲线")
        clab.setObjectName("Muted")
        cl.addWidget(clab)
        self.curve = CurveWidget()
        cl.addWidget(self.curve, 1)
        cards.addWidget(curvef, 1)
        right.addLayout(cards)

        pv_title = QLabel("实时画面")
        pv_title.setObjectName("Title")
        right.addWidget(pv_title)
        self.preview = QLabel("等待画面...")
        self.preview.setObjectName("PreviewText")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(200, 180)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored,
                                   QSizePolicy.Policy.Ignored)
        self.preview.setStyleSheet(
            "background:#000000;border:2px solid;"
            "border-color:#808080 #ffffff #ffffff #808080;")
        right.addWidget(self.preview, 1)

        log_title = QLabel("运行日志")
        log_title.setObjectName("Title")
        right.addWidget(log_title)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setFixedHeight(122)
        right.addWidget(self.log)

        outer.addWidget(self.panel)
        outer.addLayout(right, 1)

        QShortcut(QKeySequence(Qt.Key.Key_R), self, self.refresh_windows)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.on_stop)
        self.sigPause.connect(self.toggle_pause)
        self.mouse_hook = wb.MouseHook(self.state)
        self.mouse_hook.start()
        self.key_hook = wb.KeyboardHook(self.state,
                                        lambda: self.sigPause.emit())
        self.key_hook.start()
        self.preview_thread = PreviewThread(self.state)
        self.preview_thread.sigFrame.connect(self.on_frame)
        self.preview_thread.start()
        self.refresh_windows()
        self.log.appendPlainText("强化学习模式：策略网络在 Python 侧训练，Qt 界面负责抓图与静默按压")
        self.log.appendPlainText("快捷键：空格 开始/暂停（游戏窗口在前台也生效），Esc 停止，R 刷新窗口")

    def closeEvent(self, ev):
        self.stop_engine(8000)
        self.preview_thread.stop()
        self.preview_thread.wait(2000)
        self.mouse_hook.stop()
        self.key_hook.stop()
        super().closeEvent(ev)

    def stop_engine(self, timeout_ms=6000):
        state = self.state
        state.running = False
        state.paused = False
        eng = self.engine
        if eng is not None and eng.isRunning():
            eng.wait(timeout_ms)

    def refresh_windows(self):
        self.win_list.clear()
        wins = wb.list_windows()
        wins.sort(key=lambda kv: (0 if ("跳一跳" in kv[1] or "小程序" in kv[1]) else 1,
                                  kv[1]))
        for h, t in wins:
            from PySide6.QtWidgets import QListWidgetItem
            it = QListWidgetItem(t)
            it.setData(Qt.ItemDataRole.UserRole, h)
            self.win_list.addItem(it)

    def on_pick(self, item):
        self.state.target_hwnd = item.data(Qt.ItemDataRole.UserRole)
        self.status.setText("已选择窗口：%s" % item.text())

    def on_slider(self, v):
        c = v / 100.0
        self.state.base_coef = c
        self.coef_label.setText("基础系数  %.3f" % c)

    def on_manual(self, v):
        self.state.manual = v
        self.state.got = False
        self.state.preview = bool(v)
        self.log.appendPlainText("已切换到手动模式：光标会自动移到预测目标，点击即跳"
                        if v else "已切换回自动模式")

    def on_start(self):
        if self.engine and self.engine.isRunning():
            return
        if not self.state.target_hwnd:
            self.log.appendPlainText("请先选择游戏窗口")
            return
        self.state.running = True
        self.state.paused = False
        self.engine = Engine(self.state)
        self.engine.sigLog.connect(self.log.appendPlainText)
        self.engine.sigStats.connect(self.on_stats)
        self.engine.sigDet.connect(self.on_det)
        self.engine.sigReward.connect(self.on_reward)
        self.engine.sigState.connect(lambda s: self.status.setText(
            "%s ｜ %s" % (s, wb.window_title(self.state.target_hwnd or 0))))
        self.engine.start()

    def on_stop(self):
        self.stop_engine(6000)
        self.btn_pause.setText("暂停")

    def toggle_pause(self):
        if not (self.engine and self.engine.isRunning()):
            return
        self.state.paused = not self.state.paused
        self.btn_pause.setText("继续" if self.state.paused else "暂停")

    def on_stats(self, s):
        m = {"分数": s.get("score", 0), "最高分": s.get("best_score", 0),
             "局数": s.get("episodes", 0), "跳跃数": s.get("jumps", 0),
             "平均奖励": "%.3f" % s.get("mean", 0), "熵": "%.2f" % s.get("entropy", 0)}
        for k, v in m.items():
            self.card_vals[k].setText(str(v))

    def on_det(self, d):
        self.det = d or {}

    def on_reward(self, r):
        self.curve.data.append(r)
        self.curve.set_data(self.curve.data)

    def on_frame(self, qi):
        d = self.det
        pm = QPixmap.fromImage(qi)
        p = QPainter(pm)
        p.setPen(QColor("#ffffff"))
        p.setFont(QFont("Consolas", 10))
        txt = "d=%s %s c=%s" % (d.get("d", "-"), d.get("method", "-"),
                                d.get("conf", "-"))
        p.drawText(12, 22, txt)
        p.end()
        self.preview.setPixmap(pm.scaled(
            max(1, self.preview.width() - 4), max(1, self.preview.height() - 4),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))


def main():
    smoke = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = QApplication(sys.argv)
    if "Windows" in QStyleFactory.keys():
        app.setStyle("Windows")
    else:
        app.setStyle("Fusion")
    app.setFont(QFont("Tahoma", 9))
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(C_BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(C_FG))
    pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Text, QColor(C_FG))
    pal.setColor(QPalette.ColorRole.Button, QColor(C_BG))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(C_FG))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(C_ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    scr = win.screen().availableGeometry()
    win.resize(min(1120, scr.width() - 90), min(760, scr.height() - 90))
    win.move(scr.left() + (scr.width() - win.width()) // 2,
             scr.top() + (scr.height() - win.height()) // 2)
    if smoke:
        def auto():
            for i in range(win.win_list.count()):
                it = win.win_list.item(i)
                if "跳一跳" in it.text():
                    win.win_list.setCurrentItem(it)
                    win.on_pick(it)
                    win.on_start()
                    win.log.appendPlainText("smoke: 自动开始 %d 秒" % smoke)
                    return
            win.log.appendPlainText("smoke: 未找到跳一跳窗口")
        QTimer_s = None
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1200, auto)

        def finish():
            win.on_stop()
            win.close()
            from PySide6.QtCore import QTimer as _T
            _T.singleShot(400, app.quit)
        from PySide6.QtCore import QTimer
        QTimer.singleShot((smoke + 3) * 1000, finish)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
