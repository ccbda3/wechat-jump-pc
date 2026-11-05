#include "app.h"
#include <windowsx.h>
#include <timeapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

HINSTANCE g_inst = NULL;
HWND g_main = NULL;
CRITICAL_SECTION g_cs;

FrameBuf g_view = {0};
DetInfo g_det;
StatInfo g_stats;
double g_rewards[REWARD_SLOTS];
int g_reward_count = 0;
int g_reward_head = 0;

volatile LONG g_running = 0;
volatile LONG g_paused = 0;
volatile LONG g_stop_req = 0;
volatile LONG g_state = ST_STOPPED;

double g_base_coef = 2.0;
int g_greedy = 0;
HWND g_target = NULL;
wchar_t g_target_title[256] = L"";

int g_btn_hover = -1;
int g_btn_down = -1;
int g_win_sel = -1;
int g_dragging_slider = 0;

wchar_t g_log_text[MAX_LOG][256];
int g_log_count = 0;

#define C_BG      RGB(24, 24, 35)
#define C_PANEL   RGB(38, 38, 54)
#define C_CARD    RGB(43, 43, 62)
#define C_CARD2   RGB(50, 50, 72)
#define C_STROKE  RGB(60, 60, 84)
#define C_FG      RGB(205, 214, 244)
#define C_MUTED   RGB(140, 142, 172)
#define C_ACCENT  RGB(137, 180, 250)
#define C_GREEN   RGB(64, 160, 43)
#define C_GREEN_H RGB(82, 190, 60)
#define C_RED     RGB(205, 25, 60)
#define C_RED_H   RGB(232, 62, 92)
#define C_YELLOW  RGB(210, 150, 20)
#define C_YELLOW_H RGB(236, 180, 42)
#define C_TRACK   RGB(52, 52, 74)
#define C_IMG     RGB(18, 18, 27)
#define C_GOOD    RGB(120, 220, 140)

typedef struct {
    HWND hwnd;
    wchar_t title[200];
} WinItem;

static WinItem g_wins[MAX_WIN];
static int g_win_count = 0;

static double g_scale = 1.0;
static int g_dpi = 96;

static HFONT g_f_sec = NULL;
static HFONT g_f_body = NULL;
static HFONT g_f_small = NULL;
static HFONT g_f_num = NULL;
static HFONT g_f_btn = NULL;
static HFONT g_f_log = NULL;

static Layout g_layout;
static RECT g_btn[BTN_COUNT];

static int S(int v) {
    int r = (int)(v * g_scale);
    return r < 1 ? 1 : r;
}

static void append_log(const wchar_t* s) {
    EnterCriticalSection(&g_cs);
    if (g_log_count < MAX_LOG) {
        wcsncpy(g_log_text[g_log_count], s, 255);
        g_log_text[g_log_count][255] = 0;
        g_log_count++;
    } else {
        memmove(g_log_text[0], g_log_text[1], sizeof(g_log_text[0]) * (MAX_LOG - 1));
        wcsncpy(g_log_text[MAX_LOG - 1], s, 255);
        g_log_text[MAX_LOG - 1][255] = 0;
    }
    LeaveCriticalSection(&g_cs);
    if (g_main) InvalidateRect(g_main, NULL, FALSE);
}

void log_line(const wchar_t* fmt, ...) {
    wchar_t body[256];
    va_list ap;
    va_start(ap, fmt);
    _vsnwprintf(body, 255, fmt, ap);
    body[255] = 0;
    va_end(ap);
    SYSTEMTIME st;
    GetLocalTime(&st);
    wchar_t out[300];
    _snwprintf(out, 299, L"[%02d:%02d:%02d] %s", st.wHour, st.wMinute, st.wSecond, body);
    out[299] = 0;
    append_log(out);
}

void log_line_utf8(const char* s) {
    wchar_t w[300];
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w, 300);
    w[299] = 0;
    append_log(w);
}

BOOL CALLBACK enum_proc(HWND hwnd, LPARAM lp) {
    (void)lp;
    if (hwnd == g_main) return TRUE;
    if (!IsWindowVisible(hwnd)) return TRUE;
    if (GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) return TRUE;
    wchar_t title[200];
    if (GetWindowTextW(hwnd, title, 200) <= 0) return TRUE;
    RECT rc;
    if (!GetWindowRect(hwnd, &rc)) return TRUE;
    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    if (w < 160 || h < 200) return TRUE;
    if (g_win_count >= MAX_WIN) return TRUE;
    g_wins[g_win_count].hwnd = hwnd;
    wcsncpy(g_wins[g_win_count].title, title, 199);
    g_wins[g_win_count].title[199] = 0;
    g_win_count++;
    return TRUE;
}

static void refresh_windows(void) {
    g_win_count = 0;
    EnumWindows(enum_proc, 0);
    g_win_sel = -1;
    for (int i = 0; i < g_win_count; i++) {
        if (wcsstr(g_wins[i].title, L"跳一跳") || wcsstr(g_wins[i].title, L"小程序") ||
            wcsstr(g_wins[i].title, L"微信")) {
            g_win_sel = i;
            break;
        }
    }
    if (g_win_sel < 0 && g_win_count > 0) g_win_sel = 0;
    if (g_win_sel >= 0) {
        g_target = g_wins[g_win_sel].hwnd;
        wcsncpy(g_target_title, g_wins[g_win_sel].title, 255);
        g_target_title[255] = 0;
    } else {
        g_target = NULL;
        g_target_title[0] = 0;
    }
    log_line(L"已扫描到 %d 个窗口", g_win_count);
    InvalidateRect(g_main, NULL, FALSE);
}

static HFONT make_font(int pt, int weight) {
    int height = -(int)(pt * 4.0 / 3.0 * g_scale);
    return CreateFontW(height, 0, 0, 0, weight, FALSE, FALSE, FALSE,
                       DEFAULT_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
                       CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Microsoft YaHei UI");
}

static void fill_rect(HDC dc, RECT r, COLORREF c) {
    HBRUSH b = CreateSolidBrush(c);
    FillRect(dc, &r, b);
    DeleteObject(b);
}

static void setf(HDC dc, HFONT f, COLORREF c) {
    SelectObject(dc, f);
    SetTextColor(dc, c);
    SetBkMode(dc, TRANSPARENT);
}

static void round_box(HDC dc, RECT r, COLORREF fill, COLORREF stroke, int rad) {
    HBRUSH b = CreateSolidBrush(fill);
    HPEN p = CreatePen(PS_SOLID, 1, stroke);
    HGDIOBJ ob = SelectObject(dc, b);
    HGDIOBJ op = SelectObject(dc, p);
    RoundRect(dc, r.left, r.top, r.right, r.bottom, rad, rad);
    SelectObject(dc, ob);
    SelectObject(dc, op);
    DeleteObject(b);
    DeleteObject(p);
}

static void draw_text(HDC dc, RECT r, const wchar_t* s, HFONT f, COLORREF c, UINT flags) {
    setf(dc, f, c);
    DrawTextW(dc, s, -1, &r, flags);
}

static void draw_button(HDC dc, RECT r, const wchar_t* label, COLORREF base,
                        COLORREF hot, COLORREF text, int hovered, int enabled) {
    COLORREF fill = enabled ? (hovered ? hot : base) : RGB(58, 58, 74);
    round_box(dc, r, fill, fill, S(10));
    RECT tr = r;
    draw_text(dc, tr, label, g_f_btn, enabled ? text : C_MUTED,
              DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

static void draw_card(HDC dc, RECT r, const wchar_t* label, const wchar_t* value,
                      COLORREF value_color) {
    round_box(dc, r, C_CARD, C_STROKE, S(10));
    RECT lr = r;
    lr.left += S(12);
    lr.top += S(8);
    draw_text(dc, lr, label, g_f_small, C_MUTED, DT_LEFT | DT_TOP | DT_SINGLELINE);
    RECT vr = r;
    vr.left += S(12);
    vr.bottom -= S(6);
    vr.top += S(22);
    draw_text(dc, vr, value, g_f_num, value_color, DT_LEFT | DT_BOTTOM | DT_SINGLELINE);
}

static void reward_at(double* out, int* count) {
    EnterCriticalSection(&g_cs);
    *count = g_reward_count;
    for (int i = 0; i < g_reward_count; i++) {
        int idx = (g_reward_head - g_reward_count + i + REWARD_SLOTS) % REWARD_SLOTS;
        out[i] = g_rewards[idx];
    }
    LeaveCriticalSection(&g_cs);
}

static void do_layout(RECT* c, Layout* L) {
    int m = S(18);
    int left_w = S(272);
    L->panel.left = m;
    L->panel.top = m;
    L->panel.right = m + left_w;
    L->panel.bottom = c->bottom - m;
    int pad = S(14);
    int ix = L->panel.left + pad;
    int iw = left_w - 2 * pad;

    L->list_title.left = ix;
    L->list_title.top = L->panel.top + pad;
    L->list_title.right = ix + iw;
    L->list_title.bottom = L->list_title.top + S(20);

    L->list_box.left = ix;
    L->list_box.top = L->list_title.bottom + S(8);
    L->list_box.right = ix + iw;
    L->list_box.bottom = L->list_box.top + S(196);

    L->btn_refresh.left = ix;
    L->btn_refresh.top = L->list_box.bottom + S(10);
    L->btn_refresh.right = ix + iw;
    L->btn_refresh.bottom = L->btn_refresh.top + S(30);

    L->param_title.left = ix;
    L->param_title.top = L->btn_refresh.bottom + S(20);
    L->param_title.right = ix + iw;
    L->param_title.bottom = L->param_title.top + S(20);

    L->coef_label.left = ix;
    L->coef_label.top = L->param_title.bottom + S(10);
    L->coef_label.right = ix + iw;
    L->coef_label.bottom = L->coef_label.top + S(22);

    L->slider.left = ix;
    L->slider.top = L->coef_label.bottom + S(8);
    L->slider.right = ix + iw;
    L->slider.bottom = L->slider.top + S(18);

    L->coef_range.left = ix;
    L->coef_range.top = L->slider.bottom + S(2);
    L->coef_range.right = ix + iw;
    L->coef_range.bottom = L->coef_range.top + S(18);

    L->chk.left = ix;
    L->chk.top = L->coef_range.bottom + S(14);
    L->chk.right = ix + iw;
    L->chk.bottom = L->chk.top + S(24);

    L->btn_start.left = ix;
    L->btn_start.top = L->chk.bottom + S(18);
    L->btn_start.right = ix + iw;
    L->btn_start.bottom = L->btn_start.top + S(46);

    L->btn_stop.left = ix;
    L->btn_stop.top = L->btn_start.bottom + S(10);
    L->btn_stop.right = ix + iw;
    L->btn_stop.bottom = L->btn_stop.top + S(40);

    L->btn_pause.left = ix;
    L->btn_pause.top = L->btn_stop.bottom + S(8);
    L->btn_pause.right = ix + iw;
    L->btn_pause.bottom = L->btn_pause.top + S(36);

    L->status.left = ix;
    L->status.top = L->panel.bottom - S(44);
    L->status.right = ix + iw;
    L->status.bottom = L->panel.bottom - S(16);

    int rx = L->panel.right + S(16);
    int rw = c->right - m - rx;
    int gap = S(10);
    int cw = (rw - gap * 5) / 6;
    for (int i = 0; i < 6; i++) {
        L->cards[i].left = rx + i * (cw + gap);
        L->cards[i].top = m;
        L->cards[i].right = L->cards[i].left + cw;
        L->cards[i].bottom = m + S(78);
    }
    L->preview.left = rx;
    L->preview.top = m + S(78) + S(14);
    L->preview.right = c->right - m;
    L->preview.bottom = c->bottom - S(248);

    L->log_box.left = rx;
    L->log_box.top = c->bottom - S(232);
    L->log_box.right = c->right - m;
    L->log_box.bottom = c->bottom - m;

    g_btn[BTN_REFRESH] = L->btn_refresh;
    g_btn[BTN_START] = L->btn_start;
    g_btn[BTN_STOP] = L->btn_stop;
    g_btn[BTN_PAUSE] = L->btn_pause;
}

static int state_code(void) {
    if (g_paused) return ST_PAUSED;
    if (g_running) return ST_RUNNING;
    return ST_STOPPED;
}

static void draw_preview(HDC dc, Layout* L) {
    round_box(dc, L->preview, C_PANEL, C_STROKE, S(12));
    RECT tr = L->preview;
    tr.left += S(14);
    tr.top += S(10);
    tr.right -= S(120);
    tr.bottom = tr.top + S(22);
    draw_text(dc, tr, L"实时画面", g_f_sec, C_ACCENT, DT_LEFT | DT_TOP | DT_SINGLELINE);

    int st = state_code();
    const wchar_t* label = (st == ST_RUNNING) ? L"运行中" : (st == ST_PAUSED) ? L"已暂停" : L"已停止";
    COLORREF pill = (st == ST_RUNNING) ? C_GREEN : (st == ST_PAUSED) ? C_YELLOW : RGB(90, 90, 110);
    RECT pr;
    pr.right = L->preview.right - S(14);
    pr.top = L->preview.top + S(8);
    pr.bottom = pr.top + S(24);
    pr.left = pr.right - S(78);
    round_box(dc, pr, pill, pill, S(12));
    draw_text(dc, pr, label, g_f_small, RGB(255, 255, 255),
              DT_CENTER | DT_VCENTER | DT_SINGLELINE);

    RECT img;
    img.left = L->preview.left + S(12);
    img.top = L->preview.top + S(38);
    img.right = L->preview.right - S(12);
    img.bottom = L->preview.bottom - S(12);
    fill_rect(dc, img, C_IMG);

    EnterCriticalSection(&g_cs);
    int vw = g_view.w;
    int vh = g_view.h;
    int have = (g_view.data != NULL && vw > 0 && vh > 0);
    DetInfo det = g_det;
    if (have) {
        int iw = img.right - img.left;
        int ih = img.bottom - img.top;
        double sc = fmin((double)iw / vw, (double)ih / vh);
        int dw = (int)(vw * sc);
        int dh = (int)(vh * sc);
        int dx = img.left + (iw - dw) / 2;
        int dy = img.top + (ih - dh) / 2;
        BITMAPINFO bi;
        memset(&bi, 0, sizeof(bi));
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = vw;
        bi.bmiHeader.biHeight = -vh;
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        SetStretchBltMode(dc, HALFTONE);
        SetBrushOrgEx(dc, 0, 0, NULL);
        StretchDIBits(dc, dx, dy, dw, dh, 0, 0, vw, vh, g_view.data,
                      &bi, DIB_RGB_COLORS, SRCCOPY);
        LeaveCriticalSection(&g_cs);

        if (det.has_piece) {
            int px = dx + (int)(det.px * sc);
            int py = dy + (int)(det.py * sc);
            HPEN pen = CreatePen(PS_SOLID, S(2), RGB(255, 90, 90));
            HGDIOBJ op = SelectObject(dc, pen);
            HGDIOBJ ob = SelectObject(dc, GetStockObject(NULL_BRUSH));
            Ellipse(dc, px - S(11), py - S(11), px + S(11), py + S(11));
            SelectObject(dc, op);
            SelectObject(dc, ob);
            DeleteObject(pen);
        }
        if (det.has_board) {
            int bx = dx + (int)(det.bx * sc);
            int by = dy + (int)(det.by * sc);
            HPEN pen = CreatePen(PS_SOLID, S(2), RGB(90, 150, 255));
            HGDIOBJ op = SelectObject(dc, pen);
            HGDIOBJ ob = SelectObject(dc, GetStockObject(NULL_BRUSH));
            Ellipse(dc, bx - S(11), by - S(11), bx + S(11), by + S(11));
            SelectObject(dc, op);
            SelectObject(dc, ob);
            DeleteObject(pen);
        }
        if (det.has_piece && det.has_board) {
            int px = dx + (int)(det.px * sc);
            int py = dy + (int)(det.py * sc);
            int bx = dx + (int)(det.bx * sc);
            int by = dy + (int)(det.by * sc);
            HPEN pen = CreatePen(PS_SOLID, S(2), C_GOOD);
            HGDIOBJ op = SelectObject(dc, pen);
            MoveToEx(dc, px, py, NULL);
            LineTo(dc, bx, by);
            SelectObject(dc, op);
            DeleteObject(pen);
            wchar_t buf[160];
            _snwprintf(buf, 159, L"d=%.0f  %S  c=%.2f", det.d, det.method, det.conf);
            buf[159] = 0;
            RECT tr2;
            tr2.left = (px + bx) / 2 + S(6);
            tr2.top = (py + by) / 2 - S(22);
            tr2.right = tr2.left + S(240);
            tr2.bottom = tr2.top + S(20);
            draw_text(dc, tr2, buf, g_f_small, C_GOOD, DT_LEFT | DT_TOP | DT_SINGLELINE);
        }
        if (det.reward > 0.0 && det.err > 0.0) {
            wchar_t buf[160];
            _snwprintf(buf, 159, L"奖励 %.2f   偏差 %.0fpx", det.reward, det.err);
            buf[159] = 0;
            RECT tr3;
            tr3.left = img.left + S(10);
            tr3.top = img.top + S(8);
            tr3.right = tr3.left + S(300);
            tr3.bottom = tr3.top + S(22);
            draw_text(dc, tr3, buf, g_f_body, C_GOOD, DT_LEFT | DT_TOP | DT_SINGLELINE);
        }
        if (det.over) {
            RECT tr4;
            tr4.left = img.left + S(10);
            tr4.top = img.top + S(34);
            tr4.right = tr4.left + S(300);
            tr4.bottom = tr4.top + S(24);
            wchar_t buf[160];
            if (det.text[0]) {
                MultiByteToWideChar(CP_UTF8, 0, det.text, -1, buf, 160);
                buf[159] = 0;
            } else {
                wcsncpy(buf, L"结算页", 159);
                buf[159] = 0;
            }
            draw_text(dc, tr4, buf, g_f_body, RGB(255, 120, 120),
                      DT_LEFT | DT_TOP | DT_SINGLELINE);
        }
    } else {
        LeaveCriticalSection(&g_cs);
        draw_text(dc, img, L"等待画面…", g_f_body, C_MUTED,
                  DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }
}

static void draw_log(HDC dc, Layout* L) {
    round_box(dc, L->log_box, C_PANEL, C_STROKE, S(12));
    RECT tr = L->log_box;
    tr.left += S(14);
    tr.top += S(10);
    tr.bottom = tr.top + S(22);
    draw_text(dc, tr, L"运行日志", g_f_sec, C_ACCENT, DT_LEFT | DT_TOP | DT_SINGLELINE);

    int line_h = S(18);
    RECT body;
    body.left = L->log_box.left + S(14);
    body.top = tr.bottom + S(6);
    body.right = L->log_box.right - S(14);
    body.bottom = L->log_box.bottom - S(10);
    int rows = (body.bottom - body.top) / line_h;
    if (rows < 1) return;

    EnterCriticalSection(&g_cs);
    int total = g_log_count;
    int start = total - rows;
    if (start < 0) start = 0;
    for (int i = 0; i < rows && start + i < total; i++) {
        RECT r = body;
        r.top = body.top + i * line_h;
        r.bottom = r.top + line_h;
        draw_text(dc, r, g_log_text[start + i], g_f_log, C_FG,
                  DT_LEFT | DT_TOP | DT_SINGLELINE | DT_END_ELLIPSIS);
    }
    LeaveCriticalSection(&g_cs);
}

static void paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps);
    RECT cr;
    GetClientRect(hwnd, &cr);
    int w = cr.right;
    int h = cr.bottom;
    HDC mem = CreateCompatibleDC(dc);
    HBITMAP bmp = CreateCompatibleBitmap(dc, w, h);
    HGDIOBJ old = SelectObject(mem, bmp);

    fill_rect(mem, cr, C_BG);
    do_layout(&cr, &g_layout);

    round_box(mem, g_layout.panel, C_PANEL, C_STROKE, S(12));

    draw_text(mem, g_layout.list_title, L"游戏窗口", g_f_sec, C_ACCENT,
              DT_LEFT | DT_TOP | DT_SINGLELINE);
    round_box(mem, g_layout.list_box, C_CARD, C_STROKE, S(10));

    int row_h = S(24);
    int row_top = g_layout.list_box.top + S(4);
    int max_rows = (g_layout.list_box.bottom - row_top - S(4)) / row_h;
    for (int i = 0; i < g_win_count && i < max_rows; i++) {
        RECT r;
        r.left = g_layout.list_box.left + S(4);
        r.top = row_top + i * row_h;
        r.right = g_layout.list_box.right - S(4);
        r.bottom = r.top + row_h;
        if (i == g_win_sel) {
            round_box(mem, r, RGB(60, 74, 110), RGB(60, 74, 110), S(6));
        }
        RECT tr = r;
        tr.left += S(8);
        tr.right -= S(8);
        draw_text(mem, tr, g_wins[i].title, g_f_small,
                  i == g_win_sel ? RGB(235, 242, 255) : C_FG,
                  DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
    }
    if (g_win_count == 0) {
        RECT tr = g_layout.list_box;
        draw_text(mem, tr, L"未找到窗口", g_f_small, C_MUTED,
                  DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }

    draw_button(mem, g_layout.btn_refresh, L"刷新窗口", C_CARD2, RGB(64, 64, 90), C_FG,
                g_btn_hover == BTN_REFRESH, 1);

    draw_text(mem, g_layout.param_title, L"训练参数", g_f_sec, C_ACCENT,
              DT_LEFT | DT_TOP | DT_SINGLELINE);

    wchar_t buf[80];
    _snwprintf(buf, 79, L"基础系数  %.3f", g_base_coef);
    buf[79] = 0;
    draw_text(mem, g_layout.coef_label, buf, g_f_body, C_FG,
              DT_LEFT | DT_TOP | DT_SINGLELINE);

    RECT track = g_layout.slider;
    track.top = (track.top + track.bottom) / 2 - S(3);
    track.bottom = track.top + S(6);
    fill_rect(mem, track, C_TRACK);
    double frac = (g_base_coef - 0.5) / 3.5;
    int knob_x = g_layout.slider.left + (int)((g_layout.slider.right - g_layout.slider.left) * frac);
    RECT done = track;
    done.right = knob_x;
    fill_rect(mem, done, C_ACCENT);
    int ky = (g_layout.slider.top + g_layout.slider.bottom) / 2;
    HPEN pen = CreatePen(PS_SOLID, 1, C_ACCENT);
    HGDIOBJ op = SelectObject(mem, pen);
    HBRUSH kb = CreateSolidBrush(C_ACCENT);
    HGDIOBJ ob = SelectObject(mem, kb);
    Ellipse(mem, knob_x - S(7), ky - S(7), knob_x + S(7), ky + S(7));
    SelectObject(mem, op);
    SelectObject(mem, ob);
    DeleteObject(pen);
    DeleteObject(kb);

    RECT rr = g_layout.coef_range;
    draw_text(mem, rr, L"0.5", g_f_small, C_MUTED, DT_LEFT | DT_TOP | DT_SINGLELINE);
    draw_text(mem, rr, L"4.0", g_f_small, C_MUTED, DT_RIGHT | DT_TOP | DT_SINGLELINE);

    RECT cb;
    cb.left = g_layout.chk.left;
    cb.top = g_layout.chk.top + S(3);
    cb.right = cb.left + S(18);
    cb.bottom = cb.top + S(18);
    round_box(mem, cb, g_greedy ? C_ACCENT : C_CARD2, C_STROKE, S(5));
    if (g_greedy) {
        draw_text(mem, cb, L"✓", g_f_body, RGB(255, 255, 255),
                  DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    }
    RECT cl;
    cl.left = cb.right + S(8);
    cl.top = g_layout.chk.top;
    cl.right = g_layout.chk.right;
    cl.bottom = g_layout.chk.bottom;
    draw_text(mem, cl, L"贪心模式（不再随机探索）", g_f_body, C_FG,
              DT_LEFT | DT_VCENTER | DT_SINGLELINE);

    int running = (g_running != 0);
    draw_button(mem, g_layout.btn_start, running ? L"▶  正在运行" : L"▶  开始训练",
                C_GREEN, C_GREEN_H, RGB(255, 255, 255),
                g_btn_hover == BTN_START, !running);
    draw_button(mem, g_layout.btn_stop, L"■  停止", C_RED, C_RED_H, RGB(255, 255, 255),
                g_btn_hover == BTN_STOP, running);
    draw_button(mem, g_layout.btn_pause, g_paused ? L"▶  继续" : L"⏸  暂停",
                C_YELLOW, C_YELLOW_H, RGB(255, 255, 255),
                g_btn_hover == BTN_PAUSE, running);

    StatInfo st;
    EnterCriticalSection(&g_cs);
    st = g_stats;
    LeaveCriticalSection(&g_cs);

    draw_text(mem, g_layout.status, g_target_title[0] ? g_target_title : L"未选择窗口",
              g_f_small, g_target_title[0] ? C_MUTED : RGB(240, 120, 120),
              DT_LEFT | DT_TOP | DT_SINGLELINE | DT_END_ELLIPSIS);

    wchar_t v[64];
    _snwprintf(v, 63, L"%d", (int)st.episodes);
    v[63] = 0;
    draw_card(mem, g_layout.cards[0], L"局数", v, C_FG);
    _snwprintf(v, 63, L"%d", (int)st.jumps);
    v[63] = 0;
    draw_card(mem, g_layout.cards[1], L"跳跃数", v, C_FG);
    _snwprintf(v, 63, L"%.2f", st.mean);
    v[63] = 0;
    draw_card(mem, g_layout.cards[2], L"平均奖励", v, C_ACCENT);
    _snwprintf(v, 63, L"%.2f", st.best);
    v[63] = 0;
    draw_card(mem, g_layout.cards[3], L"最佳奖励", v, C_GOOD);
    _snwprintf(v, 63, L"%.2f", st.entropy);
    v[63] = 0;
    draw_card(mem, g_layout.cards[4], L"策略熵", v, C_FG);

    RECT crd = g_layout.cards[5];
    round_box(mem, crd, C_CARD, C_STROKE, S(10));
    RECT lr = crd;
    lr.left += S(12);
    lr.top += S(8);
    draw_text(mem, lr, L"奖励曲线", g_f_small, C_MUTED, DT_LEFT | DT_TOP | DT_SINGLELINE);

    double rw[REWARD_SLOTS];
    int rc = 0;
    reward_at(rw, &rc);
    RECT plot;
    plot.left = crd.left + S(10);
    plot.right = crd.right - S(10);
    plot.top = crd.top + S(26);
    plot.bottom = crd.bottom - S(9);
    if (rc < 2) {
        draw_text(mem, plot, L"暂无数据", g_f_small, C_MUTED,
                  DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    } else {
        int n = rc;
        double maxv = 1.0;
        for (int i = 0; i < n; i++) if (rw[i] > maxv) maxv = rw[i];
        HPEN lp = CreatePen(PS_SOLID, S(2), C_GOOD);
        HGDIOBJ lop = SelectObject(mem, lp);
        for (int i = 0; i < n; i++) {
            int x = plot.left + (int)((double)(plot.right - plot.left) * i / (n - 1));
            int y = plot.bottom - (int)((plot.bottom - plot.top) * (rw[i] / maxv));
            if (i == 0) MoveToEx(mem, x, y, NULL);
            else LineTo(mem, x, y);
        }
        SelectObject(mem, lop);
        DeleteObject(lp);
    }

    draw_preview(mem, &g_layout);
    draw_log(mem, &g_layout);

    BitBlt(dc, 0, 0, w, h, mem, 0, 0, SRCCOPY);
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
    EndPaint(hwnd, &ps);
}

static int hit_btn(POINT p) {
    for (int i = 0; i < BTN_COUNT; i++) {
        if (p.x >= g_btn[i].left && p.x < g_btn[i].right &&
            p.y >= g_btn[i].top && p.y < g_btn[i].bottom) return i;
    }
    return -1;
}

static void set_slider(int mx) {
    double span = g_layout.slider.right - g_layout.slider.left;
    if (span < 1) span = 1;
    double frac = (mx - g_layout.slider.left) / span;
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    g_base_coef = 0.5 + frac * 3.5;
    InvalidateRect(g_main, NULL, FALSE);
}

static void click_row(int my) {
    int row_h = S(24);
    int row_top = g_layout.list_box.top + S(4);
    int idx = (my - row_top) / row_h;
    if (idx < 0 || idx >= g_win_count) return;
    g_win_sel = idx;
    g_target = g_wins[idx].hwnd;
    wcsncpy(g_target_title, g_wins[idx].title, 255);
    g_target_title[255] = 0;
    log_line(L"已选择窗口：%s", g_wins[idx].title);
    InvalidateRect(g_main, NULL, FALSE);
}

static void do_button(int id) {
    if (id == BTN_REFRESH) {
        refresh_windows();
    } else if (id == BTN_START) {
        if (g_running) return;
        if (!g_target || !IsWindow(g_target)) {
            log_line(L"请先选择游戏窗口");
            return;
        }
        worker_start(g_target);
    } else if (id == BTN_STOP) {
        if (g_running) worker_stop();
    } else if (id == BTN_PAUSE) {
        if (!g_running) return;
        InterlockedExchange(&g_paused, g_paused ? 0 : 1);
        log_line(g_paused ? L"已暂停" : L"已继续");
    }
    InvalidateRect(g_main, NULL, FALSE);
}

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        return 0;
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT:
        paint(hwnd);
        return 0;
    case WM_SIZE:
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    case WM_MOUSEMOVE: {
        POINT p;
        p.x = GET_X_LPARAM(lp);
        p.y = GET_Y_LPARAM(lp);
        int hb = hit_btn(p);
        if (hb != g_btn_hover) {
            g_btn_hover = hb;
            InvalidateRect(hwnd, NULL, FALSE);
        }
        if (g_dragging_slider) set_slider(p.x);
        return 0;
    }
    case WM_LBUTTONDOWN: {
        POINT p;
        p.x = GET_X_LPARAM(lp);
        p.y = GET_Y_LPARAM(lp);
        if (PtInRect(&g_layout.slider, p)) {
            g_dragging_slider = 1;
            SetCapture(hwnd);
            set_slider(p.x);
            return 0;
        }
        if (PtInRect(&g_layout.chk, p)) {
            g_greedy = !g_greedy;
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        }
        if (PtInRect(&g_layout.list_box, p)) {
            click_row(p.y);
            return 0;
        }
        int hb = hit_btn(p);
        if (hb >= 0) {
            g_btn_down = hb;
            InvalidateRect(hwnd, NULL, FALSE);
        }
        return 0;
    }
    case WM_LBUTTONUP: {
        POINT p;
        p.x = GET_X_LPARAM(lp);
        p.y = GET_Y_LPARAM(lp);
        if (g_dragging_slider) {
            g_dragging_slider = 0;
            ReleaseCapture();
        }
        int hb = hit_btn(p);
        if (g_btn_down >= 0 && hb == g_btn_down) {
            int id = g_btn_down;
            g_btn_down = -1;
            do_button(id);
        } else {
            g_btn_down = -1;
        }
        return 0;
    }
    case WM_MOUSEWHEEL:
        return 0;
    case WM_KEYDOWN:
        if (wp == VK_SPACE) {
            if (!g_running) do_button(BTN_START);
            else do_button(BTN_PAUSE);
            return 0;
        }
        if (wp == VK_ESCAPE) {
            do_button(BTN_STOP);
            return 0;
        }
        if (wp == 'R') {
            do_button(BTN_REFRESH);
            return 0;
        }
        return 0;
    case APP_FRAME:
    case APP_STATS:
        InvalidateRect(hwnd, NULL, FALSE);
        return 0;
    case WM_CLOSE:
        worker_stop();
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE prev, LPSTR cmd, int show) {
    (void)prev;
    (void)cmd;
    g_inst = inst;
    srand((unsigned)GetTickCount());
    SetProcessDPIAware();
    timeBeginPeriod(1);

    HDC screen = GetDC(NULL);
    g_dpi = GetDeviceCaps(screen, LOGPIXELSY);
    ReleaseDC(NULL, screen);
    if (g_dpi <= 0) g_dpi = 96;
    RECT wa;
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
    double ds = g_dpi / 96.0;
    double fit_w = (wa.right - wa.left) * 0.58 / 1200.0;
    double fit_h = (wa.bottom - wa.top) * 0.86 / 800.0;
    g_scale = fmin(ds, fmin(fit_w, fit_h));
    if (g_scale < 0.8) g_scale = 0.8;

    InitializeCriticalSection(&g_cs);

    const wchar_t* cls = L"JumpRlWindow";
    WNDCLASSEXW wc;
    memset(&wc, 0, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
    wc.lpfnWndProc = wndproc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = NULL;
    wc.lpszClassName = cls;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    RegisterClassExW(&wc);

    int ww = S(1200);
    int wh = S(800);
    int sx = wa.right - ww - S(18);
    int sy = wa.top + S(18);
    g_main = CreateWindowExW(0, cls, L"微信跳一跳 · 强化学习训练台",
                             WS_OVERLAPPEDWINDOW, sx, sy, ww, wh,
                             NULL, NULL, inst, NULL);
    if (!g_main) return 1;

    g_f_sec = make_font(12, FW_BOLD);
    g_f_body = make_font(11, FW_NORMAL);
    g_f_small = make_font(9, FW_NORMAL);
    g_f_num = make_font(20, FW_BOLD);
    g_f_btn = make_font(11, FW_BOLD);
    g_f_log = make_font(9, FW_NORMAL);

    RECT rc;
    GetClientRect(g_main, &rc);
    do_layout(&rc, &g_layout);

    refresh_windows();
    log_line(L"强化学习模式：策略网络在 Python 侧训练，C 侧负责抓图与精确按压");
    log_line(L"奖励 = 落点离目标中心越近越高，跳到中心为 1.00，掉落为 0.00");
    log_line(L"结算页会通过中文 OCR 识别「再玩一局」并自动点击继续训练");
    log_line(L"快捷键：空格 开始/暂停，Esc 停止，R 刷新窗口");

    ShowWindow(g_main, show);
    UpdateWindow(g_main);

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    worker_stop();
    if (g_view.data) free(g_view.data);
    if (g_f_sec) DeleteObject(g_f_sec);
    if (g_f_body) DeleteObject(g_f_body);
    if (g_f_small) DeleteObject(g_f_small);
    if (g_f_num) DeleteObject(g_f_num);
    if (g_f_btn) DeleteObject(g_f_btn);
    if (g_f_log) DeleteObject(g_f_log);
    DeleteCriticalSection(&g_cs);
    timeEndPeriod(1);
    return 0;
}
