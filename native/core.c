#include "app.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mmsystem.h>

typedef struct {
    char kind[16];
    double ms, x, y, reward, err;
    int has_ms, has_x, has_y, has_reward, has_err;
    DetInfo det;
    StatInfo st;
    char text[64];
} Res;

typedef struct {
    HANDLE h_write;
    HANDLE h_read;
    HANDLE h_proc;
    HANDLE h_thread;
    char rbuf[32768];
    int rlen;
    int rpos;
    int ok;
} Child;

static Child g_child;
static volatile LONG g_child_alive = 0;
static wchar_t g_root[MAX_PATH] = L"";

static int jfind(const char* s, const char* key, int* out_pos) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(s, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t') p++;
    if (*p != ':') return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    *out_pos = (int)(p - s);
    return 1;
}

int jget_num(const char* s, const char* key, double* out) {
    int pos = 0;
    if (!jfind(s, key, &pos)) return 0;
    *out = atof(s + pos);
    return 1;
}

int jget_str(const char* s, const char* key, char* out, int max) {
    int pos = 0;
    if (!jfind(s, key, &pos)) return 0;
    if (s[pos] != '"') return 0;
    pos++;
    int n = 0;
    while (s[pos] && s[pos] != '"' && n < max - 1) {
        if (s[pos] == '\\' && s[pos + 1]) {
            pos++;
            char c = s[pos];
            if (c == 'n') c = ' ';
            else if (c == 't') c = ' ';
            else if (c == 'r') c = ' ';
            else if (c == 'u') {
                pos += 4;
                c = '?';
            }
            out[n++] = c;
        } else {
            out[n++] = s[pos];
        }
        pos++;
    }
    out[n] = 0;
    return 1;
}

static void fb_ensure(FrameBuf* fb, int w, int h) {
    int need = w * h * 4;
    if (fb->cap < need) {
        fb->data = (BYTE*)realloc(fb->data, need);
        fb->cap = need;
    }
    fb->w = w;
    fb->h = h;
}

int grab_frame(HWND hwnd, FrameBuf* fb) {
    RECT rc;
    if (!GetClientRect(hwnd, &rc)) return 0;
    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    if (w < 40 || h < 40) return 0;
    fb_ensure(fb, w, h);

    HDC hdc = GetDC(hwnd);
    if (!hdc) return 0;
    HDC mem = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, w, h);
    HGDIOBJ old = SelectObject(mem, bmp);
    int printed = PrintWindow(hwnd, mem, 3);
    if (!printed) {
        POINT pt = {0, 0};
        ClientToScreen(hwnd, &pt);
        HDC screen = GetDC(NULL);
        BitBlt(mem, 0, 0, w, h, screen, pt.x, pt.y, SRCCOPY);
        ReleaseDC(NULL, screen);
    }
    BITMAPINFO bi;
    memset(&bi, 0, sizeof(bi));
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    int got = GetDIBits(mem, bmp, 0, h, fb->data, &bi, DIB_RGB_COLORS);
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
    ReleaseDC(hwnd, hdc);
    if (!got) return 0;

    long long sum = 0;
    long n = 0;
    for (int i = 0; i < w * h; i += 89) {
        sum += fb->data[i * 4] + fb->data[i * 4 + 1] + fb->data[i * 4 + 2];
        n++;
    }
    if (n > 0 && sum / n < 3) {
        POINT pt = {0, 0};
        ClientToScreen(hwnd, &pt);
        HDC screen = GetDC(NULL);
        HDC m2 = CreateCompatibleDC(screen);
        HBITMAP b2 = CreateCompatibleBitmap(screen, w, h);
        HGDIOBJ o2 = SelectObject(m2, b2);
        BitBlt(m2, 0, 0, w, h, screen, pt.x, pt.y, SRCCOPY);
        GetDIBits(m2, b2, 0, h, fb->data, &bi, DIB_RGB_COLORS);
        SelectObject(m2, o2);
        DeleteObject(b2);
        DeleteDC(m2);
        ReleaseDC(NULL, screen);
    }
    return 1;
}

void publish_frame(FrameBuf* fb) {
    if (!fb->data) return;
    EnterCriticalSection(&g_cs);
    fb_ensure(&g_view, fb->w, fb->h);
    memcpy(g_view.data, fb->data, (size_t)fb->w * fb->h * 4);
    LeaveCriticalSection(&g_cs);
    if (g_main) PostMessage(g_main, APP_FRAME, 0, 0);
}

static double frame_diff(FrameBuf* a, FrameBuf* b) {
    if (!a->data || !b->data) return 999.0;
    if (a->w != b->w || a->h != b->h) return 999.0;
    long long sum = 0;
    long n = 0;
    int step = 4;
    for (int y = 0; y < a->h; y += step) {
        BYTE* pa = a->data + (size_t)y * a->w * 4;
        BYTE* pb = b->data + (size_t)y * b->w * 4;
        for (int x = 0; x < a->w; x += step) {
            int ia = pa[x * 4] + pa[x * 4 + 1] + pa[x * 4 + 2];
            int ib = pb[x * 4] + pb[x * 4 + 1] + pb[x * 4 + 2];
            sum += abs(ia - ib);
            n++;
        }
    }
    if (n == 0) return 999.0;
    return (double)sum / (double)n / 3.0;
}

static int wait_stable(HWND hwnd, FrameBuf* prev, FrameBuf* cur, double timeout) {
    Sleep(320);
    if (!grab_frame(hwnd, prev)) return 0;
    int stable = 0;
    DWORD t0 = GetTickCount();
    while ((GetTickCount() - t0) < (DWORD)(timeout * 1000.0)) {
        if (!g_running) return 0;
        Sleep(70);
        FrameBuf tmp = *prev;
        *prev = *cur;
        *cur = tmp;
        if (!grab_frame(hwnd, cur)) return 0;
        if (frame_diff(prev, cur) < 2.2) {
            stable++;
            if (stable >= 2) return 1;
        } else {
            stable = 0;
        }
    }
    return 0;
}

void bring_front_impl(HWND hwnd) {
    if (!hwnd || !IsWindow(hwnd)) return;
    if (GetForegroundWindow() == hwnd) return;
    if (IsIconic(hwnd)) ShowWindow(hwnd, SW_RESTORE);
    SetForegroundWindow(hwnd);
    Sleep(60);
}

static void send_mouse(DWORD flag) {
    INPUT in;
    memset(&in, 0, sizeof(in));
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = flag;
    SendInput(1, &in, sizeof(in));
}

static double press_hold(int x, int y, double ms) {
    if (ms < 80.0) ms = 80.0;
    SetCursorPos(x, y);
    Sleep(18);
    send_mouse(MOUSEEVENTF_LEFTDOWN);
    LARGE_INTEGER freq, t0, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    double target = ms / 1000.0;
    for (;;) {
        QueryPerformanceCounter(&t);
        double elapsed = (double)(t.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
        if (elapsed >= target) break;
        double remain = target - elapsed;
        if (remain > 0.003) Sleep((DWORD)((remain - 0.002) * 1000.0));
    }
    send_mouse(MOUSEEVENTF_LEFTUP);
    QueryPerformanceCounter(&t);
    return (double)(t.QuadPart - t0.QuadPart) * 1000.0 / (double)freq.QuadPart;
}

static void click_at(int x, int y) {
    SetCursorPos(x, y);
    Sleep(25);
    send_mouse(MOUSEEVENTF_LEFTDOWN);
    Sleep(45);
    send_mouse(MOUSEEVENTF_LEFTUP);
}

void press_stop_all(void) {
    send_mouse(MOUSEEVENTF_LEFTUP);
}

static int find_root(wchar_t* root) {
    if (GetModuleFileNameW(NULL, root, MAX_PATH) == 0) return 0;
    wchar_t* slash = wcsrchr(root, L'\\');
    if (!slash) return 0;
    *slash = 0;
    for (int i = 0; i < 4; i++) {
        wchar_t probe[MAX_PATH];
        wsprintfW(probe, L"%s\\ml\\worker.py", root);
        if (GetFileAttributesW(probe) != INVALID_FILE_ATTRIBUTES) return 1;
        slash = wcsrchr(root, L'\\');
        if (!slash) break;
        *slash = 0;
    }
    return 0;
}

static int child_start(void) {
    wchar_t root[MAX_PATH];
    wchar_t py[MAX_PATH];
    wchar_t script[MAX_PATH];
    wchar_t cmd[1024];
    if (!find_root(root)) {
        log_line(L"找不到项目目录（缺少 ml\\worker.py）");
        return 0;
    }
    wcscpy(g_root, root);

    wsprintfW(py, L"%s\\.venv\\Scripts\\pythonw.exe", root);
    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES) {
        wsprintfW(py, L"%s\\.venv\\Scripts\\python.exe", root);
    }
    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES) {
        log_line(L"缺少 Python 运行环境：%s", py);
        return 0;
    }
    wsprintfW(script, L"%s\\ml\\worker.py", root);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        log_line(L"缺少推理脚本：%s", script);
        return 0;
    }
    wsprintfW(cmd, L"\"%s\" \"%s\"", py, script);

    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE in_r = NULL, in_w = NULL, out_r = NULL, out_w = NULL;
    if (!CreatePipe(&in_r, &in_w, &sa, 1 << 16)) return 0;
    if (!CreatePipe(&out_r, &out_w, &sa, 1 << 16)) return 0;
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    wchar_t logpath[MAX_PATH];
    wsprintfW(logpath, L"%s\\logs\\ml_stderr.log", root);
    CreateDirectoryW(L"logs", NULL);
    HANDLE errf = CreateFileW(logpath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (errf == INVALID_HANDLE_VALUE) errf = out_w;

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = errf;
    si.wShowWindow = SW_HIDE;

    BOOL ok = CreateProcessW(py, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, root, &si, &pi);
    CloseHandle(in_r);
    CloseHandle(out_w);
    if (errf != out_w) CloseHandle(errf);
    if (!ok) {
        CloseHandle(in_w);
        CloseHandle(out_r);
        log_line(L"启动 Python 推理进程失败");
        return 0;
    }
    CloseHandle(pi.hThread);

    memset(&g_child, 0, sizeof(g_child));
    g_child.h_write = in_w;
    g_child.h_read = out_r;
    g_child.h_proc = pi.hProcess;
    g_child.ok = 1;
    InterlockedExchange(&g_child_alive, 1);
    log_line(L"Python 推理进程已启动");
    return 1;
}

static void child_stop(void) {
    if (g_child.h_write) {
        const char* quit = "{\"t\":\"quit\"}\n";
        WriteFile(g_child.h_write, quit, (DWORD)strlen(quit), NULL, NULL);
    }
    Sleep(60);
    if (g_child.h_write) {
        CloseHandle(g_child.h_write);
        g_child.h_write = NULL;
    }
    if (g_child.h_proc) {
        WaitForSingleObject(g_child.h_proc, 1500);
        if (WaitForSingleObject(g_child.h_proc, 0) != WAIT_OBJECT_0) {
            TerminateProcess(g_child.h_proc, 0);
        }
        CloseHandle(g_child.h_proc);
        g_child.h_proc = NULL;
    }
    if (g_child.h_read) {
        CloseHandle(g_child.h_read);
        g_child.h_read = NULL;
    }
    g_child.ok = 0;
    InterlockedExchange(&g_child_alive, 0);
}

static int child_write(const void* data, int len) {
    if (!g_child.h_write || len <= 0) return 0;
    const BYTE* p = (const BYTE*)data;
    int left = len;
    while (left > 0) {
        DWORD wrote = 0;
        if (!WriteFile(g_child.h_write, p, (DWORD)left, &wrote, NULL)) return 0;
        if (wrote == 0) return 0;
        p += wrote;
        left -= (int)wrote;
    }
    return 1;
}

static int child_read_line(char* out, int max) {
    if (!g_child.h_read) return 0;
    for (;;) {
        for (int i = g_child.rpos; i < g_child.rlen; i++) {
            if (g_child.rbuf[i] == '\n') {
                int len = i - g_child.rpos;
                if (len > max - 1) len = max - 1;
                memcpy(out, g_child.rbuf + g_child.rpos, len);
                out[len] = 0;
                g_child.rpos = i + 1;
                return 1;
            }
        }
        if (g_child.rpos > 0) {
            memmove(g_child.rbuf, g_child.rbuf + g_child.rpos, g_child.rlen - g_child.rpos);
            g_child.rlen -= g_child.rpos;
            g_child.rpos = 0;
        }
        if (g_child.rlen >= (int)sizeof(g_child.rbuf)) g_child.rlen = 0;
        DWORD got = 0;
        if (!ReadFile(g_child.h_read, g_child.rbuf + g_child.rlen,
                      (DWORD)(sizeof(g_child.rbuf) - g_child.rlen), &got, NULL)) return 0;
        if (got == 0) return 0;
        g_child.rlen += (int)got;
    }
}

static void apply_stats(StatInfo* st) {
    EnterCriticalSection(&g_cs);
    g_stats = *st;
    LeaveCriticalSection(&g_cs);
    if (g_main) PostMessage(g_main, APP_STATS, 0, 0);
}

static void apply_det(DetInfo* det, double reward, int has_reward) {
    EnterCriticalSection(&g_cs);
    if (det->method[0]) {
        memcpy(&g_det, det, sizeof(DetInfo));
    }
    if (has_reward) {
        g_rewards[g_reward_head] = reward;
        g_reward_head = (g_reward_head + 1) % REWARD_SLOTS;
        if (g_reward_count < REWARD_SLOTS) g_reward_count++;
    }
    LeaveCriticalSection(&g_cs);
    if (g_main) PostMessage(g_main, APP_STATS, 0, 0);
}

static int parse_res(const char* line, Res* r) {
    memset(r, 0, sizeof(*r));
    if (!jget_str(line, "kind", r->kind, sizeof(r->kind))) return 0;
    r->has_ms = jget_num(line, "ms", &r->ms);
    r->has_x = jget_num(line, "x", &r->x);
    r->has_y = jget_num(line, "y", &r->y);
    r->has_reward = jget_num(line, "reward", &r->reward);
    r->has_err = jget_num(line, "err", &r->err);
    jget_str(line, "text", r->text, sizeof(r->text));

    double v = 0;
    if (jget_num(line, "has_piece", &v)) r->det.has_piece = (int)v;
    if (jget_num(line, "has_board", &v)) r->det.has_board = (int)v;
    if (jget_num(line, "over", &v)) r->det.over = (int)v;
    if (jget_num(line, "px", &v)) r->det.px = v;
    if (jget_num(line, "py", &v)) r->det.py = v;
    if (jget_num(line, "bx", &v)) r->det.bx = v;
    if (jget_num(line, "by", &v)) r->det.by = v;
    if (jget_num(line, "d", &v)) r->det.d = v;
    if (jget_num(line, "conf", &v)) r->det.conf = v;
    if (r->has_reward) r->det.reward = r->reward;
    if (r->has_err) r->det.err = r->err;
    jget_str(line, "method", r->det.method, sizeof(r->det.method));
    jget_str(line, "ocr", r->det.text, sizeof(r->det.text));
    jget_str(line, "digits", r->det.digits, sizeof(r->det.digits));

    if (jget_num(line, "episodes", &v)) r->st.episodes = v;
    if (jget_num(line, "jumps", &v)) r->st.jumps = v;
    if (jget_num(line, "mean", &v)) r->st.mean = v;
    if (jget_num(line, "best", &v)) r->st.best = v;
    if (jget_num(line, "entropy", &v)) r->st.entropy = v;
    if (jget_num(line, "scale", &v)) r->st.scale = v;
    if (jget_num(line, "episode_jumps", &v)) r->st.episode_jumps = v;
    if (jget_num(line, "score", &v)) r->st.score = v;
    if (jget_num(line, "best_score", &v)) r->st.best_score = v;
    jget_str(line, "model", r->st.model, sizeof(r->st.model));
    return 1;
}

static int read_response(Res* r) {
    char line[32768];
    while (child_read_line(line, sizeof(line))) {
        char typ[16];
        if (jget_str(line, "t", typ, sizeof(typ)) && strcmp(typ, "log") == 0) {
            char msg[2048];
            if (jget_str(line, "msg", msg, sizeof(msg))) log_line_utf8(msg);
            continue;
        }
        parse_res(line, r);
        return 1;
    }
    return 0;
}

static int save_frame(FrameBuf* fb, const char* phase) {
    if (!fb->data || fb->w <= 0 || fb->h <= 0) return 0;
    wchar_t dir[MAX_PATH];
    wsprintfW(dir, L"%s\\logs", g_root);
    CreateDirectoryW(dir, NULL);
    wchar_t wphase[32];
    MultiByteToWideChar(CP_UTF8, 0, phase, -1, wphase, 32);
    wphase[31] = 0;
    wchar_t path[MAX_PATH];
    wsprintfW(path, L"%s\\frame_%s.bin", dir, wphase);
    HANDLE f = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, NULL,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return 0;
    DWORD total = (DWORD)(fb->w * fb->h * 4);
    BYTE* p = fb->data;
    DWORD left = total;
    int ok = 1;
    while (left > 0) {
        DWORD wrote = 0;
        if (!WriteFile(f, p, left, &wrote, NULL) || wrote == 0) {
            ok = 0;
            break;
        }
        p += wrote;
        left -= wrote;
    }
    CloseHandle(f);
    return ok;
}

static int request_frame(FrameBuf* fb, const char* phase, int id,
                         double base, int greedy, int settled, Res* r) {
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
             "{\"t\":\"frame\",\"id\":%d,\"w\":%d,\"h\":%d,\"phase\":\"%s\","
             "\"base\":%.3f,\"greedy\":%d,\"settled\":%d}\n",
             id, fb->w, fb->h, phase, base, greedy, settled);
    if (!save_frame(fb, phase)) {
        log_line(L"画面写入文件失败");
        return 0;
    }
    if (!child_write(hdr, (int)strlen(hdr))) {
        log_line(L"写入帧头失败");
        return 0;
    }
    if (!read_response(r)) {
        log_line(L"读取推理结果失败");
        return 0;
    }
    return 1;
}

static void random_point(HWND hwnd, int* ox, int* oy) {
    RECT rc;
    GetClientRect(hwnd, &rc);
    POINT pt = {0, 0};
    ClientToScreen(hwnd, &pt);
    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    if (w < 10) w = 10;
    if (h < 10) h = 10;
    double fx = 0.35 + (rand() % 1000) / 1000.0 * 0.30;
    double fy = 0.55 + (rand() % 1000) / 1000.0 * 0.23;
    *ox = pt.x + (int)(w * fx);
    *oy = pt.y + (int)(h * fy);
}

int is_game_process(HWND hwnd) {
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (!pid) return 0;
    HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!p) return 0;
    wchar_t path[MAX_PATH];
    DWORD len = MAX_PATH;
    BOOL ok = QueryFullProcessImageNameW(p, 0, path, &len);
    CloseHandle(p);
    if (!ok) return 0;
    return wcsstr(path, L"WeChatAppEx.exe") != NULL ||
           wcsstr(path, L"WeChat.exe") != NULL ||
           wcsstr(path, L"Weixin.exe") != NULL;
}

static DWORD WINAPI worker_main(LPVOID param) {
    (void)param;
    FrameBuf before = {0}, after = {0}, prev = {0}, cur = {0};
    int id = 1;
    Res res;
    g_state = ST_RUNNING;

    if (!child_start()) {
        log_line(L"推理进程启动失败，引擎停止");
        g_running = 0;
        g_state = ST_STOPPED;
        return 0;
    }
    if (!child_write("{\"t\":\"init\"}\n", (int)strlen("{\"t\":\"init\"}\n")) || !read_response(&res)) {
        log_line(L"推理进程握手失败");
        child_stop();
        g_running = 0;
        g_state = ST_STOPPED;
        return 0;
    }
    apply_stats(&res.st);

    while (g_running && !g_stop_req) {
        if (g_paused) {
            g_state = ST_PAUSED;
            Sleep(120);
            continue;
        }
        g_state = ST_RUNNING;
        HWND hwnd = g_target;
        if (!hwnd || !IsWindow(hwnd)) {
            log_line(L"请先在左侧选择一个游戏窗口");
            g_state = ST_WAITING;
            Sleep(600);
            continue;
        }
        wchar_t title[256];
        if (GetWindowTextW(hwnd, title, 256) <= 0) title[0] = 0;
        if ((!wcsstr(title, L"跳一跳") && !wcsstr(title, L"小程序") &&
             !wcsstr(title, L"微信")) || !is_game_process(hwnd)) {
            log_line(L"目标窗口不是微信小程序（%s），请重新选择", title);
            g_state = ST_WAITING;
            Sleep(700);
            continue;
        }
        bring_front_impl(hwnd);
        if (!grab_frame(hwnd, &before)) {
            Sleep(350);
            continue;
        }
        publish_frame(&before);
        DWORD t_send = GetTickCount();
        if (!request_frame(&before, "before", id, g_base_coef, g_greedy, 1, &res)) {
            log_line(L"推理进程无响应，引擎停止");
            break;
        }
        DWORD cost = GetTickCount() - t_send;
        if (cost > 2500) log_line(L"推理耗时 %.1f 秒（帧 %dx%d）", cost / 1000.0, before.w, before.h);
        apply_stats(&res.st);
        if (strcmp(res.kind, "press") == 0 && res.has_ms) {
            int sx = 0, sy = 0;
            POINT cpt = {0, 0};
            ClientToScreen(hwnd, &cpt);
            if (res.has_x && res.has_y) {
                sx = cpt.x + (int)res.x;
                sy = cpt.y + (int)res.y;
            } else {
                random_point(hwnd, &sx, &sy);
            }
            press_hold(sx, sy, res.ms);
            int stable = wait_stable(hwnd, &prev, &cur, 3.2);
            if (!grab_frame(hwnd, &after)) {
                g_running = 0;
                break;
            }
            publish_frame(&after);
            if (!request_frame(&after, "after", id, g_base_coef, g_greedy, stable, &res)) {
                log_line(L"推理进程无响应，引擎停止");
                break;
            }
            apply_stats(&res.st);
            apply_det(&res.det, res.reward, res.has_reward);
            id++;
            if (strcmp(res.kind, "restart") == 0 && res.has_x && res.has_y) {
                POINT pt = {0, 0};
                ClientToScreen(hwnd, &pt);
                int cx = pt.x + (int)res.x;
                int cy = pt.y + (int)res.y;
                bring_front_impl(hwnd);
                click_at(cx, cy);
                log_line(L"已点击 (%.0f, %.0f) 继续", res.x, res.y);
                Sleep(1500);
            }
        } else if (strcmp(res.kind, "restart") == 0 && res.has_x && res.has_y) {
            POINT pt = {0, 0};
            ClientToScreen(hwnd, &pt);
            int cx = pt.x + (int)res.x;
            int cy = pt.y + (int)res.y;
            bring_front_impl(hwnd);
            click_at(cx, cy);
            log_line(L"已点击 (%.0f, %.0f)", res.x, res.y);
            Sleep(1500);
        } else {
            apply_det(&res.det, 0.0, 0);
            Sleep(380);
        }
    }

    press_stop_all();
    child_stop();
    free(before.data);
    free(after.data);
    free(prev.data);
    free(cur.data);
    InterlockedExchange(&g_running, 0);
    InterlockedExchange(&g_paused, 0);
    g_state = ST_STOPPED;
    log_line(L"引擎已停止");
    if (g_main) PostMessage(g_main, APP_STATS, 0, 0);
    return 0;
}

static HANDLE g_thread = NULL;

void worker_start(HWND hwnd) {
    if (g_running) return;
    g_target = hwnd;
    InterlockedExchange(&g_stop_req, 0);
    InterlockedExchange(&g_paused, 0);
    InterlockedExchange(&g_running, 1);
    g_thread = CreateThread(NULL, 0, worker_main, NULL, 0, NULL);
    if (!g_thread) {
        InterlockedExchange(&g_running, 0);
        log_line(L"线程创建失败");
    }
}

void worker_stop(void) {
    if (!g_running) return;
    InterlockedExchange(&g_running, 0);
    InterlockedExchange(&g_stop_req, 1);
    if (g_child.h_write) {
        CloseHandle(g_child.h_write);
        g_child.h_write = NULL;
    }
    if (g_thread) {
        WaitForSingleObject(g_thread, 4000);
        CloseHandle(g_thread);
        g_thread = NULL;
    }
    child_stop();
    g_state = ST_STOPPED;
    if (g_main) PostMessage(g_main, APP_STATS, 0, 0);
}
