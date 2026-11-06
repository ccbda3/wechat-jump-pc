#ifndef APP_H
#define APP_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define APP_LOG   (WM_APP + 1)
#define APP_FRAME (WM_APP + 2)
#define APP_STATS (WM_APP + 3)
#define APP_HOTKEY (WM_APP + 4)

#define MAX_LOG 400
#define MAX_WIN 48
#define REWARD_SLOTS 48

#define ST_STOPPED 0
#define ST_RUNNING 1
#define ST_PAUSED 2
#define ST_WAITING 3

typedef struct {
    BYTE* data;
    int cap;
    int w;
    int h;
} FrameBuf;

typedef struct {
    int has_piece;
    int has_board;
    int over;
    double px, py, bx, by, d, conf, reward, err;
    char method[24];
    char text[64];
} DetInfo;

typedef struct {
    double episodes, jumps, mean, best, entropy, scale, reward, episode_jumps;
    double score, best_score;
    char model[40];
} StatInfo;

typedef struct {
    RECT panel;
    RECT list_title;
    RECT list_box;
    RECT btn_refresh;
    RECT param_title;
    RECT coef_label;
    RECT slider;
    RECT coef_range;
    RECT chk;
    RECT btn_start;
    RECT btn_stop;
    RECT btn_pause;
    RECT status;
    RECT cards[7];
    RECT preview;
    RECT log_box;
} Layout;

typedef enum {
    BTN_REFRESH = 0,
    BTN_START = 1,
    BTN_STOP = 2,
    BTN_PAUSE = 3,
    BTN_COUNT = 4
} BtnId;

extern HINSTANCE g_inst;
extern HWND g_main;
extern CRITICAL_SECTION g_cs;

extern FrameBuf g_view;
extern DetInfo g_det;
extern StatInfo g_stats;
extern double g_rewards[REWARD_SLOTS];
extern int g_reward_count;
extern int g_reward_head;

extern volatile LONG g_running;
extern volatile LONG g_paused;
extern volatile LONG g_stop_req;
extern volatile LONG g_state;

extern double g_base_coef;
extern int g_greedy;
extern HWND g_target;
extern wchar_t g_target_title[256];
extern int g_btn_hover;
extern int g_btn_down;
extern int g_win_sel;
extern int g_dragging_slider;

extern wchar_t g_log_text[MAX_LOG][256];
extern int g_log_count;

void log_line(const wchar_t* fmt, ...);
void log_line_utf8(const char* s);
void publish_frame(FrameBuf* fb);
int grab_frame(HWND hwnd, FrameBuf* fb);
void worker_start(HWND hwnd);
void worker_stop(void);
void press_stop_all(void);

int jget_num(const char* s, const char* key, double* out);
int jget_str(const char* s, const char* key, char* out, int max);

#endif
