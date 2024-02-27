#include "Window.h"
#include <cassert>

namespace {
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
    }

    return DefWindowProcA(hwnd, uMsg, wParam, lParam);
}
}  // namespace

Window::Window(const std::string& title, const int width, const int height)
    : mWidth(width), mHeight(height), mWindowClassName(title) {
    m_hinstance = GetModuleHandle(NULL);
    assert(m_hinstance);
    DWORD style = WS_OVERLAPPEDWINDOW | WS_CAPTION | WS_SIZEBOX;

    RECT rect;
    rect.left = 50;
    rect.top = 50;
    rect.right = mWidth + rect.left;
    rect.bottom = mHeight + rect.top;
    AdjustWindowRectEx(&rect, style, 0, 0);  // it's required because window frame takes several pixels

    {
        WNDCLASSEX wndcls = {};

        wndcls.cbSize = sizeof(wndcls);
        wndcls.lpfnWndProc = WndProc;
        wndcls.hInstance = m_hinstance;
        wndcls.hIcon = LoadIcon(NULL, IDI_APPLICATION);
        wndcls.hCursor = LoadCursor(NULL, IDC_ARROW);
        wndcls.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
        wndcls.lpszClassName = mWindowClassName.c_str();

        bool res = RegisterClassEx(&wndcls);
        assert(res);
    }

    mHwnd = CreateWindowEx(0, mWindowClassName.c_str(), mWindowClassName.c_str(), style, rect.left, rect.top,
                           rect.right - rect.left, rect.bottom - rect.top, NULL, NULL, m_hinstance, NULL);

    ShowWindow(mHwnd, SW_SHOW);
}

Window::~Window() {
    UnregisterClassA(mWindowClassName.c_str(), (HINSTANCE)::GetModuleHandle(NULL));
}
