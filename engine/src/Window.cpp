#include "Window.h"
#include <Keyboard.h>
#include <Mouse.h>
#include <cassert>

#include <imgui/backends/imgui_impl_dx12.h>
#include <imgui/backends/imgui_impl_win32.h>
#include <imgui/imgui.h>

namespace {
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            PostQuitMessage(0);
            return 0;
        case WM_ACTIVATEAPP:
            DirectX::Keyboard::ProcessMessage(uMsg, wParam, lParam);
            DirectX::Mouse::ProcessMessage(uMsg, wParam, lParam);
            break;
        case WM_ACTIVATE:
        case WM_INPUT:
        case WM_MOUSEMOVE:
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
        case WM_MOUSEWHEEL:
        case WM_XBUTTONDOWN:
        case WM_XBUTTONUP:
        case WM_MOUSEHOVER:
            DirectX::Mouse::ProcessMessage(uMsg, wParam, lParam);
            break;
        case WM_KEYDOWN:
        case WM_KEYUP:
        case WM_SYSKEYUP:
            DirectX::Keyboard::ProcessMessage(uMsg, wParam, lParam);
            break;
        case WM_MOUSEACTIVATE:
            // "click activating" the window to regain focus we don'g react on this, just focusing
            return MA_ACTIVATEANDEAT;
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

    SetProcessDPIAware();  // ignore dpi when setting cursor

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

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui::StyleColorsDark();

    ImGui_ImplWin32_Init(mHwnd);

    ShowWindow(mHwnd, SW_SHOWMAXIMIZED);

    ShowCursor(false);

    mDefaultMousePos.x = mWidth / 2;
    mDefaultMousePos.y = mHeight / 2;
    resetMousePos();
}

Window::~Window() {
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    UnregisterClassA(mWindowClassName.c_str(), (HINSTANCE)::GetModuleHandle(NULL));
}
