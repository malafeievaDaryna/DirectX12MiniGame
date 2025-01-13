#pragma once

#include <Windows.h>
#include <memory>
#include <string>

class Window {
public:
    Window(const std::string& title, int width, int height);
    ~Window();

    HWND hwnd() const {
        return mHwnd;
    }

    UINT32 width() const {
        return mWidth;
    }

    UINT32 height() const {
        return mHeight;
    }

    const POINT& resetMousePos() {
        SetCursorPos(mDefaultMousePos.x, mDefaultMousePos.y);
        return mDefaultMousePos;
    }

private:
    std::string mWindowClassName{};
    HWND mHwnd{nullptr};
    HMODULE m_hinstance{nullptr};
    bool mIsClosed{false};
    UINT32 mWidth{0u};
    UINT32 mHeight{0u};
    POINT mDefaultMousePos;
};
