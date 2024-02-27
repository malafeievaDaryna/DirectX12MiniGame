#include "DirectxRenderer.h"

static constexpr uint32_t WINDOW_WIDTH = 512u;
static constexpr uint32_t WINDOW_HEIGHT = 512u;

int main(int argc, char** argv) {
    DirectXRenderer directxRenderer;
    directxRenderer.Initialize("DirectXMiniGame", WINDOW_WIDTH, WINDOW_HEIGHT);

    /* program main loop */
    while (directxRenderer.Run())
        ;

    return 0;
}