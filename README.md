<!DOCTYPE html>
<html>
<body>

<h1>The game engine 'MiniGame' powered by DirectX12</h1>
</br> <b>Platforms:</b> Windows
</br> <b>Technologies:</b> C++ 17, DirectX12, DirectXTK12(user input), CMake, WinApi, stb_image, directxmath
<p><img src="demo.png" width="60%" height="60%"></p>
<p>This is simple game engine based on own engine run by DirectX12, DirectXTK12 utils(user input)
currently it supports such features as
<ol>
  <li>support for animated models (md5 format), the parser was implemented from scratch it's the port of OpenGL based parser of ID TECH 4 engine (Doom 3) with reworked math and rendering api;
  in short the animation is calculated by directxmath and then generated submeshes of current frame are pushed to DirectX12 pipeline </li>
  </li>
</ol></p>
<p><b>HOWTO BUILD:</b>
</br>
You have to fetch DirectXTK12 lib as git submodule before building
</br>
git submodule init && cd build
</br>
</br>
<b><i>cmake.exe ..\ -G "Visual Studio 17 2022"</i></b>
</br>
<b><i>cmake --target "ALL_BUILD" --config "Release"</i></b>
</br>
</p>
</body>
</html>
