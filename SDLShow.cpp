#include <string>
#include <iostream>
#include <time.h>

#include "SDLShow.hpp"
#include "Main.hpp"

SDL_Window* sdlWindow;
SDL_Renderer* sdlRenderer;
SDL_Texture* sdlTexture;

void sdl_show_window() {
	//init SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cout << "error, SDL video init failed" << std::endl;
		return;
	}
	SDL_CreateWindowAndRenderer(0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP, &sdlWindow, &sdlRenderer);

	sdlTexture = SDL_CreateTexture(sdlRenderer,
		//SDL_PIXELFORMAT_ARGB8888,
		//SDL_PIXELFORMAT_RGBA8888,
		SDL_PIXELFORMAT_ABGR8888,
		SDL_TEXTUREACCESS_STREAMING,
		resolution[0], resolution[1]);
	SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
	SDL_RenderClear(sdlRenderer);
	SDL_RenderPresent(sdlRenderer);

	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");  // make the scaled rendering look smoother.
	SDL_RenderSetLogicalSize(sdlRenderer, resolution[0], resolution[1]);

	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");  // make the scaled rendering look smoother.
	SDL_RenderSetLogicalSize(sdlRenderer, resolution[0], resolution[1]);
}

void sdl_update_frame(Uint32 *pixels) {
	SDL_UpdateTexture(sdlTexture, NULL, pixels, resolution[0] * sizeof(Uint32));
	SDL_RenderClear(sdlRenderer);
	SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
	SDL_RenderPresent(sdlRenderer);
}