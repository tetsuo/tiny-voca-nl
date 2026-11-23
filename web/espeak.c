/* Minimal browser adapter for eSpeak NG, GPL-3.0-or-later. */
#include <stdio.h>
#include <string.h>
#include <emscripten.h>
#include <espeak-ng/speak_lib.h>

static int discard_audio(short *audio, int count, espeak_EVENT *events) {
    (void)audio; (void)count; (void)events;
    return 0;
}

EMSCRIPTEN_KEEPALIVE int dutch_init(void) {
    if (espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, "/", espeakINITIALIZE_DONT_EXIT) <= 0)
        return -1;
    espeak_SetSynthCallback(discard_audio);
    return espeak_SetVoiceByName("nl");
}

/* Match the desktop CLI's synchronous IPA trace, including its phone separators. */
EMSCRIPTEN_KEEPALIVE int dutch_trace(const char *text) {
    FILE *trace = fopen("/phones.txt", "wb");
    if (!trace) return -1;
    espeak_SetPhonemeTrace(espeakPHONEMES_IPA | ('_' << 8), trace);
    int result = espeak_Synth(text, strlen(text) + 1, 0, POS_CHARACTER, 0, espeakCHARS_UTF8, NULL, NULL);
    espeak_SetPhonemeTrace(0, NULL);
    if (fclose(trace) != 0) return -1;
    return result;
}
