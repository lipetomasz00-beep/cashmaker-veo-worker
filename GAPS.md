# Co brakuje do działania „na 100%”

## Krytyczne braki
- Brak twardej walidacji kluczy `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `OPENAI_API_KEY` przy starcie aplikacji (część błędów wyjdzie dopiero w trakcie renderu).
- Brak timeoutu i limitu równoległych zadań (thread na każde żądanie może zabić proces przy większym ruchu).
- Brak autoryzacji endpointu `POST /render-sequence` (każdy może odpalać kosztowne joby).
- Brak retry/circuit-breaker dla kluczowych calli zewnętrznych (Veo/ElevenLabs/Whisper) poza samym pobieraniem pliku wideo.

## Stabilność i operacyjność
- Brak testów automatycznych (unit/integration) dla pipeline i endpointów.
- Brak migracji DB i strategii backup/retencji dla `renders.db`.
- Brak monitoringu metryk (czas renderu, error-rate, kolejka, zużycie CPU/RAM/dysku).
- Brak mechanizmu cleanup uruchamianego cyklicznie (teraz tylko przy starcie procesu).

## Jakość API
- Brak wersjonowania API i jednoznacznego kontraktu schematu request/response (np. OpenAPI).
- Brak idempotency key dla `POST /render-sequence`.
- Brak limtu rozmiaru danych wejściowych i walidacji `narration`.

## Bezpieczeństwo
- Brak filtracji i walidacji `webhookUrl` (ryzyko SSRF).
- Brak ograniczeń dostępu do `/videos/<filename>`.
- Brak redakcji wrażliwych danych w logach i polityki log retention.

## Środowisko uruchomieniowe
- Wymagane są binarki `ffmpeg` i `ffprobe`, ale nie są deklarowane jako dependency systemowe w repo.
- Brak `.env.example` i checklisty deploymentowej.
