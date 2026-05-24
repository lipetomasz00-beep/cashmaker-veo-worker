# cashmaker-veo-worker

## Webhook event contract (standard)

This worker sends webhook callbacks to the URL provided in `webhookUrl` from `POST /render-sequence`.

To keep Make, Vercel and custom dashboards aligned, webhook payloads are standardized to two event types:

- `render.completed`
- `render.failed`

Canonical JSON Schema is stored in:

- `docs/webhook-events.schema.json`

### `render.completed` payload

Required fields:

- `event_type` = `"render.completed"`
- `job_id` (string)
- `status` = `"success"`
- `video_url` (string, URI)
- `topic` (string)
- `video_duration` (number)
- `file_size_mb` (number)
- `speed_adjustment` (number)
- `has_subtitles` (boolean)
- `has_watermark` (boolean)
- `has_endscreen` (boolean)
- `hashtags` (array of strings, each starting with `#`)
- `source` = `"cashmaker-veo-worker"`
- `timestamp` (ISO datetime)

### `render.failed` payload

Required fields:

- `event_type` = `"render.failed"`
- `job_id` (string)
- `status` = `"failed"`
- `error` (string)
- `source` = `"cashmaker-veo-worker"`
- `timestamp` (ISO datetime)

### Optional webhook signature

If `WEBHOOK_SECRET` is configured, worker sends:

- Header: `X-Webhook-Signature`
- Format: `sha256=<hex_digest>`
- Digest source: raw request body bytes

This allows receivers (Make custom code step / Vercel endpoint) to verify authenticity.

### Hashtags in request and webhook

- `POST /render-sequence` accepts optional `hashtags` array.
- If omitted, worker auto-generates hashtags from topic + default finance tags.
- Max count is controlled by `MAX_HASHTAGS` (default `8`).

## Viral Playbook (maximize chances)

> Note: virality cannot be guaranteed for every short, but these rules are mandatory to maximize probability.

### 1) Hook in 0–2 seconds
- Start with a strong claim, number, or contrast.
- No intro/logo at the beginning.

### 2) Tempo and retention
- Target duration: 10–15 seconds.
- Visual cut/change every 1–2 seconds.
- One idea per sentence.

### 3) Subtitles + pattern interrupts
- Large font, high contrast, short subtitle lines.
- Add a visual interrupt every 2–4 seconds (zoom, B-roll, number, arrow, movement).

### 4) Conversion-oriented narrative
- Mandatory structure: Hook → Problem → Solution → single CTA.
- CTA must be specific and action-oriented (example: “Sprawdź ranking teraz…”).

### 5) A/B testing on every publish
- Minimum: 3 hook variants and 2 CTA variants per topic.
- Keep winners after 24–48 hours based on metrics.

### 6) Automatic feedback rules
- Low CTR => strengthen hook and CTA.
- Low VTR => shorten story and increase pace.
- Runtime support exists in code via `apply_optimization_rules`.

### 7) KPI gate for decisions
- Track and review: CTR, VTR 3s, VTR 50%, completion rate, shares, saves.
- No KPI review = no optimization decision.
