# Rehab Monitor — Phase 2.1 (B2C Patient Self-Service)

A privacy-centric rehabilitation monitoring system for **home use**. Each
patient signs into their own account, picks one of their prescribed
exercises, performs it in front of a webcam, and gets live AI-driven
feedback on form. All processing — pose detection, scoring, persistence
— happens on the patient's machine.

The clinician-facing patient list / patient management surface from
Phase 1 has been deprecated; sessions are now owned directly by an
authenticated user and all data is strictly per-user-scoped.

## Architecture

```
┌──────────────────── Browser (React + Vite) ────────────────────┐
│   /login  →  Token (DRF authtoken) stored in localStorage      │
│   /dashboard                                                   │
│     ├── Exercise Guides & Prompts Module                       │
│     │     • interactive cards with tutorials (text + cues)     │
│     │     • click → /monitor?exercise=<key>                    │
│     └── Last-7 Trend Chart (user-scoped GET /api/trend/)       │
│   /monitor                                                     │
│     ├── usePoseDetection (MediaPipe Pose, facingMode:user)     │
│     ├── Mirror effect (CSS scaleX(-1))                         │
│     ├── 30 FPS  → WebSocket  (token in query string)           │
│     └── 15 FPS  → in-memory buffer → POST /api/sessions/ingest │
└────────────────────────────────────────────────────────────────┘
                              │  Token <key>
                              ▼
┌──────────── Django + Channels (rehab_backend) ─────────────────┐
│   REST (DEFAULT_PERMISSION_CLASSES = IsAuthenticated)          │
│     POST /api/auth/{register,login,logout}/  GET /me/          │
│     GET  /api/sessions/                   (only request.user)  │
│     GET  /api/sessions/<id>/              (404 if not yours)   │
│     DELETE /api/sessions/<id>/                                 │
│     POST /api/sessions/ingest/        (user from request.user) │
│     GET  /api/trend/                  (last-7 for caller)      │
│   WebSocket /ws/monitor/?token=<key>                           │
│     • TokenAuthMiddleware → scope['user']                      │
│     • Anonymous → close 4401                                   │
│   Analyzer (deterministic, plug-and-play)                      │
│     • BaseAnalyzer + PlaceholderAnalyzer                       │
│     • Seed-based reproducibility                               │
└────────────────────────────────────────────────────────────────┘
                              │
                       SQLite (backend/data/rehab_local.sqlite3)
```

## Directory layout

```
DPMonitor/
├── implementation_plan.md
├── README.md
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── rehab_backend/          ← Django project (settings, urls, asgi, wsgi)
│   ├── api/
│   │   ├── auth_views.py       ← register / login / logout / me
│   │   ├── views.py            ← session list/detail/ingest, trend, health
│   │   ├── serializers.py      ← no patient FK; user resolved server-side
│   │   ├── consumers.py        ← rejects anonymous WS with close 4401
│   │   ├── ws_auth.py          ← token query-string middleware for Channels
│   │   ├── routing.py
│   │   └── tests.py            ← 23 tests
│   ├── clinical_sessions/
│   │   ├── models.py           ← Session.user FK on AUTH_USER_MODEL
│   │   └── management/commands/seed_demo.py
│   └── analyzer/
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx              ← AuthProvider + RequireAuth + Outlet shell
        ├── auth/                ← AuthContext + RequireAuth guard
        ├── api/                 ← token-aware client + token-WS client
        ├── pages/
        │   ├── Login.jsx        ← sign-in / register tabs
        │   ├── Dashboard.jsx    ← Exercise Cards + Last-7 Trend
        │   ├── LiveMonitor.jsx  ← mirrored selfie + tutorial drawer
        │   └── ReportView.jsx
        ├── components/
        │   ├── ExerciseCard.jsx ← clickable card + tutorial modal
        │   ├── JointGauges.jsx, HUD.jsx, SkeletonOverlay.jsx,
        │   │   FeedbackList.jsx, StabilityChart.jsx, ControlBar.jsx,
        │   │   MultiLineChart.jsx
        ├── hooks/               ← usePoseDetection, useSessionMachine
        ├── lib/                 ← poseUtils, thresholds, exercises (catalogue)
        └── styles/global.css
```

## Running locally

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo          # optional — preload demo user
python manage.py runserver 0.0.0.0:8000
```

After `seed_demo`, sign in as:

| field    | value      |
|----------|------------|
| username | `demo`     |
| password | `demopass` |

### Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

To override the backend URL, copy `.env.example` to `.env.local` and edit
`VITE_API_BASE` / `VITE_WS_BASE`.

## Verification

### Automated tests

```bash
cd backend
python manage.py test
```

23 tests cover:

- **Auth** — register, login, login-fails-on-bad-creds, me-requires-auth,
  logout-revokes-token (`AuthEndpointTests`)
- **Session ingest** — auth required, FK is taken from `request.user`,
  spoofed `user`/`patient` payload fields are ignored (`SessionIngestTests`)
- **Strict data isolation** — User A can only list their own sessions,
  detail/delete on User B's session 404s, anonymous list 401s
  (`DataIsolationTests`)
- **Last-7 trend** — aggregates only the caller's data (`TrendEndpointTests`)
- **Public health** — reachable without auth (`HealthEndpointTests`)
- **Deterministic analyzer** — same seed → byte-identical results, RNG
  context isolation, summary ranges (`AnalyzerReproducibilityTests`)
- **seed_demo** — creates demo user + sessions, idempotent with `--clear`
  (`SeedDemoCommandTests`)

### WebSocket auth (smoke-tested separately)

- No token → connection rejected with close code `4401`
- Invalid token → close `4401`
- Valid token → `ready` + `summary` round-trip works

### Manual end-to-end paths

#### Quick path (no webcam)

```bash
cd backend && python manage.py seed_demo --sessions 3
```

Log in as `demo / demopass`. The dashboard shows the prescribed exercise
cards plus the demo user's 3 squat sessions. Click any session to see a
fully-rendered Clinical Report.

#### Full path (with webcam)

1. Visit `/`, get redirected to `/login`. Create an account (or sign in).
2. On the Dashboard, click an exercise card — open the **Tutorial** modal
   to read the cues, then **Start session**.
3. Stand back so your full body is in frame (the chip flips
   `CALIBRATING` → click **Begin Recording** to go `ACTIVE`).
4. Perform reps. Verify the HUD count increments, the skeleton turns
   amber/red on bad form, the feedback feed scrolls, and the stability
   chart updates.
5. Click **Finish & Generate Report**. The full trajectory uploads via
   `POST /api/sessions/ingest/` and you're redirected to the report view.
6. Open the Django admin (`/admin/`, after `createsuperuser`) and confirm
   the `Session` row is linked to YOUR user — not a clinic-issued patient
   row.

## REST surface

| Auth | Method | Path | Purpose |
|---|---|---|---|
| – | GET | `/api/health/` | Liveness + analyzer metadata |
| – | POST | `/api/auth/register/` | Create account; returns `{token, user}` |
| – | POST | `/api/auth/login/` | Returns `{token, user}` |
| ✓ | POST | `/api/auth/logout/` | Revokes the caller's token |
| ✓ | GET  | `/api/auth/me/` | Whoami |
| ✓ | GET  | `/api/sessions/` | List the **caller's** sessions |
| ✓ | GET  | `/api/sessions/<id>/` | Caller's session + trajectory (404 if not theirs) |
| ✓ | DELETE | `/api/sessions/<id>/` | Delete caller's session (cascades) |
| ✓ | POST | `/api/sessions/ingest/` | Batch upload; user from token |
| ✓ | GET  | `/api/trend/` | Last-7 aggregate for caller |

## Data contracts

| Direction | Channel | Shape |
|---|---|---|
| Browser → Backend | `WS /ws/monitor/?token=<key>` | `{type:"frame", frame_index, timestamp_ms, points:{landmark:[x,y,z,v]}, angles:{joint:deg}}` |
| Backend → Browser | `WS /ws/monitor/` | `{type:"result", frame_index, count, quality_score, is_compensatory, feedback:[…], diagnostics}` |
| Browser → Backend | `POST /api/sessions/ingest/` | Summary + `frames:[…]` at 15 FPS — **no patient field** |
| Backend → Browser | `GET /api/sessions/<id>/` | Session detail + nested `trajectory.frames` |

## Privacy & determinism

- Database at `backend/data/rehab_local.sqlite3`. No external services.
- Random Seed flows through `analyzer/seed.py::apply_global_seed` and is
  surfaced in the UI status bar + every report so a reviewer can
  reproduce a result trivially.
- Strict data isolation: queryset-level `filter(user=request.user)` plus
  Token-authenticated WebSockets mean foreign data is unreachable from
  the API surface — not even as a 403.

## Phase 2.1 deprecations

| Removed | Replaced by |
|---|---|
| `patients` Django app + `PatientProfile` model | `auth.User` (via `AUTH_USER_MODEL`) |
| `GET/POST /api/patients/...` endpoints | `POST /api/auth/{register,login}/` |
| `patient` field in ingest payload | Token → `request.user` server-side |
| New Patient page, Patient Detail page | Login page + per-user Dashboard |
| Patient list on Dashboard / sidebar | Exercise Guides & Prompts module |
