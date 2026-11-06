"use client";

import { useEffect, useState } from "react";

type Track = { title: string; artists?: string; album?: string; year?: string | number; image_url?: string; stream_url?: string; song_id?: string; recommendation_reason?: string };
type Shelf = { title: string; items: Track[] };
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [shelves, setShelves] = useState<Shelf[]>([]);
  const [recent, setRecent] = useState<Track[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Track[]>([]);
  const [queue, setQueue] = useState<Track[]>([]);
  const [now, setNow] = useState<Track | null>(null);
  const [loading, setLoading] = useState(false);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem("cozybeats_token");
    if (saved) { setToken(saved); void loadMe(saved); }
  }, []);

  async function loadMe(accessToken: string) {
    const response = await fetch(`${API}/v1/auth/me`, { headers: { Authorization: `Bearer ${accessToken}` } });
    if (!response.ok) { logout(); return; }
    const body = await response.json();
    setEmail(body.user.email);
    void Promise.all([loadHome(accessToken), loadHistory(accessToken)]);
  }

  async function loadHome(accessToken = token) {
    if (!accessToken) return;
    const response = await fetch(`${API}/v1/home`, { headers: { Authorization: `Bearer ${accessToken}` } });
    if (response.ok) setShelves((await response.json()).shelves || []);
  }

  async function loadHistory(accessToken = token) {
    if (!accessToken) return;
    const response = await fetch(`${API}/v1/history`, { headers: { Authorization: `Bearer ${accessToken}` } });
    if (response.ok) setRecent((await response.json()).items || []);
  }

  async function submitAuth() {
    setAuthError("");
    const endpoint = authMode === "login" ? "login" : "register";
    const response = await fetch(`${API}/v1/auth/${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email: authEmail, password }) });
    const body = await response.json();
    if (!response.ok) { setAuthError(body.detail || "Unable to authenticate"); return; }
    window.localStorage.setItem("cozybeats_token", body.token);
    setToken(body.token); setEmail(body.user.email); setPassword(""); void Promise.all([loadHome(body.token), loadHistory(body.token)]);
  }

  function logout() {
    window.localStorage.removeItem("cozybeats_token");
    setToken(null); setEmail(""); setShelves([]); setRecent([]); setQueue([]); setNow(null);
  }

  async function search() {
    if (!query.trim() || !token) return;
    setLoading(true);
    try {
      const response = await fetch(`${API}/v1/search?q=${encodeURIComponent(query)}&limit=24`, { headers: { Authorization: `Bearer ${token}` } });
      if (response.ok) setResults((await response.json()).items || []);
    } finally { setLoading(false); }
  }

  async function play(track: Track) {
    setNow(track);
    setRecent(current => [track, ...current.filter(item => (item.song_id || item.title) !== (track.song_id || track.title))].slice(0, 20));
    if (!token) return;
    setRecommendationLoading(true);
    try {
      const response = await fetch(`${API}/v1/recommendations`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` }, body: JSON.stringify({ seed: track, limit: 24 }) });
      if (response.ok) setQueue((await response.json()).items || []);
    } finally { setRecommendationLoading(false); }
    void fetch(`${API}/v1/events`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` }, body: JSON.stringify({ event_type: "play", track_id: track.song_id || track.title, payload: track }) });
  }

  function playNext() {
    const next = queue[0];
    if (next) void play(next);
  }

  if (!token) return <AuthScreen mode={authMode} setMode={setAuthMode} email={authEmail} setEmail={setAuthEmail} password={password} setPassword={setPassword} error={authError} onSubmit={submitAuth} />;
  return <main className="app-shell"><aside className="sidebar"><div className="brand"><span className="brand-dot" />CozyBeats</div><nav><button className="nav-active">⌂ Home</button><button onClick={() => document.getElementById("search-input")?.focus()}>⌕ Search</button><button>♡ Your library</button><button onClick={() => document.getElementById("recently-listened")?.scrollIntoView({ behavior: "smooth" })}>◷ Listening history</button></nav><div className="account"><span>Signed in</span><b>{email}</b><button onClick={logout}>Log out</button></div></aside><section className="content"><header className="topbar"><div><p className="eyebrow">Good evening, listener</p><h1>Find your next favorite.</h1></div><div className="profile-wrap"><button className="profile" aria-label="Open account menu" onClick={() => setAccountOpen(open => !open)}>{email.slice(0, 1).toUpperCase()}</button>{accountOpen && <div className="account-menu"><span>Signed in as</span><b>{email}</b><button onClick={logout}>Log out</button></div>}</div></header><nav className="mobile-nav"><button onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>⌂ Home</button><button onClick={() => document.getElementById("search-input")?.focus()}>⌕ Search</button><button onClick={() => document.getElementById("recently-listened")?.scrollIntoView({ behavior: "smooth" })}>◷ Recent</button><button onClick={() => setAccountOpen(open => !open)}>◉ Account</button></nav><div className="searchbar"><input id="search-input" value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === "Enter" && void search()} placeholder="Search songs, artists, albums..." /><button onClick={() => void search()}>{loading ? "..." : "Search"}</button></div><section className="hero"><div><span className="pill">PERSONALIZED FOR {email.toUpperCase()}</span><h2>A mix that moves<br />with you.</h2><p>Play a song and recommendations will appear in the “Made for you” shelf.</p></div><div className="hero-orb">✦</div></section>{recent.length > 0 && <div id="recently-listened"><Shelf title="Recently listened" tracks={recent} onPlay={play} /></div>}{results.length > 0 && <Shelf title="Search results" tracks={results} onPlay={play} />}{recommendationLoading && <div className="recommendation-loading">✦ Building your fresh mix…</div>}{queue.length > 0 && <Shelf title={`Made for you · ${queue.length} tracks`} tracks={queue} onPlay={play} recommendation />}{shelves.map(shelf => <Shelf key={shelf.title} title={shelf.title} tracks={shelf.items} onPlay={play} />)}{now && <div className="player"><div className="player-art"><Artwork track={now} eager /></div><div className="player-meta"><b>{now.title}</b><span>{now.artists || "Unknown artist"}</span></div><button className="next-button" onClick={playNext} disabled={!queue.length}>Next <span>›</span></button><audio controls autoPlay src={now.stream_url} /></div>}</section></main>;
}

function AuthScreen({ mode, setMode, email, setEmail, password, setPassword, error, onSubmit }: { mode: "login" | "register"; setMode: (mode: "login" | "register") => void; email: string; setEmail: (value: string) => void; password: string; setPassword: (value: string) => void; error: string; onSubmit: () => void }) { return <main className="auth-shell"><div className="auth-card"><div className="brand"><span className="brand-dot" />CozyBeats</div><p className="eyebrow">YOUR SOUND, EVOLVING</p><h1>{mode === "login" ? "Welcome back." : "Create your account."}</h1><p className="muted">Your recommendations, history and library stay connected to your account.</p><input value={email} onChange={e => setEmail(e.target.value)} placeholder="Email address" type="email" /><input value={password} onChange={e => setPassword(e.target.value)} placeholder="Password · 8+ characters" type="password" onKeyDown={e => e.key === "Enter" && onSubmit()} />{error && <p className="error">{error}</p>}<button className="primary-action" onClick={onSubmit}>{mode === "login" ? "Log in" : "Create account"}</button><button className="switch-auth" onClick={() => setMode(mode === "login" ? "register" : "login")}>{mode === "login" ? "New to CozyBeats? Create account" : "Already have an account? Log in"}</button></div></main>; }

function Artwork({ track, eager = false }: { track: Track; eager?: boolean }) {
  const [failed, setFailed] = useState(false);
  const rawUrl = track.image_url?.trim() || "";
  const imageUrl = rawUrl
    .replace(/^http:\/\//i, "https://")
    .replace(/(?:50|150|250)x(?:50|150|250)/gi, "500x500");
  const initials = track.title.split(/\s+/).filter(Boolean).slice(0, 2).map(word => word[0]).join("").toUpperCase() || "♪";
  if (!imageUrl || failed) {
    const seed = `${track.title}-${track.artists || "CozyBeats"}`;
    const hue = [...seed].reduce((sum, char) => sum + char.charCodeAt(0), 0) % 360;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="hsl(${hue} 65% 32%)"/><stop offset="1" stop-color="hsl(${(hue + 70) % 360} 62% 22%)"/></linearGradient></defs><rect width="500" height="500" fill="url(#g)"/><circle cx="80" cy="70" r="150" fill="white" opacity=".12"/><text x="250" y="285" fill="white" opacity=".92" font-size="92" font-family="Arial" font-weight="700" text-anchor="middle">${initials}</text><text x="250" y="435" fill="white" opacity=".7" font-size="22" font-family="Arial" text-anchor="middle">COZYBEATS</text></svg>`;
    return <img className="generated-art" src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`} alt={`Artwork for ${track.title}`} />;
  }
  return <img src={imageUrl} alt={track.title} loading={eager ? "eager" : "lazy"} decoding="async" referrerPolicy="no-referrer" onError={() => setFailed(true)} />;
}

function Shelf({ title, tracks, onPlay, recommendation = false }: { title: string; tracks: Track[]; onPlay: (track: Track) => void; recommendation?: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const visibleTracks = tracks.slice(0, expanded ? 20 : 5);
  return <section className="shelf">
    <div className="shelf-heading"><div><h3>{title}</h3><span>{recommendation ? "Ranked for your taste" : "Curated live picks"}</span></div>{tracks.length > 5 && <button className="view-all" onClick={() => setExpanded(open => !open)}>{expanded ? "Show less" : `View all ${tracks.length} songs`}</button>}</div>
    <div className="cards">{visibleTracks.map((track, index) => <article className="card" key={`${track.song_id || track.title}-${index}`} onClick={() => onPlay(track)}><div className="art"><Artwork track={track} /><button className="play" aria-label={`Play ${track.title}`}>▶</button></div><b>{track.title}</b><small>{track.artists || "Unknown artist"}</small>{track.album && <small>{track.album}{track.year ? ` · ${track.year}` : ""}</small>}{recommendation && <em>{track.recommendation_reason || "Similar vibe"}</em>}</article>)}</div>
  </section>;
}
