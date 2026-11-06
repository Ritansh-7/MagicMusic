"use client";

import { useState } from "react";

type Track = { title: string; artists?: string; album?: string; year?: string | number; image_url?: string; stream_url?: string; song_id?: string; recommendation_reason?: string };
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Track[]>([]);
  const [queue, setQueue] = useState<Track[]>([]);
  const [now, setNow] = useState<Track | null>(null);
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!query.trim()) return;
    setLoading(true);
    const response = await fetch(`${API}/v1/search?q=${encodeURIComponent(query)}&limit=24`);
    const body = await response.json();
    setResults(body.items || []);
    setLoading(false);
  }

  async function play(track: Track) {
    setNow(track);
    try {
      const response = await fetch(`${API}/v1/recommendations`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed: track, user_id: "demo-user", limit: 20 }) });
      const body = await response.json();
      setQueue(body.items || []);
    } catch { setQueue([]); }
  }

  return <main className="app-shell">
    <aside className="sidebar"><div className="brand"><span className="brand-dot" />CozyBeats</div><nav><button className="nav-active">⌂ Home</button><button>⌕ Search</button><button>♡ Your library</button></nav><div className="sidebar-note">Your sound, evolving.<br /><span>Research mode · API connected</span></div></aside>
    <section className="content"><header className="topbar"><div><p className="eyebrow">Good evening, listener</p><h1>Find your next favorite.</h1></div><div className="profile">CB</div></header>
      <div className="searchbar"><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === "Enter" && search()} placeholder="Search songs, artists, albums..." /><button onClick={search}>{loading ? "..." : "Search"}</button></div>
      <section className="hero"><div><span className="pill">LIVE DISCOVERY</span><h2>A mix that moves<br />with you.</h2><p>Play a song and let CozyBeats build an explainable, taste-aware queue.</p></div><div className="hero-orb">✦</div></section>
      {results.length > 0 && <Shelf title="Search results" tracks={results} onPlay={play} />}
      {queue.length > 0 && <Shelf title={`Made for you · ${queue.length} tracks`} tracks={queue} onPlay={play} recommendation />}
      {now && <div className="player"><div className="player-art">{now.image_url ? <img src={now.image_url} alt={now.title} /> : "♪"}</div><div className="player-meta"><b>{now.title}</b><span>{now.artists || "Unknown artist"}</span></div><audio controls autoPlay src={now.stream_url} /></div>}
    </section>
  </main>;
}

function Shelf({ title, tracks, onPlay, recommendation = false }: { title: string; tracks: Track[]; onPlay: (track: Track) => void; recommendation?: boolean }) {
  return <section className="shelf"><div className="shelf-heading"><h3>{title}</h3><span>{recommendation ? "Ranked for your taste" : "Live catalog"}</span></div><div className="cards">{tracks.slice(0, 20).map((track, index) => <article className="card" key={`${track.song_id || track.title}-${index}`} onClick={() => onPlay(track)}><div className="art">{track.image_url ? <img src={track.image_url} alt={track.title} /> : <span>♪</span>}<button className="play">▶</button></div><b>{track.title}</b><small>{track.artists || "Unknown artist"}</small>{recommendation && <em>{track.recommendation_reason || "Similar vibe"}</em>}</article>)}</div></section>;
}
