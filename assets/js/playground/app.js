/**
 * The playground, ported from the local Node app.
 *
 * Identical game logic; the only change is that every API call goes through
 * API_BASE. Point it at a deployed instance for the full live experience --
 * all ten agents generating fresh content -- or leave it empty, in which case
 * playground-offline.js intercepts the same calls and serves pre-baked packs
 * for the games whose content can be settled in advance.
 */
const API_BASE = (window.PLAYGROUND_API || "").replace(/\/$/, "");
const api = (p) => API_BASE + p;

/* The piano samples are static files, not an API call, so they are served from
   the site itself even when a live agent server is configured. */
const AUDIO_BASE = (window.PLAYGROUND_AUDIO || "/audio/piano").replace(/\/$/, "");

/**
 * Front end for the agent playground.
 *
 * Two responsibilities: stream the agent trace while a run is in flight, and
 * render whatever artifact comes back into something you can actually play.
 * Game state that matters (answers, groupings, the key) lives on the server —
 * this file only ever knows what a fair player would know.
 */

const TINTS = {
  spotify: "var(--spotify)",
  piano: "var(--piano)",
  spy: "var(--spy)",
  profiler: "var(--profiler)",
  mafia: "var(--mafia)",
  alibi: "var(--alibi)",
  poetry: "var(--poetry)",
  haggle: "var(--haggle)",
  wordle: "var(--wordle)",
  connections: "var(--connections)",
  trivia: "var(--trivia)",
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const form = $("#ask-form");
const promptEl = $("#prompt");
const sendBtn = $("#send");
const stage = $("#stage");
const traceWrap = $("#trace-wrap");
const traceList = $("#trace");
const traceTitle = $("#trace-title");
const traceToggle = $("#trace-toggle");
const spark = $("#spark");

let busy = false;
// What produced the panel on screen, so it can be run again without retyping.
let lastRun = null;
// Titles served this session, so a rerun can be told what to avoid.
let seenTitles = [];

// --------------------------------------------------- playlist destination

const SAVE_KEY = "playground.savePlaylist";
const SAVE_NOTES = {
  none: "Tracks are resolved against the real catalog and shown here. Nothing is written to Spotify.",
  private:
    "Saved to the Spotify account, but kept off its public profile. Note that Spotify's “private” means undiscoverable, not access-controlled — anyone with the link can still open it.",
  public: "Saved and listed on the account's public profile, so it shows up for anyone following it.",
  collaborative:
    "Saved so anyone with the link can add and remove tracks — open it once from your own account and follow it to keep editing. Spotify requires collaborative playlists to be non-public, so these do not appear on the profile.",
};

const optsWrap = $("#ask-opts");
const optsNote = $("#opts-note");
let saveMode = localStorage.getItem(SAVE_KEY) ?? "none";

function paintSaveMode() {
  for (const btn of optsWrap.querySelectorAll(".seg-btn")) {
    btn.setAttribute("aria-checked", String(btn.dataset.save === saveMode));
  }
  optsNote.textContent = SAVE_NOTES[saveMode] ?? "";
}

for (const btn of optsWrap.querySelectorAll(".seg-btn")) {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    saveMode = btn.dataset.save;
    localStorage.setItem(SAVE_KEY, saveMode);
    paintSaveMode();
  });
}

// ------------------------------------------------------------------ roster

async function loadRoster() {
  try {
    const res = await fetch(api("/api/agents"));
    const { agents, publicMode } = await res.json();
    const roster = $("#roster");
    const chips = $("#chips");

    for (const a of agents) {
      const card = el("button", "agent-card");
      card.type = "button";
      card.style.setProperty("--tint", TINTS[a.id] ?? "var(--accent)");
      const name = el("div", "name");
      name.append(el("span", "dot"), document.createTextNode(a.label));
      card.append(name, el("div", "blurb", a.blurb));
      card.addEventListener("click", () => {
        promptEl.value = a.examples[0] ?? "";
        promptEl.focus();
        autosize();
      });
      roster.append(card);
    }

    // One example per agent, so the chip row advertises the whole crew.
    for (const a of agents) {
      const example = a.examples[Math.min(1, a.examples.length - 1)];
      if (!example) continue;
      const chip = el("button", "chip", example.length > 52 ? example.slice(0, 50) + "…" : example);
      chip.type = "button";
      chip.style.setProperty("--tint", TINTS[a.id] ?? "var(--accent)");
      chip.title = example;
      chip.addEventListener("click", () => {
        promptEl.value = example;
        autosize();
        form.requestSubmit();
      });
      chips.append(chip);
    }

    // In public mode the server refuses writes outright, so offering the save
    // options would be a lie. Show them disabled rather than hiding them, so
    // it's clear the capability exists and why it's off.
    if (publicMode) {
      saveMode = "none";
      for (const btn of optsWrap.querySelectorAll(".seg-btn")) {
        if (btn.dataset.save !== "none") btn.disabled = true;
      }
      SAVE_NOTES.none = "This site runs in public mode: playlists are resolved and shown, never written to anyone's Spotify account.";
    }
    optsWrap.hidden = false;
    paintSaveMode();

    $("#foot-note").textContent = publicMode
      ? "Public mode — playlists are previewed, not saved to anyone's account."
      : "Private mode — the Spotify agent can save playlists to the owner's account.";
  } catch {
    $("#foot-note").textContent = "Could not reach the server.";
  }
}

// ------------------------------------------------------------------- trace

function resetTrace() {
  traceList.replaceChildren();
  traceList.hidden = false;
  traceToggle.setAttribute("aria-expanded", "true");
  traceWrap.hidden = false;
  spark.className = "spark busy";
  traceTitle.textContent = "Routing…";
}

function traceLine(tag, body, fail) {
  const li = el("li", fail ? "fail" : "");
  li.append(el("span", "tag", tag), el("span", "body", body));
  traceList.append(li);
  traceList.scrollTop = traceList.scrollHeight;
}

function onTrace(e) {
  switch (e.type) {
    case "agent_start": {
      const li = el("li", "head", `▸ ${e.label}`);
      traceList.append(li);
      traceTitle.textContent = `${e.label} is working…`;
      break;
    }
    case "tool_call":
      traceLine("→", `${e.tool}(${summarizeArgs(e.args)})`);
      break;
    case "tool_result":
      traceLine(e.ok ? "←" : "✗", e.summary, !e.ok);
      break;
    case "note":
      traceLine("·", e.message);
      break;
    case "agent_done":
      traceLine("✓", `${(e.ms / 1000).toFixed(1)}s · ${e.steps} steps · ${e.usage.total_tokens} tokens`);
      break;
    case "agent_error":
      traceLine("✗", e.message, true);
      break;
  }
}

function summarizeArgs(args) {
  if (!args || typeof args !== "object") return "";
  const parts = [];
  for (const [k, v] of Object.entries(args)) {
    const val = Array.isArray(v) ? `[${v.length}]` : typeof v === "string" ? `"${v.slice(0, 34)}"` : String(v);
    parts.push(`${k}: ${val}`);
    if (parts.join(", ").length > 70) break;
  }
  return parts.join(", ");
}

traceToggle.addEventListener("click", () => {
  const open = traceToggle.getAttribute("aria-expanded") === "true";
  traceToggle.setAttribute("aria-expanded", String(!open));
  traceList.hidden = open;
});

// --------------------------------------------------------------- submitting

function autosize() {
  promptEl.style.height = "auto";
  promptEl.style.height = Math.min(promptEl.scrollHeight, 128) + "px";
}
promptEl.addEventListener("input", autosize);
promptEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const prompt = promptEl.value.trim();
  if (!prompt || busy) return;

  busy = true;
  sendBtn.disabled = true;
  stage.replaceChildren();
  resetTrace();

  try {
    const res = await fetch(api("/api/ask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, savePlaylist: saveMode }),
    });

    if (!res.ok || !res.body) {
      const { error } = await res.json().catch(() => ({ error: "Request failed." }));
      throw new Error(error ?? "Request failed.");
    }

    await readEventStream(res.body, {
      trace: onTrace,
      result: (data) => {
        spark.className = "spark done";
        traceTitle.textContent = `Done — ${(data.ms / 1000).toFixed(1)}s · ${data.usage.total_tokens} tokens`;
        traceList.hidden = true;
        traceToggle.setAttribute("aria-expanded", "false");
        render(data);
      },
      error: (data) => {
        throw new Error(data.message);
      },
    });
  } catch (err) {
    spark.className = "spark err";
    traceTitle.textContent = "Failed";
    stage.replaceChildren(el("div", "err", err.message || String(err)));
  } finally {
    busy = false;
    sendBtn.disabled = false;
  }
});

/** Minimal SSE reader — the browser's EventSource can't do POST. */
async function readEventStream(body, handlers) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      if (chunk.startsWith(":")) continue;

      let event = "message";
      const dataLines = [];
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      }
      if (dataLines.length === 0) continue;
      handlers[event]?.(JSON.parse(dataLines.join("\n")));
    }
  }
}

// ---------------------------------------------------------------- rendering

function panel(agentId, who, title, intro) {
  const p = el("section", "panel");
  p.style.setProperty("--tint", TINTS[agentId] ?? "var(--accent)");
  const head = el("div", "panel-head");
  head.append(el("span", "who", who));

  // Uniform "give me another" on every panel: same specialist, same brief, new
  // output. Cheaper than a fresh ask because the routing step is skipped.
  if (lastRun) {
    const again = el("button", "again", "Another");
    again.type = "button";
    again.title = "Same kind of thing, freshly generated";
    again.addEventListener("click", () => rerun(again));
    head.append(again);
  }

  p.append(head, el("h2", null, title));
  if (intro) p.append(el("p", "intro", intro));
  return p;
}

async function rerun(btn) {
  if (!lastRun || busy) return;
  busy = true;
  if (btn) {
    btn.disabled = true;
    btn.textContent = "…";
  }
  sendBtn.disabled = true;
  resetTrace();
  try {
    const res = await fetch(api("/api/again"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...lastRun, seen: seenTitles, savePlaylist: saveMode }),
    });
    if (!res.ok || !res.body) {
      const { error } = await res.json().catch(() => ({ error: "Request failed." }));
      throw new Error(error ?? "Request failed.");
    }
    stage.replaceChildren();
    await readEventStream(res.body, {
      trace: onTrace,
      result: (data) => {
        spark.className = "spark done";
        traceTitle.textContent = `Done — ${(data.ms / 1000).toFixed(1)}s · ${data.usage.total_tokens} tokens`;
        traceList.hidden = true;
        traceToggle.setAttribute("aria-expanded", "false");
        render(data);
      },
      error: (data) => {
        throw new Error(data.message);
      },
    });
  } catch (err) {
    spark.className = "spark err";
    traceTitle.textContent = "Failed";
    stage.replaceChildren(el("div", "err", err.message || String(err)));
  } finally {
    busy = false;
    sendBtn.disabled = false;
  }
}

function render(data) {
  const { agent, label, result, artifacts } = data;
  lastRun = agent && data.route ? { agent, task: data.route.task } : null;
  const title = artifacts && Object.values(artifacts)[0] && (Object.values(artifacts)[0].title || Object.values(artifacts)[0].name);
  if (title) seenTitles = [...seenTitles, title].slice(-8);

  if (!agent) {
    stage.replaceChildren(el("div", "err", result.message));
    return;
  }
  if (artifacts.playlist) return stage.replaceChildren(renderPlaylist(result, artifacts.playlist, label));
  if (artifacts.piano) return stage.replaceChildren(renderPiano(result, artifacts.piano, label));
  if (artifacts.spy) return stage.replaceChildren(renderSpy(result, artifacts.spy, label));
  if (artifacts.mafia) return stage.replaceChildren(renderMafia(result, artifacts.mafia, label));
  if (artifacts.poetry) return stage.replaceChildren(renderPoetry(result, artifacts.poetry, label));
  if (artifacts.profile) return stage.replaceChildren(renderProfile(result, artifacts.profile, label));
  if (artifacts.alibi) return stage.replaceChildren(renderAlibi(result, artifacts.alibi, label));
  if (artifacts.haggle) return stage.replaceChildren(renderHaggle(result, artifacts.haggle, label));
  if (artifacts.wordle) return stage.replaceChildren(renderWordle(result, artifacts.wordle, label));
  if (artifacts.connections) return stage.replaceChildren(renderConnections(result, artifacts.connections, label));
  if (artifacts.trivia) return stage.replaceChildren(renderTrivia(result, artifacts.trivia, label));

  // Reached when an agent finished but its publishing tool never succeeded —
  // show what it did say rather than a dead end.
  const words = result?.commentary ?? result?.intro ?? result?.message;
  stage.replaceChildren(
    el("div", "err", words ?? "The agent finished but produced nothing to show. Try rephrasing."),
  );
}

// ------------------------------------------------------------- playlist UI

function renderPlaylist(result, pl, label) {
  const p = panel("spotify", label, pl.name, result.commentary);
  p.querySelector("h2")?.classList.add("slug");
  const minutes = Math.round(pl.tracks.reduce((s, t) => s + t.durationMs, 0) / 60000);

  // The character sketch — this is also what Spotify shows as the description.
  if (pl.description) {
    const bio = el("div", "bio");
    if (result.character_world) bio.append(el("div", "bio-world", result.character_world));
    bio.append(el("p", null, pl.description));
    p.append(bio);
  }

  const meta = el("div", "meta");
  meta.append(el("span", "pill", `${pl.tracks.length} tracks`), document.createTextNode(" "));
  meta.append(el("span", "pill", `${minutes} min`), document.createTextNode(" "));
  // Reported from the artifact — what the tool actually did — not from what
  // was requested, so a save that silently didn't happen can't read as one.
  meta.append(
    el(
      "span",
      "pill",
      pl.mode !== "created"
        ? "preview — not saved"
        : pl.visibility === "public"
          ? "saved · on profile"
          : pl.visibility === "collaborative"
            ? "saved · collaborative"
            : "saved · off profile",
    ),
  );
  p.append(meta);

  const list = el("ol", "tracks");
  pl.tracks.forEach((t, i) => {
    const li = el("li", t.uncertain ? "iffy" : "");
    if (t.uncertain) li.title = "Fuzzy catalog match — worth a listen to confirm";
    const a = el("a");
    a.href = t.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.append(el("span", "who", t.artist), document.createTextNode(" — "), el("span", "ttl", t.title));
    li.append(el("span", "n", String(i + 1).padStart(2, "0")), a, el("span", "yr", t.year));
    list.append(li);
  });
  p.append(list);

  if (pl.url) {
    const open = el("a", "open-spotify", "Open in Spotify");
    open.href = pl.url;
    open.target = "_blank";
    open.rel = "noopener noreferrer";
    p.append(open);
  }
  if (pl.missing?.length) {
    p.append(el("div", "missing", `Not on Spotify: ${pl.missing.join(" · ")}`));
  }
  return p;
}


// -------------------------------------------------------------- profiler UI

function renderProfile(result, suite, label) {
  const p = panel("profiler", label, "The Profiler", result.intro ?? suite.intro);

  const progress = el("div", "meta");
  const stageEl = el("div", "task-stage");
  const results = el("div");
  p.append(progress, stageEl, results);

  let i = 0;
  const answers = {};

  function drawProgress() {
    progress.textContent = i < suite.tasks.length
      ? `Task ${i + 1} of ${suite.tasks.length}`
      : "";
  }

  function drawTask() {
    drawProgress();
    stageEl.replaceChildren();
    if (i >= suite.tasks.length) return;
    const t = suite.tasks[i];

    const card = el("div", "task");
    card.append(el("h3", "task-title", t.title));
    card.append(el("p", "task-prompt", t.prompt));

    const row = el("div", "crow-actions");

    if (t.input.kind === "number") {
      const inp = el("input", "offer-input");
      inp.type = "number";
      inp.min = String(t.input.min);
      inp.max = String(t.input.max);
      inp.placeholder = t.input.label;
      const go = el("button", "btn primary", "Lock it in");
      go.type = "button";
      const submit = () => {
        const v = Number(inp.value);
        if (!Number.isFinite(v)) return;
        send(t.id, Math.max(t.input.min, Math.min(t.input.max, Math.round(v))));
      };
      go.addEventListener("click", submit);
      inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); submit(); } });
      row.append(inp, go);
      card.append(row);
      setTimeout(() => inp.focus(), 30);
    } else {
      // Holt-Laury ladder: pick safe or risky on each row; the switch point is
      // the measurement, so the whole ladder is answered before submitting.
      const picks = new Array(t.input.rows.length).fill(null);
      const ladder = el("div", "ladder");
      t.input.rows.forEach((text, idx) => {
        const r = el("div", "ladder-row");
        r.append(el("span", "ladder-n", String(idx + 1)));
        r.append(el("span", "ladder-text", text));
        const seg = el("div", "seg");
        for (const [val, lab] of [["A", "Safer"], ["B", "Riskier"]]) {
          const b = el("button", "seg-btn", lab);
          b.type = "button";
          b.addEventListener("click", () => {
            picks[idx] = val;
            for (const sib of seg.querySelectorAll(".seg-btn")) sib.setAttribute("aria-checked", "false");
            b.setAttribute("aria-checked", "true");
            go.disabled = picks.some((x) => x === null);
          });
          seg.append(b);
        }
        r.append(seg);
        ladder.append(r);
      });
      const go = el("button", "btn primary", "Done");
      go.type = "button";
      go.disabled = true;
      go.addEventListener("click", () => {
        const first = picks.indexOf("B");
        send(t.id, first < 0 ? t.input.rows.length : first);
      });
      row.append(go);
      card.append(ladder, row);
    }

    card.append(el("p", "task-paradigm", t.paradigm));
    stageEl.append(card);
  }

  async function send(taskId, value) {
    answers[taskId] = value;
    stageEl.replaceChildren(el("p", "task-prompt", "\u2026"));
    try {
      const res = await fetch(api(`/api/play/profile/${suite.id}/answer`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: taskId, value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      i++;
      if (data.complete) { stageEl.replaceChildren(); progress.textContent = ""; drawResults(data); }
      else drawTask();
    } catch (err) {
      stageEl.replaceChildren(el("p", "task-prompt", err.message || "Lost the connection."));
    }
  }

  function drawResults(data) {
    results.replaceChildren();

    const table = el("table", "bargain");
    for (const m of data.measures) {
      const r = el("tr");
      r.append(el("th", null, m.label));
      r.append(el("td", null, `${m.value.toFixed(2)} ${m.unit}  [${m.low.toFixed(2)}, ${m.high.toFixed(2)}]`));
      table.append(r);
    }
    results.append(el("h3", "task-title", "What was measured"), table);

    results.append(el("h3", "task-title", "What that suggests"));
    for (const reading of data.readings) {
      const card = el("div", "reading");
      const head = el("div", "reading-head");
      head.append(el("span", "conf", `${Math.round(reading.confidence * 100)}% confidence`));
      if (reading.n > 0) {
        head.append(el("span", "conf-n", `${Math.round(reading.agreement * 100)}% agreed (n=${reading.n})`));
      }
      const vote = el("span", "vote");
      for (const [ok, glyph, title] of [[true, "\u25b2", "This fits"], [false, "\u25bc", "This misses"]]) {
        const b = el("button", "thumb", glyph);
        b.type = "button";
        b.title = title;
        b.addEventListener("click", async () => {
          for (const sib of vote.querySelectorAll(".thumb")) sib.disabled = true;
          b.classList.add(ok ? "up" : "down");
          await fetch(api(`/api/play/profile/${suite.id}/feedback`), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ readingId: reading.id, agreed: ok }),
          }).catch(() => {});
        });
        vote.append(b);
      }
      head.append(vote);
      card.append(head, el("p", "reading-text", reading.text), el("p", "reading-basis", reading.basis));
      results.append(card);
    }

    const caveat = el("div", "caveat");
    caveat.append(el("p", null,
      "These are four decisions in a few minutes. They measure how you played these particular " +
      "games, which is not the same as who you are, and one short session cannot establish a " +
      "stable trait. Nothing here is diagnostic, and it should not be used to judge anyone \u2014 " +
      "including yourself \u2014 for anything that matters."));
    results.append(caveat);
  }

  drawTask();
  return p;
}

// --------------------------------------------------------------- poetry UI

function renderPoetry(result, shelf, label) {
  const p = panel("poetry", label, shelf.poet, null);

  if (result.substituted === "yes") {
    const swap = el("div", "swap");
    swap.append(el("b", null, "A different poet: "), document.createTextNode(
      "the one you asked for isn't in the archive, which only carries work that's out of copyright."));
    p.append(swap);
  }

  const note = el("p", "intro", result.note);
  p.append(note);

  const sheet = el("div", "poem");
  const heading = el("h3", "poem-title");
  const body = el("div", "poem-body");
  sheet.append(heading, body);
  p.append(sheet);

  const actions = el("div", "crow-actions");
  const more = el("button", "btn", "Read another");
  more.type = "button";
  const left = el("span", "lives");
  actions.append(more, left);
  const status = el("div", "meta");
  p.append(actions, status);

  let remaining = Math.max(0, (shelf.available ?? 1) - 1);

  function show(title, lines) {
    heading.textContent = title;
    body.replaceChildren();
    // Blank entries are stanza breaks in the source, so they carry meaning.
    for (const line of lines) {
      const row = el("p", line.trim() ? "verse" : "verse gap");
      row.textContent = line || " ";
      body.append(row);
    }
    left.textContent = remaining > 0 ? `${remaining} more in the archive` : "";
    more.disabled = remaining === 0;
    sheet.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  more.addEventListener("click", async () => {
    more.disabled = true;
    status.textContent = "Turning the page…";
    status.style.color = "var(--ink-faint)";
    try {
      const res = await fetch(api(`/api/play/poetry/${shelf.id}/more`), { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      if (data.done) {
        remaining = 0;
        status.textContent = data.error;
        more.disabled = true;
        return;
      }
      remaining = data.remaining;
      show(data.title, data.lines);
      status.textContent = "";
    } catch (err) {
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
      more.disabled = false;
    }
  });

  show(shelf.title, shelf.lines);
  return p;
}


/**
 * The debrief panel. Two registers behind a toggle: "what to do differently"
 * and "what was measured". Both come from the server as plain strings computed
 * from the game's own numbers, so neither can contradict the scoreboard.
 */
function lessonPanel(lesson) {
  if (!lesson || (!lesson.plain.length && !lesson.technical.length)) return null;
  const box = el("div", "lesson");
  const head = el("div", "lesson-head");
  head.append(el("span", "lesson-title", "What that tells you"));

  const seg = el("div", "seg lesson-seg");
  const body = el("div", "lesson-body");
  let mode = localStorage.getItem("playground.lessonMode") === "technical" ? "technical" : "plain";

  const paint = () => {
    for (const b of seg.querySelectorAll(".seg-btn")) {
      b.setAttribute("aria-checked", String(b.dataset.mode === mode));
    }
    body.replaceChildren();
    for (const line of lesson[mode]) body.append(el("p", null, line));
    body.className = "lesson-body" + (mode === "technical" ? " tech" : "");
  };

  for (const [m, label] of [["plain", "Plain"], ["technical", "Technical"]]) {
    const b = el("button", "seg-btn", label);
    b.type = "button";
    b.dataset.mode = m;
    b.addEventListener("click", () => {
      mode = m;
      localStorage.setItem("playground.lessonMode", m);
      paint();
    });
    seg.append(b);
  }

  head.append(seg);
  box.append(head, body);
  paint();
  return box;
}

// ---------------------------------------------------------------- alibi UI

function renderAlibi(result, puzzle, label) {
  const p = panel("alibi", label, "The Alibi", result.brief);

  const brief = el("div", "case-brief");
  brief.append(
    el("div", "case-crime", puzzle.crime),
    el("div", "case-where", `${puzzle.setting} · ${puzzle.hour}`),
  );
  p.append(brief);

  const list = el("div", "statements");
  const actions = el("div", "crow-actions");
  const askInput = el("input", "spy-ask");
  askInput.type = "text";
  askInput.placeholder = "Press someone — pick them, then ask…";
  askInput.maxLength = 200;
  const left = el("span", "lives");
  actions.append(askInput, left);
  const status = el("div", "meta");
  p.append(list, actions, status);

  let suspects = puzzle.suspects;
  let pressesLeft = puzzle.pressesLeft;
  let over = false;
  let selected = null;

  const updateLeft = () => {
    left.textContent = over ? "" : `${pressesLeft} question${pressesLeft === 1 ? "" : "s"} left`;
    askInput.disabled = over || pressesLeft === 0;
  };

  function draw() {
    list.replaceChildren();
    for (const s of suspects) {
      const card = el("div", "statement");
      if (selected === s.name) card.classList.add("picked");
      const head = el("div", "statement-head");
      head.append(el("span", "suspect-name", s.name));
      head.append(el("span", "claim", `${s.place} · ${s.with === "alone" ? "alone" : "with " + s.with}`));

      if (!over) {
        const press = el("button", "btn small", selected === s.name ? "Selected" : "Press");
        press.type = "button";
        press.disabled = pressesLeft === 0;
        press.addEventListener("click", () => {
          selected = s.name;
          askInput.focus();
          draw();
        });
        const accuse = el("button", "btn accuse", "Accuse");
        accuse.type = "button";
        accuse.addEventListener("click", () => doAccuse(s.name));
        head.append(press, accuse);
      }
      card.append(head, el("p", "suspect-line", s.statement));
      for (const extra of s.pressed) card.append(el("p", "suspect-line later", extra));
      list.append(card);
    }
  }

  async function doPress() {
    const question = askInput.value.trim();
    if (!selected) {
      status.textContent = "Pick who you're pressing first.";
      status.style.color = "var(--warn)";
      return;
    }
    if (question.length < 3 || over || pressesLeft === 0) return;
    askInput.disabled = true;
    status.textContent = `Pressing ${selected}…`;
    status.style.color = "var(--ink-faint)";
    try {
      const res = await fetch(api(`/api/play/alibi/${puzzle.id}/press`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selected, question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      suspects = suspects.map((s) => (s.name === data.name ? { ...s, pressed: [...s.pressed, data.answer] } : s));
      pressesLeft = data.pressesLeft;
      askInput.value = "";
      status.textContent = `You asked ${data.name}: “${question}”`;
      draw();
    } catch (err) {
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    } finally {
      updateLeft();
    }
  }

  async function doAccuse(name) {
    if (over) return;
    over = true;
    updateLeft();
    try {
      const res = await fetch(api(`/api/play/alibi/${puzzle.id}/accuse`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      draw();
      for (const card of list.querySelectorAll(".statement")) {
        const who = card.querySelector(".suspect-name").textContent;
        if (who === data.culprit) card.classList.add("was-spy");
        else if (data.clash.includes(who)) card.classList.add("wrongly-accused");
      }
      const box = el("div", data.correct ? "verdict good" : "verdict bad");
      box.append(el("strong", null, data.correct ? `Right — it was ${data.culprit}.` : `No. It was ${data.culprit}.`));
      box.append(el("p", null, `The two that couldn't both be true: ${data.explain}`));
      p.append(box);
      status.textContent = "";
    } catch (err) {
      over = false;
      updateLeft();
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    }
  }

  askInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      doPress();
    }
  });

  draw();
  updateLeft();
  status.textContent = "Two of these accounts can't both be true.";
  return p;
}

// --------------------------------------------------------------- haggle UI

function renderHaggle(result, initial, label) {
  const p = panel("haggle", label, initial.item, result.pitch);
  let game = initial;

  const card = el("div", "stall");
  card.append(
    el("div", "stall-blurb", game.blurb),
    el("div", "stall-price", `${game.merchant} is asking ${game.currency}${game.asking}`),
  );
  p.append(card);

  const thread = el("div", "thread");
  const actions = el("div", "crow-actions");
  const input = el("input", "offer-input");
  input.type = "number";
  input.min = "1";
  input.max = String(game.asking);
  input.placeholder = "Your offer";
  const offerBtn = el("button", "btn primary", "Offer");
  const walk = el("button", "btn", "Walk away");
  const left = el("span", "lives");
  offerBtn.type = walk.type = "button";
  actions.append(input, offerBtn, walk, left);
  const status = el("div", "meta", result.advice ?? "");
  p.append(thread, actions, status);

  const MOOD = ["walking out", "cold", "losing patience", "businesslike", "amiable"];
  const updateLeft = () => {
    if (game.over) {
      left.textContent = "";
    } else {
      const p = Math.max(0, Math.min(4, game.patience ?? 4));
      left.replaceChildren(
        el("span", `mood m${p}`, MOOD[p]),
        document.createTextNode(` · ${game.roundsLeft} offer${game.roundsLeft === 1 ? "" : "s"} left`),
      );
    }
    input.disabled = offerBtn.disabled = walk.disabled = game.over;
  };

  function draw() {
    thread.replaceChildren();
    for (const o of game.offers) {
      const row = el("div", `bubble ${o.from}`);
      row.append(
        el("span", "bubble-who", o.from === "you" ? "You" : game.merchant),
        el("span", "bubble-amount", `${game.currency}${o.amount}`),
      );
      if (o.line) row.append(el("p", "bubble-line", o.line));
      thread.append(row);
    }
  }

  async function offer() {
    const amount = Math.round(Number(input.value));
    if (!Number.isFinite(amount) || amount <= 0 || game.over) return;
    input.disabled = offerBtn.disabled = true;
    status.textContent = `${game.merchant} considers…`;
    status.style.color = "var(--ink-faint)";
    // Show the offer immediately; the reply follows.
    game.offers = [...game.offers, { from: "you", amount, line: "…" }];
    draw();
    try {
      const res = await fetch(api(`/api/play/haggle/${game.id}/offer`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      game = { ...game, ...data };
      input.value = "";
      draw();
      if (data.turn.accepted) {
        const box = el("div", "verdict good");
        box.append(el("strong", null,
          "Deal at " + game.currency + data.turn.settled + " — " +
          Math.round((1 - data.turn.settled / game.asking) * 100) + "% off the asking price."));
        p.append(box);

        // The bargaining-theory read-out. Percentages here are shares of the
        // surplus (the gap between reserve and asking), not discounts — the two
        // are easy to confuse and mean very different things.
        const b = data.bargain;
        if (b) {
          const t = el("table", "bargain");
          const row = (k, v, cls) => {
            const r = el("tr");
            r.append(el("th", null, k), el("td", cls || null, v));
            t.append(r);
          };
          row("zone of agreement",
            game.currency + b.zopaLow + " – " + game.currency + b.zopaHigh +
            "  (" + game.currency + (b.zopaHigh - b.zopaLow) + " of surplus)");
          row("you captured", Math.round(b.captured * 100) + "% of it",
            b.captured >= b.rubinstein ? "good" : "meh");
          row("Nash solution", Math.round(b.nash * 100) + "%  (symmetric split)");
          row("Rubinstein SPE", (b.rubinstein * 100).toFixed(1) + "%  (first mover, δ = " + b.discount + ")");
          box.append(t);
          box.append(el("p", null, b.verdict));
        }
        const lp = lessonPanel(data.lesson);
        if (lp) p.append(lp);
        status.textContent = "";
      } else if (data.turn.walked) {
        const box = el("div", "verdict bad");
        box.append(el("strong", null, `${game.merchant} walked away.`));
        box.append(el("p", null,
          `You pushed too hard, too low. They'd have taken ${game.currency}${data.turn.floor} ` +
          `from someone who'd made a serious offer.`));
        p.append(box);
        const lpw = lessonPanel(data.lesson);
        if (lpw) p.append(lpw);
        status.textContent = "";
      } else if (data.over) {
        const box = el("div", "verdict bad");
        box.append(el("strong", null, "No deal — you're out of offers."));
        box.append(el("p", null, `They'd have taken ${game.currency}${data.turn.floor}.`));
        p.append(box);
        status.textContent = "";
      } else {
        // Tell the player how that landed, so the mechanic is legible.
        const READ = {
          insulting: ["That landed badly — they didn't move at all.", "var(--bad)"],
          low: ["Still low. They barely shifted.", "var(--warn)"],
          close: ["That got their attention.", "var(--ok)"],
        };
        const [msg, colour] = READ[data.turn.read] ?? ["", "var(--ink-faint)"];
        status.textContent = msg;
        status.style.color = colour;
      }
    } catch (err) {
      game.offers = game.offers.filter((o) => o.line !== "…");
      draw();
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    } finally {
      updateLeft();
    }
  }

  offerBtn.addEventListener("click", offer);
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      offer();
    }
  });
  walk.addEventListener("click", () => {
    game.over = true;
    updateLeft();
    status.textContent = "You walked. They didn't come after you.";
  });

  draw();
  updateLeft();
  return p;
}

// ---------------------------------------------------------------- mafia UI

const ROLE_LABEL = { mafia: "mafia", doctor: "the doctor", villager: "a villager", detective: "the detective" };

function renderMafia(result, initial, label) {
  const p = panel("mafia", label, "Mafia", result.scene);
  let game = initial;

  const setting = el("div", "mafia-setting", game.setting);
  const phaseBar = el("div", "phase-bar");
  const board = el("div", "board");
  const record = el("details", "record");
  const actions = el("div", "crow-actions");
  const status = el("div", "meta");
  record.append(el("summary", null, "What's happened so far"));
  const recordList = el("ol", "record-list");
  record.append(recordList);
  p.append(setting, phaseBar, board, actions, status, record);

  let busyTurn = false;

  const living = () => [
    ...(game.youAlive ? [{ name: "You", alive: true, role: null, said: [], you: true }] : []),
    ...game.players,
  ];

  /** What the visitor has proven about a player, if anything. */
  function verdictFor(name) {
    const found = game.found.find((f) => f.name === name);
    if (!found) return null;
    return found.mafia ? "mafia" : "clean";
  }

  function drawRecord() {
    recordList.replaceChildren();
    for (const line of game.log) recordList.append(el("li", null, line));
    record.open = game.log.length > 0 && game.phase !== "over";
  }

  function drawPhase() {
    phaseBar.className = `phase-bar ${game.phase}`;
    phaseBar.replaceChildren();
    const title =
      game.phase === "over"
        ? game.winner === "town"
          ? "The town wins"
          : "The mafia win"
        : game.phase === "night"
          ? `Night ${game.day + 1}`
          : `Day ${game.day}`;
    phaseBar.append(el("span", "phase-name", title));
    const hint =
      game.phase === "over"
        ? game.youAlive
          ? ""
          : "You didn't make it."
        : game.phase === "night"
          ? "Investigate one of them. You'll learn whether they're mafia."
          : "They've spoken. Vote someone out.";
    if (hint) phaseBar.append(el("span", "phase-hint", hint));
    if (game.belief && game.phase !== "over") {
      const bl = game.belief;
      phaseBar.append(el("span", "phase-bits",
        bl.viable + "/" + bl.total + " hypotheses \u00b7 " + bl.entropy.toFixed(2) + " bits"));
    }
  }

  function drawBoard() {
    board.replaceChildren();
    for (const pl of living()) {
      const card = el("div", "player");
      if (!pl.alive) card.classList.add("dead");
      if (pl.you) card.classList.add("you");

      const head = el("div", "player-head");
      head.append(el("span", "player-name", pl.you ? "You" : pl.name));

      const verdict = pl.you ? null : verdictFor(pl.name);
      if (pl.you) head.append(el("span", "tag detective", "detective"));
      else if (!pl.alive && pl.role) head.append(el("span", `tag ${pl.role}`, ROLE_LABEL[pl.role]));
      else if (verdict) head.append(el("span", `tag ${verdict}`, verdict === "mafia" ? "you know: mafia" : "you know: clean"));

      const bel = game.belief && game.belief.players.find((x) => x.name === pl.name);
      if (bel && !pl.you && pl.alive) {
        const pct = Math.round(bel.p * 100);
        const chip = el("span", "pmaf " + (pct >= 100 ? "sure" : pct === 0 ? "clear" : ""), pct + "%");
        chip.title = "Posterior probability this player is mafia, from hard evidence only";
        head.append(chip);
      }
      if (pl.alive && !pl.you && game.phase !== "over") {
        const act = el("button", "btn small", game.phase === "night" ? "Investigate" : "Vote out");
        act.type = "button";
        act.disabled = busyTurn;
        act.addEventListener("click", () => (game.phase === "night" ? doNight(pl.name) : doVote(pl.name)));
        head.append(act);
      }
      card.append(head);

      const latest = pl.said?.[pl.said.length - 1];
      if (latest && pl.alive) {
        card.append(el("p", "player-line", latest.text));
        card.append(el("p", "player-vote", `votes: ${latest.vote}`));
      }
      board.append(card);
    }
  }

  function drawActions() {
    actions.replaceChildren();
    if (game.phase === "over") {
      const again = el("button", "btn primary", "Deal again");
      again.type = "button";
      again.addEventListener("click", () => {
        promptEl.value = "play mafia";
        form.requestSubmit();
      });
      actions.append(again);
      return;
    }
    if (game.phase === "night") {
      const skip = el("button", "btn", "Investigate nobody");
      skip.type = "button";
      skip.disabled = busyTurn;
      skip.addEventListener("click", () => doNight(null));
      actions.append(skip);
    }
    const counts = el("span", "lives");
    const aliveAI = game.players.filter((x) => x.alive).length;
    counts.textContent = `${aliveAI + (game.youAlive ? 1 : 0)} still at the table`;
    actions.append(counts);
  }

  function redraw() {
    drawPhase();
    drawBoard();
    drawActions();
    drawRecord();
  }

  function apply(data) {
    game = { ...game, ...data };
  }

  async function doNight(target) {
    if (busyTurn || game.phase !== "night") return;
    busyTurn = true;
    redraw();
    status.textContent = target ? `Looking into ${target}…` : "You keep your head down…";
    status.style.color = "var(--ink-faint)";
    try {
      const res = await fetch(api(`/api/play/mafia/${game.id}/night`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      apply(data);
      const n = data.night;
      const parts = [];
      if (n.investigated) {
        parts.push(
          n.investigated.mafia
            ? `${n.investigated.name} is mafia.`
            : `${n.investigated.name} is not mafia.`,
        );
      }
      parts.push(n.killed ? `${n.killed === "You" ? "You were" : n.killed + " was"} killed.` : "Nobody died — someone was protected.");
      status.textContent = parts.join("  ");
      status.style.color = n.investigated?.mafia ? "var(--bad)" : "var(--ink-faint)";
    } catch (err) {
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    } finally {
      busyTurn = false;
      redraw();
    }
  }

  async function doVote(target) {
    if (busyTurn || game.phase !== "day") return;
    busyTurn = true;
    redraw();
    try {
      const res = await fetch(api(`/api/play/mafia/${game.id}/vote`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      apply(data);
      const v = data.vote;
      status.textContent = v.eliminated
        ? `${v.eliminated} was voted out — ${ROLE_LABEL[v.role] ?? v.role}.`
        : "The vote was tied. Nobody went.";
      status.style.color = v.role === "mafia" ? "var(--ok)" : "var(--ink-faint)";
      if (data.winner) {
        const box = el("div", data.winner === "town" ? "verdict good" : "verdict bad");
        box.append(el("strong", null, data.winner === "town" ? "You got them all." : "The mafia have the table."));
        box.append(
          el("p", null,
            "They were: " +
              game.players.filter((x) => x.role === "mafia").map((x) => x.name).join(" and ") +
              "."),
        );
        p.append(box);
        const lm = lessonPanel(data.lesson);
        if (lm) p.append(lm);
      }
    } catch (err) {
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    } finally {
      busyTurn = false;
      redraw();
    }
  }

  redraw();
  status.textContent = result.opening ?? "Night one. Choose someone to investigate.";
  return p;
}

// ------------------------------------------------------------------ spy UI

function renderSpy(result, game, label) {
  const p = panel("spy", label, "Who is the spy?", result.scene);

  const brief = el("div", "spy-brief");
  brief.append(
    el("span", "spy-brief-label", "The word everyone but one of them was given"),
    el("strong", "spy-word", game.word),
  );
  p.append(brief);

  const room = el("div", "room");
  p.append(room);

  const askRow = el("div", "crow-actions");
  const askInput = el("input", "spy-ask");
  askInput.type = "text";
  askInput.placeholder = "Ask the room something…";
  askInput.maxLength = 200;
  const askBtn = el("button", "btn", "Ask");
  const left = el("span", "lives");
  askBtn.type = "button";
  askRow.append(askInput, askBtn, left);

  const status = el("div", "meta");
  p.append(askRow, status);

  let suspects = game.suspects;
  let questionsLeft = game.questionsLeft;
  let over = false;
  let busyAsking = false;

  const updateLeft = () => {
    left.textContent = over
      ? ""
      : `${questionsLeft} question${questionsLeft === 1 ? "" : "s"} left`;
    askInput.disabled = askBtn.disabled = over || questionsLeft === 0 || busyAsking;
  };

  function draw() {
    room.replaceChildren();
    suspects.forEach((s) => {
      const card = el("div", "suspect");
      const head = el("div", "suspect-head");
      head.append(el("span", "suspect-name", s.name));
      const accuse = el("button", "btn accuse", "Accuse");
      accuse.type = "button";
      accuse.disabled = over;
      accuse.addEventListener("click", () => doAccuse(s.name, card));
      head.append(accuse);
      card.append(head);
      s.said.forEach((line, i) => {
        const said = el("p", "suspect-line", line);
        if (i > 0) said.classList.add("later");
        card.append(said);
      });
      room.append(card);
    });
  }

  async function doAsk() {
    const question = askInput.value.trim();
    if (question.length < 3 || over || busyAsking) return;
    busyAsking = true;
    updateLeft();
    status.textContent = "The room considers…";
    status.style.color = "var(--ink-faint)";
    try {
      const res = await fetch(api(`/api/play/spy/${game.id}/ask`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      suspects = data.suspects;
      questionsLeft = data.questionsLeft;
      askInput.value = "";
      status.textContent = `You asked: “${question}”`;
      draw();
    } catch (err) {
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    } finally {
      busyAsking = false;
      updateLeft();
    }
  }

  async function doAccuse(name, card) {
    if (over) return;
    over = true;
    updateLeft();
    try {
      const res = await fetch(api(`/api/play/spy/${game.id}/accuse`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      draw();
      for (const el2 of room.querySelectorAll(".suspect")) {
        const who = el2.querySelector(".suspect-name").textContent;
        if (who === data.spy) el2.classList.add("was-spy");
        else if (who === name) el2.classList.add("wrongly-accused");
        el2.querySelector(".accuse").disabled = true;
      }

      const verdict = el("div", data.correct ? "verdict good" : "verdict bad");
      verdict.append(
        el("strong", null, data.correct ? `Got them — ${data.spy} was the spy.` : `Wrong. It was ${data.spy}.`),
      );
      verdict.append(
        el("p", null, `They were never told “${data.word}”. All they had was: ${data.category}`),
      );
      p.append(verdict);
      status.textContent = "";
    } catch (err) {
      over = false;
      updateLeft();
      status.textContent = err.message || "Lost the connection.";
      status.style.color = "var(--bad)";
    }
  }

  askBtn.addEventListener("click", doAsk);
  askInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      doAsk();
    }
  });

  draw();
  updateLeft();
  status.textContent = "Three of them know the word. One is guessing.";
  return p;
}

// ---------------------------------------------------------------- piano UI

const BLACK = new Set([1, 3, 6, 8, 10]);
const isBlack = (midi) => BLACK.has(((midi % 12) + 12) % 12);

/**
 * Key geometry is fixed in pixels so falling notes line up with keys exactly,
 * but the width scales to the range: a one-octave lesson gets fat, easy-to-hit
 * keys instead of a narrow strip stranded in a wide panel.
 */
const keyWidth = (whiteCount) => (whiteCount <= 8 ? 62 : whiteCount <= 15 ? 46 : 38);
const LANE_H = 300;
const PX_PER_SEC = 110;
const LEAD_IN = 2.4;  // seconds of fall before the first note lands

/** Hit windows in seconds, either side of the note's true time. */
const PERFECT = 0.11;
const GOOD = 0.24;
const LATE_MISS = 0.34;

/**
 * Voice management: notes sustain while held, like a real key.
 *
 * A single fixed envelope per press makes every note a pluck of the same
 * length no matter how long you hold it. So each sounding note is a voice that
 * lives from note-on to note-off — attack, a slight decay to a sustain level,
 * then a short release when the key comes up.
 */
let audio = null;
let master = null;
let reverbSend = null;
let pianoWave = null;
let noiseBuf = null;
const voices = new Map(); // midi -> { oscs, amp, guard }

/** Nothing rings forever if a keyup is lost to a blur or a dropped pointer. */
const MAX_SUSTAIN = 14;

/**
 * Relative strength of each harmonic. A single naked waveform is what makes a
 * synth read as 8-bit; a piano's body comes from a decaying stack of partials,
 * so the tone is built as one PeriodicWave rather than one oscillator shape.
 */
function makePianoWave(ctx) {
  const partials = [0, 1, 0.62, 0.38, 0.24, 0.15, 0.1, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012, 0.008];
  return ctx.createPeriodicWave(new Float32Array(partials.length), Float32Array.from(partials));
}

/**
 * Impulse response for the room. The noise is smoothed as it's generated —
 * raw white noise convolves into a metallic, spring-reverb rasp, which was a
 * good part of what sounded "weird".
 */
function makeImpulse(ctx, seconds, decay) {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = buf.getChannelData(ch);
    let smooth = 0;
    for (let i = 0; i < len; i++) {
      // One-pole lowpass over the noise: darker, closer to a real room tail.
      smooth = smooth * 0.72 + (Math.random() * 2 - 1) * 0.28;
      data[i] = smooth * Math.pow(1 - i / len, decay);
    }
  }
  return buf;
}

function makeNoise(ctx, seconds) {
  const len = Math.floor(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
  return buf;
}

function ensureAudio() {
  if (audio) {
    if (audio.state === "suspended") audio.resume();
    return audio;
  }
  audio = new (window.AudioContext || window.webkitAudioContext)();

  // Master chain: a gentle limiter keeps overlapping voices from clipping.
  // Light touch — at ratio 6 this pumped audibly between notes.
  const comp = audio.createDynamicsCompressor();
  comp.threshold.value = -8;
  comp.knee.value = 20;
  comp.ratio.value = 2.5;
  comp.attack.value = 0.006;
  comp.release.value = 0.35;
  master = audio.createGain();
  master.gain.value = 0.8;
  master.connect(comp).connect(audio.destination);

  const convolver = audio.createConvolver();
  convolver.buffer = makeImpulse(audio, 1.8, 2.6);
  reverbSend = audio.createGain();
  reverbSend.gain.value = 0.19;
  reverbSend.connect(convolver).connect(master);

  pianoWave = makePianoWave(audio);
  noiseBuf = makeNoise(audio, 0.25);
  return audio;
}

// ------------------------------------------------------- sampled piano

/**
 * Recorded Yamaha C5 (Salamander, public domain), sampled in minor thirds.
 *
 * Synthesis was never going to reach this: a real recording carries the
 * inharmonic partials, per-partial decay and soundboard resonance that make a
 * piano a piano. Only the pitches a lesson actually touches are fetched, so a
 * page load pulls ~4-8 files rather than all thirty.
 */
const SAMPLE_MIDI = [
  21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90,
  93, 96, 99, 102, 105, 108,
];
const SAMPLE_FILE = {
  21: "A0", 24: "C1", 27: "Ds1", 30: "Fs1", 33: "A1", 36: "C2", 39: "Ds2", 42: "Fs2", 45: "A2",
  48: "C3", 51: "Ds3", 54: "Fs3", 57: "A3", 60: "C4", 63: "Ds4", 66: "Fs4", 69: "A4", 72: "C5",
  75: "Ds5", 78: "Fs5", 81: "A5", 84: "C6", 87: "Ds6", 90: "Fs6", 93: "A6", 96: "C7", 99: "Ds7",
  102: "Fs7", 105: "A7", 108: "C8",
};

/**
 * Three of the sixteen recorded velocity layers: soft, medium and firm. A soft
 * strike on a real piano is not merely quieter, it is a different timbre — the
 * hammer barely engages the string's upper partials — which is why one layer
 * with the treble rolled off never quite convinces.
 */
const LAYERS = ["p", "m", "f"];
const LAYER_NAME = { p: "soft", m: "medium", f: "firm" };

const buffers = new Map(); // "midi:layer" -> AudioBuffer
const pending = new Map();
let samplesReady = false;

const nearestSample = (midi) =>
  SAMPLE_MIDI.reduce((best, m) => (Math.abs(m - midi) < Math.abs(best - midi) ? m : best));

const layerFor = (velocity) => (velocity < 0.55 ? "p" : velocity < 0.85 ? "m" : "f");

async function loadSample(sampleMidi, layer) {
  const key = `${sampleMidi}:${layer}`;
  if (buffers.has(key)) return buffers.get(key);
  if (pending.has(key)) return pending.get(key);
  const ctx = ensureAudio();
  const job = (async () => {
    const res = await fetch(`${AUDIO_BASE}/${SAMPLE_FILE[sampleMidi]}_${layer}.m4a`);
    if (!res.ok) throw new Error(`${SAMPLE_FILE[sampleMidi]}_${layer} ${res.status}`);
    const buf = await ctx.decodeAudioData(await res.arrayBuffer());
    buffers.set(key, buf);
    return buf;
  })();
  pending.set(key, job);
  return job;
}

/** Fetch one velocity layer for every pitch a set of notes reaches into. */
async function loadSamplesFor(midis, layer = "m") {
  const needed = [...new Set(midis.map(nearestSample))];
  const results = await Promise.allSettled(needed.map((m) => loadSample(m, layer)));
  const ok = results.filter((r) => r.status === "fulfilled").length;
  if (ok > 0) samplesReady = true;
  return { requested: needed.length, loaded: ok };
}

/** Best available buffer for this pitch, preferring the requested layer. */
function pickBuffer(sampleMidi, layer) {
  const order = [layer, "m", ...LAYERS];
  for (const l of order) {
    const buf = buffers.get(`${sampleMidi}:${l}`);
    if (buf) return { buf, layer: l };
  }
  return null;
}

/**
 * Play a recorded note. Velocity shapes brightness as well as level: this
 * rendering has one velocity layer, so a softer strike is approximated by
 * rolling off the top end, which is roughly what a real soft strike does.
 */
function sampleOn(midi, velocity = 1) {
  const ctx = ensureAudio();
  const src = nearestSample(midi);
  const want = layerFor(velocity);
  const hit = pickBuffer(src, want);
  if (!hit) return null;
  const t = ctx.currentTime;

  const node = ctx.createBufferSource();
  node.buffer = hit.buf;
  // Never more than 1.5 semitones from a real sample, so shifting is inaudible.
  node.playbackRate.value = Math.pow(2, (midi - src) / 12);

  const amp = ctx.createGain();
  amp.gain.setValueAtTime(0.0001, t);
  // The recorded layer already carries most of the dynamic difference, so this
  // only trims the gap between the requested layer and the one available.
  const trim = hit.layer === want ? 1 : 0.55 + 0.55 * velocity;
  amp.gain.linearRampToValueAtTime(0.42 * trim, t + 0.004);

  node.connect(amp);
  amp.connect(master);
  amp.connect(reverbSend);
  node.start(t);

  const guard = setTimeout(() => noteOff(midi), MAX_SUSTAIN * 1000);
  voices.set(midi, { sampled: true, node, amp, guard });
  return true;
}

function noteOn(midi, velocity = 1) {
  ensureAudio();
  if (voices.has(midi)) noteOff(midi, true);
  if (samplesReady && sampleOn(midi, velocity)) return;
  synthOn(midi);
}

/** Fallback tone generator, used only if the samples fail to load. */
function synthOn(midi) {
  const ctx = ensureAudio();
  const t = ctx.currentTime;
  const freq = 440 * Math.pow(2, (midi - 69) / 12);

  const amp = ctx.createGain();
  const filter = ctx.createBiquadFilter();
  filter.type = "lowpass";
  filter.Q.value = 0.4;
  // Brightness closes as the note ages, because a string's upper partials die
  // first. Kept shallow and slow: a wide fast sweep reads as a synth filter
  // "wow" rather than a piano.
  filter.frequency.setValueAtTime(Math.min(8000, freq * 11), t);
  filter.frequency.setTargetAtTime(Math.max(freq * 5, 500), t, 0.9);

  // Standing in for a note's multiple strings. Real unisons are detuned by
  // barely a cent or two; at ±3.5 the beating was slow enough to hear as a
  // wobble, which was most of the "weird".
  const oscs = [];
  for (const [detune, level] of [[0, 1], [1.4, 0.42], [-1.6, 0.38]]) {
    const osc = ctx.createOscillator();
    osc.setPeriodicWave(pianoWave);
    osc.frequency.value = freq;
    osc.detune.value = detune;
    const g = ctx.createGain();
    g.gain.value = level;
    osc.connect(g).connect(filter);
    osc.start(t);
    oscs.push(osc);
  }

  // A short filtered noise burst for the hammer strike.
  const hammer = ctx.createBufferSource();
  hammer.buffer = noiseBuf;
  const hp = ctx.createBiquadFilter();
  hp.type = "bandpass";
  hp.frequency.value = Math.min(freq * 5, 5200);
  hp.Q.value = 0.7;
  const hg = ctx.createGain();
  // Barely there. At 0.22 the burst read as an audible chirp on every note.
  hg.gain.setValueAtTime(0.07, t);
  hg.gain.exponentialRampToValueAtTime(0.0001, t + 0.028);
  hammer.connect(hp).connect(hg).connect(filter);
  hammer.start(t);
  hammer.stop(t + 0.09);

  // The envelope: near-instant strike, then a continuous exponential decay that
  // never plateaus. A flat sustain is precisely what made this sound like a
  // game console — real strings start dying the moment they are struck.
  const peak = 0.26 * (1 - Math.max(0, midi - 60) / 260);
  amp.gain.setValueAtTime(0.0001, t);
  amp.gain.linearRampToValueAtTime(peak, t + 0.004);
  // Bass notes ring far longer than treble ones.
  const tau = 2.2 * Math.pow(2, -(midi - 60) / 22);
  amp.gain.setTargetAtTime(0.0001, t + 0.005, tau);

  filter.connect(amp);
  amp.connect(master);
  amp.connect(reverbSend);

  const guard = setTimeout(() => noteOff(midi), MAX_SUSTAIN * 1000);
  voices.set(midi, { oscs, hammer, amp, guard });
}

function noteOff(midi, immediate = false) {
  const v = voices.get(midi);
  if (!v) return;
  voices.delete(midi);
  clearTimeout(v.guard);
  const t = audio.currentTime;
  // Damper felt. Generous on purpose: a short tail makes every tap a detached
  // blip, which is what made the whole thing sound chopped up. Real dampers
  // take a moment to settle, and the overlap is what reads as legato.
  const tail = immediate ? 0.03 : 0.55;
  v.amp.gain.cancelScheduledValues(t);
  v.amp.gain.setValueAtTime(Math.max(v.amp.gain.value, 0.0001), t);
  v.amp.gain.exponentialRampToValueAtTime(0.0001, t + tail);

  if (v.sampled) {
    try {
      v.node.stop(t + tail + 0.05);
    } catch {
      /* already ended */
    }
    return;
  }
  for (const osc of v.oscs) osc.stop(t + tail + 0.04);
  try {
    v.hammer.stop(t + tail);
  } catch {
    /* already finished */
  }
}

function allNotesOff() {
  for (const midi of [...voices.keys()]) noteOff(midi);
}

/** Fixed-length note, for playback rather than performance. */
function pluck(midi, seconds) {
  noteOn(midi);
  setTimeout(() => noteOff(midi), Math.max(seconds, 0.12) * 1000);
}

// A lost keyup (tab switch, window blur) would otherwise leave a note ringing.
window.addEventListener("blur", allNotesOff);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) allNotesOff();
});

function midiToName(midi) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

/**
 * Computer-keyboard mapping, two octaves up from the lesson's lowest C, in the
 * layout every piano app uses: white keys along the home row, black keys on the
 * row above. Clicking is too slow to play in time, so this is what makes the
 * falling mode playable — but only if you can see which letter to press, which
 * is why the letter is printed on every key and on every falling block.
 */
const KEY_ROWS = "awsedftgyhujkolp;'".split("");
const KEY_SEMITONES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

/** Printable form of the mapped key — ";" and "'" need to read clearly. */
const KEY_GLYPH = { ";": ";", "'": "'" };

function renderPiano(result, lesson, label) {
  const p = panel("piano", label, lesson.title, result.teacher_note);

  const meta = el("div", "meta");
  for (const t of [lesson.attribution, `${lesson.tempo} bpm`, `${lesson.hand} hand`, lesson.difficulty]) {
    meta.append(el("span", "pill", t), document.createTextNode(" "));
  }
  p.append(meta);

  if (result.substituted === "yes") {
    const swap = el("div", "swap");
    swap.append(el("b", null, "Different piece: "), document.createTextNode(
      "the one you asked for is still in copyright, so its notes can't be reproduced here."));
    p.append(swap);
  }

  // --- geometry: extend to whole octaves so it looks like a piano ---
  const low = lesson.lowMidi - (((lesson.lowMidi % 12) + 12) % 12);
  let high = lesson.highMidi;
  while (((high % 12) + 12) % 12 !== 11) high++;

  const whites = [];
  for (let m = low; m <= high; m++) if (!isBlack(m)) whites.push(m);
  const WHITE_W = keyWidth(whites.length);
  const BLACK_W = Math.round(WHITE_W * 0.6);
  const boardW = whites.length * WHITE_W;

  // midi -> computer key letter, so both the keys and the falling blocks can
  // show what to press. Without this the mapping is invisible and unguessable.
  const letterFor = new Map();
  KEY_ROWS.forEach((ch, i) => {
    const midi = low + KEY_SEMITONES[i];
    if (midi <= high) letterFor.set(midi, KEY_GLYPH[ch] ?? ch.toUpperCase());
  });

  /** left offset and width for any midi note in range */
  const geom = new Map();
  whites.forEach((m, i) => {
    geom.set(m, { left: i * WHITE_W, width: WHITE_W - 2 });
    if (isBlack(m + 1) && m + 1 <= high) {
      geom.set(m + 1, { left: (i + 1) * WHITE_W - BLACK_W / 2 - 1, width: BLACK_W });
    }
  });

  // --- schedule: absolute time of every sounding note ---
  const beat = 60 / lesson.tempo;
  const sched = [];
  let t = 0;
  for (const n of lesson.notes) {
    const dur = n.beats * beat;
    if (n.midi !== null && geom.has(n.midi)) {
      sched.push({ midi: n.midi, name: n.name, time: t, dur, state: "pending", el: null });
    }
    t += dur;
  }
  const songEnd = t;

  // --- DOM ---
  const wrap = el("div", "piano-wrap");
  const inner = el("div", "piano-inner");
  inner.style.width = `${boardW}px`;

  const lane = el("div", "fall-lane");
  lane.style.height = `${LANE_H}px`;
  const hitline = el("div", "hitline");
  lane.append(hitline);

  const board = el("div", "piano");
  const keys = new Map();
  const whiteRow = el("div", "keys-white");
  whites.forEach((m) => {
    const k = el("button", "wkey");
    k.type = "button";
    k.dataset.midi = String(m);
    k.style.width = `${WHITE_W - 2}px`;
    if (letterFor.has(m)) k.append(el("span", "key-letter", letterFor.get(m)));
    if (((m % 12) + 12) % 12 === 0) k.append(el("span", "octave-label", `C${Math.floor(m / 12) - 1}`));
    keys.set(m, k);
    whiteRow.append(k);
  });
  board.append(whiteRow);
  for (const [m, g] of geom) {
    if (!isBlack(m)) continue;
    const b = el("button", "bkey");
    b.type = "button";
    b.dataset.midi = String(m);
    b.style.left = `${g.left}px`;
    b.style.width = `${g.width}px`;
    if (letterFor.has(m)) b.append(el("span", "key-letter dark", letterFor.get(m)));
    keys.set(m, b);
    board.append(b);
  }

  // Falling blocks, positioned once and moved with transform each frame.
  for (const s of sched) {
    const g = geom.get(s.midi);
    const block = el("div", isBlack(s.midi) ? "fnote black" : "fnote");
    block.style.left = `${g.left}px`;
    block.style.width = `${g.width}px`;
    const h = Math.max(s.dur * PX_PER_SEC - 3, 16);
    block.style.height = `${h}px`;
    // The letter is what you act on, so it sits at the bottom edge — the part
    // that meets the line. The note name is secondary and only if it fits.
    if (h > 40 && !isBlack(s.midi)) block.append(el("span", "fnote-name", s.name));
    block.append(el("span", "fnote-key", letterFor.get(s.midi) ?? s.name));
    s.el = block;
    lane.append(block);
  }

  inner.append(lane, board);
  wrap.append(inner);
  p.append(wrap);

  // --- controls ---
  const controls = el("div", "crow-actions");
  const startBtn = el("button", "btn primary", "Start");
  const listenBtn = el("button", "btn", "Listen");
  const speedSel = el("select", "speed");
  for (const [v, tag] of [[0.5, "0.5×"], [0.75, "0.75×"], [1, "1×"], [1.25, "1.25×"], [1.5, "1.5×"]]) {
    const o = el("option", null, tag);
    o.value = String(v);
    if (v === 1) o.selected = true;
    speedSel.append(o);
  }
  const score = el("span", "lives");
  startBtn.type = listenBtn.type = "button";
  controls.append(startBtn, listenBtn, speedSel, score);

  const judge = el("div", "judge");
  p.append(controls, judge);

  const hint = el("div", "meta");
  hint.textContent = "Each block shows the keyboard letter to press. Hit it as the block reaches the line — or click the keys.";
  p.append(hint);

  // --- state ---
  let running = false;
  let startedAt = 0;
  let rate = 1;
  let raf = 0;
  const tally = { perfect: 0, good: 0, missed: 0, combo: 0, best: 0 };

  const now = () => (performance.now() - startedAt) / 1000 * rate - LEAD_IN;

  function updateScore() {
    const hit = tally.perfect + tally.good;
    const total = hit + tally.missed;
    const pct = total ? Math.round((tally.perfect + tally.good * 0.6) / total * 100) : 100;
    score.textContent = total
      ? `${pct}% · ${tally.perfect} perfect · ${tally.good} good · ${tally.missed} missed · combo ${tally.best}`
      : `${sched.length} notes`;
  }

  function flashJudge(text, cls) {
    judge.textContent = text;
    judge.className = `judge ${cls} show`;
    setTimeout(() => judge.classList.remove("show"), 320);
  }

  function reset() {
    cancelAnimationFrame(raf);
    running = false;
    for (const s of sched) {
      s.state = "pending";
      s.el.className = isBlack(s.midi) ? "fnote black" : "fnote";
      s.el.style.transform = `translateY(${LANE_H}px)`;
    }
    tally.perfect = tally.good = tally.missed = tally.combo = tally.best = 0;
    updateScore();
    startBtn.textContent = "Start";
  }

  function frame() {
    const elapsed = now();
    for (const s of sched) {
      // y is where the note's bottom edge sits: it reaches the line at s.time.
      const y = LANE_H - (s.time - elapsed) * PX_PER_SEC;
      const h = parseFloat(s.el.style.height);
      if (y < -h - 40 || y > LANE_H + 400) {
        s.el.style.visibility = "hidden";
      } else {
        s.el.style.visibility = "visible";
        s.el.style.transform = `translateY(${y - h}px)`;
      }
      if (s.state === "pending" && elapsed > s.time + LATE_MISS) {
        s.state = "missed";
        s.el.classList.add("missed");
        tally.missed++;
        tally.combo = 0;
        updateScore();
      }
    }
    if (elapsed > songEnd + 1.2) {
      running = false;
      startBtn.textContent = "Play again";
      const hit = tally.perfect + tally.good;
      flashJudge(hit === sched.length && tally.missed === 0 ? "Clean run!" : "Done", "ok");
      return;
    }
    raf = requestAnimationFrame(frame);
  }

  async function start() {
    ensureAudio(); // unlock audio on the gesture
    // First press pays for the samples; afterwards they're cached in memory.
    if (!samplesReady) {
      startBtn.disabled = true;
      startBtn.textContent = "Loading piano…";
      await loadPiano();
      startBtn.disabled = false;
    }
    reset();
    running = true;
    rate = Number(speedSel.value);
    startedAt = performance.now();
    startBtn.textContent = "Restart";
    raf = requestAnimationFrame(frame);
  }

  /**
   * Pull only the pitches this lesson touches, medium layer first.
   *
   * Waiting on all three velocity layers would triple the time before you can
   * play. The medium layer alone is enough to start; soft and firm arrive in
   * the background and simply begin getting used, since pickBuffer falls back
   * to whatever is loaded.
   */
  async function loadPiano() {
    const midis = sched.map((s) => s.midi);
    try {
      const { requested, loaded } = await loadSamplesFor(midis, "m");
      if (loaded < requested) status.textContent = `Loaded ${loaded}/${requested} piano samples.`;
      for (const layer of ["f", "p"]) {
        loadSamplesFor(midis, layer).catch(() => {});
      }
    } catch {
      status.textContent = "Couldn't load the piano samples — using the fallback tone.";
    }
  }

  function release(midi) {
    keys.get(midi)?.classList.remove("down");
    noteOff(midi);
  }

  function press(midi) {
    const key = keys.get(midi);
    if (!key) return;
    // The lit state lasts as long as the note does — released in release().
    key.classList.add("down");

    // Velocity from timing: a note struck right on the beat sounds firmer than
    // one scrambled at the edge of the window, which is true of playing too.
    let velocity = 0.85;
    if (running) {
      const elapsed = now();
      const nearest = sched
        .filter((s) => s.midi === midi && s.state === "pending")
        .map((s) => Math.abs(s.time - elapsed))
        .sort((a, b) => a - b)[0];
      if (nearest !== undefined) velocity = nearest <= PERFECT ? 1 : nearest <= GOOD ? 0.8 : 0.6;
    }
    noteOn(midi, velocity);

    if (!running) return;
    const elapsed = now();
    // Nearest pending note for this key, inside the widest window.
    let best = null;
    for (const s of sched) {
      if (s.midi !== midi || s.state !== "pending") continue;
      const delta = Math.abs(s.time - elapsed);
      if (delta <= GOOD && (!best || delta < Math.abs(best.time - elapsed))) best = s;
    }
    if (!best) {
      tally.combo = 0;
      flashJudge("✗", "bad");
      updateScore();
      return;
    }
    const delta = Math.abs(best.time - elapsed);
    best.state = "hit";
    best.el.classList.add(delta <= PERFECT ? "perfect" : "good");
    if (delta <= PERFECT) tally.perfect++;
    else tally.good++;
    tally.combo++;
    tally.best = Math.max(tally.best, tally.combo);
    flashJudge(delta <= PERFECT ? "Perfect" : "Good", delta <= PERFECT ? "ok" : "meh");
    updateScore();
  }

  // pointerdown/up rather than click, so a held mouse button sustains the note.
  for (const [midi, key] of keys) {
    key.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      key.setPointerCapture?.(ev.pointerId);
      press(midi);
    });
    for (const end of ["pointerup", "pointercancel", "pointerleave"]) {
      key.addEventListener(end, () => release(midi));
    }
  }

  startBtn.addEventListener("click", start);
  speedSel.addEventListener("change", () => { if (!running) rate = Number(speedSel.value); });

  listenBtn.addEventListener("click", async () => {
    if (running) return;
    listenBtn.disabled = true;
    ensureAudio();
    if (!samplesReady) { listenBtn.textContent = "Loading…"; await loadPiano(); listenBtn.textContent = "Listen"; }
    const r = Number(speedSel.value);
    for (const n of lesson.notes) {
      const seconds = (n.beats * beat) / r;
      if (n.midi !== null) {
        const key = keys.get(n.midi);
        key?.classList.add("sounding");
        // Ring well past the slot rather than stopping at it: on a piano the
        // string keeps sounding until the damper drops, so notes overlap.
        // Cutting at 92% of the beat put a gap between every pair of notes.
        pluck(n.midi, seconds + 1.8);
        setTimeout(() => key?.classList.remove("sounding"), seconds * 900);
      }
      await new Promise((res) => setTimeout(res, seconds * 1000));
    }
    listenBtn.disabled = false;
  });

  // Computer keyboard, relative to the lesson's lowest C.
  const keyMap = new Map();
  KEY_ROWS.forEach((ch, i) => {
    const midi = low + KEY_SEMITONES[i];
    if (geom.has(midi)) keyMap.set(ch, midi);
  });
  const held = new Map(); // ev.key -> midi
  const detach = () => {
    document.removeEventListener("keydown", onDown);
    document.removeEventListener("keyup", onUp);
    allNotesOff();
  };
  const onDown = (ev) => {
    if (!document.body.contains(p)) return detach();
    if (document.activeElement === promptEl || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const midi = keyMap.get(ev.key.toLowerCase());
    // ev.repeat guards the OS key-repeat storm while a key is held down.
    if (midi === undefined || ev.repeat || held.has(ev.key)) return;
    held.set(ev.key, midi);
    ev.preventDefault();
    press(midi);
  };
  const onUp = (ev) => {
    const midi = held.get(ev.key);
    if (midi === undefined) return;
    held.delete(ev.key);
    release(midi);
  };
  document.addEventListener("keydown", onDown);
  document.addEventListener("keyup", onUp);

  reset();
  return p;
}

// --------------------------------------------------------------- wordle UI

function renderWordle(result, game, label) {
  const p = panel("wordle", label, result.theme, result.intro);

  const hint = el("div", "hint-box");
  hint.append(el("b", null, "Clue: "), document.createTextNode(game.hint));
  p.append(hint);

  const status = el("div", "meta");
  p.append(status);

  // Information-theoretic read-out: how much each guess actually bought.
  const ledger = el("table", "info-ledger");
  const head = el("tr");
  for (const h of ["guess", "left", "bits", "vs best"]) head.append(el("th", null, h));
  ledger.append(head);
  ledger.hidden = true;

  const board = el("div", "wboard");
  const cells = [];
  for (let r = 0; r < game.maxGuesses; r++) {
    const row = el("div", "wrow");
    const rowCells = [];
    for (let c = 0; c < game.length; c++) {
      const cell = el("div", "wcell");
      rowCells.push(cell);
      row.append(cell);
    }
    cells.push(rowCells);
    board.append(row);
  }
  p.append(board);

  const KEYS = ["qwertyuiop", "asdfghjkl", "↵zxcvbnm⌫"];
  const keyEls = new Map();
  const kbd = el("div", "kbd");
  for (const rowStr of KEYS) {
    const row = el("div", "krow");
    for (const ch of rowStr) {
      const wide = ch === "↵" || ch === "⌫";
      const k = el("button", wide ? "key wide" : "key", ch === "↵" ? "enter" : ch);
      k.type = "button";
      k.addEventListener("click", () => press(ch === "↵" ? "Enter" : ch === "⌫" ? "Backspace" : ch));
      if (!wide) keyEls.set(ch, k);
      row.append(k);
    }
    kbd.append(row);
  }
  p.append(kbd, ledger);

  let row = 0;
  let current = "";
  let over = false;

  function paint() {
    cells[row]?.forEach((cell, i) => {
      cell.textContent = current[i] ?? "";
      cell.classList.toggle("filled", i < current.length);
    });
  }

  function say(text, bad) {
    status.textContent = text;
    status.style.color = bad ? "var(--bad)" : "var(--ink-faint)";
  }

  async function submit() {
    if (current.length !== 5) return say("Five letters.", true);
    let data;
    try {
      const res = await fetch(api(`/api/play/wordle/${game.id}/guess`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ guess: current }),
      });
      data = await res.json();
      if (!res.ok) {
        cells[row].forEach((c) => c.classList.add("shake"));
        setTimeout(() => cells[row]?.forEach((c) => c.classList.remove("shake")), 340);
        return say(data.error, true);
      }
    } catch {
      return say("Lost the connection.", true);
    }

    // Capture the letters now: the staggered callbacks below fire after
    // `current` has been cleared for the next row.
    const guessed = current;

    if (data.info) {
      const i = data.info;
      const row = el("tr");
      row.append(el("td", "mono", guessed.toUpperCase()));
      row.append(el("td", "mono", `${i.before.toLocaleString()} → ${i.after.toLocaleString()}`));
      // Actual bits can beat or trail the expectation — that gap is the luck.
      const lucky = i.actual >= i.expected;
      row.append(el("td", `mono ${lucky ? "good" : "meh"}`, `${i.actual.toFixed(2)} / ${i.expected.toFixed(2)}`));
      row.append(
        el("td", "mono", i.best ? `${i.best.word} ${i.best.bits.toFixed(2)}` : "—"),
      );
      ledger.append(row);
      ledger.hidden = false;
    }

    data.marks.forEach((mark, i) => {
      const cell = cells[row][i];
      // Stagger the reveal; it reads as a flip rather than an instant answer.
      setTimeout(() => {
        cell.classList.remove("filled");
        cell.classList.add(mark);
        const key = keyEls.get(guessed[i]);
        // Green is sticky — a letter known-correct elsewhere must not be
        // downgraded to yellow or grey by a later guess.
        if (key && !key.classList.contains("correct")) {
          key.classList.remove("present", "absent");
          key.classList.add(mark);
        }
      }, i * 110);
    });

    row++;
    current = "";

    if (data.solved) {
      over = true;
      setTimeout(() => {
        say(`Got it in ${row}.`);
        p.append(reveal(guessed.toUpperCase(), data.fact));
      }, 620);
    } else if (row >= game.maxGuesses) {
      over = true;
      const res = await fetch(api(`/api/play/wordle/${game.id}/reveal`));
      const { answer, fact } = await res.json();
      setTimeout(() => {
        say(`Out of guesses — it was ${answer}.`, true);
        p.append(reveal(answer, fact));
      }, 620);
    } else {
      say(`${game.maxGuesses - row} guesses left.`);
    }
  }

  function reveal(answer, fact) {
    const box = el("div", "hint-box");
    box.append(el("b", null, `${answer} — `), document.createTextNode(fact));
    return box;
  }

  function press(key) {
    if (over) return;
    if (key === "Enter") return void submit();
    if (key === "Backspace") {
      current = current.slice(0, -1);
      return paint();
    }
    if (/^[a-zA-Z]$/.test(key) && current.length < 5) {
      current += key.toLowerCase();
      paint();
    }
  }

  const onKey = (ev) => {
    if (!document.body.contains(p)) return document.removeEventListener("keydown", onKey);
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (document.activeElement === promptEl) return;
    press(ev.key);
  };
  document.addEventListener("keydown", onKey);

  say(`${game.maxGuesses} guesses. Type or tap.`);
  return p;
}

// ---------------------------------------------------------- connections UI

function renderConnections(result, game, label) {
  const p = panel("connections", label, game.title, result.intro);

  const solvedWrap = el("div");
  const grid = el("div", "cgrid");
  const actions = el("div", "crow-actions");
  const submitBtn = el("button", "btn primary", "Submit");
  const shuffleBtn = el("button", "btn", "Shuffle");
  const lives = el("span", "lives");
  const status = el("div", "meta");

  submitBtn.type = shuffleBtn.type = "button";
  submitBtn.disabled = true;
  actions.append(submitBtn, shuffleBtn, lives);
  p.append(solvedWrap, grid, actions, status);

  let words = [...game.words];
  let selected = new Set();
  let mistakes = 0;
  let solvedCount = 0;
  let over = false;

  const updateLives = () => {
    lives.textContent = `${4 - mistakes} mistake${4 - mistakes === 1 ? "" : "s"} left`;
  };

  function draw() {
    grid.replaceChildren();
    for (const w of words) {
      const tile = el("button", "ctile", w);
      tile.type = "button";
      if (selected.has(w)) tile.classList.add("sel");
      tile.addEventListener("click", () => {
        if (over) return;
        if (selected.has(w)) selected.delete(w);
        else if (selected.size < 4) selected.add(w);
        submitBtn.disabled = selected.size !== 4;
        draw();
      });
      grid.append(tile);
    }
  }

  function band(group) {
    const b = el("div", `cband d${group.difficulty}`);
    b.append(el("div", "lbl", group.label), el("div", "mem", group.words.join(" · ")));
    return b;
  }

  shuffleBtn.addEventListener("click", () => {
    // Fisher-Yates over whatever tiles remain.
    for (let i = words.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [words[i], words[j]] = [words[j], words[i]];
    }
    draw();
  });

  submitBtn.addEventListener("click", async () => {
    if (selected.size !== 4 || over) return;
    const picked = [...selected];
    submitBtn.disabled = true;

    let data;
    try {
      const res = await fetch(api(`/api/play/connections/${game.id}/guess`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words: picked }),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.error);
    } catch (err) {
      status.textContent = err.message || "Lost the connection.";
      return;
    }

    if (data.correct) {
      solvedWrap.append(band(data));
      words = words.filter((w) => !data.words.includes(w));
      selected.clear();
      solvedCount++;
      status.textContent = data.label;
      draw();
      if (solvedCount === 4) {
        over = true;
        status.textContent = mistakes === 0 ? "Clean sweep." : "Solved.";
        p.append(trapNote());
      }
      return;
    }

    mistakes++;
    updateLives();
    grid.classList.add("shake");
    setTimeout(() => grid.classList.remove("shake"), 340);
    status.textContent = data.oneAway ? "One away." : "Not a group.";
    selected.clear();
    draw();

    if (mistakes >= 4) {
      over = true;
      const res = await fetch(api(`/api/play/connections/${game.id}/reveal`));
      const { groups } = await res.json();
      solvedWrap.replaceChildren(...groups.map(band));
      grid.replaceChildren();
      status.textContent = "Out of mistakes — here they all are.";
      p.append(trapNote());
    }
  });

  function trapNote() {
    const box = el("div", "hint-box");
    box.append(el("b", null, "The trap: "), document.createTextNode(result.trap));
    return box;
  }

  updateLives();
  status.textContent = "Find four groups of four.";
  draw();
  return p;
}

// --------------------------------------------------------------- trivia UI

function renderTrivia(result, quiz, label) {
  const p = panel("trivia", label, quiz.title, result.intro);
  const score = el("div", "meta");
  const scoreLine = el("span", "score");
  score.append(scoreLine);
  p.append(score);

  let right = 0;
  let answered = 0;

  const updateScore = () => {
    scoreLine.textContent = `${right} / ${answered} correct` + (answered === quiz.questions.length ? " — done" : "");
  };

  quiz.questions.forEach((q, qi) => {
    const block = el("div", "q");
    const qtext = el("div", "qtext");
    qtext.append(el("span", "n", `${qi + 1}.`), document.createTextNode(q.question));
    block.append(qtext);

    const choices = el("div", "choices");
    const buttons = q.choices.map((c, ci) => {
      const b = el("button", "choice", c);
      b.type = "button";
      b.addEventListener("click", async () => {
        buttons.forEach((x) => (x.disabled = true));
        let data;
        try {
          const res = await fetch(api(`/api/play/trivia/${quiz.id}/answer`), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index: qi, choice: ci }),
          });
          data = await res.json();
          if (!res.ok) throw new Error(data.error);
        } catch (err) {
          block.append(el("div", "why", err.message || "Lost the connection."));
          return;
        }

        buttons[data.answerIndex].classList.add("right");
        if (!data.correct) b.classList.add("wrong");
        else right++;
        answered++;
        updateScore();
        block.append(el("div", "why", data.explanation));
        if (data.ability) {
          scoreLine.textContent += "  \u00b7  " + data.ability.text;
        }
        const lt = lessonPanel(data.lesson);
        if (lt) p.append(lt);
      });
      choices.append(b);
      return b;
    });

    block.append(choices);
    p.append(block);
  });

  updateScore();
  return p;
}

loadRoster();
autosize();
