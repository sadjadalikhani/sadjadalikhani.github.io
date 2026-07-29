/**
 * Offline mode: the playground with no backend.
 *
 * When PLAYGROUND_API is unset there is nowhere to run an agent, so this
 * intercepts the same /api/ calls the app already makes and answers them from
 * pre-baked JSON packs. The app's renderers are untouched — they never knew
 * where an artifact came from — which is why a puzzle plays identically here
 * and against a live server.
 *
 * The judging that normally happens server-side has to happen here instead.
 * That means the answers are in the page, so a determined visitor can read
 * them. That is an honest trade for a static host: the alternative is no games
 * at all. With a backend configured, answers stay server-side as designed.
 *
 * Only games whose content can be settled in advance are available. Mafia, the
 * Spy Game and the Haggle generate dialogue turn by turn and are not faked
 * here — offline they say so rather than pretending.
 */
(function () {
  "use strict";
  if (window.PLAYGROUND_API) return; // live backend: nothing to intercept

  const PACKS = window.PLAYGROUND_PACKS || "/assets/json";
  const cache = {};
  const state = {};

  async function pack(name) {
    if (!cache[name]) {
      const r = await fetch(PACKS + "/pack-" + name + ".json");
      if (!r.ok) throw new Error("pack " + name + " unavailable");
      cache[name] = await r.json();
    }
    return cache[name];
  }

  /* Rotate through a pack rather than repeating, and remember across reloads. */
  function nextIndex(name, len) {
    const key = "playground.seen." + name;
    let seen = 0;
    try { seen = Number(localStorage.getItem(key) || 0); } catch (e) { /* private mode */ }
    try { localStorage.setItem(key, String(seen + 1)); } catch (e) { /* ignore */ }
    return seen % len;
  }

  const id = () => Math.random().toString(36).slice(2, 11);
  const json = (body, status) =>
    new Response(JSON.stringify(body), {
      status: status || 200,
      headers: { "Content-Type": "application/json" },
    });

  /**
   * /api/ask and /api/again are event streams, not JSON.
   *
   * The app reads them with readEventStream and renders on the `result` event,
   * so answering with a plain JSON body parses as an empty stream and puts
   * nothing on the page. Everything else in the API is ordinary JSON.
   *
   * The trace events are honest about what actually happened here: files were
   * read, no model was called.
   */
  function sse(envelope, steps) {
    const enc = new TextEncoder();
    const frame = (event, data) => enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    return new Response(
      new ReadableStream({
        start(controller) {
          for (const s of steps || []) controller.enqueue(frame("trace", s));
          controller.enqueue(frame("result", envelope));
          controller.close();
        },
      }),
      { status: 200, headers: { "Content-Type": "text/event-stream" } },
    );
  }

  /** Turn a JSON envelope from serve()/unavailable() into the stream the app wants. */
  async function streamed(res, steps) {
    const body = await res.json();
    if (!res.ok) return json(body, res.status);
    return sse(body, steps);
  }

  /* Server-side rules, reimplemented for offline judging. ------------------ */

  function markGuess(guess, answer) {
    const g = guess.toLowerCase().split("");
    const a = answer.toLowerCase().split("");
    const marks = new Array(g.length).fill("absent");
    const pool = {};
    for (let i = 0; i < a.length; i++) {
      if (g[i] === a[i]) marks[i] = "correct";
      else pool[a[i]] = (pool[a[i]] || 0) + 1;
    }
    for (let i = 0; i < g.length; i++) {
      if (marks[i] === "correct") continue;
      if (pool[g[i]] > 0) { marks[i] = "present"; pool[g[i]]--; }
    }
    return marks;
  }

  function patternCode(guess, answer) {
    const m = markGuess(guess, answer);
    let code = 0;
    for (let i = 0; i < 5; i++) {
      code += (m[i] === "correct" ? 2 : m[i] === "present" ? 1 : 0) * Math.pow(3, i);
    }
    return code;
  }

  /* H(g) = -sum p(f) log2 p(f) over the feedback partition. */
  function expectedBits(guess, candidates) {
    if (candidates.length <= 1) return 0;
    const buckets = {};
    for (const c of candidates) {
      const k = patternCode(guess, c);
      buckets[k] = (buckets[k] || 0) + 1;
    }
    let h = 0;
    for (const k in buckets) {
      const p = buckets[k] / candidates.length;
      h -= p * Math.log2(p);
    }
    return h;
  }

  let WORDS = null;
  let OPENERS = null;
  async function dictionary() {
    if (!WORDS) {
      const r = await fetch(PACKS + "/words5.txt");
      WORDS = (await r.text()).split("\n").map((w) => w.trim()).filter((w) => w.length === 5);
    }
    return WORDS;
  }
  async function openers() {
    if (!OPENERS) {
      try {
        const r = await fetch(PACKS + "/wordle-openers.json");
        OPENERS = (await r.json()).map((x) => x.w);
      } catch (e) { OPENERS = []; }
    }
    return OPENERS;
  }

  /* The poetry archive, straight from the browser. ------------------------
     poetrydb.org answers with Access-Control-Allow-Origin: *, so the verse on
     the page is the real published text fetched live, exactly as the server
     agent would fetch it. Only the commentary is pre-written. That also makes
     the shelf effectively unlimited rather than a dozen frozen poems. */

  const POETRY_DB = "https://poetrydb.org";

  async function titlesBy(poet) {
    const r = await fetch(POETRY_DB + "/author/" + encodeURIComponent(poet) + "/title");
    const j = await r.json();
    return Array.isArray(j) ? j.map((t) => t.title).filter(Boolean) : [];
  }

  async function getPoem(poet, title) {
    const r = await fetch(
      POETRY_DB + "/author,title/" + encodeURIComponent(poet) + ";" + encodeURIComponent(title));
    const j = await r.json();
    const p = Array.isArray(j) ? j[0] : null;
    return p && p.lines && p.lines.length ? { title: p.title, lines: p.lines } : null;
  }

  /* Same rule as the server: prefer something of readable length, and never
     repeat a poem this shelf has already served. */
  async function pickPoem(poet, titles, skip, maxLines) {
    const remaining = titles.filter((t) => skip.indexOf(t) === -1);
    for (const title of remaining.slice(0, 12)) {
      try {
        const poem = await getPoem(poet, title);
        if (poem && poem.lines.length >= 4 && poem.lines.length <= (maxLines || 60)) return poem;
      } catch (e) { /* try the next */ }
    }
    for (const title of remaining.slice(0, 6)) {
      try {
        const poem = await getPoem(poet, title);
        if (poem) return poem;
      } catch (e) { /* keep going */ }
    }
    return null;
  }

  /* Routes ---------------------------------------------------------------- */

  const OFFLINE_AGENTS = ["wordle", "connections", "trivia", "alibi", "piano", "poetry"];

  async function ask(prompt) {
    const p = (prompt || "").toLowerCase();

    // Asking for a live-only game has to win over every keyword below, or
    // "let's play mafia" gets caught by the piano's "play" and quietly hands
    // back the wrong thing instead of saying it can't.
    if (/\b(mafia|spy|haggle|bargain|negotiat|playlist|spotify)\b/.test(p)) return unavailable(p);

    let want = OFFLINE_AGENTS.find((a) => p.indexOf(a) !== -1);
    if (!want) {
      if (/word|five letter|5 letter/.test(p)) want = "wordle";
      else if (/group|grid|categor/.test(p)) want = "connections";
      else if (/quiz|trivia|question/.test(p)) want = "trivia";
      else if (/alibi|suspect|crime|who did/.test(p)) want = "alibi";
      else if (/poem|poet|verse|sonnet|read me/.test(p)) want = "poetry";
      else if (/song|tune|keys|teach me|melody|lesson|piano/.test(p)) want = "piano";
    }
    // Nothing recognisable and nothing live-only asked for — "surprise me" is a
    // reasonable request, so answer it with a game rather than a refusal.
    if (!want) want = OFFLINE_AGENTS[nextIndex("any", OFFLINE_AGENTS.length)];
    return serve(want, prompt);
  }

  /**
   * What to show in the trace panel. It stays truthful: this reports reading a
   * file, not thinking, because dressing a JSON lookup up as agent reasoning
   * would misrepresent where the content came from.
   */
  function trace(prompt) {
    return [
      { type: "agent_start", label: "Offline concierge" },
      { type: "note", message: "No agent server configured — serving pre-generated content." },
      {
        type: "tool_call",
        tool: "read_pack",
        args: { prompt: String(prompt || "").slice(0, 60) },
      },
      { type: "tool_result", ok: true, summary: "Loaded from the site's content packs." },
      { type: "agent_done", ms: 0, steps: 1, usage: { total_tokens: 0 } },
    ];
  }

  /** Say plainly what this copy can and cannot do, naming the reason. */
  function unavailable(p) {
    const live = /\b(playlist|spotify)\b/.test(p)
      ? "Building a Spotify playlist needs an account and a live server."
      : "Mafia, the Spy Game and the Haggle write their dialogue as you play, so they need a " +
        "live server — they aren't faked here.";
    return json({
      route: { agent: "none", task: p, reason: "no backend configured" },
      agent: null, label: null, artifacts: {}, usage: { total_tokens: 0 }, ms: 0,
      result: {
        message:
          live + " Six agents do work in this copy: ask for a wordle, a connections grid, a " +
          "quiz, an alibi case, a piano lesson, or a poem.",
      },
    });
  }

  async function serve(agent, task) {
    if (agent === "poetry") return servePoetry(task);

    const data = await pack(agent);
    const list = data.puzzles || data.quizzes || data.cases || data.lessons;
    const item = list[nextIndex(agent, list.length)];
    const gid = id();
    state[gid] = { agent: agent, item: item, history: [] };

    const envelope = {
      route: { agent: agent, task: task, reason: "from the pack" },
      agent: agent,
      usage: { total_tokens: 0 },
      ms: 0,
    };

    if (agent === "piano") {
      // Baked whole, so the lesson the renderer gets is byte-for-byte the one
      // the agent published.
      return json(Object.assign(envelope, {
        label: "Piano Tutor",
        result: item.result,
        artifacts: { piano: Object.assign({}, item.artifact, { id: gid }) },
      }));
    }

    if (agent === "wordle") {
      return json(Object.assign(envelope, {
        label: "Wordle Smith",
        result: { theme: item.theme, intro: item.intro, difficulty: item.difficulty },
        artifacts: { wordle: { id: gid, theme: item.theme, hint: item.hint, length: 5, maxGuesses: 6 } },
      }));
    }
    if (agent === "connections") {
      return json(Object.assign(envelope, {
        label: "Connections Setter",
        result: { title: item.title, intro: item.intro, trap: item.trap },
        artifacts: { connections: { id: gid, title: item.title, words: item.words } },
      }));
    }
    if (agent === "trivia") {
      return json(Object.assign(envelope, {
        label: "Quizmaster",
        result: { title: item.title, intro: item.intro, difficulty: "solid" },
        artifacts: {
          trivia: {
            id: gid, title: item.title,
            questions: item.questions.map((q) => ({ question: q.question, choices: q.choices })),
          },
        },
      }));
    }
    return json(Object.assign(envelope, {
      label: "The Alibi",
      result: { brief: item.brief, difficulty: "fair" },
      artifacts: {
        alibi: {
          id: gid, crime: item.crime, setting: item.setting, hour: item.hour, pressesLeft: 0,
          suspects: item.suspects.map((s) => ({
            name: s.name, statement: s.statement, place: s.place, with: s.with, pressed: [],
          })),
        },
      },
    }));
  }

  /**
   * A reading. The poet is whoever the visitor named, matched first against the
   * shelves that have written commentary and then against the whole archive —
   * so asking for someone obscure still works, it just arrives without a note.
   */
  async function servePoetry(task) {
    const data = await pack("poetry");
    const shelves = data.shelves;
    const q = String(task || "").toLowerCase();

    let shelf = shelves.filter((s) => {
      const name = s.poet.toLowerCase();
      const surname = name.split(/[\s,]+/).filter(Boolean).pop();
      return q.indexOf(name) !== -1 || (surname && surname.length > 3 && q.indexOf(surname) !== -1);
    })[0];

    let poet = shelf ? shelf.poet : null;
    let note = shelf ? shelf.note : null;
    let substituted = "no";

    // Named someone we have no commentary for? Try the archive itself before
    // giving up — an unannotated real poem beats the wrong poet.
    if (!shelf && /\b(by|from)\b/.test(q)) {
      const asked = q.split(/\bby\b|\bfrom\b/).pop().replace(/[^a-z\s'-]/g, "").trim();
      if (asked.length > 2) {
        try {
          const r = await fetch(POETRY_DB + "/author");
          const all = (await r.json()).authors || [];
          const hit =
            all.filter((a) => a.toLowerCase() === asked)[0] ||
            all.filter((a) => a.toLowerCase().indexOf(asked) !== -1)[0];
          if (hit) {
            poet = hit;
            note =
              "This copy has no backend, so there's no written introduction for " + hit +
              " — but the archive has them, and the poem below is the real text.";
          }
        } catch (e) { /* archive unreachable; fall through to the shelf */ }
      }
    }

    if (!poet) {
      shelf = shelves[nextIndex("poetry", shelves.length)];
      poet = shelf.poet;
      note = shelf.note;
      // Only claim a substitution when the visitor actually asked for someone.
      if (/\b(by|from)\b/.test(q)) substituted = "yes";
    }

    let titles = [];
    try { titles = await titlesBy(poet); } catch (e) { /* offline archive */ }

    let opening = null;
    if (titles.length) opening = await pickPoem(poet, titles, [], 60);
    // The archive is unreachable: fall back to the poem baked with the shelf.
    if (!opening && shelf) opening = shelf.opening;
    if (!opening) {
      return json({ error: "The poetry archive is unreachable right now." }, 503);
    }

    const gid = id();
    state[gid] = { agent: "poetry", poet: poet, titles: titles, read: [opening.title] };

    return json({
      route: { agent: "poetry", task: task, reason: "from the archive" },
      agent: "poetry",
      label: "The Poetry Shelf",
      usage: { total_tokens: 0 },
      ms: 0,
      result: { note: note, substituted: substituted },
      artifacts: {
        poetry: {
          id: gid,
          poet: poet,
          title: opening.title,
          lines: opening.lines,
          available: Math.max(1, titles.length),
        },
      },
    });
  }

  async function play(kind, gid, action, body) {
    const s = state[gid];
    if (!s) return json({ error: "That game has expired. Ask for a new one." }, 404);

    if (kind === "poetry" && action === "more") {
      const poem = await pickPoem(s.poet, s.titles, s.read, 60);
      if (!poem) return json({ done: true, error: "That's the whole shelf for " + s.poet + "." });
      s.read.push(poem.title);
      return json({
        title: poem.title,
        lines: poem.lines,
        remaining: Math.max(0, s.titles.length - s.read.length),
      });
    }

    const item = s.item;

    if (kind === "wordle" && action === "guess") {
      const guess = String(body.guess || "").toLowerCase();
      const words = await dictionary();
      if (!/^[a-z]{5}$/.test(guess)) return json({ error: "Guesses are five letters, a-z." }, 400);
      if (words.indexOf(guess) === -1) return json({ error: '"' + guess.toUpperCase() + "\" isn't in the word list." }, 400);

      const answer = item.answer.toLowerCase();
      let before = words;
      for (const h of s.history) before = before.filter((w) => patternCode(h.guess, w) === h.code);
      const code = patternCode(guess, answer);
      const after = before.filter((w) => patternCode(guess, w) === code);

      let best = null;
      if (before.length > 2) {
        const pool = before.slice(0, 1200).concat(await openers());
        let bw = "", bb = -1;
        for (const w of pool) {
          const b = expectedBits(w, before);
          if (b > bb) { bb = b; bw = w; }
        }
        best = { word: bw, bits: bb };
      }
      const info = {
        before: before.length, after: after.length,
        expected: expectedBits(guess, before),
        actual: after.length ? Math.log2(before.length / after.length) : 0,
        best: best,
      };
      s.history.push({ guess: guess, code: code });

      const solved = guess === answer;
      return json(Object.assign({ marks: markGuess(guess, answer), solved: solved, info: info },
        solved ? { fact: item.fact } : {}));
    }

    if (kind === "wordle" && action === "reveal") {
      return json({ answer: item.answer.toUpperCase(), fact: item.fact });
    }

    if (kind === "connections" && action === "guess") {
      const picked = (body.words || []).map((w) => String(w).toUpperCase());
      if (picked.length !== 4) return json({ error: "Pick exactly four." }, 400);
      for (const g of item.groups) {
        const overlap = g.words.filter((w) => picked.indexOf(w) !== -1).length;
        if (overlap === 4) return json({ correct: true, label: g.label, words: g.words, difficulty: g.difficulty });
        if (overlap === 3) return json({ correct: false, oneAway: true });
      }
      return json({ correct: false, oneAway: false });
    }
    if (kind === "connections" && action === "reveal") return json({ groups: item.groups });

    if (kind === "trivia" && action === "answer") {
      const q = item.questions[Number(body.index)];
      if (!q) return json({ error: "No such question." }, 400);
      return json({
        correct: Number(body.choice) === q.answerIndex,
        answerIndex: q.answerIndex,
        explanation: q.explanation,
        difficulty: q.b,
      });
    }

    if (kind === "alibi" && action === "press") {
      return json({ error: "Pressing a suspect needs the live agent — they answer as you ask." }, 400);
    }
    if (kind === "alibi" && action === "accuse") {
      const name = String(body.name || "");
      const a = item.suspects.filter((x) => x.name === item.clash[0])[0];
      const b = item.suspects.filter((x) => x.name === item.clash[1])[0];
      const say = (x) => x.place + (x.with === "alone" ? ", alone" : ", with " + x.with);
      return json({
        correct: name === item.culprit,
        culprit: item.culprit,
        clash: item.clash,
        explain: a.name + " said " + say(a) + ". " + b.name + " said " + say(b) + ".",
      });
    }

    return json({ error: "That needs the live agent." }, 400);
  }

  /* Intercept only our own API paths; everything else passes through. */
  const realFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.indexOf("/api/") !== 0) return realFetch(input, init);

    let body = {};
    try { if (init && init.body) body = JSON.parse(init.body); } catch (e) { /* no body */ }

    if (url === "/api/agents") {
      return json({
        publicMode: true,
        agents: [
          { id: "wordle", label: "Wordle Smith", blurb: "A themed five-letter puzzle, scored on the information each guess buys.", examples: ["a wordle"] },
          { id: "connections", label: "Connections Setter", blurb: "Sixteen words hiding four groups, with one-away feedback.", examples: ["a connections grid"] },
          { id: "trivia", label: "Quizmaster", blurb: "A quiz that estimates your ability rather than counting answers.", examples: ["a quiz"] },
          { id: "alibi", label: "The Alibi", blurb: "Four accounts of one hour. Exactly two cannot both be true.", examples: ["an alibi case"] },
          { id: "piano", label: "Piano Tutor", blurb: "A falling-note lesson on a sampled piano, for a piece out of copyright.", examples: ["teach me Für Elise"] },
          { id: "poetry", label: "The Poetry Shelf", blurb: "A reading from the public-domain archive — the real text, not a recollection of it.", examples: ["read me something by Blake"] },
        ],
      });
    }
    if (url === "/api/ask") return streamed(await ask(body.prompt), trace(body.prompt));
    if (url === "/api/again") return streamed(await serve(body.agent, body.task), trace(body.task));

    const m = url.match(/^\/api\/play\/(\w+)\/([\w-]+)\/(\w+)$/);
    if (m) return play(m[1], m[2], m[3], body);

    return json({ error: "Not available without a backend." }, 404);
  };
})();
