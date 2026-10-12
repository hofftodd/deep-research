"use strict";

const ACTIVE = ["queued", "planning", "researching", "writing"];
const POLL_MS = 2500;

const state = {
  health: null,
  topics: [],
  currentId: null,
  current: null,
  lastEventId: 0,
  tab: null,
  pollTimer: null,
};

const $ = (sel, root = document) => root.querySelector(sel);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function fmtTime(iso) {
  return new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function duration(a, b) {
  if (!a) return "";
  const secs = Math.max(0, Math.round(((b ? new Date(b) : new Date()) - new Date(a)) / 1000));
  const m = Math.floor(secs / 60), s = secs % 60;
  return m ? `${m}m ${s}s` : `${s}s`;
}

/* ---------- health ---------- */
async function loadHealth() {
  const box = $("#health");
  try {
    state.health = await api("/api/health");
    const h = state.health;
    box.classList.toggle("ok", h.llm_ok);
    box.classList.toggle("bad", !h.llm_ok);
    $(".health-text", box).textContent = h.llm_ok
      ? `${h.model} · search: ${h.search_provider}`
      : `Model server unreachable (${h.llm_base_url})`;
    box.title = h.llm_ok ? `Connected to ${h.llm_base_url}` : (h.llm_error || "");
  } catch (e) {
    box.classList.add("bad");
    $(".health-text", box).textContent = "Server unreachable";
  }
}

/* ---------- topic list ---------- */
async function loadTopics() {
  try {
    state.topics = await api("/api/topics");
  } catch (_) { return; }
  const list = $("#topic-list");
  list.replaceChildren();
  $("#topic-empty").classList.toggle("hidden", state.topics.length > 0);
  for (const t of state.topics) {
    const meta = el("div", { class: "t-meta" }, el("span", { class: `badge ${t.status}`, text: t.status }));
    if (ACTIVE.includes(t.status) && t.status !== "queued") {
      const bar = el("div", { class: "mini-progress" }, el("div"));
      bar.firstChild.style.width = `${t.progress}%`;
      meta.append(bar);
    } else {
      meta.append(el("span", { text: fmtDate(t.created_at) }));
    }
    const a = el("a", { href: `#/topic/${t.id}` }, el("div", { class: "t-title", text: t.title }), meta);
    if (t.id === state.currentId) a.classList.add("active");
    list.append(el("li", {}, a));
  }
}

/* ---------- new topic ---------- */
function showNew() {
  state.currentId = null;
  state.current = null;
  const main = $("#main");
  main.replaceChildren($("#tpl-new").content.cloneNode(true));
  const form = $("#new-form");
  const h = state.health;
  const select = form.elements.model;
  const models = h && h.models && h.models.length ? h.models : [h ? h.model : ""];
  if (h && !models.includes(h.model)) models.unshift(h.model);
  for (const m of models) select.append(el("option", { value: m, text: m }));
  if (h) {
    select.value = h.model;
    for (const [k, v] of Object.entries(h.defaults || {})) {
      if (form.elements[k]) form.elements[k].placeholder = v;
    }
  }
  form.elements.title.focus();
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const body = { title: fd.get("title"), description: fd.get("description"), model: fd.get("model") || "" };
    for (const k of ["max_iterations", "max_questions", "max_sources", "sources_per_question"]) {
      if (fd.get(k)) body[k] = Number(fd.get(k));
    }
    const btn = form.querySelector("button[type=submit]");
    btn.disabled = true;
    try {
      const topic = await api("/api/topics", { method: "POST", body: JSON.stringify(body) });
      location.hash = `#/topic/${topic.id}`;
    } catch (e) {
      const err = $("#new-error");
      err.textContent = e.message;
      err.classList.remove("hidden");
      btn.disabled = false;
    }
  });
  loadTopics();
}

/* ---------- topic detail ---------- */
function showTopic(id) {
  state.currentId = id;
  state.current = null;
  state.lastEventId = 0;
  state.tab = null;
  const main = $("#main");
  main.replaceChildren($("#tpl-topic").content.cloneNode(true));
  for (const b of main.querySelectorAll(".tabs button")) {
    b.addEventListener("click", () => setTab(b.dataset.tab));
  }
  $(".act-cancel").addEventListener("click", () => topicAction("cancel"));
  $(".act-retry").addEventListener("click", () => topicAction("retry"));
  $(".act-delete").addEventListener("click", async () => {
    if (!confirm("Delete this topic and its report?")) return;
    await api(`/api/topics/${id}`, { method: "DELETE" });
    location.hash = "#/";
  });
  refreshTopic(true);
  loadTopics();
}

function setTab(name) {
  state.tab = name;
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("active", b.dataset.tab === name);
  for (const d of document.querySelectorAll(".tab-body")) d.classList.toggle("active", d.dataset.body === name);
}

async function topicAction(action) {
  try {
    await api(`/api/topics/${state.currentId}/${action}`, { method: "POST" });
    if (action === "retry") {
      state.lastEventId = 0;
      $(".events").replaceChildren();
      setTab("activity");
    }
    await refreshTopic();
    loadTopics();
  } catch (e) {
    alert(e.message);
  }
}

async function refreshTopic(first = false) {
  const id = state.currentId;
  if (id == null) return;
  let t;
  try {
    t = await api(`/api/topics/${id}`);
  } catch (e) {
    $("#main").replaceChildren(el("section", { class: "panel narrow" },
      el("h1", { text: "Topic not found" }), el("p", { class: "muted", text: e.message })));
    state.currentId = null;
    return;
  }
  if (id !== state.currentId) return;
  const prevStatus = state.current && state.current.status;
  state.current = t;
  renderTopic(t);
  await loadEvents();
  if (first || (prevStatus && prevStatus !== t.status && t.status === "completed")) {
    setTab(t.status === "completed" ? "report" : (state.tab || "activity"));
  } else if (!state.tab) {
    setTab("activity");
  }
}

function renderTopic(t) {
  document.title = `${t.title} · Deep Research`;
  $(".topic-title").textContent = t.title;
  $(".topic-desc").textContent = t.description || "";
  const active = ACTIVE.includes(t.status);

  const badge = $(".status-card .badge");
  badge.className = `badge ${t.status}`;
  badge.textContent = t.status;
  $(".status-card").className = `status-card ${t.status}`;
  $(".phase").textContent = t.phase || (t.status === "queued" ? "Waiting for the worker…" : "");
  $(".progress .bar").style.width = `${t.status === "completed" ? 100 : t.progress}%`;
  const used = t.sources.filter((s) => s.status === "used").length;
  const stats = [
    `Model: ${t.model}`,
    `${t.questions.length} question(s)`,
    `${used} source(s) used of ${t.sources.length} examined`,
    `${t.note_count} note(s)`,
    `Created ${fmtDate(t.created_at)}`,
  ];
  if (t.started_at) stats.push(`Elapsed ${duration(t.started_at, active ? null : t.completed_at || t.updated_at)}`);
  $(".stats").textContent = stats.join(" · ");
  const err = $(".topic-error");
  err.textContent = t.error || "";
  err.classList.toggle("hidden", !t.error);

  $(".act-cancel").classList.toggle("hidden", !active);
  $(".act-retry").classList.toggle("hidden", active);
  $(".act-retry").textContent = t.status === "completed" ? "Re-run" : "Retry";
  const dl = $(".act-download");
  dl.classList.toggle("hidden", !t.report_md);
  dl.href = `/api/topics/${t.id}/report.md`;

  // Report
  const report = $(".report");
  if (t.report_html) {
    if (report.dataset.rendered !== t.updated_at) {
      report.innerHTML = t.report_html; // server-rendered; model text is escaped server-side
      report.dataset.rendered = t.updated_at;
    }
  } else {
    report.dataset.rendered = "";
    const msg = active ? "The report will appear here when research is complete."
      : t.status === "failed" ? "Research failed before a report was written. See Activity for details."
      : "No report was produced.";
    report.replaceChildren(el("p", { class: "placeholder", text: msg }));
  }

  // Plan
  const qs = $(".questions");
  qs.replaceChildren();
  if (!t.questions.length) qs.append(el("p", { class: "placeholder", text: "The plan has not been created yet." }));
  for (const q of t.questions) {
    const label = q.status === "answered" ? "completed" : q.status === "unanswered" ? "failed" : "queued";
    const meta = el("div", { class: "q-meta" },
      el("span", { class: `badge ${label}`, text: q.status }),
      `round ${q.iteration} · ${q.note_count} note(s)` + (q.queries.length ? ` · queries: ${q.queries.join(" | ")}` : ""));
    qs.append(el("li", {}, el("div", { text: q.text }), meta));
  }

  // Sources
  const tbody = $(".sources tbody");
  tbody.replaceChildren();
  if (!t.sources.length) {
    tbody.append(el("tr", {}, el("td", { colspan: "3", class: "placeholder", text: "No sources examined yet." })));
  }
  for (const s of t.sources) {
    const link = el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer", text: s.title || s.url });
    const label = { used: "completed", failed: "failed", irrelevant: "cancelled" }[s.status] || "queued";
    const status = el("td", {}, el("span", { class: `badge ${label}`, text: s.status }));
    if (s.error) status.append(el("div", { class: "small muted", text: s.error }));
    tbody.append(el("tr", {},
      el("td", { class: "src" }, link, el("div", { class: "url", text: s.url })),
      status,
      el("td", { class: "num", text: String(s.note_count) })));
  }
}

async function loadEvents() {
  const id = state.currentId;
  let events;
  try {
    events = await api(`/api/topics/${id}/events?after=${state.lastEventId}`);
  } catch (_) { return; }
  if (id !== state.currentId) return;
  const list = $(".events");
  const nearBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 40;
  for (const e of events) {
    list.append(el("li", { class: e.level },
      el("time", { datetime: e.ts, text: fmtTime(e.ts) }),
      el("span", { class: "msg", text: e.message })));
    state.lastEventId = e.id;
  }
  if (events.length && nearBottom && state.tab === "activity") window.scrollTo(0, document.body.scrollHeight);
}

/* ---------- routing & polling ---------- */
function route() {
  const m = location.hash.match(/^#\/topic\/(\d+)/);
  if (m) showTopic(Number(m[1]));
  else showNew();
}

function poll() {
  const anyActive = state.topics.some((t) => ACTIVE.includes(t.status));
  if (anyActive) loadTopics();
  if (state.current && ACTIVE.includes(state.current.status)) refreshTopic();
}

window.addEventListener("hashchange", route);
(async function init() {
  await loadHealth();
  await loadTopics();
  route();
  state.pollTimer = setInterval(poll, POLL_MS);
  setInterval(loadHealth, 30000);
})();
