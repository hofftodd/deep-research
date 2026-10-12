"""Deterministic stand-ins for the model server, search engine and web."""

from __future__ import annotations

import re

from deep_research.search import FetchError, Page, SearchResult

PAGES = {
    "https://example.org/battery-basics": (
        "Battery basics",
        "Sodium-ion batteries use sodium ions as charge carriers instead of lithium ions. "
        "Sodium is roughly a thousand times more abundant than lithium in the Earth's crust. "
        "The first commercial sodium-ion cells shipped in 2023 from several Chinese manufacturers. " * 3,
    ),
    "https://example.com/grid-storage": (
        "Grid storage economics",
        "Analysts estimate sodium-ion cell costs could fall 20 percent below LFP cells by 2030. "
        "Energy density of current sodium-ion cells is between 140 and 160 Wh/kg. "
        "Grid storage projects value low cost and cycle life over energy density. " * 3,
    ),
    "https://news.example.net/sodium": (
        "Sodium-ion news",
        "A 100 MWh sodium-ion storage station was connected to the grid in Hubei in 2024. "
        "Operators reported stable performance in cold weather conditions. " * 4,
    ),
}


class FakeSearcher:
    def __init__(self, *_):
        self.queries = []

    def search(self, query, limit=None):
        self.queries.append(query)
        return [SearchResult(url, title, text[:80]) for url, (title, text) in PAGES.items()] + [
            SearchResult("https://broken.example/404", "Broken", "nothing here"),
        ]


class FakeFetcher:
    def __init__(self, *_):
        pass

    def fetch(self, url):
        if url not in PAGES:
            raise FetchError("HTTP 404")
        title, text = PAGES[url]
        return Page(url, title, text)


class FakeLLM:
    """Answers based on which prompt it receives, the way the agent would use a real model."""

    def __init__(self, model="fake-model", cite_bogus=False, uncited_paragraph=False,
                 unsupported=(), wrong_source=None, outline=True):
        self.model = model
        self.cite_bogus = cite_bogus
        self.uncited_paragraph = uncited_paragraph
        self.unsupported = unsupported      # statements containing these words fail verification
        self.wrong_source = wrong_source    # (word, source_id): statements with word belong to source_id
        self.outline = outline              # False: return an unusable outline
        self.calls = 0
        self.modes = []
        self.prompts = []
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.warnings = []

    def pop_warnings(self):
        out, self.warnings = self.warnings, []
        return out

    def with_model(self, model):
        self.model = model
        return self

    def list_models(self):
        return ["fake-model", "other-model"]

    def chat_json(self, system, user, mode="fast", retries=2):
        self.calls += 1
        self.stats["calls"] += 1
        self.modes.append(mode)
        self.prompts.append(user)
        if "research questions that together" in user:
            return {"questions": ["What are sodium-ion batteries?", "How do they perform for grid storage?"]}
        if "web search queries" in user:
            return {"queries": ["sodium ion battery", "sodium ion grid storage"]}
        if "Rank the results" in user:
            return {"ranking": [2, 0, 1]}
        if "Extract the facts" in user:
            text = user.split("---BEGIN SOURCE TEXT---\n")[1].split("\n---END SOURCE TEXT---")[0]
            first = re.split(r"(?<=\.) ", text)[0]
            return {"notes": [
                {"claim": f"Source states: {first}", "quote": first},
                {"claim": "A fabricated claim", "quote": "this sentence does not appear anywhere in the page"},
            ]}
        if "planning the structure of a research report" in user:
            if not self.outline:
                return {"sections": [{"heading": "Nothing", "notes": ["N9999"]}]}
            ids = re.findall(r"^N(\d+) ", user, re.M)
            half = max(1, len(ids) // 2)
            return {"sections": [
                {"heading": "## How sodium-ion cells work", "notes": [f"N{i}" for i in ids[:half]]},
                {"heading": "Costs and deployment", "notes": [int(i) for i in ids[half:]] + ["N424242"]},
            ]}
        if "Check each numbered statement" in user:
            results = []
            for num, statement in re.findall(r"^(\d+)\. (.*)$", user, re.M):
                r = {"id": int(num), "verdict": "supported", "reason": ""}
                if any(w in statement for w in self.unsupported):
                    r.update(verdict="unsupported", reason="not in evidence")
                elif self.wrong_source and self.wrong_source[0] in statement:
                    r.update(verdict="wrong_source", sources=[f"S{self.wrong_source[1]}"])
                results.append(r)
            return {"results": results}
        if "Assess whether this is enough" in user:
            return {"sufficient": True, "reasoning": "Both questions are covered.", "follow_up_questions": []}
        raise AssertionError(f"unexpected JSON prompt: {user[:200]}")

    def chat(self, messages, json_mode=False, mode="fast"):
        self.calls += 1
        self.stats["calls"] += 1
        self.modes.append(mode)
        user = messages[-1]["content"]
        self.prompts.append(user)
        # Only cite ids from the notes / report body, not the "[S4]" example in the instructions.
        material = user.split("---")[1] if "executive summary" in user else user
        ids = re.findall(r"^\[S(\d+)\]", material, re.M) or re.findall(r"\[S(\d+)", material)
        a = ids[0] if ids else "1"
        b = ids[-1] if ids else "1"
        if "executive summary" in user:
            text = f"Sodium-ion batteries are emerging for grid storage [S{a}]. Costs may undercut LFP [S{b}]."
            if self.cite_bogus:
                text += " A trailing claim with a made-up source [S999]."
            return text
        if "one section of a research report" in user:
            text = f"## Findings\n\nSodium is abundant and cheap [S{a}].\n\nGrid projects favour cost [S{b}, S{a}]."
            if self.cite_bogus:
                text += "\n\nAn invented fact [S999]."
            if self.uncited_paragraph:
                text += "\n\nThis paragraph has no citation at all."
            return text
        raise AssertionError(f"unexpected prompt: {user[:200]}")
