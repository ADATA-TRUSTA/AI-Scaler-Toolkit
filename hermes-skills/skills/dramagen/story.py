#!/usr/bin/env python
"""dramagen/story.py -- premise in, machine-readable short-drama template out.

Runs on the co-resident local LLM (TRUSTA-AST, or any OpenAI-compatible
/v1/chat/completions endpoint). The craft rules baked into the prompts are
condensed from the MIT-licensed `ai-short-drama-screenwriter` skill
(https://github.com/POUND0423/AI-drama-pound); the full text lives in
references/screenwriting.md.

The output is drama.json -- the one contract produce.py renders through the
imggen and vidgen skills. Everything the renderers need is decided HERE, while
a language model is in the loop, so produce.py never has to guess:

  * dialogue and action stay Traditional Chinese (they are the script)
  * image_prompt / motion_prompt are ENGLISH, because realvis and flux cannot
    read CJK and fail SILENTLY on it (a clean picture of the wrong thing)
  * every character carries an ascii `en_name`, so an English prompt can name
    them and produce.py can rewrite that name into H3's "Picture N"

Usage:
  story.py "<premise>" --episodes 1 --shots 5 --out <project dir>
  story.py "<premise>" --style cartoon --silent       # every shot -> silent Wan i2v
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def _load_conf():
    """Fold hermes-comfy.conf into os.environ, the way imggen and vidgen do.

    Without this, a machine that configures the backend in that file (the whole
    point of the file) is configuring it only for the two skills that import
    the engine -- this one would silently keep talking to localhost. Finding
    the engine is best-effort: dramagen does not otherwise need it, so a
    missing engine is not an error here, just no conf.
    """
    env = os.environ.get("HERMES_COMFY_ENGINE", "").strip()
    cands = [Path(env)] if env else []
    marker = Path(__file__).resolve().parent / "engine_dir.txt"
    if marker.is_file():
        try:
            cands.append(Path(os.path.expanduser(marker.read_text(encoding="utf-8").strip())))
        except OSError:
            pass
    # A linked install (install.py --link) runs this file straight from the pack,
    # whose engine/ is two levels up; nothing is written next to the script.
    cands.append(Path(__file__).resolve().parents[2] / "engine")
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        home = Path(os.environ.get("HERMES_HOME") or Path(local) / "hermes")
    else:
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    cands += [home / "comfy-engine"]
    for c in cands:
        if (c / "paths.py").is_file():
            try:
                sys.path.insert(0, str(c))
                import paths        # noqa: F401  -- import folds conf into environ
                return str(c)
            except Exception:
                return None
    return None


_load_conf()

# Where the local LLM answers. DRAMAGEN_LLM_URL wins; otherwise reuse the
# backend vidgen already talks to, so a new machine configures one address
# instead of two. Both may come from hermes-comfy.conf, loaded just above.
_AST = os.environ.get("VIDGEN_AST_URL", "").strip().rstrip("/")
DEF_URL = (os.environ.get("DRAMAGEN_LLM_URL", "").strip()
           or (_AST + "/v1" if _AST else "")
           or "http://127.0.0.1:8571/v1")
SCHEMA_VERSION = 1

# ---------------------------------------------------------------- LLM client


def resolve_model(base, explicit):
    """Ask the endpoint what it is serving rather than hardcoding a model id.

    The same skill has to work against any OpenAI-compatible server, and
    on this box the served id has already changed once (Qwen3.5-35B ->
    Nemotron-3.5-30B) without anything else moving.
    """
    if explicit:
        return explicit
    env = os.environ.get("DRAMAGEN_LLM_MODEL", "").strip()
    if env:
        return env
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=15) as f:
            data = json.load(f)
        return data["data"][0]["id"]
    except Exception as e:
        raise SystemExit(json.dumps({"status": "error", "error":
            "cannot reach the local LLM at %s/models (%s). Start the local "
            "inference service, or point DRAMAGEN_LLM_URL at one." % (base, e)},
            ensure_ascii=False))


class LLM:
    """Chat client tuned for a *reasoning* model behind a thin OpenAI shim.

    The one setting that matters here is `enable_thinking: false`. The local
    model (Nemotron 3.5 Lightning) thinks by default, the gateway in front of
    it drops `reasoning_content` and returns only `content`, and thinking
    tokens still count against max_tokens. Measured on this box, same request:

        thinking on   12.9 s, 700 completion tokens, finish_reason=length,
                      content = ""            <- the whole budget went to
                                                 reasoning nobody ever sees
        thinking off   2.2 s,  67 completion tokens, finish_reason=stop,
                      content = the JSON we asked for

    That is not a tuning preference, it is the difference between working and
    not: a story template is 1-3k tokens of JSON, so with thinking on the
    visible answer is truncated mid-object every time. `--think` restores the
    old behaviour for anyone who wants it and is willing to pay for headroom.
    """

    def __init__(self, base, model, max_tokens, temperature,
                 timeout=900, verbose=True, think=False):
        self.url = base.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens = model, max_tokens
        self.temperature, self.timeout, self.verbose = temperature, timeout, verbose
        self.think = think
        self.calls, self.tokens, self.wall = 0, 0, 0.0
        self.last_finish = None

    def ask(self, system, user, temperature=None):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if not self.think:
            # Forwarded verbatim into the Jinja chat template. Servers that do
            # not understand it ignore it, so this stays safe on endpoints
            # whose model has no thinking mode at all.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode("utf-8")   # payload is full of CJK
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as f:
            data = json.load(f)
        dt = time.time() - t0
        self.calls += 1
        self.wall += dt
        usage = data.get("usage") or {}
        self.tokens += int(usage.get("completion_tokens") or 0)
        choice = data["choices"][0]
        self.last_finish = choice.get("finish_reason")
        msg = choice["message"]
        # Some servers surface the hidden trace as reasoning_content; prefer
        # the real answer but do not return nothing if only that came back.
        out = (msg.get("content") or "").strip()
        if not out:
            out = (msg.get("reasoning_content") or "").strip()
        if self.verbose:
            print("[story] llm %.1fs, %s completion tokens, finish=%s"
                  % (dt, usage.get("completion_tokens"), choice.get("finish_reason")),
                  file=sys.stderr, flush=True)
        if not out and choice.get("finish_reason") == "length":
            raise RuntimeError(
                "the model spent its whole %d-token budget without producing "
                "visible output -- it is thinking, and this gateway discards "
                "the reasoning. Raise --max-tokens, or drop --think."
                % self.max_tokens)
        return out


# ------------------------------------------------------------ JSON salvage


def extract_json(text):
    """Pull the first complete JSON object out of a model reply.

    A 30B local model does not reliably honour "JSON only": it fences the
    block, prefixes a sentence, or appends a note. Salvaging is cheaper than a
    retry (a retry costs a full minute of generation here), so try hard before
    giving up.
    """
    if not text:
        raise ValueError("empty reply")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply: %s" % t[:200])
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = t[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    # Trailing commas are the single most common defect.
                    return json.loads(re.sub(r",(\s*[}\]])", r"\1", blob))
    raise ValueError("unterminated JSON object: %s" % t[start:start + 200])


def ask_json(llm, system, user, tries=3):
    """Ask for JSON, and on a parse failure show the model its own mistake.

    A retry costs a full generation, so the follow-up says *why* it failed --
    and when the reply was cut off at the token cap, asking again for the same
    thing would just get cut off again, so that retry asks for a shorter one.
    """
    last, hint = "", ""
    for n in range(tries):
        reply = llm.ask(system, user if n == 0 else
                        user + "\n\n上一次的輸出無法解析為 JSON,錯誤是:%s\n%s"
                               "請只重新輸出一個合法的 JSON 物件,不要任何解釋文字。"
                               % (last, hint))
        try:
            return extract_json(reply)
        except Exception as e:
            last = str(e)[:300]
            hint = ("這次的回覆在 %d token 上限被截斷,請把文字寫得更短。\n"
                    % llm.max_tokens) if llm.last_finish == "length" else ""
            print("[story] JSON parse failed (try %d/%d, finish=%s): %s"
                  % (n + 1, tries, llm.last_finish, last), file=sys.stderr, flush=True)
    raise RuntimeError("the local LLM never returned parseable JSON: %s" % last)


# ------------------------------------------------------------ text helpers

CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")


def has_cjk(s):
    return bool(CJK.search(s or ""))


def slug(s, fallback):
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or fallback


# ------------------------------------------------------------ craft prompts

# The drama's language is a user choice, not a property of this skill. Three
# things move together and must not drift apart: the language the model WRITES
# the script in, the code recorded in drama.json, and the phrase produce.py
# puts in the video prompt so the cast SPEAKS that language. Resolve all three
# from one flag.
LANGUAGES = {
    "zh-hant": ("繁體中文", "zh-Hant", "Mandarin Chinese with a Taiwanese accent"),
    "zh-tw":   ("繁體中文", "zh-Hant", "Mandarin Chinese with a Taiwanese accent"),
    "zh-hans": ("简体中文", "zh-Hans", "Mandarin Chinese"),
    "zh-cn":   ("简体中文", "zh-Hans", "Mandarin Chinese"),
    "zh":      ("中文", "zh", "Mandarin Chinese"),
    "ja":      ("日本語", "ja", "Japanese"),
    "japanese": ("日本語", "ja", "Japanese"),
    "ko":      ("한국어", "ko", "Korean"),
    "korean":  ("한국어", "ko", "Korean"),
    "en":      ("English", "en", "English"),
    "english": ("English", "en", "English"),
    "es":      ("Español", "es", "Spanish"),
    "spanish": ("Español", "es", "Spanish"),
    "th":      ("ภาษาไทย", "th", "Thai"),
    "yue":     ("廣東話", "yue", "Cantonese"),
    "cantonese": ("廣東話", "yue", "Cantonese"),
}


def resolve_language(arg):
    """(what to write in, what to record, how to say it to a video model).

    An unknown value is not an error -- it is passed straight through, so
    `--language "Brazilian Portuguese"` works without this table growing to
    cover every language on earth.
    """
    key = (arg or "zh-Hant").strip()
    hit = LANGUAGES.get(key.lower())
    return hit if hit else (key, key, key)


CRAFT = """你是短劇編劇。規則(不可違反):
- 劇本文字(action、dialogue、標題)一律使用 %(LANG)s;每場要能被看見或聽見,不用內心獨白代替戲劇事件。
- 每場至少改變資訊、關係、目標、風險或情緒其中一項。
- 場景標頭格式固定為「N. 內景／外景・地點・時間」,例如「1. 內景・便利商店・夜」。
- 反轉必須有前文線索;集尾鉤子要留下一個問題或代價,不是單純中斷。
- 臺詞短、有目的,避免解說背景。
- 輸出只有一個 JSON 物件,不要 markdown 圍欄、不要任何說明文字。"""

VISUAL = """畫面提示詞規則(這關係到生圖生影片會不會生錯,務必遵守):
- image_prompt 與 motion_prompt 一律用**英文**,不可出現任何中文字。
  (出圖模型 RealVisXL / FLUX 的文字編碼器讀不懂中文,送中文進去不會報錯,
   而是安靜地生出一張完全無關的圖。)
- image_prompt 描述**一張靜止畫面**:誰、在哪、什麼光線、鏡位、氣氛。不要寫動作過程。
- motion_prompt 描述**這一鏡怎麼動**:運鏡與人物動作,一到兩句,不要重述整段劇情。
- 角色在英文提示詞裡一律用它的 en_name 稱呼,不要用劇本裡的本名。"""


def stage_brief(llm, args):
    lang = args.script_language
    schema = {
        "title": "劇名(%s)" % lang,
        "genre": "類型(%s)" % lang,
        "logline": "一句話故事(%s)" % lang,
        "audience": "目標觀眾(%s)" % lang,
        "style": "english visual style keywords for the whole drama",
        "characters": [{
            "id": "ascii_snake_case_id",
            "name": "角色名(用該語言寫)",
            "en_name": "這個角色的英文人名,例如 A-Qiang 或 Uncle Chen;不可以跟 id 一樣",
            "brief": "身分/目標/阻力(%s,一句)" % lang,
            "appearance": "English head-and-shoulders description: age, ethnicity, "
                          "hair, clothing, distinguishing feature",
        }],
        "beats": ["第 1 集的節拍(%s)" % lang],
    }
    user = (
        "故事前提:%s\n"
        "集數:%d,每集約 %d 個鏡頭,每鏡約 %s 秒。\n"
        "整體視覺風格:%s\n\n"
        "%s\n\n"
        "請產出創作簡報與角色設定,嚴格照這個 JSON 結構(欄位名稱不可改):\n%s\n\n"
        "限制:角色 2 到 %d 人;beats 給 %d 條,每集一條;"
        "characters[].id 只能用小寫英文與底線。"
        % (args.premise, args.episodes, args.shots, args.seconds, args.style,
           VISUAL, json.dumps(schema, ensure_ascii=False, indent=2),
           args.max_characters, args.episodes)
    )
    return ask_json(llm, CRAFT % {"LANG": lang}, user)


DIALOGUE_OPTIONAL = ("每鏡最多 2 句臺詞,沒有臺詞就給空陣列(空鏡、物件特寫本來就不該有人說話);")
DIALOGUE_EVERY = ("**每一鏡都必須有臺詞**,1 到 2 句,dialogue 不可以是空陣列;"
                  "臺詞的說話者一定要出現在該鏡的 characters[] 裡;"
                  "所以不要寫純物件特寫或空景 —— 每一鏡都要有人在場說話;")


def normalise_characters(llm, brief):
    """Clean the cast BEFORE the shots are written, not after.

    The brief comes back from a language model, so any field can be missing --
    and the shot stage reads name/en_name to tell the model who it may cast.
    Doing the cleanup at the end instead cost a whole run to a KeyError on the
    one brief that omitted `name`.
    """
    chars, seen = [], set()
    for i, c in enumerate(brief.get("characters") or []):
        if not isinstance(c, dict):
            continue
        cid = slug(str(c.get("id") or ""), "char%d" % (i + 1))
        while cid in seen:
            cid += "_x"
        seen.add(cid)
        # en_name is what an English prompt calls this person, so it has to
        # read like a name. The model happily echoes the snake_case id back
        # ("delivery_boy"), which then lands verbatim in an image prompt.
        en = str(c.get("en_name") or "").strip()
        if not en or has_cjk(en) or en.lower() == cid:
            en = cid.replace("_", " ").title()
        elif "_" in en:
            en = en.replace("_", " ").title()
        appearance = str(c.get("appearance") or "").strip()
        # The model sometimes copies the schema's instruction into the value
        # ("English head-and-shoulders description: early 20s, ..."), which
        # then rides along into every image prompt this character appears in.
        appearance = re.sub(r"^\s*English[^:]{0,60}description\s*:\s*", "",
                            appearance, flags=re.I)
        if has_cjk(appearance):
            appearance = translate_prompt(llm, appearance, "character appearance")
        chars.append({"id": cid, "name": str(c.get("name") or en).strip(),
                      "en_name": en, "brief": str(c.get("brief") or "").strip(),
                      "appearance": appearance})
    return chars


def stage_shots(llm, args, brief, chars, ep_index, beat):
    lang = args.script_language
    cast = [{"id": c["id"], "name": c["name"], "en_name": c["en_name"],
             "brief": c["brief"]} for c in chars]
    schema = {
        "episode_title": "本集集名(%s)" % lang,
        "shots": [{
            "heading": "1. 內景・地點・夜",
            "action": "現在正在發生的可見動作(%s,1-2 句)" % lang,
            "dialogue": [{"character": "角色名(%s)" % lang, "line": "臺詞(%s)" % lang}],
            "characters": ["character id"],
            "image_prompt": "English still-frame description of this shot",
            "motion_prompt": "English camera move and character motion",
            "seconds": args.seconds,
        }],
        "hook": "集尾鉤子(%s,一句)" % lang,
    }
    user = (
        "劇名:%s\n類型:%s\n一句話故事:%s\n整體視覺風格:%s\n"
        "角色:%s\n\n本集(第 %d 集)要完成的節拍:%s\n\n%s\n\n"
        "請寫出本集的分鏡,嚴格照這個 JSON 結構(欄位名稱不可改):\n%s\n\n"
        "限制:剛好 %d 個鏡頭;heading 從 1 開始連號;%s"
        "characters[] 只填**這一鏡畫面裡真的看得見的人**,"
        "不要把整集的角色都列上去 —— 這個欄位會決定出圖時畫面裡出現誰。"
        % (brief.get("title", ""), brief.get("genre", ""), brief.get("logline", ""),
           brief.get("style", args.style), json.dumps(cast, ensure_ascii=False),
           ep_index, beat, VISUAL,
           json.dumps(schema, ensure_ascii=False, indent=2), args.shots,
           DIALOGUE_EVERY if args.all_dialogue else DIALOGUE_OPTIONAL)
    )
    return ask_json(llm, CRAFT % {"LANG": lang}, user)


def translate_prompt(llm, zh, kind):
    """Last resort when the model wrote a visual prompt in Chinese.

    Cheaper and far more accurate than letting produce.py fall back to a
    CJK-capable but slow image model for that one shot.
    """
    out = llm.ask("You translate film shot descriptions into English. "
                  "Reply with the English translation only, one line, no quotes.",
                  "Translate this %s into English:\n%s" % (kind, zh), temperature=0.0)
    return out.strip().strip('"').splitlines()[0] if out else zh


# ------------------------------------------------------------ normalisation


def normalise(llm, args, brief, chars, episodes):
    valid = set(c["id"] for c in chars)
    by_name = dict((c["name"], c["id"]) for c in chars)

    eps, n = [], 0
    for ei, ep in enumerate(episodes, 1):
        shots = []
        for si, sh in enumerate(ep.get("shots") or [], 1):
            n += 1
            ip = str(sh.get("image_prompt") or "").strip()
            mp = str(sh.get("motion_prompt") or "").strip()
            if has_cjk(ip):
                ip = translate_prompt(llm, ip, "still frame description")
            if has_cjk(mp):
                mp = translate_prompt(llm, mp, "camera and motion description")
            ids = [c for c in (sh.get("characters") or []) if c in valid]
            dlg = []
            for d in (sh.get("dialogue") or []):
                if not isinstance(d, dict):
                    continue
                line = str(d.get("line") or "").strip()
                if not line:
                    continue
                who = str(d.get("character") or "").strip()
                dlg.append({"character": who, "line": line})
                if who in by_name and by_name[who] not in ids:
                    ids.append(by_name[who])
            # r2v (MiniMax-H3) is the default wherever it is possible at all,
            # i.e. wherever the shot has a character to reference. Measured on
            # this box at 480p portrait: H3 renders 5 s in 255 s against Wan's
            # 288 s, adds native audio, needs no per-shot keyframe, and -- the
            # part that actually matters -- holds the face across independent
            # clips, which independently generated keyframes never will.
            # H3 cannot run without a reference, so an empty frame (an object
            # close-up, a landscape) still goes to Wan i2v.
            mode = "i2v" if (args.silent or not ids) else "r2v"
            try:
                secs = float(sh.get("seconds") or args.seconds)
            except (TypeError, ValueError):
                secs = float(args.seconds)
            shots.append({
                "id": "e%02ds%02d" % (ei, si),
                "heading": str(sh.get("heading") or "%d." % si).strip(),
                "action": str(sh.get("action") or "").strip(),
                "dialogue": dlg,
                "characters": ids,
                "image_prompt": ip or "cinematic still frame",
                "motion_prompt": mp or "slow push in, subtle movement",
                "seconds": max(1.0, min(secs, 20.0)),
                "mode": mode,
            })
        eps.append({"index": ei,
                    "title": str(ep.get("episode_title") or "").strip(),
                    "hook": str(ep.get("hook") or "").strip(),
                    "shots": shots})
    return {
        "schema": SCHEMA_VERSION,
        "title": str(brief.get("title") or "未命名短劇").strip(),
        "genre": str(brief.get("genre") or "").strip(),
        "logline": str(brief.get("logline") or "").strip(),
        "audience": str(brief.get("audience") or "").strip(),
        "premise": args.premise,
        "style": str(brief.get("style") or args.style).strip(),
        "aspect": "portrait" if args.portrait else "landscape",
        # `language` records what the script is in; `speech_language` is the
        # phrase produce.py hands the video model so the cast speaks it. Kept
        # separate because a code is for records and a prompt needs prose.
        "language": args.language_code,
        "speech_language": args.speech_language,
        "characters": chars,
        "episodes": eps,
        "shot_count": n,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": {"model": llm.model, "url": llm.url},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate a short-drama template with the local LLM.")
    ap.add_argument("premise", help="story premise, in any language")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--shots", type=int, default=5, help="shots per episode")
    ap.add_argument("--seconds", type=float, default=4.0, help="seconds per shot")
    ap.add_argument("--style",
                    default="cinematic realistic, shallow depth of field, moody lighting",
                    help="English visual style for the whole drama")
    ap.add_argument("--language", default="zh-Hant",
                    help="language the script is written and spoken in. A code "
                         "(zh-Hant, ja, en, ko, yue, es, th) or any plain name "
                         "(\"Brazilian Portuguese\"). Default: zh-Hant.")
    ap.add_argument("--portrait", action="store_true", help="vertical drama (9:16)")
    ap.add_argument("--silent", action="store_true",
                    help="render every shot with Wan i2v instead: no audio, and "
                         "the characters' faces drift between shots. Only worth it "
                         "for mood reels with no recurring cast.")
    ap.add_argument("--all-dialogue", action="store_true", dest="all_dialogue",
                    help="every shot must carry dialogue -- no silent object "
                         "close-ups. Every shot then also has a speaker, so every "
                         "shot goes to r2v and is spoken aloud.")
    ap.add_argument("--max-characters", type=int, default=3, dest="max_characters")
    ap.add_argument("--out", help="write drama.json here (default: stdout only)")
    ap.add_argument("--llm-url", default=DEF_URL, dest="llm_url")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=0, dest="max_tokens",
                    help="0 = size it from --shots (a shot is ~350 tokens of JSON, "
                         "and a reply truncated at the cap is a wasted minute)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first (6x slower here, and the "
                         "gateway throws the reasoning away -- raise --max-tokens too)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        a.script_language, a.language_code, a.speech_language = resolve_language(a.language)
        if not a.max_tokens:
            a.max_tokens = max(4096, 400 * a.shots + 1500)
        model = resolve_model(a.llm_url, a.model)
        llm = LLM(a.llm_url, model, a.max_tokens, a.temperature,
                  verbose=not a.quiet, think=a.think)
        brief = stage_brief(llm, a)
        chars = normalise_characters(llm, brief)
        if not chars:
            raise RuntimeError("the model returned no usable characters")
        beats = brief.get("beats") or []
        episodes = []
        for i in range(1, a.episodes + 1):
            beat = beats[i - 1] if i - 1 < len(beats) else "延續前一集,推進主線並留下鉤子"
            episodes.append(stage_shots(llm, a, brief, chars, i, beat))
        drama = normalise(llm, a, brief, chars, episodes)
        out = None
        if a.out:
            p = Path(a.out)
            if p.is_dir() or not p.suffix:
                p.mkdir(parents=True, exist_ok=True)
                p = p / "drama.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(drama, ensure_ascii=False, indent=2), encoding="utf-8")
            out = str(p.resolve())
        print(json.dumps({"status": "ok", "file": out, "title": drama["title"],
                          "shots": drama["shot_count"],
                          "llm_calls": llm.calls, "llm_seconds": round(llm.wall, 1),
                          "drama": None if out else drama}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "error": "%s: %s" % (type(e).__name__, e)},
                         ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
