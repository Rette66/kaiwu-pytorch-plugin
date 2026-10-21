"""Review GitHub evidence with DeepSeek without executing contributor code."""

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


MARKER = "<!-- kaiwu-ai-review -->"
VERDICTS = {
    "recommend": "值得继续合并审查", "changes": "需要修改",
    "needs_evidence": "需要补证据", "duplicate": "重复或已经解决",
    "out_of_scope": "需要维护者决定是否接纳",
}
MAX_FILES = 8
MAX_INPUT = 90000


class ReviewError(ValueError):
    """A fixed diagnostic that is safe to show without provider data or credentials."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward authorization headers through an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Leave redirects to the caller as HTTP errors."""
        return None


def request_json(url, token, method="GET", payload=None, *, timeout=180):
    """Use fixed API hosts and bounded responses; never execute returned data."""
    headers = {"Accept": "application/json", "Content-Type": "application/json",
               "User-Agent": "kaiwu-ai-review"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        content = response.read(5_000_001)
    if len(content) > 5_000_000:
        raise ReviewError("API 回答超过读取大小限制")
    return json.loads(content)


def github(path, token, method="GET", payload=None):
    """Address GitHub paths, never URLs supplied by an issue or a model."""
    return request_json(f"https://api.github.com{path}", token, method, payload)


def item_version(item):
    """Identify relevant text and both PR revisions, excluding comment activity."""
    fields = {name: item.get(name) for name in ("title", "body", "state")}
    for name in ("head", "base"):
        fields[name] = item.get(name, {}).get("sha")
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def resolve_target(event_name, event, repository, token):
    """Resolve manual, issue and canonical CI events without trusting fork metadata."""
    prefix = f"/repos/{repository}"
    if event_name == "workflow_dispatch":
        number = int(event["inputs"]["number"])
        if number < 1:
            raise ValueError("Target number must be positive")
        return number, {}
    if event_name == "issues":
        trusted = event["issue"].get("author_association") in {"OWNER", "MEMBER", "COLLABORATOR"}
        number = (event["issue"]["number"]
                  if trusted and event["issue"]["user"]["type"] != "Bot" else None)
        return number, {}
    if event_name != "workflow_run":
        return None, {}
    run = github(f"{prefix}/actions/runs/{int(event['workflow_run']['id'])}", token)
    if run["event"] != "pull_request" or run["path"] != ".github/workflows/ci.yml":
        return None, {}
    pulls = run["pull_requests"]
    if not pulls:
        pulls = github(f"{prefix}/commits/{run['head_sha']}/pulls", token)
    for pull in pulls:
        number = pull["number"]
        current = github(f"{prefix}/pulls/{number}", token)
        if (current["state"] == "open" and not current["draft"]
                and current["head"]["sha"] == run["head_sha"]
                and current["base"]["repo"]["full_name"] == repository):
            return number, {key: run.get(key) for key in
                            ("id", "head_sha", "status", "conclusion", "html_url")}
    return None, {}


def read_source(repository, source, token, incomplete):
    """Read a pinned GitHub blob; unavailable and shortened files stay visible."""
    filename, revision = source["filename"], source["ref"]
    path = urllib.parse.quote(filename, safe="/")
    ref = urllib.parse.quote(revision, safe="")
    try:
        blob = github(f"/repos/{repository}/contents/{path}?ref={ref}", token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        if not source.get("absent_ok"):
            incomplete.append(f"{filename}: source not found")
        return ""
    if blob.get("encoding") != "base64":
        incomplete.append(f"{filename}: source unavailable")
        return ""
    content = base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
    if len(content) > 5000:
        incomplete.append(f"{filename}@{revision[:12]}: source excerpt only")
        if source.get("patch"):
            ranges = re.findall(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@",
                                source["patch"], re.MULTILINE)
            indices = set()
            for start in ranges:
                line = int(start[source.get("side", 1)]) - 1
                indices.update(range(max(0, line - 15), line + 35))
            if indices:
                content = "\n".join(text for index, text in enumerate(content.splitlines())
                                    if index in indices)
    return content[:5000]


def collect_packet(repository, number, token, ci_result):
    """Collect code, base revisions and related issues as untrusted evidence."""
    prefix = f"/repos/{repository}"
    issue = github(f"{prefix}/issues/{number}", token)
    is_pull = "pull_request" in issue
    item = github(f"{prefix}/pulls/{number}", token) if is_pull else issue
    packet = {"repository": repository, "number": number,
              "kind": "pr" if is_pull else "issue", "version": item_version(item),
              "item": {key: item.get(key) for key in ("title", "body", "state")},
              "incomplete": [], "files": [], "ci": ci_result, "related": []}
    info = github(prefix, token)
    revision = github(f"{prefix}/commits/{info['default_branch']}", token)["sha"]
    packet["default_sha"] = revision
    if is_pull:
        packet["head_sha"] = item["head"]["sha"]
        packet["base_sha"] = item["base"]["sha"]
        changed = github(f"{prefix}/pulls/{number}/files?per_page=100", token)
        changed.sort(key=lambda entry: (not entry["filename"].endswith(".py"),
                                       entry["filename"].startswith("tests/")))
        packet["changed_files"] = [entry["filename"] for entry in changed]
        if item["changed_files"] > MAX_FILES:
            packet["incomplete"].append(
                f"Only {MAX_FILES} of {item['changed_files']} changed files included")
        for entry in changed[:MAX_FILES]:
            filename = entry["filename"]
            patch_text = entry.get("patch", "")
            if not patch_text or len(patch_text) > 5000:
                packet["incomplete"].append(f"{filename}: patch missing or shortened")
            packet["files"].append({
                "filename": filename, "patch": patch_text[:5000],
                "base": read_source(repository, {
                    "filename": entry.get("previous_filename", filename),
                    "ref": item["base"]["sha"], "patch": patch_text, "side": 0,
                    "absent_ok": entry["status"] == "added",
                }, token, packet["incomplete"]),
                "head": read_source(repository, {
                    "filename": filename, "ref": item["head"]["sha"], "patch": patch_text,
                    "absent_ok": entry["status"] == "removed",
                }, token, packet["incomplete"]),
            })
        if not ci_result:
            runs = github(f"{prefix}/actions/runs?event=pull_request&per_page=100", token)
            matching = [run for run in runs["workflow_runs"]
                        if run["head_sha"] == item["head"]["sha"]
                        and run["path"] == ".github/workflows/ci.yml"]
            if matching:
                packet["ci"] = {key: matching[0].get(key) for key in
                                ("id", "status", "conclusion", "html_url")}
    else:
        referenced = re.findall(r"[\w./-]+\.(?:py|md|yml|toml)", item.get("body") or "")
        filenames = [name for name in dict.fromkeys(referenced)
                     if name.startswith(("src/", "example/", "tests/", ".github/"))
                     and ".." not in name.split("/")]
        if not filenames:
            packet["incomplete"].append("Issue has no repository source paths to verify")
        for filename in filenames[:MAX_FILES]:
            packet["files"].append({"filename": filename, "patch": "", "base": "",
                                    "head": read_source(repository, {
                                        "filename": filename, "ref": revision,
                                    }, token, packet["incomplete"])})
    packet["project_readme_excerpt"] = read_source(repository, {
        "filename": "README_ZH.md", "ref": revision,
    }, token, [])
    words = re.findall(r"[\w-]{3,}", item["title"])
    terms = [word for word in words if word.lower() not in
             {"fix", "feat", "docs", "test", "perf", "example", "examples", "the", "and"}]
    if terms:
        query = urllib.parse.quote(f"repo:{repository} in:title " + " ".join(terms[:2]))
        found = github(f"/search/issues?q={query}&per_page=15", token)
        packet["related"] = [{key: row.get(key) for key in ("number", "title", "state")}
                             for row in found["items"] if row["number"] != number]
        packet["related_search"] = "Title search only; absence does not prove uniqueness"
    if len(json.dumps(packet, ensure_ascii=False)) > MAX_INPUT:
        raise ValueError("Review context exceeds the per-request input limit")
    return packet


def report_schema():
    """Describe the JSON report format for the model and local checks."""
    text = {"type": "string"}
    finding = {"type": "object", "additionalProperties": False,
               "properties": {name: text for name in
                              ("severity", "location", "trigger", "evidence", "detail")}}
    finding["required"] = list(finding["properties"])
    properties = {
        "verdict": {"type": "string", "enum": list(VERDICTS)}, "summary": text,
        "models": {"type": "array", "items": text},
        "findings": {"type": "array", "items": finding},
        "next_steps": {"type": "array", "items": text},
    }
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def parse_response(response):
    """Reject API failures, refusals and incomplete JSON rather than invent success."""
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ReviewError("DeepSeek 没有返回唯一的审核结果")
    reason = choices[0].get("finish_reason")
    if reason != "stop":
        reasons = {"length": "DeepSeek 回答达到输出长度限制，未能完整生成",
                   "content_filter": "DeepSeek 因内容限制中止回答",
                   "insufficient_system_resource": "DeepSeek 因服务资源不足中止回答",
                   "aborted": "DeepSeek 中止了回答"}
        raise ReviewError(reasons.get(reason, "DeepSeek 没有正常结束回答"))
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip() or message.get("refusal"):
        raise ReviewError("DeepSeek 没有返回审核正文，或拒绝了本次审核")
    return json.loads(content)


def validate_report(report, packet):
    """Only publish bounded reports with quotations from the collected code."""
    if (not isinstance(report, dict) or set(report) != set(report_schema()["required"])
            or not isinstance(report["verdict"], str) or report["verdict"] not in VERDICTS):
        raise ReviewError("DeepSeek 返回的审核字段不符合要求")
    if not isinstance(report["summary"], str) or len(report["summary"]) > 5000:
        raise ReviewError("DeepSeek 返回的审核摘要格式或长度不符合要求")
    for name in ("models", "next_steps"):
        if (not isinstance(report[name], list) or len(report[name]) > 10
                or any(not isinstance(value, str) or len(value) > 2000
                       for value in report[name])):
            raise ReviewError("DeepSeek 返回的模型或后续步骤格式或长度不符合要求")
    if not isinstance(report["findings"], list) or len(report["findings"]) > 6:
        raise ReviewError("DeepSeek 返回的问题列表格式不正确，或超过 6 条")
    for finding in report["findings"]:
        if (not isinstance(finding, dict)
                or set(finding) != {"severity", "location", "trigger", "evidence", "detail"}
                or any(not isinstance(value, str) or not value or len(value) > 2000
                       for value in finding.values())
                or finding["severity"] not in {"high", "medium", "low"}):
            raise ReviewError("DeepSeek 返回的问题缺少必填内容，或格式或长度不符合要求")
        sources = [entry for entry in packet["files"]
                   if entry["filename"] == finding["location"]]
        if not sources or not any(finding["evidence"] in entry.get(key, "")
                                  for entry in sources for key in ("base", "head", "patch")):
            raise ReviewError("DeepSeek 返回的问题路径或原文引用与实际读取的代码不匹配")
    if len(render_report(report, packet)) > 60000:
        raise ReviewError("审核报告超过评论长度限制")


def failure_reason(error):
    """Expose fixed diagnostics and HTTP status codes, never arbitrary error text."""
    if isinstance(error, ReviewError):
        return str(error)
    if isinstance(error, json.JSONDecodeError):
        return "DeepSeek 回答不是有效的 JSON，无法读取"
    if isinstance(error, urllib.error.HTTPError):
        hint = {401: "请检查密钥", 402: "请检查账户余额", 429: "请求过于频繁，请稍后重试"}
        return f"DeepSeek 请求失败（HTTP {error.code}）；{hint.get(error.code, '请稍后重试')}"
    if isinstance(error, TimeoutError):
        return "DeepSeek 请求超时，请稍后重试"
    if isinstance(error, urllib.error.URLError):
        return "DeepSeek 网络请求失败，请稍后重试"
    return "DeepSeek 请求或回答无法处理，请检查配置后重试"


def escaped(value):
    """Keep model text visible and prevent user or team mention notifications."""
    return html.escape(str(value)).replace("@", "@\u200b")


def render_report(report, packet):
    """Report analysis separately from CI and explicitly unexecuted verification."""
    verdict = report["verdict"]
    ci_result = packet["ci"]
    if verdict == "recommend" and (packet["incomplete"] or packet["kind"] == "issue"
                                  or ci_result.get("conclusion") != "success"):
        verdict = "needs_evidence"
    lines = [MARKER, f"### AI 审核：{VERDICTS[verdict]}", "", escaped(report["summary"]),
             "", "涉及模型／功能：" + "、".join(map(escaped, report["models"])), ""]
    for finding in report["findings"]:
        lines.extend([f"- **{escaped(finding['severity'])} · {escaped(finding['location'])}**："
                      f"{escaped(finding['detail'])}",
                      f"  触发条件：{escaped(finding['trigger'])}",
                      f"  原文依据：<code>{escaped(finding['evidence'])}</code>"])
    lines.extend(["", "**运行与覆盖情况**", "",
                  "- 本审核只读取代码和已有检查结果，没有执行 PR 代码，也没有独立复现实验。",
                  f"- 已有 CI：{escaped(ci_result.get('conclusion') or '未完成或未运行')}。"
                  "CI 成功不代表所有测试均执行，也不证明模型效果。"])
    for gap in packet["incomplete"]:
        lines.append(f"- 未覆盖：{escaped(gap)}")
    lines.extend(["", "**下一步**", ""] +
                 [f"- {escaped(step)}" for step in report["next_steps"]])
    lines.extend(["", f"审核对象：{packet['repository']}#{packet['number']}；"
                  f"版本：{packet.get('head_sha', packet.get('default_sha', ''))[:12]}。",
                  "此结果仅供审查参考，不会自动批准、合并或关闭。"])
    return "\n".join(lines)


def publish_report(packet, body, token):
    """Publish on the hosting fork only, after rechecking the reviewed revision."""
    prefix = f"/repos/{packet['repository']}"
    endpoint = "pulls" if packet["kind"] == "pr" else "issues"
    current = github(f"{prefix}/{endpoint}/{packet['number']}", token)
    if current["state"] != "open" or item_version(current) != packet["version"]:
        return False
    if packet["kind"] == "issue":
        info = github(prefix, token)
        current_sha = github(f"{prefix}/commits/{info['default_branch']}", token)["sha"]
        if current_sha != packet["default_sha"]:
            return False
    page = 1
    while True:
        comments = github(f"{prefix}/issues/{packet['number']}/comments?per_page=100&page={page}",
                          token)
        for comment in comments:
            if (comment["user"]["login"] == "github-actions[bot]"
                    and comment["body"].startswith(MARKER)):
                github(f"{prefix}/issues/comments/{comment['id']}", token, "PATCH", {"body": body})
                return True
        if len(comments) < 100:
            break
        page += 1
    github(f"{prefix}/issues/{packet['number']}/comments", token, "POST", {"body": body})
    return True


def main():
    """Collect, review or publish in separate jobs with separate credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review", "publish", "collect"))
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    host = os.environ["GITHUB_REPOSITORY"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    if args.mode == "publish":
        packet = json.loads(Path("review-packet.json").read_text(encoding="utf-8"))
        if packet["repository"] == host:
            publish_report(packet, Path("review.md").read_text(encoding="utf-8"), token)
        return
    repository = event.get("inputs", {}).get("repository") or host
    allowed = {host, github(f"/repos/{host}", token).get("parent", {}).get("full_name")}
    if repository not in allowed:
        raise ValueError("Review target must be this repository or its upstream parent")
    number, ci_result = resolve_target(os.environ["GITHUB_EVENT_NAME"], event, repository, token)
    if number is None:
        return
    packet = collect_packet(repository, number, token, ci_result)
    Path("review-packet.json").write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
    if args.mode == "collect":
        print(f"Collected {repository}#{number}; {len(packet['files'])} source files")
        return
    rules = Path(__file__).with_name("ai-review-rules.md").read_text(encoding="utf-8")
    payload = {
        "model": os.environ.get("DEEPSEEK_REVIEW_MODEL") or "deepseek-flash",
        "max_tokens": 32768, "reasoning_effort": "low",
        "messages": [
            {"role": "system", "content": rules + "\n\n只返回符合以下结构的 JSON 对象：\n"
             + json.dumps(report_schema(), ensure_ascii=False)},
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ReviewError("未配置 DEEPSEEK_API_KEY，请在仓库 Secrets 中添加密钥")
        response = request_json("https://api.deepseek.com/chat/completions", key, "POST", payload,
                                timeout=480)
        report = parse_response(response)
        validate_report(report, packet)
    except (ValueError, urllib.error.URLError, TimeoutError) as error:
        # Do not expose provider response bodies, request headers or credentials.
        reason = failure_reason(error)
        report = {"verdict": "needs_evidence", "summary": f"审核未完成：{reason}。不能作为通过依据。",
                  "models": [], "findings": [],
                  "next_steps": ["根据上述失败原因处理后重新运行。"]}
        Path("review.md").write_text(render_report(report, packet), encoding="utf-8")
        raise RuntimeError(f"AI review did not complete: {reason}; see review.md") from None
    body = render_report(report, packet)
    Path("review.md").write_text(body, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(body)


if __name__ == "__main__":
    main()
