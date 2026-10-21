"""Behavior and permission boundaries of the repository review workflow."""

import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error


SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "ai_review.py"
SPEC = importlib.util.spec_from_file_location("ai_review", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class TestReviewPipeline(unittest.TestCase):
    """Keep model claims separate from execution and publication privileges."""

    def setUp(self):
        """Use a complete, small, independently inspectable review packet."""
        self.packet = {
            "repository": "owner/repo", "number": 7, "kind": "pr",
            "version": "version-one", "incomplete": [],
            "item": {"title": "Fix", "body": "", "state": "open"},
            "files": [{"filename": "src/model.py", "patch": "-bad\n+good",
                       "base": "bad", "head": "good"}],
            "ci": {"status": "completed", "conclusion": "success"},
            "related": [],
        }
        self.report = {
            "verdict": "needs_evidence", "summary": "Needs an independent check.",
            "models": ["RBM"], "findings": [], "next_steps": ["Check energy."],
        }

    def test_completed_response_extracts_only_assistant_text(self):
        """Reasoning and other output items are not a substitute for a report."""
        response = {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(self.report),
            "reasoning_content": "Internal analysis is not the JSON report.",
        }}]}
        self.assertEqual(review.parse_response(response), self.report)

    def test_incomplete_and_refused_responses_do_not_pass(self):
        """Partial responses, API errors and refusals cannot become success."""
        for response in (
            {"choices": []},
            {"error": {"message": "private"}},
            {"choices": [{"finish_reason": "length", "message": {
                "content": json.dumps(self.report)}}]},
            {"choices": [{"finish_reason": "content_filter", "message": {"content": None}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                review.parse_response(response)

    def test_report_requires_valid_structure_and_actual_source_evidence(self):
        """JSON mode must not bypass field checks or accept invented quotations."""
        for report in (None, [], "text", {}, {**self.report, "verdict": []},
                       {**self.report, "findings": ["invented problem"]}):
            with self.subTest(report=report), self.assertRaises(ValueError):
                review.validate_report(report, self.packet)
        self.report["findings"] = [{
            "severity": "high", "location": "src/model.py", "trigger": "On load",
            "evidence": "invented code", "detail": "Overwrites weights.",
        }]
        with self.assertRaises(ValueError):
            review.validate_report(self.report, self.packet)
        self.report["findings"][0]["evidence"] = "bad"
        review.validate_report(self.report, self.packet)

    def test_unknown_source_and_empty_evidence_are_rejected(self):
        """A location must refer to a collected source file."""
        for location, evidence in (("other.py", "bad"), ("src/model.py", "")):
            self.report["findings"] = [{
                "severity": "high", "location": location, "trigger": "On load",
                "evidence": evidence, "detail": "Problem.",
            }]
            with self.subTest(location=location), self.assertRaises(ValueError):
                review.validate_report(self.report, self.packet)

    def test_partial_context_cannot_recommend_merge(self):
        """Truncated large PRs remain explicitly unverified."""
        self.packet["incomplete"] = ["Only 8 of 25 files included"]
        self.report["verdict"] = "recommend"
        body = review.render_report(self.report, self.packet)
        self.assertIn("需要补证据", body)
        self.assertIn("Only 8 of 25 files included", body)

    def test_missing_or_failed_ci_cannot_recommend_merge(self):
        """No checks and failed checks cannot be presented as passed tests."""
        self.report["verdict"] = "recommend"
        for ci_result in ({}, {"status": "completed", "conclusion": "failure"}):
            self.packet["ci"] = ci_result
            body = review.render_report(self.report, self.packet)
            self.assertIn("需要补证据", body)
            self.assertIn("没有执行 PR 代码", body)

    def test_summary_escapes_mentions_and_hidden_html(self):
        """Model output cannot notify users or hide arbitrary HTML in comments."""
        self.report["summary"] = "@everyone <script>do()</script>"
        body = review.render_report(self.report, self.packet)
        self.assertNotIn("@everyone", body)
        self.assertNotIn("<script>", body)
        self.assertIn(review.MARKER, body)

    def test_version_changes_when_base_or_issue_text_changes(self):
        """A PR head alone is insufficient to identify the reviewed state."""
        item = {"title": "Fix", "body": "one", "state": "open",
                "head": {"sha": "aaa"}, "base": {"sha": "bbb"}}
        original = review.item_version(item)
        changed = copy.deepcopy(item)
        changed["base"]["sha"] = "ccc"
        self.assertNotEqual(original, review.item_version(changed))
        changed = copy.deepcopy(item)
        changed["body"] = "two"
        self.assertNotEqual(original, review.item_version(changed))

    def test_publisher_skips_changed_or_closed_target(self):
        """A stale run must not overwrite a report for a newer revision."""
        with patch.object(review, "github") as github:
            github.return_value = {"title": "Changed", "body": "", "state": "open"}
            self.assertFalse(review.publish_report(self.packet, "report", "token"))
            self.assertEqual(github.call_count, 1)

    def test_publisher_updates_only_its_own_comment(self):
        """Copied markers in a contributor comment are never edited."""
        item = {"title": "Fix", "body": "", "state": "open"}
        self.packet["kind"] = "issue"
        self.packet["version"] = review.item_version(item)
        self.packet["default_sha"] = "main-sha"
        comments = [
            {"id": 4, "body": review.MARKER, "user": {"login": "contributor"}},
            {"id": 8, "body": review.MARKER, "user": {"login": "github-actions[bot]"}},
        ]
        responses = [item, {"default_branch": "main"}, {"sha": "main-sha"}, comments, {}]
        with patch.object(review, "github", side_effect=responses) as github:
            self.assertTrue(review.publish_report(self.packet, "new report", "token"))
            args = github.call_args.args
            self.assertEqual(args[0], "/repos/owner/repo/issues/comments/8")
            self.assertEqual(args[2], "PATCH")

    def test_untrusted_workflow_is_ignored(self):
        """Only the known CI workflow can supply automatic review targets."""
        event = {"workflow_run": {"id": 12}}
        with patch.object(review, "github", return_value={
            "event": "pull_request", "path": ".github/workflows/attacker.yml",
            "pull_requests": [{"number": 7}],
        }):
            self.assertEqual(review.resolve_target("workflow_run", event, "owner/repo", "t"),
                             (None, {}))

    def test_manual_target_requires_positive_number(self):
        """Dispatch input never becomes a URL or shell command."""
        for number in ("-1", "0", "$(touch marker)"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                review.resolve_target("workflow_dispatch", {"inputs": {"number": number}},
                                      "owner/repo", "t")

    def test_canonical_ci_run_accepts_only_the_current_open_pr(self):
        """Do not spend a model call on a CI result for an old revision."""
        run = {"event": "pull_request", "path": ".github/workflows/ci.yml",
               "pull_requests": [{"number": 7}], "head_sha": "new",
               "id": 12, "status": "completed", "conclusion": "success"}
        current = {"state": "open", "draft": False, "head": {"sha": "new"},
                   "base": {"repo": {"full_name": "owner/repo"}}}
        event = {"workflow_run": {"id": 12}}
        with patch.object(review, "github", side_effect=[run, current]):
            number, ci_result = review.resolve_target("workflow_run", event, "owner/repo", "t")
        self.assertEqual(number, 7)
        self.assertEqual(ci_result["conclusion"], "success")
        current["head"]["sha"] = "newer"
        with patch.object(review, "github", side_effect=[run, current]):
            self.assertEqual(review.resolve_target("workflow_run", event, "owner/repo", "t"),
                             (None, {}))

    def test_external_issue_edits_do_not_trigger_paid_requests(self):
        """Public issue authors cannot consume the maintainer's API budget."""
        event = {"issue": {"number": 7, "user": {"type": "User"},
                           "author_association": "NONE"}}
        self.assertEqual(review.resolve_target("issues", event, "owner/repo", "t"),
                         (None, {}))
        event["issue"]["author_association"] = "OWNER"
        self.assertEqual(review.resolve_target("issues", event, "owner/repo", "t"), (7, {}))
        event["issue"]["user"]["type"] = "Bot"
        self.assertEqual(review.resolve_target("issues", event, "owner/repo", "t"),
                         (None, {}))

    def test_changed_default_branch_invalidates_issue_review(self):
        """An unchanged Issue can still be fixed while AI reads old source."""
        item = {"title": "Fix", "body": "", "state": "open"}
        self.packet.update(kind="issue", version=review.item_version(item), default_sha="old")
        with patch.object(review, "github", side_effect=[
            item, {"default_branch": "main"}, {"sha": "new"},
        ]) as github:
            self.assertFalse(review.publish_report(self.packet, "old report", "token"))
            self.assertEqual(github.call_count, 3)

    def test_missing_source_is_explicit_except_for_added_or_removed_files(self):
        """Legitimate absent revisions differ from an unverifiable source path."""
        source = {"filename": "src/model.py", "ref": "sha"}
        missing = urllib.error.HTTPError("url", 404, "Not found", {}, None)
        with patch.object(review, "github", side_effect=missing):
            gaps = []
            self.assertEqual(review.read_source("owner/repo", source, "t", gaps), "")
            self.assertTrue(gaps)
            source["absent_ok"] = True
            gaps = []
            review.read_source("owner/repo", source, "t", gaps)
            self.assertFalse(gaps)

    def test_large_sources_include_changed_region_instead_of_only_file_header(self):
        """Changes near the bottom must remain inspectable within the input budget."""
        lines = ["padding " * 12 for _ in range(300)]
        lines[140] = "ising_bias = correct_bias"
        content = base64.b64encode("\n".join(lines).encode()).decode()
        source = {"filename": "src/model.py", "ref": "sha", "patch": "@@ -140 +140 @@"}
        blob = {"encoding": "base64", "content": content}
        with patch.object(review, "github", return_value=blob):
            gaps = []
            excerpt = review.read_source("owner/repo", source, "t", gaps)
        self.assertIn("ising_bias = correct_bias", excerpt)
        self.assertLessEqual(len(excerpt), 5000)
        self.assertTrue(gaps)

    def test_review_command_uses_structured_api_without_repository_write(self):
        """Exercise the command's API contract and real report files offline."""
        response = {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(self.report),
        }}]}
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.chdir(temporary)
                event = Path(temporary) / "event.json"
                event.write_text(json.dumps({"inputs": {"number": "7"}}), encoding="utf-8")
                env = {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_EVENT_NAME": "workflow_dispatch",
                       "GITHUB_EVENT_PATH": str(event), "GITHUB_TOKEN": "read-token",
                       "DEEPSEEK_API_KEY": "api-test-token", "DEEPSEEK_REVIEW_MODEL": "",
                       "GITHUB_STEP_SUMMARY": "summary.md"}
                with (
                    patch.dict(os.environ, env),
                    patch.object(sys, "argv", ["ai_review", "review"]),
                    patch.object(review, "github", return_value={}) as github,
                    patch.object(review, "collect_packet", return_value=self.packet),
                    patch.object(review, "request_json", return_value=response) as request,
                ):
                    review.main()
                self.assertTrue(Path("review-packet.json").exists())
                self.assertIn("没有执行 PR 代码", Path("review.md").read_text(encoding="utf-8"))
                self.assertEqual(github.call_count, 1)
                self.assertEqual(request.call_args.args[0],
                                 "https://api.deepseek.com/chat/completions")
                self.assertEqual(request.call_args.args[1], "api-test-token")
                payload = request.call_args.args[3]
                self.assertEqual(payload["model"], "deepseek-flash")
                self.assertEqual(payload["response_format"], {"type": "json_object"})
                self.assertEqual(payload["max_tokens"], 32768)
                self.assertEqual(request.call_args.kwargs["timeout"], 480)
                self.assertEqual(payload["reasoning_effort"], "low")
                self.assertEqual(payload["messages"][0]["role"], "system")
                self.assertIn('"required"', payload["messages"][0]["content"])
                self.assertIn("summary 最多 300 字", payload["messages"][0]["content"])
                self.assertEqual(payload["messages"][1]["role"], "user")
                self.assertNotIn("tools", payload)
        finally:
            os.chdir(previous)

    def test_publisher_command_never_comments_on_upstream(self):
        """Reports from an upstream manual trial stay in the fork's artifacts."""
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.chdir(temporary)
                Path("review-packet.json").write_text(json.dumps(self.packet), encoding="utf-8")
                Path("event.json").write_text("{}", encoding="utf-8")
                env = {"GITHUB_REPOSITORY": "my/fork", "GITHUB_EVENT_PATH": "event.json"}
                with (
                    patch.dict(os.environ, env),
                    patch.object(sys, "argv", ["ai_review", "publish"]),
                    patch.object(review, "github") as github,
                ):
                    review.main()
                github.assert_not_called()
        finally:
            os.chdir(previous)


class TestReviewFailures(unittest.TestCase):
    """Keep failure diagnostics actionable without exposing private API data."""

    def test_failed_review_explains_the_cause_without_exposing_private_data(self):
        """An incomplete run must keep an actionable, credential-free reason."""
        private = "private-api-key-and-provider-content"
        packet = {"repository": "owner/repo", "number": 7, "kind": "pr",
                  "ci": {}, "files": [], "incomplete": []}
        cases = [
            ({"choices": [{"finish_reason": "length"}]}, "输出长度限制"),
            ({"choices": []}, "没有返回唯一"),
            ({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
             "没有返回审核正文"),
            ({"choices": [{"finish_reason": "stop", "message": {"content": private}}]},
             "不是有效的 JSON"),
            ({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
             "审核字段不符合要求"),
            (urllib.error.HTTPError(private, 401, private, {}, None), "HTTP 401"),
            (urllib.error.URLError(private), "网络请求失败"),
            (TimeoutError(private), "请求超时"),
            (ValueError(private), "请求或回答无法处理"),
        ]
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.chdir(temporary)
                Path("event.json").write_text('{"inputs": {"number": "7"}}', encoding="utf-8")
                env = {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_EVENT_NAME": "workflow_dispatch",
                       "GITHUB_EVENT_PATH": "event.json", "DEEPSEEK_API_KEY": private}
                for response, reason in cases:
                    with (
                        self.subTest(reason=reason), patch.dict(os.environ, env),
                        patch.object(sys, "argv", ["ai_review", "review"]),
                        patch.object(review, "github", return_value={}),
                        patch.object(review, "collect_packet", return_value=packet),
                        patch.object(review, "request_json", side_effect=(
                            response if isinstance(response, Exception) else None),
                            return_value=response),
                        self.assertRaises(RuntimeError) as failure,
                    ):
                        review.main()
                    body = Path("review.md").read_text(encoding="utf-8")
                    self.assertIn(reason, str(failure.exception))
                    self.assertIn(reason, body)
                    self.assertIn("审核未完成", body)
                    self.assertNotIn(private, body + str(failure.exception))
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
