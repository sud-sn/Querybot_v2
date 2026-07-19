"""
tests/test_stop_query.py

Stop-button feature: a user can cancel an in-flight chat question.

  1. WebAdapter.send_lock exists and genuinely serializes concurrent sends
     (the WS receive loop now runs question-handling as a background task,
     so two coroutines can call adapter.send_* at the same time — ASGI does
     not guarantee that's safe without a lock).
  2. asyncio.Task.cancel() semantics used by the cancel handler behave as
     the implementation assumes (CancelledError propagates through a
     try/except Exception, is not swallowed).
  3. gateway/webhooks.py wiring: the "cancel" message type, per-connection
     task tracking, background-task dispatch, and cleanup-on-disconnect
     are all present (marker-test convention already used in this file's
     WsFallbackErrorTests / test_runtime_wiring.py).
  4. Frontend wiring: portal_chat.html has the stop-button markup, mode
     toggle, and the sendMessage()/processingActive double-send guard.
"""
import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SendLockSerializesConcurrentSendsTests(unittest.TestCase):
    def test_send_lock_is_an_asyncio_lock(self):
        from gateway.web_adapter import WebAdapter

        class FakeWS:
            async def send_json(self, payload):
                pass

        adapter = WebAdapter(FakeWS(), "acct", "user1")
        self.assertIsInstance(adapter.send_lock, asyncio.Lock)

    def test_concurrent_sends_do_not_interleave(self):
        from gateway.web_adapter import WebAdapter

        events: list[str] = []

        class SlowFakeWS:
            async def send_json(self, payload):
                events.append(f"start:{payload['type']}")
                await asyncio.sleep(0)  # yield control — the classic interleaving window
                events.append(f"end:{payload['type']}")

        async def run():
            adapter = WebAdapter(SlowFakeWS(), "acct", "user1")
            await asyncio.gather(
                adapter.send_message(None, "hello"),
                adapter.send_status(None, "thinking", "Thinking"),
            )

        asyncio.run(run())
        # Serialized: one call's start+end must appear as an adjacent pair,
        # never start:A, start:B, end:A, end:B (which would mean the two
        # sends interleaved on the wire).
        self.assertEqual(events[0].split(":")[0], "start")
        self.assertEqual(events[1].split(":")[0], "end")
        self.assertEqual(events[0].split(":")[1], events[1].split(":")[1])
        self.assertEqual(events[2].split(":")[0], "start")
        self.assertEqual(events[3].split(":")[0], "end")

    def test_all_send_methods_hold_the_lock(self):
        # Every public send_* / upload_file method must acquire send_lock —
        # a method that forgot to would reintroduce the interleaving risk
        # invisibly (no test would catch it except reading the source).
        src = (ROOT / "gateway" / "web_adapter.py").read_text(encoding="utf-8")
        for method in (
            "send_message", "send_status", "send_chart", "send_assistant_response",
            "send_analysis_response", "send_clarification_prompt", "upload_file",
        ):
            start = src.index(f"async def {method}(")
            body = src[start:start + 700]
            self.assertIn("async with self.send_lock:", body, method)


class CancellationSemanticsTests(unittest.TestCase):
    """Sanity-checks the exact asyncio pattern _run_main_question relies on:
    a cancelled task raises CancelledError past a bare `except Exception`,
    and `await task` after `.cancel()` raises CancelledError to the caller."""

    def test_cancelled_error_not_caught_by_except_exception(self):
        caught: list[str] = []

        async def victim():
            try:
                await asyncio.sleep(10)
            except Exception as e:  # noqa — deliberately mirrors the pipeline's broad catches
                caught.append(f"exception:{e}")
                raise

        async def run():
            task = asyncio.create_task(victim())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertEqual(caught, [])  # except Exception never fired

    def test_awaiting_cancelled_task_raises_cancelled_error_to_caller(self):
        async def run():
            task = asyncio.create_task(asyncio.sleep(10))
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
                return "no-error"
            except asyncio.CancelledError:
                return "cancelled"

        self.assertEqual(asyncio.run(run()), "cancelled")


class WebhooksWiringTests(unittest.TestCase):
    """Marker tests — same convention as WsFallbackErrorTests in
    tests/test_defect_fixes.py."""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")

    def test_cancel_message_type_handled(self):
        self.assertIn('if msg_type == "cancel":', self.src)
        self.assertIn("current_query_task.cancel()", self.src)
        self.assertIn('"type": "system"', self.src)
        self.assertIn("Query stopped.", self.src)

    def test_main_question_runs_as_background_task_not_awaited_inline(self):
        self.assertIn("current_query_task = asyncio.create_task(", self.src)
        self.assertIn("_run_main_question(text, table_hint, schema_hint)", self.src)
        # The dispatch call now lives inside _run_main_question, not awaited
        # directly in the receive loop — confirm the loop body's own tail
        # (after the create_task call) goes straight to the except blocks,
        # with no inline "await dispatch(...)" of its own.
        loop_tail_start = self.src.index("current_query_task = asyncio.create_task(")
        loop_tail = self.src[loop_tail_start:self.src.index("except WebSocketDisconnect:")]
        self.assertNotIn("await dispatch(", loop_tail)

    def test_new_question_supersedes_unfinished_one(self):
        anchor = self.src.index("current_query_task = asyncio.create_task(")
        head = self.src[max(0, anchor - 300):anchor]
        self.assertIn("if current_query_task and not current_query_task.done():", head)
        self.assertIn("current_query_task.cancel()", head)

    def test_task_cleaned_up_on_disconnect(self):
        anchor = self.src.index("except WebSocketDisconnect:")
        tail = self.src[anchor:anchor + 900]
        self.assertIn("finally:", tail)
        self.assertIn("current_query_task.cancel()", tail)

    def test_run_main_question_never_silently_swallows_errors(self):
        start = self.src.index("async def _run_main_question(")
        body = self.src[start:start + 3000]
        self.assertIn("except asyncio.CancelledError:", body)
        self.assertIn("except Exception as e:", body)
        self.assertIn('"type": "assistant_error"', body)

    def test_asyncio_imported(self):
        self.assertIn("import asyncio", self.src)


class WebSocketPayloadNormalizationTests(unittest.TestCase):
    def test_plain_text_is_trimmed(self):
        from gateway.webhooks import _ws_text_value

        self.assertEqual(_ws_text_value("  monthly receipts  "), "monthly receipts")

    def test_structured_question_is_recovered(self):
        from gateway.webhooks import _ws_text_value

        value = {"question": "Show monthly receipts by supplier", "kind": "history"}
        self.assertEqual(
            _ws_text_value(value, "text", "question", "value", "label"),
            "Show monthly receipts by supplier",
        )

    def test_structured_table_hint_is_recovered(self):
        from gateway.webhooks import _ws_text_value

        value = {"fqn": "CHATBOT_DB.PHARMA_LAB.F_PURCHASE_RECEIPT"}
        self.assertEqual(
            _ws_text_value(value, "fqn", "table_hint", "table", "value"),
            "CHATBOT_DB.PHARMA_LAB.F_PURCHASE_RECEIPT",
        )

    def test_unknown_object_is_ignored_instead_of_stringified(self):
        from gateway.webhooks import _ws_text_value

        self.assertEqual(_ws_text_value({"unexpected": ["value"]}, "text"), "")


class FrontendWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")

    def test_stop_button_markup_present(self):
        self.assertIn('id="sendBtnIcon"', self.html)
        self.assertIn("send-btn-icon", self.html)

    def test_mode_toggle_function_present(self):
        self.assertIn("function _setSendButtonMode(isStop)", self.html)
        self.assertIn("btn.classList.toggle('is-stop', isStop)", self.html)
        self.assertIn("btn.onclick = isStop ? stopQuery : sendMessage", self.html)

    def test_stop_query_sends_cancel_message(self):
        self.assertIn("function stopQuery()", self.html)
        start = self.html.index("function stopQuery()")
        body = self.html[start:start + 400]
        self.assertIn("JSON.stringify({ type: 'cancel' })", body)
        self.assertIn("if (!wsReady) return;", body)

    def test_structured_table_hint_is_normalized_before_send(self):
        start = self.html.index("function sendMessage(tableHint)")
        body = self.html[start:start + 1800]
        self.assertIn("normalizedTableHint", body)
        self.assertIn("tableHint?.fqn", body)
        self.assertIn("payload.table_hint = normalizedTableHint", body)

    def test_set_processing_drives_button_mode(self):
        start = self.html.index("function setProcessing(active)")
        body = self.html[start:start + 500]
        self.assertIn("_setSendButtonMode(!!active)", body)

    def test_send_message_blocked_while_processing(self):
        start = self.html.index("function sendMessage(tableHint)")
        body = self.html[start:start + 400]
        self.assertIn("processingActive", body)

    def test_stop_mode_css_present(self):
        self.assertIn(".send-btn.is-stop{", self.html)
        self.assertIn(".stop-icon{", self.html)


if __name__ == "__main__":
    unittest.main()
