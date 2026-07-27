"""
core/notification_scheduler.py — background loop driving due alert checks
and due report digests. Modeled on
core/log_export.py::scheduled_log_export_loop; these tests mirror the same
shape without actually sleeping (loop is cancelled after one iteration).
"""

import asyncio
import unittest
from unittest.mock import patch

import core.notification_scheduler as ns


class RunDueNotificationsOnceTests(unittest.TestCase):

    def test_calls_run_due_alert_checks(self):
        with (
            patch("core.alert_engine.run_due_alert_checks") as mock_alerts,
            patch("core.report_engine.run_due_report_digests"),
        ):
            ns.run_due_notifications_once()
        mock_alerts.assert_called_once()

    def test_calls_run_due_report_digests(self):
        with (
            patch("core.alert_engine.run_due_alert_checks"),
            patch("core.report_engine.run_due_report_digests") as mock_digests,
        ):
            ns.run_due_notifications_once()
        mock_digests.assert_called_once()

    def test_alert_check_exception_does_not_propagate(self):
        with (
            patch("core.alert_engine.run_due_alert_checks", side_effect=Exception("boom")),
            patch("core.report_engine.run_due_report_digests") as mock_digests,
        ):
            ns.run_due_notifications_once()  # must not raise
        mock_digests.assert_called_once()  # alert failure must not block digests

    def test_report_digest_exception_does_not_propagate(self):
        with (
            patch("core.alert_engine.run_due_alert_checks") as mock_alerts,
            patch("core.report_engine.run_due_report_digests", side_effect=Exception("boom")),
        ):
            ns.run_due_notifications_once()  # must not raise
        mock_alerts.assert_called_once()


class ScheduledNotificationLoopTests(unittest.TestCase):

    def test_loop_runs_check_then_sleeps_then_cancellable(self):
        calls = []

        def _fake_once():
            calls.append(1)

        async def _run_one_iteration():
            with (
                patch("core.notification_scheduler.run_due_notifications_once", side_effect=_fake_once),
                patch("core.notification_scheduler.asyncio.sleep", side_effect=asyncio.CancelledError),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await ns.scheduled_notification_loop(poll_seconds=0)

        asyncio.run(_run_one_iteration())
        self.assertEqual(calls, [1])

    def test_loop_survives_a_failing_iteration_and_keeps_going(self):
        # First iteration's to_thread call raises; loop must log and continue
        # to sleep rather than propagating (mirrors scheduled_log_export_loop).
        call_count = {"n": 0}

        async def _fake_to_thread(fn, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("iteration 1 failed")
            raise asyncio.CancelledError  # stop after iteration 2

        real_sleep = asyncio.sleep

        async def _run():
            with (
                patch("core.notification_scheduler.asyncio.to_thread", side_effect=_fake_to_thread),
                patch("core.notification_scheduler.asyncio.sleep", new=lambda *_: real_sleep(0)),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await ns.scheduled_notification_loop(poll_seconds=0)

        asyncio.run(_run())
        self.assertEqual(call_count["n"], 2)


if __name__ == "__main__":
    unittest.main()
