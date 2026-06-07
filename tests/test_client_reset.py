import unittest
from unittest.mock import MagicMock, patch

from admin.routes import client_reset


class ClientResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_reset_removes_files_vectors_and_client_row(self):
        request = MagicMock()

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("shutil.rmtree") as remove_tree,
            patch("core.vector_store.delete_kb_for_client") as delete_vectors,
            patch("admin.routes.store.delete_client") as delete_client,
        ):
            response = await client_reset(request, "Demo")

        remove_tree.assert_called_once()
        delete_vectors.assert_called_once_with("Demo")
        delete_client.assert_called_once_with("Demo")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/admin/clients")

    async def test_vector_cleanup_failure_does_not_block_client_reset(self):
        request = MagicMock()

        with (
            patch("admin.routes._is_auth", return_value=True),
            patch("shutil.rmtree"),
            patch(
                "core.vector_store.delete_kb_for_client",
                side_effect=RuntimeError("qdrant unavailable"),
            ),
            patch("admin.routes.store.delete_client") as delete_client,
        ):
            response = await client_reset(request, "Demo")

        delete_client.assert_called_once_with("Demo")
        self.assertEqual(response.status_code, 303)
