"""Abstract base class for file storage backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class FileStorage(ABC):
    """Abstract base for file storage backends."""

    @abstractmethod
    async def store(
        self,
        file_data: bytes,
        key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Store file and return storage key.

        Args:
            file_data: Raw file bytes
            key: Storage key (e.g., "banks/{bank_id}/files/{file_id}.pdf")
            metadata: Optional metadata to store with file

        Returns:
            Storage key that can be used to retrieve the file
        """
        pass

    async def store_stream(
        self,
        key: str,
        stream: AsyncIterator[bytes],
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Store file from an async byte stream.

        Default implementation buffers stream into memory and calls :meth:`store`.
        Streaming-capable backends override this so a large file is never held whole.
        """
        chunks: list[bytes] = []
        async for chunk in stream:
            if chunk:
                chunks.append(chunk)
        return await self.store(b"".join(chunks), key, metadata=metadata)

    @abstractmethod
    async def retrieve(self, key: str) -> bytes:
        """
        Retrieve file by storage key.

        Args:
            key: Storage key

        Returns:
            File data as bytes

        Raises:
            FileNotFoundError: If file does not exist
        """
        pass

    async def retrieve_stream(self, key: str) -> AsyncIterator[bytes]:
        """Retrieve file as an async stream of bytes.

        Default implementation calls :meth:`retrieve` and yields the full bytes.
        Streaming-capable backends override this so a large file is never held whole.
        """
        data = await self.retrieve(key)
        yield data

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete file by storage key.

        Args:
            key: Storage key
        """
        pass

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every file whose key starts with ``prefix``; return how many.

        How a bank's files go when the bank does: its keys share one prefix
        (see :func:`~remember_api.engine.storage.bank_storage_prefix`), so the
        sweep needs no row to enumerate them. Not abstract, so a backend written
        before this existed still loads; callers treat it as best-effort.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support delete_prefix")

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if file exists.

        Args:
            key: Storage key

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_download_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Get a URL for downloading the file.

        For PostgreSQL storage, this might be a relative API path.
        For S3, this would be a pre-signed URL.

        Args:
            key: Storage key
            expires_in: Expiration time in seconds (may be ignored for some backends)

        Returns:
            Download URL or path
        """
        pass

    async def get_size(self, key: str) -> int | None:
        """Return the size of the file in bytes if known/supported, or None."""
        return None
