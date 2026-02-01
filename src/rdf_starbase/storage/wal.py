"""
Write-Ahead Log (WAL) for RDF-StarBase.

Provides crash recovery and durability guarantees by logging all
mutations before applying them to the main storage.

Key concepts:
- Every mutation is written to the WAL before being applied
- On startup, any uncommitted WAL entries are replayed
- Periodic checkpoints flush the WAL and sync with main storage
- Supports atomic multi-operation transactions

WAL Entry Format (binary):
    [4 bytes] entry_length (excluding this field)
    [1 byte]  entry_type (INSERT=1, DELETE=2, CHECKPOINT=3, TXN_BEGIN=4, TXN_COMMIT=5, TXN_ABORT=6)
    [8 bytes] sequence_number (monotonic)
    [8 bytes] transaction_id (0 for auto-commit)
    [8 bytes] timestamp (microseconds since epoch)
    [N bytes] payload (entry-type specific)
    [4 bytes] checksum (CRC32)

File Layout:
    storage_path/
        wal/
            wal_000001.log    # First WAL segment
            wal_000002.log    # Second segment (after rotation)
            checkpoint.json   # Last checkpoint state
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Callable, Any, Iterator
import struct
import json
import zlib
import threading
import os
import time

import polars as pl


class WALEntryType(IntEnum):
    """Types of WAL entries."""
    INSERT = 1          # Insert single triple
    DELETE = 2          # Delete triple(s)
    CHECKPOINT = 3      # Checkpoint marker
    TXN_BEGIN = 4       # Transaction start
    TXN_COMMIT = 5      # Transaction commit
    TXN_ABORT = 6       # Transaction abort/rollback
    BATCH_INSERT = 7    # Batch insert (columnar)


@dataclass
class WALEntry:
    """A single WAL entry."""
    entry_type: WALEntryType
    sequence_number: int
    transaction_id: int
    timestamp: int  # microseconds since epoch
    payload: bytes
    
    # Header: length(4) + type(1) + seq(8) + txn(8) + ts(8) = 29 bytes
    # Footer: checksum(4) = 4 bytes
    HEADER_SIZE = 29
    FOOTER_SIZE = 4
    HEADER_FORMAT = "<IBQQQ"  # little-endian: uint32, uint8, uint64, uint64, uint64
    
    def serialize(self) -> bytes:
        """Serialize entry to bytes."""
        payload_len = len(self.payload)
        entry_len = 1 + 8 + 8 + 8 + payload_len + 4  # type + seq + txn + ts + payload + checksum
        
        header = struct.pack(
            self.HEADER_FORMAT,
            entry_len,
            self.entry_type,
            self.sequence_number,
            self.transaction_id,
            self.timestamp,
        )
        
        data = header + self.payload
        checksum = zlib.crc32(data) & 0xFFFFFFFF
        
        return data + struct.pack("<I", checksum)
    
    @classmethod
    def deserialize(cls, data: bytes, offset: int = 0) -> tuple["WALEntry", int]:
        """
        Deserialize entry from bytes.
        
        Returns:
            Tuple of (WALEntry, bytes_consumed)
        """
        # Read header
        entry_len, entry_type, seq, txn_id, timestamp = struct.unpack_from(
            cls.HEADER_FORMAT, data, offset
        )
        
        # Calculate payload size
        payload_size = entry_len - 1 - 8 - 8 - 8 - 4  # total - type - seq - txn - ts - checksum
        
        # Read payload
        payload_start = offset + cls.HEADER_SIZE
        payload = data[payload_start:payload_start + payload_size]
        
        # Read and verify checksum
        checksum_offset = payload_start + payload_size
        stored_checksum = struct.unpack_from("<I", data, checksum_offset)[0]
        
        computed_data = data[offset:checksum_offset]
        computed_checksum = zlib.crc32(computed_data) & 0xFFFFFFFF
        
        if stored_checksum != computed_checksum:
            raise ValueError(f"WAL checksum mismatch at offset {offset}")
        
        entry = cls(
            entry_type=WALEntryType(entry_type),
            sequence_number=seq,
            transaction_id=txn_id,
            timestamp=timestamp,
            payload=payload,
        )
        
        bytes_consumed = 4 + entry_len  # length field + entry content
        return entry, bytes_consumed


@dataclass
class InsertPayload:
    """Payload for INSERT entries."""
    graph_id: int
    subject_id: int
    predicate_id: int
    object_id: int
    flags: int
    source_id: int
    confidence: float
    process_id: int
    
    FORMAT = "<QQQQHQdQ"  # g, s, p, o, flags, source, confidence, process
    SIZE = struct.calcsize(FORMAT)
    
    def serialize(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.graph_id,
            self.subject_id,
            self.predicate_id,
            self.object_id,
            self.flags,
            self.source_id,
            self.confidence,
            self.process_id,
        )
    
    @classmethod
    def deserialize(cls, data: bytes) -> "InsertPayload":
        g, s, p, o, flags, source, conf, process = struct.unpack(cls.FORMAT, data)
        return cls(g, s, p, o, flags, source, conf, process)


@dataclass
class DeletePayload:
    """Payload for DELETE entries."""
    graph_id: Optional[int]
    subject_id: Optional[int]
    predicate_id: Optional[int]
    object_id: Optional[int]
    
    # Use -1 (max uint64) as sentinel for "any"
    SENTINEL = 0xFFFFFFFFFFFFFFFF
    FORMAT = "<QQQQ"
    SIZE = struct.calcsize(FORMAT)
    
    def serialize(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.graph_id if self.graph_id is not None else self.SENTINEL,
            self.subject_id if self.subject_id is not None else self.SENTINEL,
            self.predicate_id if self.predicate_id is not None else self.SENTINEL,
            self.object_id if self.object_id is not None else self.SENTINEL,
        )
    
    @classmethod
    def deserialize(cls, data: bytes) -> "DeletePayload":
        g, s, p, o = struct.unpack(cls.FORMAT, data)
        return cls(
            g if g != cls.SENTINEL else None,
            s if s != cls.SENTINEL else None,
            p if p != cls.SENTINEL else None,
            o if o != cls.SENTINEL else None,
        )


@dataclass
class BatchInsertPayload:
    """Payload for batch INSERT entries (columnar format)."""
    graph_ids: List[int]
    subject_ids: List[int]
    predicate_ids: List[int]
    object_ids: List[int]
    flags: int
    source_id: int
    confidence: float
    process_id: int
    
    def serialize(self) -> bytes:
        """Serialize batch to bytes."""
        n = len(self.subject_ids)
        # Header: count(8) + flags(2) + source(8) + confidence(8) + process(8)
        header = struct.pack("<QHQdQ", n, self.flags, self.source_id, self.confidence, self.process_id)
        # Column data: n * 4 uint64s (g, s, p, o)
        col_format = f"<{n}Q"
        g_data = struct.pack(col_format, *self.graph_ids)
        s_data = struct.pack(col_format, *self.subject_ids)
        p_data = struct.pack(col_format, *self.predicate_ids)
        o_data = struct.pack(col_format, *self.object_ids)
        return header + g_data + s_data + p_data + o_data
    
    @classmethod
    def deserialize(cls, data: bytes) -> "BatchInsertPayload":
        """Deserialize batch from bytes."""
        # Read header
        header_size = struct.calcsize("<QHQdQ")
        n, flags, source_id, confidence, process_id = struct.unpack("<QHQdQ", data[:header_size])
        
        # Read columns
        col_format = f"<{n}Q"
        col_size = struct.calcsize(col_format)
        offset = header_size
        
        g_ids = list(struct.unpack(col_format, data[offset:offset+col_size]))
        offset += col_size
        s_ids = list(struct.unpack(col_format, data[offset:offset+col_size]))
        offset += col_size
        p_ids = list(struct.unpack(col_format, data[offset:offset+col_size]))
        offset += col_size
        o_ids = list(struct.unpack(col_format, data[offset:offset+col_size]))
        
        return cls(g_ids, s_ids, p_ids, o_ids, flags, source_id, confidence, process_id)


@dataclass
class CheckpointState:
    """Checkpoint state persisted to disk."""
    last_sequence: int
    last_committed_txn: int
    checkpoint_time: str  # ISO format
    wal_segment: int
    
    def to_dict(self) -> dict:
        return {
            "last_sequence": self.last_sequence,
            "last_committed_txn": self.last_committed_txn,
            "checkpoint_time": self.checkpoint_time,
            "wal_segment": self.wal_segment,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        return cls(
            last_sequence=d["last_sequence"],
            last_committed_txn=d["last_committed_txn"],
            checkpoint_time=d["checkpoint_time"],
            wal_segment=d["wal_segment"],
        )


class WriteAheadLog:
    """
    Write-Ahead Log for durability and crash recovery.
    
    Usage:
        wal = WriteAheadLog(Path("data/wal"))
        
        # Log an insert before applying
        wal.log_insert(g, s, p, o, flags, source, conf, process)
        
        # Apply to storage...
        
        # Periodic checkpoint
        wal.checkpoint()
        
        # On startup, replay uncommitted
        for entry in wal.replay():
            apply_entry(entry)
    """
    
    SEGMENT_MAX_SIZE = 64 * 1024 * 1024  # 64 MB per segment
    CHECKPOINT_FILE = "checkpoint.json"
    WAL_PREFIX = "wal_"
    WAL_SUFFIX = ".log"
    
    def __init__(
        self,
        wal_dir: Path | str,
        segment_max_size: int = None,
        sync_mode: str = "full",  # "full", "normal", "off"
    ):
        """
        Initialize WAL.
        
        Args:
            wal_dir: Directory for WAL files
            segment_max_size: Max size per WAL segment before rotation
            sync_mode: Sync mode for durability
                - "full": fsync after every write (safest, slowest)
                - "normal": fsync after commits only
                - "off": no fsync (fastest, least safe)
        """
        self.wal_dir = Path(wal_dir)
        self.segment_max_size = segment_max_size or self.SEGMENT_MAX_SIZE
        self.sync_mode = sync_mode
        
        # State
        self._sequence = 0
        self._current_segment = 0
        self._current_file: Optional[Any] = None
        self._current_file_size = 0
        self._lock = threading.Lock()
        
        # Transaction state
        self._active_txn: Optional[int] = None
        self._next_txn_id = 1
        
        # Checkpoint state
        self._last_checkpoint: Optional[CheckpointState] = None
        
        # Initialize
        self._init_wal()
    
    def _init_wal(self) -> None:
        """Initialize WAL directory and state."""
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint if exists
        checkpoint_path = self.wal_dir / self.CHECKPOINT_FILE
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                self._last_checkpoint = CheckpointState.from_dict(json.load(f))
                self._sequence = self._last_checkpoint.last_sequence
                self._current_segment = self._last_checkpoint.wal_segment
                self._next_txn_id = self._last_checkpoint.last_committed_txn + 1
        
        # Find latest segment if checkpoint doesn't exist
        if self._last_checkpoint is None:
            segments = self._list_segments()
            if segments:
                self._current_segment = max(segments)
                # Scan to find last sequence
                self._sequence = self._scan_last_sequence()
    
    def _list_segments(self) -> List[int]:
        """List all WAL segment numbers."""
        segments = []
        for f in self.wal_dir.glob(f"{self.WAL_PREFIX}*{self.WAL_SUFFIX}"):
            try:
                num = int(f.stem.replace(self.WAL_PREFIX, ""))
                segments.append(num)
            except ValueError:
                pass
        return sorted(segments)
    
    def _segment_path(self, segment_num: int) -> Path:
        """Get path for a segment number."""
        return self.wal_dir / f"{self.WAL_PREFIX}{segment_num:06d}{self.WAL_SUFFIX}"
    
    def _scan_last_sequence(self) -> int:
        """Scan current segment to find last sequence number."""
        path = self._segment_path(self._current_segment)
        if not path.exists():
            return 0
        
        last_seq = 0
        with open(path, "rb") as f:
            data = f.read()
            offset = 0
            while offset < len(data):
                try:
                    entry, consumed = WALEntry.deserialize(data, offset)
                    last_seq = entry.sequence_number
                    offset += consumed
                except (struct.error, ValueError):
                    break
        
        return last_seq
    
    def _get_current_file(self) -> Any:
        """Get or open current WAL file."""
        if self._current_file is None:
            path = self._segment_path(self._current_segment)
            self._current_file = open(path, "ab")
            self._current_file_size = path.stat().st_size if path.exists() else 0
        return self._current_file
    
    def _rotate_segment(self) -> None:
        """Rotate to a new WAL segment."""
        if self._current_file:
            self._current_file.close()
            self._current_file = None
        
        self._current_segment += 1
        self._current_file_size = 0
    
    def _write_entry(self, entry: WALEntry) -> None:
        """Write entry to WAL."""
        data = entry.serialize()
        
        # Check if rotation needed
        if self._current_file_size + len(data) > self.segment_max_size:
            self._rotate_segment()
        
        f = self._get_current_file()
        f.write(data)
        self._current_file_size += len(data)
        
        # Sync based on mode
        if self.sync_mode == "full":
            f.flush()
            os.fsync(f.fileno())
        elif self.sync_mode == "normal" and entry.entry_type == WALEntryType.TXN_COMMIT:
            f.flush()
            os.fsync(f.fileno())
    
    def _now_micros(self) -> int:
        """Get current timestamp in microseconds."""
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000)
    
    def log_insert(
        self,
        graph_id: int,
        subject_id: int,
        predicate_id: int,
        object_id: int,
        flags: int = 0,
        source_id: int = 0,
        confidence: float = 1.0,
        process_id: int = 0,
        txn_id: Optional[int] = None,
    ) -> int:
        """
        Log an insert operation.
        
        Returns:
            Sequence number of the logged entry
        """
        with self._lock:
            self._sequence += 1
            
            payload = InsertPayload(
                graph_id=graph_id,
                subject_id=subject_id,
                predicate_id=predicate_id,
                object_id=object_id,
                flags=flags,
                source_id=source_id,
                confidence=confidence,
                process_id=process_id,
            )
            
            entry = WALEntry(
                entry_type=WALEntryType.INSERT,
                sequence_number=self._sequence,
                transaction_id=txn_id or self._active_txn or 0,
                timestamp=self._now_micros(),
                payload=payload.serialize(),
            )
            
            self._write_entry(entry)
            return self._sequence
    
    def log_delete(
        self,
        graph_id: Optional[int] = None,
        subject_id: Optional[int] = None,
        predicate_id: Optional[int] = None,
        object_id: Optional[int] = None,
        txn_id: Optional[int] = None,
    ) -> int:
        """
        Log a delete operation.
        
        Returns:
            Sequence number of the logged entry
        """
        with self._lock:
            self._sequence += 1
            
            payload = DeletePayload(
                graph_id=graph_id,
                subject_id=subject_id,
                predicate_id=predicate_id,
                object_id=object_id,
            )
            
            entry = WALEntry(
                entry_type=WALEntryType.DELETE,
                sequence_number=self._sequence,
                transaction_id=txn_id or self._active_txn or 0,
                timestamp=self._now_micros(),
                payload=payload.serialize(),
            )
            
            self._write_entry(entry)
            return self._sequence
    
    def log_batch_insert(
        self,
        graph_ids: List[int],
        subject_ids: List[int],
        predicate_ids: List[int],
        object_ids: List[int],
        flags: int = 0,
        source_id: int = 0,
        confidence: float = 1.0,
        process_id: int = 0,
        txn_id: Optional[int] = None,
    ) -> int:
        """
        Log a batch insert operation (columnar format).
        
        More efficient than multiple log_insert calls for bulk ingestion.
        Single WAL entry for entire batch.
        
        Returns:
            Sequence number of the logged entry
        """
        with self._lock:
            self._sequence += 1
            
            payload = BatchInsertPayload(
                graph_ids=graph_ids,
                subject_ids=subject_ids,
                predicate_ids=predicate_ids,
                object_ids=object_ids,
                flags=flags,
                source_id=source_id,
                confidence=confidence,
                process_id=process_id,
            )
            
            entry = WALEntry(
                entry_type=WALEntryType.BATCH_INSERT,
                sequence_number=self._sequence,
                transaction_id=txn_id or self._active_txn or 0,
                timestamp=self._now_micros(),
                payload=payload.serialize(),
            )
            
            self._write_entry(entry)
            return self._sequence

    def begin_transaction(self) -> int:
        """
        Begin a new transaction.
        
        Returns:
            Transaction ID
        """
        with self._lock:
            if self._active_txn is not None:
                raise RuntimeError("Nested transactions not supported")
            
            self._sequence += 1
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            self._active_txn = txn_id
            
            entry = WALEntry(
                entry_type=WALEntryType.TXN_BEGIN,
                sequence_number=self._sequence,
                transaction_id=txn_id,
                timestamp=self._now_micros(),
                payload=b"",
            )
            
            self._write_entry(entry)
            return txn_id
    
    def commit_transaction(self, txn_id: int) -> None:
        """Commit a transaction."""
        with self._lock:
            if self._active_txn != txn_id:
                raise ValueError(f"Transaction {txn_id} is not active")
            
            self._sequence += 1
            
            entry = WALEntry(
                entry_type=WALEntryType.TXN_COMMIT,
                sequence_number=self._sequence,
                transaction_id=txn_id,
                timestamp=self._now_micros(),
                payload=b"",
            )
            
            self._write_entry(entry)
            self._active_txn = None
    
    def abort_transaction(self, txn_id: int) -> None:
        """Abort/rollback a transaction."""
        with self._lock:
            if self._active_txn != txn_id:
                raise ValueError(f"Transaction {txn_id} is not active")
            
            self._sequence += 1
            
            entry = WALEntry(
                entry_type=WALEntryType.TXN_ABORT,
                sequence_number=self._sequence,
                transaction_id=txn_id,
                timestamp=self._now_micros(),
                payload=b"",
            )
            
            self._write_entry(entry)
            self._active_txn = None
    
    def checkpoint(self) -> CheckpointState:
        """
        Create a checkpoint.
        
        This marks the current position as fully persisted,
        allowing earlier WAL segments to be recycled.
        """
        with self._lock:
            self._sequence += 1
            
            entry = WALEntry(
                entry_type=WALEntryType.CHECKPOINT,
                sequence_number=self._sequence,
                transaction_id=0,
                timestamp=self._now_micros(),
                payload=b"",
            )
            
            self._write_entry(entry)
            
            # Ensure file is synced
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())
            
            # Save checkpoint state
            state = CheckpointState(
                last_sequence=self._sequence,
                last_committed_txn=self._next_txn_id - 1,
                checkpoint_time=datetime.now(timezone.utc).isoformat(),
                wal_segment=self._current_segment,
            )
            
            checkpoint_path = self.wal_dir / self.CHECKPOINT_FILE
            with open(checkpoint_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            
            self._last_checkpoint = state
            return state
    
    def replay(
        self,
        from_sequence: Optional[int] = None,
    ) -> Iterator[tuple[WALEntry, Any]]:
        """
        Replay WAL entries for recovery.
        
        Args:
            from_sequence: Start from this sequence (exclusive).
                          If None, starts from last checkpoint.
        
        Yields:
            Tuple of (WALEntry, decoded_payload)
        """
        start_seq = from_sequence
        if start_seq is None and self._last_checkpoint:
            start_seq = self._last_checkpoint.last_sequence
        
        start_segment = 0
        if self._last_checkpoint:
            start_segment = self._last_checkpoint.wal_segment
        
        # Track transaction state for filtering
        committed_txns: set[int] = set()
        aborted_txns: set[int] = set()
        pending_entries: dict[int, list[tuple[WALEntry, Any]]] = {}
        
        # First pass: identify committed/aborted transactions
        for segment in self._list_segments():
            if segment < start_segment:
                continue
            
            path = self._segment_path(segment)
            if not path.exists():
                continue
            
            with open(path, "rb") as f:
                data = f.read()
                offset = 0
                while offset < len(data):
                    try:
                        entry, consumed = WALEntry.deserialize(data, offset)
                        offset += consumed
                        
                        if start_seq is not None and entry.sequence_number <= start_seq:
                            continue
                        
                        if entry.entry_type == WALEntryType.TXN_COMMIT:
                            committed_txns.add(entry.transaction_id)
                        elif entry.entry_type == WALEntryType.TXN_ABORT:
                            aborted_txns.add(entry.transaction_id)
                    except (struct.error, ValueError):
                        break
        
        # Second pass: yield entries from committed transactions (or auto-commit)
        for segment in self._list_segments():
            if segment < start_segment:
                continue
            
            path = self._segment_path(segment)
            if not path.exists():
                continue
            
            with open(path, "rb") as f:
                data = f.read()
                offset = 0
                while offset < len(data):
                    try:
                        entry, consumed = WALEntry.deserialize(data, offset)
                        offset += consumed
                        
                        if start_seq is not None and entry.sequence_number <= start_seq:
                            continue
                        
                        # Skip control entries
                        if entry.entry_type in (
                            WALEntryType.TXN_BEGIN,
                            WALEntryType.TXN_COMMIT,
                            WALEntryType.TXN_ABORT,
                            WALEntryType.CHECKPOINT,
                        ):
                            continue
                        
                        # Check transaction state
                        txn = entry.transaction_id
                        if txn != 0:  # Part of a transaction
                            if txn in aborted_txns:
                                continue  # Skip aborted
                            if txn not in committed_txns:
                                continue  # Skip uncommitted
                        
                        # Decode payload
                        payload = None
                        if entry.entry_type == WALEntryType.INSERT:
                            payload = InsertPayload.deserialize(entry.payload)
                        elif entry.entry_type == WALEntryType.DELETE:
                            payload = DeletePayload.deserialize(entry.payload)
                        
                        yield entry, payload
                        
                    except (struct.error, ValueError):
                        break
    
    def truncate_before(self, segment: int) -> int:
        """
        Remove WAL segments before the given segment.
        
        Returns:
            Number of segments removed
        """
        removed = 0
        for seg in self._list_segments():
            if seg < segment:
                path = self._segment_path(seg)
                if path.exists():
                    path.unlink()
                    removed += 1
        return removed
    
    def close(self) -> None:
        """Close the WAL."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()
                self._current_file.close()
                self._current_file = None
    
    def stats(self) -> dict:
        """Get WAL statistics."""
        segments = self._list_segments()
        total_size = sum(
            self._segment_path(s).stat().st_size
            for s in segments
            if self._segment_path(s).exists()
        )
        
        return {
            "sequence": self._sequence,
            "current_segment": self._current_segment,
            "segment_count": len(segments),
            "total_size_bytes": total_size,
            "active_transaction": self._active_txn,
            "last_checkpoint": self._last_checkpoint.to_dict() if self._last_checkpoint else None,
        }
