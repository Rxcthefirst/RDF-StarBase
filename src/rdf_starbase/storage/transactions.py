"""
ACID Transaction support for RDF-StarBase.

Provides transactional semantics with:
- Atomicity: All changes in a transaction succeed or none do
- Consistency: Store remains valid after transaction
- Isolation: Concurrent readers see consistent snapshots
- Durability: Committed transactions are persisted

Uses Write-Ahead Log (WAL) for crash recovery and durability.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional
import time

if TYPE_CHECKING:
    from rdf_starbase.store import TripleStore

from rdf_starbase.storage.wal import WriteAheadLog, InsertPayload, DeletePayload
from rdf_starbase.storage.facts import FactFlags, DEFAULT_GRAPH_ID


class TransactionState(IntEnum):
    """Transaction lifecycle states."""
    PENDING = auto()      # Created but not started
    ACTIVE = auto()       # In progress
    COMMITTED = auto()    # Successfully completed
    ABORTED = auto()      # Rolled back
    FAILED = auto()       # Error during commit


class IsolationLevel(IntEnum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = 1  # Can see uncommitted changes (fast)
    READ_COMMITTED = 2    # See only committed changes (default)
    SNAPSHOT = 3          # Consistent snapshot at transaction start
    SERIALIZABLE = 4      # Full isolation (slowest)


@dataclass
class TransactionStats:
    """Statistics for a transaction."""
    txn_id: int
    start_time: float
    end_time: Optional[float] = None
    inserts: int = 0
    deletes: int = 0
    state: TransactionState = TransactionState.PENDING
    
    @property
    def duration_ms(self) -> float:
        """Transaction duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


@dataclass
class PendingChange:
    """A change staged for commit."""
    operation: str  # "insert" or "delete"
    payload: Any
    wal_seq: int  # WAL sequence number for this change


class Transaction:
    """
    A single ACID transaction.
    
    Usage:
        with store.transaction() as txn:
            txn.add_triple(s, p, o, source="...", confidence=0.9)
            # If exception raised, automatically rolls back
        # Commits automatically on clean exit
    """
    
    def __init__(
        self,
        store: "TripleStore",
        wal: WriteAheadLog,
        txn_id: int,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
    ):
        self._store = store
        self._wal = wal
        self._txn_id = txn_id
        self._isolation = isolation
        self._state = TransactionState.PENDING
        self._lock = Lock()
        
        # Pending changes
        self._inserts: list[PendingChange] = []
        self._deletes: list[PendingChange] = []
        
        # Stats
        self._stats = TransactionStats(
            txn_id=txn_id,
            start_time=time.time(),
        )
    
    @property
    def txn_id(self) -> int:
        return self._txn_id
    
    @property
    def state(self) -> TransactionState:
        return self._state
    
    @property
    def stats(self) -> TransactionStats:
        return self._stats
    
    def _check_active(self):
        """Ensure transaction is active."""
        if self._state != TransactionState.ACTIVE:
            raise RuntimeError(
                f"Transaction {self._txn_id} is {self._state.name}, not ACTIVE"
            )
    
    def begin(self) -> "Transaction":
        """Start the transaction."""
        with self._lock:
            if self._state != TransactionState.PENDING:
                raise RuntimeError(f"Cannot begin transaction in state {self._state.name}")
            
            # Log transaction begin to WAL
            self._wal.begin_transaction()
            self._state = TransactionState.ACTIVE
            return self
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: Any,
        source: str = "unknown",
        confidence: float = 1.0,
        process: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> None:
        """
        Add a triple within this transaction.
        
        The triple is logged to WAL but not applied to the store
        until commit is called.
        """
        with self._lock:
            self._check_active()
            
            # Intern terms (this is safe, term dict is append-only)
            s_id = self._store._intern_term(subject, is_uri_hint=True)
            p_id = self._store._intern_term(predicate, is_uri_hint=True)
            o_id = self._store._intern_term(obj)
            g_id = (self._store._term_dict.intern_iri(graph) 
                   if graph else DEFAULT_GRAPH_ID)
            
            source_id = (self._store._term_dict.intern_literal(source) 
                        if source else 0)
            process_id = (self._store._term_dict.intern_literal(process) 
                         if process else 0)
            
            # Log to WAL
            seq = self._wal.log_insert(
                graph_id=g_id,
                subject_id=s_id,
                predicate_id=p_id,
                object_id=o_id,
                flags=int(FactFlags.ASSERTED),
                source_id=source_id,
                confidence=confidence,
                process_id=process_id,
                txn_id=self._txn_id,
            )
            
            # Record pending change
            payload = InsertPayload(
                graph_id=g_id,
                subject_id=s_id,
                predicate_id=p_id,
                object_id=o_id,
                flags=int(FactFlags.ASSERTED),
                source_id=source_id,
                confidence=confidence,
                process_id=process_id,
            )
            self._inserts.append(PendingChange("insert", payload, seq))
            self._stats.inserts += 1
    
    def delete_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> None:
        """
        Mark triples for deletion within this transaction.
        
        Deletions are logged to WAL but not applied until commit.
        """
        with self._lock:
            self._check_active()
            
            # Look up term IDs
            s_id = None
            p_id = None
            o_id = None
            g_id = None
            
            if subject:
                s_id = self._store._term_dict.lookup_iri(subject)
            if predicate:
                p_id = self._store._term_dict.lookup_iri(predicate)
            if obj:
                o_id = self._store._term_dict.lookup_iri(obj)
                if o_id is None:
                    o_id = self._store._term_dict.lookup_literal(obj)
            if graph:
                g_id = self._store._term_dict.lookup_iri(graph)
            
            # Log to WAL
            seq = self._wal.log_delete(
                graph_id=g_id,
                subject_id=s_id,
                predicate_id=p_id,
                object_id=o_id,
                txn_id=self._txn_id,
            )
            
            # Record pending change
            payload = DeletePayload(
                graph_id=g_id,
                subject_id=s_id,
                predicate_id=p_id,
                object_id=o_id,
            )
            self._deletes.append(PendingChange("delete", payload, seq))
            self._stats.deletes += 1
    
    def commit(self) -> None:
        """
        Commit all pending changes.
        
        Changes are applied to the store only after WAL commit succeeds.
        """
        with self._lock:
            self._check_active()
            
            try:
                # 1. Log commit to WAL (ensures durability)
                self._wal.commit_transaction(self._txn_id)
                
                # 2. Apply inserts to store
                for change in self._inserts:
                    p = change.payload
                    self._store._fact_store.add_fact(
                        s=p.subject_id,
                        p=p.predicate_id,
                        o=p.object_id,
                        g=p.graph_id,
                        flags=FactFlags(p.flags),
                        source=p.source_id,
                        confidence=p.confidence,
                        process=p.process_id,
                    )
                
                # 3. Apply deletes to store
                for change in self._deletes:
                    p = change.payload
                    self._store._fact_store.mark_deleted(
                        s=p.subject_id,
                        p=p.predicate_id,
                        o=p.object_id,
                        g=p.graph_id,
                    )
                
                # 4. Invalidate caches
                self._store._invalidate_cache()
                
                self._state = TransactionState.COMMITTED
                self._stats.state = TransactionState.COMMITTED
                self._stats.end_time = time.time()
                
            except Exception as e:
                self._state = TransactionState.FAILED
                self._stats.state = TransactionState.FAILED
                self._stats.end_time = time.time()
                raise RuntimeError(f"Transaction commit failed: {e}") from e
    
    def rollback(self) -> None:
        """
        Abort and roll back all pending changes.
        
        Since changes aren't applied until commit, rollback just
        discards pending changes and logs abort to WAL.
        """
        with self._lock:
            if self._state not in (TransactionState.ACTIVE, TransactionState.PENDING):
                return  # Already finished
            
            try:
                if self._state == TransactionState.ACTIVE:
                    self._wal.abort_transaction(self._txn_id)
            except Exception:
                pass  # Best effort
            
            self._inserts.clear()
            self._deletes.clear()
            self._state = TransactionState.ABORTED
            self._stats.state = TransactionState.ABORTED
            self._stats.end_time = time.time()
    
    def __enter__(self) -> "Transaction":
        return self.begin()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Exception occurred - rollback
            self.rollback()
            return False  # Re-raise exception
        else:
            # Clean exit - commit
            self.commit()
            return False


class TransactionManager:
    """
    Manages transactions for a TripleStore.
    
    Features:
    - Transaction lifecycle management
    - WAL integration for durability
    - Crash recovery on startup
    - Checkpoint management
    """
    
    def __init__(
        self,
        store: "TripleStore",
        wal_dir: Optional[Path] = None,
        sync_mode: str = "normal",
        auto_checkpoint: bool = True,
        checkpoint_interval: int = 1000,
    ):
        """
        Initialize transaction manager.
        
        Args:
            store: The TripleStore to manage transactions for
            wal_dir: Directory for WAL files (None = disabled)
            sync_mode: WAL sync mode ("full", "normal", "off")
            auto_checkpoint: Create checkpoints automatically
            checkpoint_interval: Operations between checkpoints
        """
        self._store = store
        self._wal_dir = Path(wal_dir) if wal_dir else None
        self._sync_mode = sync_mode
        self._auto_checkpoint = auto_checkpoint
        self._checkpoint_interval = checkpoint_interval
        
        self._wal: Optional[WriteAheadLog] = None
        self._lock = RLock()
        self._next_txn_id = 1
        self._active_transactions: dict[int, Transaction] = {}
        self._operations_since_checkpoint = 0
        
        # Initialize WAL if directory provided
        if self._wal_dir:
            self._init_wal()
    
    def _init_wal(self) -> None:
        """Initialize WAL and perform crash recovery if needed."""
        self._wal = WriteAheadLog(
            self._wal_dir,
            sync_mode=self._sync_mode,
        )
        
        # Restore next transaction ID from WAL state
        stats = self._wal.stats()
        if stats["active_transaction"]:
            self._next_txn_id = stats["active_transaction"] + 1
    
    def recover(self) -> int:
        """
        Replay WAL to recover uncommitted changes.
        
        Call this after loading the store from disk to restore
        any committed but not-yet-persisted changes.
        
        Returns:
            Number of recovered entries applied
        """
        if not self._wal:
            return 0
        
        count = 0
        for entry, payload in self._wal.replay():
            if isinstance(payload, InsertPayload):
                self._store._fact_store.add_fact(
                    s=payload.subject_id,
                    p=payload.predicate_id,
                    o=payload.object_id,
                    g=payload.graph_id,
                    flags=FactFlags(payload.flags),
                    source=payload.source_id,
                    confidence=payload.confidence,
                    process=payload.process_id,
                )
                count += 1
            elif isinstance(payload, DeletePayload):
                self._store._fact_store.mark_deleted(
                    s=payload.subject_id,
                    p=payload.predicate_id,
                    o=payload.object_id,
                    g=payload.graph_id,
                )
                count += 1
        
        if count > 0:
            self._store._invalidate_cache()
        
        return count
    
    def begin(
        self,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
    ) -> Transaction:
        """
        Begin a new transaction.
        
        Args:
            isolation: Isolation level for the transaction
            
        Returns:
            A new Transaction object
        """
        with self._lock:
            if not self._wal:
                raise RuntimeError("WAL not initialized - cannot create transaction")
            
            txn_id = self._next_txn_id
            self._next_txn_id += 1
            
            txn = Transaction(
                store=self._store,
                wal=self._wal,
                txn_id=txn_id,
                isolation=isolation,
            )
            
            self._active_transactions[txn_id] = txn
            return txn
    
    @contextmanager
    def transaction(
        self,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
    ) -> Generator[Transaction, None, None]:
        """
        Context manager for transactions.
        
        Usage:
            with txn_manager.transaction() as txn:
                txn.add_triple(s, p, o, source="...")
            # Auto-commits on clean exit, rolls back on exception
        """
        txn = self.begin(isolation)
        try:
            txn.begin()
            yield txn
            txn.commit()
        except Exception:
            txn.rollback()
            raise
        finally:
            with self._lock:
                self._active_transactions.pop(txn.txn_id, None)
            self._maybe_checkpoint()
    
    def _maybe_checkpoint(self) -> None:
        """Create checkpoint if threshold reached."""
        if not self._auto_checkpoint or not self._wal:
            return
        
        self._operations_since_checkpoint += 1
        if self._operations_since_checkpoint >= self._checkpoint_interval:
            self.checkpoint()
    
    def checkpoint(self) -> None:
        """
        Create a WAL checkpoint.
        
        This marks the current state as durable, allowing
        old WAL segments to be truncated.
        """
        if not self._wal:
            return
        
        with self._lock:
            self._wal.checkpoint()
            self._operations_since_checkpoint = 0
    
    def close(self) -> None:
        """Close the transaction manager and WAL."""
        with self._lock:
            # Abort any active transactions
            for txn in list(self._active_transactions.values()):
                txn.rollback()
            self._active_transactions.clear()
            
            if self._wal:
                self._wal.close()
                self._wal = None
    
    def stats(self) -> dict:
        """Get transaction manager statistics."""
        return {
            "wal_enabled": self._wal is not None,
            "wal_stats": self._wal.stats() if self._wal else None,
            "active_transactions": len(self._active_transactions),
            "next_txn_id": self._next_txn_id,
            "operations_since_checkpoint": self._operations_since_checkpoint,
        }
