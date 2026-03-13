# Repository Stats Optimization Guide

## Problem

When displaying multiple repositories in the UI, making API calls to fetch stats for ALL repositories doesn't scale well:

- **Previous behavior**: `/repositories` endpoint computed live stats for all loaded repositories
- **Performance impact**: With large datasets (e.g., GLEIF with 30M+ triples), computing stats for each loaded repository was expensive
- **Scalability issue**: Each stats computation scans the fact store to count unique subjects, predicates, etc.

## Solution

Implemented a three-tier strategy for repository stats:

### 1. Lightweight Repository Listing

**API**: `GET /repositories?include_stats=false` (default)

```bash
curl http://localhost:8000/api/repositories
```

Returns repository metadata with **persisted stats** (fast, no computation):
- Uses cached stats from last save
- No live computation for loaded repositories
- Ideal for initial dashboard load

### 2. Bulk Stats Refresh (On-Demand)

**API**: `GET /repositories/bulk-stats`

```bash
curl http://localhost:8000/api/repositories/bulk-stats
```

Returns **live stats** for all repositories in a single call:
- Checks which repositories are loaded in memory
- Computes live stats only for loaded repositories
- Returns persisted stats for unloaded repositories
- Avoids N separate API calls

Response format:
```json
{
  "repositories": {
    "MyRepo": {
      "name": "MyRepo",
      "triple_count": 30000000,
      "subject_count": 5000000,
      "predicate_count": 150,
      "loaded": true
    },
    "AnotherRepo": {
      "name": "AnotherRepo", 
      "triple_count": 1000,
      "subject_count": 200,
      "predicate_count": 10,
      "loaded": false
    }
  }
}
```

### 3. Stats Caching

**Implementation**: `TripleStore.stats()`

Stats computation is cached in memory:
- Cache is computed on first call
- Cache is invalidated only when store is modified (add/delete operations)
- Subsequent calls return cached results

## UI Implementation

The Dashboard now uses a smart refresh strategy:

### Initial Load
```javascript
// Load repositories with persisted stats (fast)
const loadRepositories = async () => {
  const data = await fetchJson('/repositories')
  setRepositories(data.repositories)
}
```

### Manual Refresh
```javascript
// Refresh live stats for all repos (on-demand)
const refreshAllStats = async () => {
  const data = await fetchJson('/repositories/bulk-stats')
  const statsMap = data.repositories
  
  // Merge with existing repositories
  setRepositories(prev => prev.map(repo => ({
    ...repo,
    triple_count: statsMap[repo.name].triple_count,
    subject_count: statsMap[repo.name].subject_count,
    predicate_count: statsMap[repo.name].predicate_count,
    loaded: statsMap[repo.name].loaded,
  })))
}
```

### UI Controls

**Refresh All Stats Button**: 
- Located in Dashboard repository list header
- Calls `bulk-stats` endpoint to update all repository cards
- Shows loading state during refresh

**Individual Repo Stats**:
- Detailed stats panel shows live stats for selected repository
- Has its own refresh button for that repository

## Performance Comparison

### Before Optimization

```
/repositories call with 5 loaded repos (each with 10M+ triples):
- Computes stats for all 5 repos
- 5 × ~200ms per stats() call = ~1 second
- Called on every dashboard load and refresh
```

### After Optimization

```
Initial load:
- /repositories (include_stats=false): ~10ms (metadata only)
- No stats computation

User clicks "Refresh Stats":
- /repositories/bulk-stats: ~1 second (computed once)
- Stats displayed until next manual refresh
```

**Result**: Dashboard loads 100x faster, stats refresh is user-controlled

## Best Practices

### For UI Developers

1. **Use `/repositories` for initial load** - Shows persisted stats (good enough for dashboard)
2. **Add manual refresh** - Let users trigger bulk-stats when needed
3. **Don't poll stats** - Stats don't change frequently enough to justify polling
4. **Cache in state** - Store bulk-stats results in React state, refresh only on user action

### For API Developers

1. **Make endpoints optional** - Use query parameters (e.g., `?include_stats=true`) for expensive operations
2. **Provide bulk endpoints** - One `/bulk-stats` call is better than N `/stats` calls
3. **Cache aggressively** - Stats change only when data changes, cache until invalidated
4. **Document performance** - Explain when to use each endpoint

## Migration Guide

If you have existing UI code making individual stats calls:

### ❌ Before (N+1 query problem)
```javascript
// Load all repositories
const repos = await fetch('/api/repositories').then(r => r.json())

// Fetch stats for each (N separate calls!)
for (const repo of repos.repositories) {
  const stats = await fetch(`/api/repositories/${repo.name}/stats`).then(r => r.json())
  repo.stats = stats
}
```

### ✅ After (Single bulk call)
```javascript
// Load repositories (metadata only)
const repos = await fetch('/api/repositories').then(r => r.json())

// Fetch all stats in one call
const bulkStats = await fetch('/api/repositories/bulk-stats').then(r => r.json())

// Merge stats
const enriched = repos.repositories.map(repo => ({
  ...repo,
  ...bulkStats.repositories[repo.name]
}))
```

## API Reference

### GET /repositories

**Query Parameters**:
- `include_stats` (boolean, default: `false`) - Compute live stats for loaded repositories

**Response**:
```json
{
  "count": 2,
  "repositories": [
    {
      "name": "MyRepo",
      "description": "My knowledge graph",
      "triple_count": 1000,
      "subject_count": 200,
      "predicate_count": 10,
      "reasoning_level": "rdfs",
      "created_at": "2025-01-15T12:00:00"
    }
  ]
}
```

### GET /repositories/bulk-stats

**Response**:
```json
{
  "repositories": {
    "MyRepo": {
      "name": "MyRepo",
      "description": "My knowledge graph",
      "triple_count": 30000000,
      "subject_count": 5000000,
      "predicate_count": 150,
      "object_count": 8000000,
      "graph_count": 1,
      "source_count": 50,
      "loaded": true,
      "created_at": "2025-01-15T12:00:00"
    }
  }
}
```

## Common Issues

### Stats are stale

**Symptom**: Dashboard shows old triple counts after importing data

**Solution**: Click the "Refresh Stats" button to trigger bulk-stats call

### Stats refresh is slow

**Symptom**: Bulk-stats takes several seconds

**Cause**: Multiple large repositories are loaded in memory

**Solution**: This is expected - computing stats for 30M+ triples takes time. Stats are cached, so subsequent refreshes are instant until data changes.

### Stats don't update after import

**Symptom**: Imported data but stats unchanged

**Cause**: Stats cache not invalidated, or using persisted stats

**Solution**: 
1. Ensure `_invalidate_cache()` is called after imports
2. Click "Refresh Stats" in UI to fetch live stats
3. Save repository to persist updated stats

## Future Enhancements

Potential improvements for even better performance:

1. **Background stats computation** - Compute stats asynchronously after imports
2. **Incremental stats** - Update stats incrementally on adds/deletes instead of full recomputation
3. **Stats snapshots** - Periodically save stats snapshots to avoid cold-start computation
4. **Approximate stats** - For very large datasets, use sampling for faster (approximate) counts
5. **Stats streaming** - Stream stats as they're computed for responsive UI

## Related Documentation

- [BULK_LOAD_GUIDE.md](BULK_LOAD_GUIDE.md) - Bulk loading large datasets
- [CRASH_RECOVERY.md](CRASH_RECOVERY.md) - Recovery performance with large triple counts
- [architecture/storage.md](architecture/storage.md) - Storage architecture and indexing
