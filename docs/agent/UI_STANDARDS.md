# RDF-StarBase — UI Development Standards

> Guidelines for building consistent, accessible, high-performance UI features

## Tech Stack (Locked)

| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| **Framework** | React | 18.x | Hooks-only, no class components |
| **Build** | Vite | 5.x | Fast HMR, ESM-first |
| **Visualization** | D3.js | 7.x | For graphs, charts, node-link diagrams |
| **Code Editor** | Monaco | 4.x | SPARQL, Turtle, RML syntax highlighting |
| **Icons** | Lucide React | Latest | Consistent icon set |
| **Styling** | CSS Variables | - | No CSS-in-JS; vanilla CSS with variables |

**Do NOT add:**
- CSS frameworks (Tailwind, Bootstrap) — we use semantic CSS
- State management libraries (Redux, Zustand) — React state + context is sufficient
- UI component libraries (MUI, Chakra) — we build our own for consistency

---

## Project Structure

```
src/ui/
├── src/
│   ├── components/           # Reusable UI components
│   │   ├── common/           # Buttons, inputs, modals, etc.
│   │   ├── visualization/    # D3-based graph components
│   │   ├── editors/          # Monaco-based editors
│   │   └── features/         # Feature-specific components
│   ├── hooks/                # Custom React hooks
│   ├── utils/                # Helper functions
│   ├── styles/               # Global CSS, variables, themes
│   ├── api/                  # API client functions
│   ├── App.jsx
│   └── main.jsx
├── public/
└── package.json
```

---

## Component Patterns

### File Naming
- Components: `PascalCase.jsx` (e.g., `GraphCanvas.jsx`)
- Hooks: `useCamelCase.js` (e.g., `useGraphLayout.js`)
- Utils: `camelCase.js` (e.g., `sparqlHelpers.js`)
- CSS: `component-name.css` (kebab-case, colocated with component)

### Component Structure
```jsx
// ============================================================================
// ComponentName — Brief description
// ============================================================================

import { useState, useCallback, useRef } from 'react'
import './component-name.css'

/**
 * ComponentName - Detailed description
 * 
 * @param {Object} props
 * @param {string} props.requiredProp - Description
 * @param {string} [props.optionalProp] - Description with default
 */
export function ComponentName({ 
  requiredProp,
  optionalProp = 'default',
  onAction,
  children 
}) {
  // 1. State declarations
  const [state, setState] = useState(null)
  
  // 2. Refs
  const containerRef = useRef(null)
  
  // 3. Callbacks (memoized)
  const handleClick = useCallback(() => {
    onAction?.(state)
  }, [state, onAction])
  
  // 4. Effects (if needed)
  
  // 5. Render
  return (
    <div 
      ref={containerRef}
      className="component-name"
      role="region"
      aria-label="Component description"
    >
      {children}
    </div>
  )
}

export default ComponentName
```

### Props Pattern
- Use destructuring with defaults
- Callbacks prefixed with `on` (e.g., `onClick`, `onSelect`)
- Boolean props: `isLoading`, `hasError`, `canEdit`
- Collections: plural names (`items`, `nodes`, `edges`)

---

## Design System

### CSS Variables (Theme Tokens)

```css
:root {
  /* Colors - Light Theme */
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f8f9fa;
  --color-bg-tertiary: #e9ecef;
  --color-bg-elevated: #ffffff;
  
  --color-text-primary: #212529;
  --color-text-secondary: #6c757d;
  --color-text-muted: #adb5bd;
  
  --color-border: #dee2e6;
  --color-border-focus: #0d6efd;
  
  --color-accent: #0d6efd;
  --color-accent-hover: #0b5ed7;
  --color-success: #198754;
  --color-warning: #ffc107;
  --color-error: #dc3545;
  
  /* Spacing Scale (8px base) */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  --space-2xl: 48px;
  
  /* Typography */
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  
  --text-xs: 11px;
  --text-sm: 13px;
  --text-md: 14px;
  --text-lg: 16px;
  --text-xl: 20px;
  --text-2xl: 24px;
  
  /* Borders & Radius */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-full: 9999px;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
  --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
  
  /* Transitions */
  --transition-fast: 150ms ease;
  --transition-normal: 250ms ease;
  
  /* Z-Index Scale */
  --z-dropdown: 100;
  --z-modal: 200;
  --z-tooltip: 300;
  --z-toast: 400;
}

/* Dark Theme Override */
[data-theme="dark"] {
  --color-bg-primary: #1a1a2e;
  --color-bg-secondary: #16213e;
  --color-bg-tertiary: #0f3460;
  --color-bg-elevated: #1f1f3a;
  
  --color-text-primary: #e9ecef;
  --color-text-secondary: #adb5bd;
  --color-text-muted: #6c757d;
  
  --color-border: #2d2d44;
  --color-border-focus: #4dabf7;
  
  --color-accent: #4dabf7;
  --color-accent-hover: #339af0;
}
```

### Semantic CSS Classes

```css
/* Buttons */
.btn { /* base button styles */ }
.btn-primary { /* accent color */ }
.btn-secondary { /* neutral */ }
.btn-danger { /* destructive actions */ }
.btn-ghost { /* minimal/icon buttons */ }
.btn-sm, .btn-lg { /* size variants */ }

/* Inputs */
.input { /* text inputs */ }
.input-error { /* validation error state */ }
.select { /* dropdowns */ }
.checkbox, .radio { /* form controls */ }

/* Layout */
.panel { /* card-like container */ }
.panel-header, .panel-body, .panel-footer { }
.toolbar { /* horizontal action bar */ }
.sidebar { /* vertical navigation */ }
.split-view { /* resizable panes */ }

/* States */
.is-loading { }
.is-disabled { }
.is-selected { }
.is-active { }
.has-error { }
```

---

## Accessibility Requirements

### WCAG 2.1 AA Compliance (Mandatory)

1. **Keyboard Navigation**
   - All interactive elements focusable via Tab
   - Logical focus order (no focus traps)
   - Visible focus indicators (`:focus-visible`)
   - Escape closes modals/dropdowns

2. **ARIA Labels**
   ```jsx
   // ✅ Good
   <button aria-label="Delete node" onClick={onDelete}>
     <TrashIcon />
   </button>
   
   // ❌ Bad
   <button onClick={onDelete}>
     <TrashIcon />
   </button>
   ```

3. **Color Contrast**
   - Text: minimum 4.5:1 ratio
   - Large text (18px+): minimum 3:1 ratio
   - UI components: minimum 3:1 ratio

4. **Screen Reader Support**
   - Announce dynamic content with `aria-live`
   - Use semantic HTML (`<nav>`, `<main>`, `<aside>`)
   - Label form inputs with `<label>` or `aria-labelledby`

5. **Motion**
   ```css
   @media (prefers-reduced-motion: reduce) {
     * {
       animation-duration: 0.01ms !important;
       transition-duration: 0.01ms !important;
     }
   }
   ```

---

## Performance Guidelines

### Rendering
- Memoize expensive computations with `useMemo`
- Memoize callbacks passed to children with `useCallback`
- Use `React.memo()` for pure components receiving complex props
- Virtualize long lists (100+ items) with windowing

### D3 Visualizations
```jsx
// ✅ Good: D3 only for calculations, React for DOM
const positions = useMemo(() => 
  d3.forceSimulation(nodes).stop().tick(100),
  [nodes]
)

return (
  <svg>
    {positions.map(node => (
      <circle key={node.id} cx={node.x} cy={node.y} />
    ))}
  </svg>
)

// ❌ Avoid: D3 direct DOM manipulation (breaks React)
useEffect(() => {
  d3.select(svgRef.current)
    .selectAll('circle')
    .data(nodes)
    .join('circle')
}, [nodes])
```

### Bundle Size
- Dynamic imports for heavy features:
  ```jsx
  const MonacoEditor = lazy(() => import('@monaco-editor/react'))
  ```
- Tree-shake D3 (import specific modules):
  ```js
  // ✅ Good
  import { forceSimulation, forceLink } from 'd3-force'
  
  // ❌ Bad
  import * as d3 from 'd3'
  ```

---

## Feature-Specific Standards

### Starchart (Visual RML Mapper)

```
┌─────────────────────────────────────────────────────────────┐
│ Starchart — RML/R2RML Visual Mapper                    [×]  │
├─────────────┬───────────────────────────────────────────────┤
│ Sources     │          Canvas (drag-drop mapping)           │
│ ─────────── │  ┌─────────┐     ┌─────────────────┐          │
│ □ users.csv │  │ Column  │────▶│ Ontology Class  │          │
│ □ orders.db │  │  [id]   │     │   foaf:Person   │          │
│             │  └─────────┘     └─────────────────┘          │
│ Ontology    │                                               │
│ ─────────── │                                               │
│ ▸ foaf:     │  ┌─────────┐     ┌─────────────────┐          │
│ ▸ schema:   │  │ [name]  │────▶│  foaf:name      │          │
│             │  └─────────┘     └─────────────────┘          │
├─────────────┴───────────────────────────────────────────────┤
│ [Validate Mapping] [Generate RML] [Test with Sample]        │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- `MappingCanvas` — D3 force-directed node-link diagram
- `SourcePanel` — File/database source browser
- `OntologyPanel` — Class/property tree
- `MappingNode` — Draggable column/property node
- `MappingEdge` — Connection with transformation options

**Interactions:**
- Drag column from source → drop on ontology property
- Double-click edge → edit transformation (CONCAT, SPLIT, etc.)
- Right-click node → context menu
- Ctrl+Z/Y → undo/redo

### ONTOP (Virtualized Data)

```
┌─────────────────────────────────────────────────────────────┐
│ ONTOP — Virtual Data Connections                            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ + Add Connection                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌────────────────────┐  ┌────────────────────┐              │
│ │ 🐘 production_db   │  │ 🐬 legacy_mysql    │              │
│ │ PostgreSQL 14      │  │ MySQL 8.0          │              │
│ │ ● Connected        │  │ ○ Disconnected     │              │
│ │ 12 tables mapped   │  │ [Connect]          │              │
│ │ [Manage] [Test]    │  │                    │              │
│ └────────────────────┘  └────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- `ConnectionCard` — Status, actions for each data source
- `ConnectionWizard` — Multi-step modal for new connections
- `SchemaInspector` — Browse tables/columns from source
- `QueryPreview` — Show generated SQL for SPARQL queries

### Protégé-like Ontology Editor

```
┌─────────────────────────────────────────────────────────────┐
│ Ontology Editor — myontology.owl                    [Save]  │
├─────────────┬───────────────────────────────────────────────┤
│ Classes     │ Class: foaf:Person                            │
│ ─────────── │ ────────────────────────────────────────────  │
│ ▾ owl:Thing │ URI: http://xmlns.com/foaf/0.1/Person         │
│   ▾ foaf:   │                                               │
│     Person ◀│ Superclass: owl:Thing                         │
│     Org     │                                               │
│   ▸ schema: │ Properties:                                   │
│             │ ┌──────────────┬──────────┬─────────────────┐ │
│ Properties  │ │ Property     │ Range    │ Cardinality     │ │
│ ─────────── │ ├──────────────┼──────────┼─────────────────┤ │
│ foaf:name   │ │ foaf:name    │ xsd:str  │ 1..*            │ │
│ foaf:knows  │ │ foaf:knows   │ Person   │ 0..*            │ │
│ foaf:mbox   │ │ foaf:mbox    │ xsd:any  │ 0..1            │ │
│             │ └──────────────┴──────────┴─────────────────┘ │
├─────────────┴───────────────────────────────────────────────┤
│ [+ Add Class] [+ Add Property] [Visualize] [Validate SHACL] │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- `ClassTree` — Hierarchical class browser with drag-reorder
- `PropertyList` — Domain/range editor
- `ClassEditor` — Detail panel for selected class
- `OntologyGraph` — D3 visualization of class hierarchy
- `RestrictionEditor` — OWL restrictions (someValuesFrom, etc.)

### Governance Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│ Governance — Data Quality & Compliance                      │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│ │  98.2%      │ │  3 pending  │ │  12 active  │             │
│ │  Quality    │ │  Changes    │ │  Policies   │             │
│ └─────────────┘ └─────────────┘ └─────────────┘             │
│                                                             │
│ Recent Activity                                             │
│ ─────────────────────────────────────────────────────────── │
│ ⚠ Schema change pending approval     [Review]   2h ago     │
│ ✓ Data import completed              [Details]  4h ago     │
│ ✓ SHACL validation passed                       yesterday  │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- `MetricCard` — KPI display with trend
- `ActivityFeed` — Timeline of governance events
- `PolicyEditor` — Rule builder for access/retention policies
- `ChangeRequest` — Approval workflow UI
- `LineageGraph` — PROV-O visualization

---

## Testing Standards

### Component Tests (Vitest + Testing Library)
```jsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { MappingNode } from './MappingNode'

describe('MappingNode', () => {
  it('renders label correctly', () => {
    render(<MappingNode label="user_id" type="column" />)
    expect(screen.getByText('user_id')).toBeInTheDocument()
  })

  it('calls onDragStart when dragged', () => {
    const onDragStart = vi.fn()
    render(<MappingNode label="test" onDragStart={onDragStart} />)
    fireEvent.dragStart(screen.getByRole('button'))
    expect(onDragStart).toHaveBeenCalled()
  })
})
```

### Visual Regression (Optional)
- Storybook for component catalog
- Chromatic or Percy for visual diff testing

---

## Error Handling

### API Errors
```jsx
const [error, setError] = useState(null)

const handleAction = async () => {
  try {
    setError(null)
    await api.performAction()
  } catch (err) {
    setError(err.message || 'An unexpected error occurred')
  }
}

return (
  <>
    {error && (
      <div role="alert" className="error-banner">
        {error}
        <button onClick={() => setError(null)} aria-label="Dismiss">
          <CloseIcon />
        </button>
      </div>
    )}
  </>
)
```

### Loading States
```jsx
// Always show loading indicator for async operations
{isLoading ? (
  <div className="loading-spinner" aria-label="Loading..." />
) : (
  <Content />
)}
```

### Empty States
```jsx
// Provide helpful empty states, not just blank
{items.length === 0 && (
  <div className="empty-state">
    <EmptyIcon />
    <h3>No mappings yet</h3>
    <p>Drag a column from the source panel to get started.</p>
    <button className="btn btn-primary">Add Source</button>
  </div>
)}
```

---

## Code Review Checklist

Before merging UI changes:

- [ ] Follows component structure pattern
- [ ] Uses CSS variables (no hardcoded colors/spacing)
- [ ] Keyboard accessible (Tab, Enter, Escape)
- [ ] ARIA labels on icon-only buttons
- [ ] Loading and error states handled
- [ ] Empty states are helpful
- [ ] No console warnings/errors
- [ ] Performance: no unnecessary re-renders
- [ ] Dark theme tested
- [ ] Mobile-responsive (if applicable)

---

## Quick Reference

| Need | Use |
|------|-----|
| Graph visualization | D3 force layout + React SVG |
| Code editing | Monaco Editor |
| Icons | Lucide React |
| Modals | Portal + focus trap |
| Tooltips | CSS `:hover` or Floating UI |
| Forms | Native HTML + CSS |
| Tables | `<table>` with sticky headers |
| Drag & drop | Native HTML5 Drag API |
| Undo/redo | Command pattern with history stack |

---

*Last updated: February 2026*
