import { useState, useRef } from 'react'
import { CloseIcon } from './Icons'

// API base URL - in dev mode with Vite, use proxy; in production, use root
const API_BASE = import.meta.env.DEV ? '/api' : ''

export default function ImportExport({ repositoryName, onDataChanged, theme }) {
  const [activeTab, setActiveTab] = useState('import')
  const [importing, setImporting] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [importText, setImportText] = useState('')
  const [importFormat, setImportFormat] = useState('turtle')
  const [exportFormat, setExportFormat] = useState('turtle')
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(null)
  const fileInputRef = useRef(null)

  const isDark = theme === 'dark'

  const handleFileSelect = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    
    setSelectedFile(file)
    setError(null)
    setSuccess(null)
    
    // Auto-detect format from extension
    const ext = file.name.split('.').pop()?.toLowerCase()
    const formatMap = {
      'ttl': 'turtle',
      'turtle': 'turtle',
      'nt': 'ntriples',
      'ntriples': 'ntriples',
      'nq': 'nquads',
      'nquads': 'nquads',
      'n3': 'n3',
      'trig': 'trig',
      'trix': 'trix',
      'rdf': 'rdfxml',
      'xml': 'rdfxml',
      'rdfxml': 'rdfxml',
      'jsonld': 'jsonld',
      'json': 'jsonld',
      'ndjsonld': 'ndjsonld',
      'ndjson': 'ndjsonld',
      'rdfjson': 'rdfjson',
      'brf': 'binaryrdf',
    }
    const detectedFormat = formatMap[ext] || 'turtle'
    setImportFormat(detectedFormat)
  }

  // Fast file upload using multipart form
  const handleFileUpload = async () => {
    if (!repositoryName || !selectedFile) return

    try {
      setImporting(true)
      setError(null)
      setSuccess(null)
      setUploadProgress('Uploading...')

      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('format', importFormat)

      const response = await fetch(`${API_BASE}/repositories/${repositoryName}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || 'Upload failed')
      }

      const result = await response.json()
      const timing = result.timing || {}
      setSuccess(
        `✓ Imported ${result.triples_added?.toLocaleString() || 0} triples\n` +
        `⚡ ${timing.triples_per_second?.toLocaleString() || 0} triples/sec\n` +
        `⏱ Parse: ${timing.parse_seconds || 0}s, Insert: ${timing.insert_seconds || 0}s`
      )
      setSelectedFile(null)
      setUploadProgress(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
      if (onDataChanged) onDataChanged()
    } catch (err) {
      setError(err.message)
      setUploadProgress(null)
    } finally {
      setImporting(false)
    }
  }

  // Text-based import (for pasted data)
  const handleImport = async () => {
    if (!repositoryName || !importText.trim()) return

    try {
      setImporting(true)
      setError(null)
      setSuccess(null)

      const response = await fetch(`${API_BASE}/repositories/${repositoryName}/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: importText,
          format: importFormat
        })
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || 'Import failed')
      }

      const result = await response.json()
      setSuccess(`Imported ${result.triples_added || 0} triples successfully!`)
      setImportText('')
      if (onDataChanged) onDataChanged()
    } catch (err) {
      setError(err.message)
    } finally {
      setImporting(false)
    }
  }

  const handleExport = async () => {
    if (!repositoryName) return

    try {
      setExporting(true)
      setError(null)

      const response = await fetch(
        `${API_BASE}/repositories/${repositoryName}/export?format=${exportFormat}`
      )

      if (!response.ok) {
        throw new Error('Export failed')
      }

      const data = await response.text()
      
      // Create download
      const blob = new Blob([data], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      
      const extensions = {
        turtle: 'ttl',
        ntriples: 'nt',
        rdfxml: 'rdf',
        jsonld: 'jsonld'
      }
      a.download = `${repositoryName}.${extensions[exportFormat] || 'txt'}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      setSuccess(`Exported data as ${exportFormat}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className={`import-export ${isDark ? 'dark' : 'light'}`}>
      <div className="ie-tabs">
        <button 
          className={`ie-tab ${activeTab === 'import' ? 'active' : ''}`}
          onClick={() => setActiveTab('import')}
        >
          Import
        </button>
        <button 
          className={`ie-tab ${activeTab === 'export' ? 'active' : ''}`}
          onClick={() => setActiveTab('export')}
        >
          Export
        </button>
      </div>

      {(error || success) && (
        <div className={`ie-message ${error ? 'error' : 'success'}`}>
          {error || success}
          <button className="close-btn" onClick={() => { setError(null); setSuccess(null) }}>
            <CloseIcon size={14} />
          </button>
        </div>
      )}

      {activeTab === 'import' && (
        <div className="ie-content">
          <div className="ie-row">
            <label>Format:</label>
            <select 
              value={importFormat} 
              onChange={e => setImportFormat(e.target.value)}
              className="ie-select"
            >
              <optgroup label="Turtle Family">
                <option value="turtle">Turtle / Turtle★ (.ttl)</option>
                <option value="n3">Notation3 (.n3)</option>
                <option value="trig">TriG / TriG★ (.trig)</option>
              </optgroup>
              <optgroup label="N-Triples Family">
                <option value="ntriples">N-Triples / N-Triples★ (.nt)</option>
                <option value="nquads">N-Quads (.nq)</option>
              </optgroup>
              <optgroup label="JSON Formats">
                <option value="jsonld">JSON-LD (.jsonld)</option>
                <option value="ndjsonld">NDJSON-LD (.ndjsonld)</option>
                <option value="rdfjson">RDF/JSON (.json)</option>
              </optgroup>
              <optgroup label="XML Formats">
                <option value="rdfxml">RDF/XML (.rdf)</option>
                <option value="trix">TriX (.trix)</option>
              </optgroup>
              <optgroup label="Binary">
                <option value="binaryrdf">Binary RDF (.brf)</option>
              </optgroup>
            </select>
          </div>

          {/* File Upload */}
          <div className="ie-file-row">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".ttl,.nt,.nq,.n3,.trig,.trix,.rdf,.xml,.jsonld,.json,.ndjsonld,.ndjson,.rdfjson,.brf"
              style={{ display: 'none' }}
            />
            <button 
              className="ie-btn secondary"
              onClick={() => fileInputRef.current?.click()}
            >
              Choose File
            </button>
            <span className="ie-file-name">
              {selectedFile ? `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)` : 'No file selected'}
            </span>
          </div>

          {selectedFile && (
            <button 
              className="ie-btn primary"
              onClick={handleFileUpload}
              disabled={importing || !repositoryName}
            >
              {importing ? (uploadProgress || 'Importing...') : 'Import File'}
            </button>
          )}

          {selectedFile && !repositoryName && (
            <div className="ie-warning">Select a repository first</div>
          )}

          <div className="ie-divider">
            <span>or paste data</span>
          </div>

          <textarea
            className="ie-textarea"
            value={importText}
            onChange={e => setImportText(e.target.value)}
            placeholder={`Paste ${importFormat.toUpperCase()} data here...`}
            rows={8}
          />

          <button 
            className="ie-btn secondary"
            onClick={handleImport}
            disabled={importing || !importText.trim() || !repositoryName}
          >
            {importing ? 'Importing...' : 'Import Pasted Data'}
          </button>
        </div>
      )}

      {activeTab === 'export' && (
        <div className="ie-content">
          <div className="ie-row">
            <label>Format:</label>
            <select 
              value={exportFormat} 
              onChange={e => setExportFormat(e.target.value)}
              className="ie-select"
            >
              <option value="turtle">Turtle (.ttl)</option>
              <option value="ntriples">N-Triples (.nt)</option>
              <option value="rdfxml">RDF/XML (.rdf)</option>
              <option value="jsonld">JSON-LD (.jsonld)</option>
            </select>
          </div>

          <p className="ie-description">
            Export all triples from <strong>{repositoryName || 'the repository'}</strong> 
            in the selected format.
          </p>

          <button 
            className="ie-btn primary"
            onClick={handleExport}
            disabled={exporting || !repositoryName}
          >
            <DownloadIcon size={16} />
            {exporting ? 'Exporting...' : 'Download Export'}
          </button>
        </div>
      )}

      <style>{`
        .import-export {
          padding: 1rem;
          height: 100%;
          display: flex;
          flex-direction: column;
        }
        
        .import-export.dark {
          background: #1e1e2e;
          color: #cdd6f4;
        }
        
        .import-export.light {
          background: #f8f9fa;
          color: #1e1e1e;
        }
        
        .ie-tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }
        
        .ie-tab {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.6rem;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.85rem;
          font-weight: 500;
          transition: all 0.2s;
        }
        
        .import-export.dark .ie-tab {
          background: #313244;
          color: #a6adc8;
        }
        
        .import-export.light .ie-tab {
          background: #e9ecef;
          color: #495057;
        }
        
        .ie-tab.active {
          background: var(--accent-color) !important;
          color: white !important;
        }
        
        .ie-message {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.5rem 0.75rem;
          border-radius: 6px;
          margin-bottom: 1rem;
          font-size: 0.85rem;
        }
        
        .ie-message.error {
          background: rgba(243, 139, 168, 0.2);
          color: #f38ba8;
        }
        
        .ie-message.success {
          background: rgba(166, 227, 161, 0.2);
          color: #a6e3a1;
        }
        
        .ie-message .close-btn {
          background: none;
          border: none;
          color: inherit;
          cursor: pointer;
          opacity: 0.7;
        }
        
        .ie-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        
        .ie-row {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }
        
        .ie-row label {
          font-size: 0.85rem;
          font-weight: 500;
          min-width: 60px;
        }
        
        .ie-select {
          flex: 1;
          padding: 0.5rem;
          border-radius: 6px;
          font-size: 0.85rem;
          border: 1px solid var(--border-color);
        }
        
        .import-export.dark .ie-select {
          background: #313244;
          color: #cdd6f4;
          border-color: #45475a;
        }
        
        .import-export.light .ie-select {
          background: white;
          color: #1e1e1e;
          border-color: #dee2e6;
        }
        
        .ie-textarea {
          flex: 1;
          padding: 0.75rem;
          border-radius: 6px;
          font-family: 'Fira Code', monospace;
          font-size: 0.85rem;
          resize: none;
          border: 1px solid var(--border-color);
        }
        
        .import-export.dark .ie-textarea {
          background: #313244;
          color: #cdd6f4;
          border-color: #45475a;
        }
        
        .import-export.light .ie-textarea {
          background: white;
          color: #1e1e1e;
          border-color: #dee2e6;
        }
        
        .ie-textarea:focus {
          outline: none;
          border-color: var(--accent-color);
        }
        
        .ie-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.6rem 1rem;
          border: none;
          border-radius: 6px;
          font-size: 0.85rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .ie-btn.primary {
          background: var(--accent-color);
          color: white;
        }
        
        .ie-btn.primary:hover:not(:disabled) {
          filter: brightness(1.1);
        }
        
        .ie-btn.secondary {
          background: transparent;
          border: 1px dashed var(--border-color);
        }
        
        .import-export.dark .ie-btn.secondary {
          color: #a6adc8;
          border-color: #45475a;
        }
        
        .import-export.light .ie-btn.secondary {
          color: #495057;
          border-color: #dee2e6;
        }
        
        .ie-btn.secondary:hover:not(:disabled) {
          border-style: solid;
        }
        
        .ie-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .ie-description {
          font-size: 0.85rem;
          opacity: 0.8;
          line-height: 1.5;
        }
        
        .ie-file-row {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }
        
        .ie-file-name {
          font-size: 0.85rem;
          opacity: 0.8;
        }
        
        .ie-divider {
          display: flex;
          align-items: center;
          text-align: center;
          margin: 0.5rem 0;
        }
        
        .ie-divider::before,
        .ie-divider::after {
          content: '';
          flex: 1;
          border-bottom: 1px solid var(--border-color);
        }
        
        .ie-divider span {
          padding: 0 0.75rem;
          font-size: 0.75rem;
          opacity: 0.6;
        }
        
        .ie-message.success {
          white-space: pre-line;
        }
        
        .ie-warning {
          font-size: 0.85rem;
          color: #f9a825;
          padding: 0.5rem;
          text-align: center;
        }
      `}</style>
    </div>
  )
}
