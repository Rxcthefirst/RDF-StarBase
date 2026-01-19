import { useRef, useEffect } from 'react'
import Editor from '@monaco-editor/react'

// SPARQL language definition for Monaco
const SPARQL_LANGUAGE = {
  defaultToken: '',
  ignoreCase: true,
  tokenPostfix: '.sparql',

  keywords: [
    'SELECT', 'DISTINCT', 'REDUCED', 'AS', 'CONSTRUCT', 'DESCRIBE', 'ASK',
    'FROM', 'NAMED', 'WHERE', 'ORDER', 'BY', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
    'OPTIONAL', 'GRAPH', 'UNION', 'FILTER', 'BIND', 'VALUES', 'MINUS',
    'GROUP', 'HAVING', 'SERVICE', 'SILENT', 'PREFIX', 'BASE',
    'INSERT', 'DELETE', 'DATA', 'INTO', 'LOAD', 'CLEAR', 'DROP', 'CREATE',
    'ADD', 'MOVE', 'COPY', 'WITH', 'USING', 'DEFAULT', 'ALL',
    'EXISTS', 'NOT', 'IN', 'IF', 'COALESCE', 'BOUND', 'BNODE', 'IRI', 'URI',
    'STR', 'LANG', 'LANGMATCHES', 'DATATYPE', 'STRLEN', 'SUBSTR', 'UCASE', 'LCASE',
    'STRSTARTS', 'STRENDS', 'CONTAINS', 'ENCODE_FOR_URI', 'CONCAT', 'REPLACE', 'REGEX',
    'ABS', 'ROUND', 'CEIL', 'FLOOR', 'RAND', 'NOW', 'YEAR', 'MONTH', 'DAY',
    'HOURS', 'MINUTES', 'SECONDS', 'TIMEZONE', 'TZ', 'MD5', 'SHA1', 'SHA256', 'SHA384', 'SHA512',
    'isIRI', 'isURI', 'isBLANK', 'isLITERAL', 'isNUMERIC', 'sameTerm',
    'COUNT', 'SUM', 'MIN', 'MAX', 'AVG', 'SAMPLE', 'GROUP_CONCAT', 'SEPARATOR',
    'true', 'false', 'a'
  ],

  operators: [
    '&&', '||', '!', '=', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '^'
  ],

  symbols: /[=><!~?:&|+\-*\/\^%]+/,

  tokenizer: {
    root: [
      // Prefixed names
      [/[a-zA-Z_][\w-]*:[\w-]*/, 'identifier.prefixed'],
      
      // Variables
      [/\?[\w]+/, 'variable'],
      [/\$[\w]+/, 'variable'],
      
      // URIs
      [/<[^>]*>/, 'string.uri'],
      
      // Keywords
      [/[a-zA-Z_][\w]*/, {
        cases: {
          '@keywords': 'keyword',
          '@default': 'identifier'
        }
      }],
      
      // Whitespace
      { include: '@whitespace' },
      
      // Strings
      [/"([^"\\]|\\.)*"/, 'string'],
      [/'([^'\\]|\\.)*'/, 'string'],
      
      // Numbers
      [/\d+\.\d*([eE][\-+]?\d+)?/, 'number.float'],
      [/\.\d+([eE][\-+]?\d+)?/, 'number.float'],
      [/\d+[eE][\-+]?\d+/, 'number.float'],
      [/\d+/, 'number'],
      
      // Delimiters
      [/[{}()\[\]]/, '@brackets'],
      [/[;,.]/, 'delimiter'],
      
      // Operators
      [/@symbols/, {
        cases: {
          '@operators': 'operator',
          '@default': ''
        }
      }],
    ],

    whitespace: [
      [/[ \t\r\n]+/, 'white'],
      [/#.*$/, 'comment'],
    ],
  },
}

// Dark theme for SPARQL
const SPARQL_DARK_THEME = {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'keyword', foreground: 'c586c0', fontStyle: 'bold' },
    { token: 'variable', foreground: '9cdcfe' },
    { token: 'string', foreground: 'ce9178' },
    { token: 'string.uri', foreground: '4ec9b0' },
    { token: 'number', foreground: 'b5cea8' },
    { token: 'comment', foreground: '6a9955' },
    { token: 'operator', foreground: 'd4d4d4' },
    { token: 'identifier.prefixed', foreground: 'dcdcaa' },
    { token: 'identifier', foreground: 'd4d4d4' },
  ],
  colors: {
    'editor.background': '#1e1e2e',
    'editor.foreground': '#cdd6f4',
    'editor.lineHighlightBackground': '#313244',
    'editorCursor.foreground': '#f5e0dc',
    'editor.selectionBackground': '#45475a',
  }
}

// Light theme for SPARQL
const SPARQL_LIGHT_THEME = {
  base: 'vs',
  inherit: true,
  rules: [
    { token: 'keyword', foreground: 'af00db', fontStyle: 'bold' },
    { token: 'variable', foreground: '001080' },
    { token: 'string', foreground: 'a31515' },
    { token: 'string.uri', foreground: '0070c1' },
    { token: 'number', foreground: '098658' },
    { token: 'comment', foreground: '008000' },
    { token: 'operator', foreground: '000000' },
    { token: 'identifier.prefixed', foreground: '795e26' },
    { token: 'identifier', foreground: '000000' },
  ],
  colors: {
    'editor.background': '#ffffff',
    'editor.foreground': '#1e1e1e',
  }
}

export default function SparqlEditor({ 
  value, 
  onChange, 
  onExecute,
  theme = 'dark',
  height = '200px',
  readOnly = false 
}) {
  const editorRef = useRef(null)
  const monacoRef = useRef(null)

  const handleEditorWillMount = (monaco) => {
    // Register SPARQL language
    monaco.languages.register({ id: 'sparql' })
    monaco.languages.setMonarchTokensProvider('sparql', SPARQL_LANGUAGE)
    
    // Register themes
    monaco.editor.defineTheme('sparql-dark', SPARQL_DARK_THEME)
    monaco.editor.defineTheme('sparql-light', SPARQL_LIGHT_THEME)
    
    // Register completion provider
    monaco.languages.registerCompletionItemProvider('sparql', {
      provideCompletionItems: (model, position) => {
        const word = model.getWordUntilPosition(position)
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn
        }
        
        const suggestions = [
          // Keywords
          ...['SELECT', 'WHERE', 'OPTIONAL', 'FILTER', 'UNION', 'MINUS', 'BIND', 
              'ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT', 'OFFSET', 'PREFIX',
              'INSERT DATA', 'DELETE DATA', 'CONSTRUCT', 'DESCRIBE', 'ASK',
              'VALUES', 'GRAPH', 'SERVICE', 'DISTINCT', 'REDUCED'].map(kw => ({
            label: kw,
            kind: monaco.languages.CompletionItemKind.Keyword,
            insertText: kw,
            range
          })),
          // Functions
          ...['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'SAMPLE', 'GROUP_CONCAT',
              'BOUND', 'IF', 'COALESCE', 'STR', 'LANG', 'DATATYPE',
              'STRLEN', 'SUBSTR', 'UCASE', 'LCASE', 'CONTAINS', 'STRSTARTS', 'STRENDS',
              'REGEX', 'REPLACE', 'isIRI', 'isLiteral', 'isBlank', 'isNumeric'].map(fn => ({
            label: fn,
            kind: monaco.languages.CompletionItemKind.Function,
            insertText: fn + '($0)',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range
          })),
          // Snippets
          {
            label: 'SELECT query',
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: 'SELECT ?s ?p ?o\nWHERE {\n  ?s ?p ?o .\n}\nLIMIT 100',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
            documentation: 'Basic SELECT query template'
          },
          {
            label: 'INSERT DATA',
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: 'PREFIX ex: <http://example.org/>\n\nINSERT DATA {\n  ex:subject ex:predicate "value" .\n}',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
            documentation: 'INSERT DATA template'
          },
          {
            label: 'CONSTRUCT query',
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: 'CONSTRUCT {\n  ?s ?p ?o .\n}\nWHERE {\n  ?s ?p ?o .\n}',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
            documentation: 'CONSTRUCT query template'
          },
        ]
        
        return { suggestions }
      }
    })
    
    monacoRef.current = monaco
  }

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor
    
    // Add keyboard shortcut for execution
    editor.addAction({
      id: 'execute-query',
      label: 'Execute Query',
      keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
      run: () => {
        if (onExecute) onExecute()
      }
    })
    
    // Format on Shift+Alt+F
    editor.addAction({
      id: 'format-query',
      label: 'Format Query',
      keybindings: [monaco.KeyMod.Shift | monaco.KeyMod.Alt | monaco.KeyCode.KeyF],
      run: (ed) => {
        // Basic formatting: uppercase keywords
        const text = ed.getValue()
        const keywords = ['SELECT', 'WHERE', 'FROM', 'PREFIX', 'OPTIONAL', 'UNION', 
                         'FILTER', 'BIND', 'VALUES', 'MINUS', 'GRAPH', 'ORDER', 'BY',
                         'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'INSERT', 'DELETE',
                         'DATA', 'CONSTRUCT', 'DESCRIBE', 'ASK', 'DISTINCT', 'AS']
        let formatted = text
        keywords.forEach(kw => {
          const regex = new RegExp(`\\b${kw}\\b`, 'gi')
          formatted = formatted.replace(regex, kw)
        })
        ed.setValue(formatted)
      }
    })
  }

  return (
    <Editor
      height={height}
      language="sparql"
      theme={theme === 'dark' ? 'sparql-dark' : 'sparql-light'}
      value={value}
      onChange={onChange}
      beforeMount={handleEditorWillMount}
      onMount={handleEditorDidMount}
      options={{
        minimap: { enabled: false },
        fontSize: 14,
        fontFamily: "'Fira Code', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace",
        fontLigatures: true,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 2,
        wordWrap: 'on',
        readOnly,
        padding: { top: 10, bottom: 10 },
        renderLineHighlight: 'line',
        cursorBlinking: 'smooth',
        smoothScrolling: true,
        contextmenu: true,
        quickSuggestions: true,
        suggestOnTriggerCharacters: true,
      }}
    />
  )
}
