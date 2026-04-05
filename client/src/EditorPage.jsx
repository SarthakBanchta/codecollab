import { useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { io } from 'socket.io-client'
import Editor from '@monaco-editor/react'

const socket = io(import.meta.env.VITE_SERVER_URL, {
  transports: ['websocket', 'polling']
})

const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
]

const LANGUAGES = [
  { id: 63, name: 'JavaScript', monacoLang: 'javascript' },
  { id: 71, name: 'Python',     monacoLang: 'python'     },
  { id: 54, name: 'C++',        monacoLang: 'cpp'        },
  { id: 62, name: 'Java',       monacoLang: 'java'       },
  { id: 50, name: 'C',          monacoLang: 'c'          },
]

function EditorPage() {
  const { roomId } = useParams()
  const editorRef = useRef(null)
  const monacoRef = useRef(null)
  const isRemoteChange = useRef(false)
  const currentVersion = useRef(0)
  const decorationIds = useRef({})
  const userColors = useRef({})
  const throttleTimer = useRef(null)

  const [users, setUsers] = useState([])
  const [selectedLang, setSelectedLang] = useState(LANGUAGES[0])
  const [output, setOutput] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [outputOpen, setOutputOpen] = useState(false)

  useEffect(() => {
    socket.emit('join-room', { roomId })

    socket.on('init-document', ({ content, version }) => {
      if (editorRef.current) {
        isRemoteChange.current = true
        editorRef.current.setValue(content)
        isRemoteChange.current = false
        currentVersion.current = version
      }
    })

    socket.on('code-change', ({ changes, version }) => {
      if (editorRef.current) {
        isRemoteChange.current = true
        editorRef.current.executeEdits('remote', changes)
        isRemoteChange.current = false
        currentVersion.current = version
      }
    })

    socket.on('ack', ({ version }) => {
      currentVersion.current = version
    })

    socket.on('room-users', (userList) => {
      setUsers(userList)
      userList.forEach((userId, index) => {
        if (!userColors.current[userId]) {
          userColors.current[userId] = COLORS[index % COLORS.length]
        }
      })
    })

    socket.on('cursor-move', ({ userId, position }) => {
      if (editorRef.current && monacoRef.current) {
        renderRemoteCursor(userId, position)
      }
    })

    socket.on('cursor-leave', ({ userId }) => {
      removeRemoteCursor(userId)
    })

    return () => {
      socket.off('init-document')
      socket.off('code-change')
      socket.off('ack')
      socket.off('room-users')
      socket.off('cursor-move')
      socket.off('cursor-leave')
    }
  }, [roomId])

  const renderRemoteCursor = (userId, position) => {
    const editor = editorRef.current
    const monaco = monacoRef.current
    if (!editor || !monaco) return

    const color = userColors.current[userId] || '#FF6B6B'
    const styleId = `cursor-style-${userId.slice(0, 8)}`

    if (!document.getElementById(styleId)) {
      const style = document.createElement('style')
      style.id = styleId
      style.innerHTML = `
        .cursor-${userId.slice(0, 8)} {
          border-left: 2px solid ${color};
          position: relative;
        }
        .cursor-${userId.slice(0, 8)}::after {
          content: '${userId.slice(0, 6)}';
          background: ${color};
          color: black;
          font-size: 10px;
          padding: 1px 4px;
          border-radius: 2px;
          position: absolute;
          top: -18px;
          left: 0;
          white-space: nowrap;
          pointer-events: none;
        }
      `
      document.head.appendChild(style)
    }

    const newDecorations = editor.deltaDecorations(
      decorationIds.current[userId] || [],
      [{
        range: new monaco.Range(
          position.lineNumber, position.column,
          position.lineNumber, position.column
        ),
        options: {
          className: `cursor-${userId.slice(0, 8)}`,
          stickiness: monaco.editor.TrackedRangeStickiness.NeverGrowsWhenTypingAtEdges
        }
      }]
    )
    decorationIds.current[userId] = newDecorations
  }

  const removeRemoteCursor = (userId) => {
    if (!editorRef.current) return
    editorRef.current.deltaDecorations(decorationIds.current[userId] || [], [])
    delete decorationIds.current[userId]
  }

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor
    monacoRef.current = monaco
    socket.emit('request-document', { roomId })

    editor.onDidChangeModelContent((event) => {
      if (isRemoteChange.current) return
      socket.emit('code-change', {
        roomId,
        changes: event.changes,
        baseVersion: currentVersion.current
      })
    })

    editor.onDidChangeCursorPosition((event) => {
      if (throttleTimer.current) return
      throttleTimer.current = setTimeout(() => {
        socket.emit('cursor-move', {
          roomId,
          position: {
            lineNumber: event.position.lineNumber,
            column: event.position.column
          }
        })
        throttleTimer.current = null
      }, 50)
    })
  }

  const handleLanguageChange = (e) => {
    const lang = LANGUAGES.find(l => l.id === parseInt(e.target.value))
    setSelectedLang(lang)
  }

  const runCode = async () => {
    if (!editorRef.current) return
    const code = editorRef.current.getValue()
    setIsRunning(true)
    setOutputOpen(true)
    setOutput('Running...')

    try {
      const res = await fetch('http://localhost:3001/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, languageId: selectedLang.id })
      })
      const data = await res.json()
      setOutput(`[${data.status}]\n\n${data.output}`)
    } catch {
      setOutput('Failed to connect to execution server.')
    }

    setIsRunning(false)
  }

  const copyLink = () => {
    navigator.clipboard.writeText(window.location.href)
    alert('Room link copied!')
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: '#1e1e1e' }}>
      {/* Toolbar */}
      <div style={{
        padding: '10px 20px',
        background: '#2d2d2d',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #444'
      }}>
        <h2 style={{ margin: 0, fontSize: '18px' }}>CodeCollab</h2>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <select
            value={selectedLang.id}
            onChange={handleLanguageChange}
            style={{
              padding: '5px 10px',
              background: '#3c3c3c',
              color: 'white',
              border: '1px solid #555',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {LANGUAGES.map(l => (
              <option key={l.id} value={l.id}>{l.name}</option>
            ))}
          </select>

          <button
            onClick={runCode}
            disabled={isRunning}
            style={{
              padding: '6px 18px',
              background: isRunning ? '#555' : '#2ea043',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              fontWeight: 'bold'
            }}
          >
            {isRunning ? 'Running...' : '▶ Run'}
          </button>

          <div style={{ display: 'flex', gap: '6px' }}>
            {users.map((userId, i) => (
              <div
                key={userId}
                title={userId}
                style={{
                  width: '28px', height: '28px',
                  borderRadius: '50%',
                  background: COLORS[i % COLORS.length],
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '11px', color: '#000', fontWeight: 'bold'
                }}
              >
                {userId.slice(0, 2).toUpperCase()}
              </div>
            ))}
          </div>

          <button
            onClick={copyLink}
            style={{
              padding: '6px 16px',
              background: '#0078d4',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Copy Link
          </button>
        </div>
      </div>

      {/* Editor + Output panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Editor
          height={outputOpen ? '65%' : '100%'}
          language={selectedLang.monacoLang}
          defaultValue="// Start coding..."
          theme="vs-dark"
          onMount={handleEditorDidMount}
        />

        {outputOpen && (
          <div style={{
            height: '35%',
            background: '#1a1a1a',
            borderTop: '1px solid #444',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{
              padding: '6px 16px',
              background: '#2d2d2d',
              color: '#aaa',
              fontSize: '12px',
              display: 'flex',
              justifyContent: 'space-between'
            }}>
              <span>OUTPUT</span>
              <span
                onClick={() => setOutputOpen(false)}
                style={{ cursor: 'pointer', color: '#ff6b6b' }}
              >
                ✕ Close
              </span>
            </div>
            <pre style={{
              flex: 1,
              margin: 0,
              padding: '12px 16px',
              color: '#d4d4d4',
              fontFamily: 'monospace',
              fontSize: '13px',
              overflowY: 'auto',
              whiteSpace: 'pre-wrap'
            }}>
              {output}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default EditorPage