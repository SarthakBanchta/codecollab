import { useEffect, useRef } from 'react'
import { io } from 'socket.io-client'
import Editor from '@monaco-editor/react'

const socket = io('http://localhost:3001')

function App() {
  const editorRef = useRef(null)
  const isRemoteChange = useRef(false)

  useEffect(() => {
    socket.on('code-change', ({ changes }) => {
      if (editorRef.current) {
        isRemoteChange.current = true
        editorRef.current.executeEdits('remote', changes)
        isRemoteChange.current = false
      }
    })

    socket.on('init-document', (content) => {
      if (editorRef.current) {
        isRemoteChange.current = true
        editorRef.current.setValue(content)
        isRemoteChange.current = false
      }
    })

    return () => {
      socket.off('code-change')
      socket.off('init-document')
    }
  }, [])

  const handleEditorDidMount = (editor) => {
    editorRef.current = editor

    // Editor is ready — now ask server for current document
    socket.emit('request-document')

    editor.onDidChangeModelContent((event) => {
      if (isRemoteChange.current) return
      socket.emit('code-change', { changes: event.changes })
    })
  }

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '12px 20px', background: '#1e1e1e', color: 'white' }}>
        <h2 style={{ margin: 0 }}>CodeCollab</h2>
      </div>
      <Editor
        height="100%"
        defaultLanguage="javascript"
        defaultValue="// Start coding..."
        theme="vs-dark"
        onMount={handleEditorDidMount}
      />
    </div>
  )
}

export default App