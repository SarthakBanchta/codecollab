import { Routes, Route, useNavigate } from 'react-router-dom'
import { v4 as uuidv4 } from 'uuid'
import EditorPage from './EditorPage'

function Home() {
  const navigate = useNavigate()

  const createRoom = () => {
    const roomId = uuidv4()
    navigate(`/room/${roomId}`)
  }

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#1e1e1e',
      color: 'white'
    }}>
      <h1 style={{ marginBottom: '8px' }}>CodeCollab</h1>
      <p style={{ color: '#888', marginBottom: '32px' }}>
        Real-time collaborative code editor
      </p>
      <button
        onClick={createRoom}
        style={{
          padding: '12px 28px',
          background: '#0078d4',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontSize: '16px',
          cursor: 'pointer'
        }}
      >
        Create New Room
      </button>
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/room/:roomId" element={<EditorPage />} />
    </Routes>
  )
}

export default App