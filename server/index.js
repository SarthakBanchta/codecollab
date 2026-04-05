require('dotenv').config()
const express = require('express')
const http = require('http')
const { Server } = require('socket.io')
const cors = require('cors')

const app = express()
app.use(cors())

const axios = require('axios')
app.use(express.json())

app.post('/execute', async (req, res) => {
  const { code, languageId } = req.body

  try {
    // Submit code to Judge0
    const submitRes = await axios.post(
      'https://ce.judge0.com/submissions?wait=true',
      {
        source_code: code,
        language_id: languageId,
        stdin: ''
      },
      {
        headers: { 'Content-Type': 'application/json' }
      }
    )

    const { stdout, stderr, compile_output, status } = submitRes.data

    res.json({
      output: stdout || stderr || compile_output || 'No output',
      status: status.description
    })

  } catch (err) {
    res.status(500).json({ output: 'Execution failed. Try again.', status: 'Error' })
  }
})

const server = http.createServer(app)

const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  },
  transports: ['websocket', 'polling']
})

const rooms = {}

function getRoom(roomId) {
  if (!rooms[roomId]) {
    rooms[roomId] = {
      document: '// Start coding...',
      history: [],
      users: []
    }
  }
  return rooms[roomId]
}

io.on('connection', (socket) => {
  console.log('User connected:', socket.id)
  let currentRoom = null

  socket.on('join-room', ({ roomId }) => {
    currentRoom = roomId
    socket.join(roomId)
    const room = getRoom(roomId)
    if (!room.users.includes(socket.id)) {
      room.users.push(socket.id)
    }
    io.to(roomId).emit('room-users', room.users)
    console.log(`${socket.id} joined room ${roomId}`)
  })

  socket.on('request-document', ({ roomId }) => {
    const room = getRoom(roomId)
    socket.emit('init-document', {
      content: room.document,
      version: room.history.length
    })
  })

  socket.on('code-change', ({ roomId, changes, baseVersion }) => {
    const room = getRoom(roomId)
    const concurrentOps = room.history.slice(baseVersion)

    let transformedChanges = changes
    for (const serverOp of concurrentOps) {
      transformedChanges = transformedChanges.map(change =>
        transformChange(change, serverOp, room.document)
      )
    }

    transformedChanges.forEach(change => {
      room.document = applyChange(room.document, change)
    })

    room.history.push(...transformedChanges)

    socket.to(roomId).emit('code-change', {
      changes: transformedChanges,
      version: room.history.length
    })

    socket.emit('ack', { version: room.history.length })
  })

  socket.on('cursor-move', ({ roomId, position }) => {
    socket.to(roomId).emit('cursor-move', {
      userId: socket.id,
      position
    })
  })

  socket.on('disconnect', () => {
    if (currentRoom && rooms[currentRoom]) {
      rooms[currentRoom].users = rooms[currentRoom].users.filter(
        id => id !== socket.id
      )
      io.to(currentRoom).emit('room-users', rooms[currentRoom].users)
      io.to(currentRoom).emit('cursor-leave', { userId: socket.id })
    }
    console.log('User disconnected:', socket.id)
  })
})

function transformChange(opA, opB, document) {
  const lines = document.split('\n')

  function toOffset(line, column) {
    let offset = 0
    for (let i = 0; i < line - 1; i++) {
      offset += lines[i].length + 1
    }
    return offset + column - 1
  }

  function toRange(start, end) {
    let offset = 0
    let startLine = 1, startCol = 1, endLine = 1, endCol = 1
    for (let i = 0; i < lines.length; i++) {
      const lineEnd = offset + lines[i].length
      if (offset <= start && start <= lineEnd) {
        startLine = i + 1
        startCol = start - offset + 1
      }
      if (offset <= end && end <= lineEnd) {
        endLine = i + 1
        endCol = end - offset + 1
      }
      offset += lines[i].length + 1
    }
    return {
      startLineNumber: startLine,
      startColumn: startCol,
      endLineNumber: endLine,
      endColumn: endCol
    }
  }

  const aStart = toOffset(opA.range.startLineNumber, opA.range.startColumn)
  const aEnd = toOffset(opA.range.endLineNumber, opA.range.endColumn)
  const bStart = toOffset(opB.range.startLineNumber, opB.range.startColumn)
  const bEnd = toOffset(opB.range.endLineNumber, opB.range.endColumn)
  const bNetChange = opB.text.length - (bEnd - bStart)

  if (bStart >= aEnd) return opA
  if (bEnd <= aStart) return { ...opA, range: toRange(aStart + bNetChange, aEnd + bNetChange) }
  return { ...opA, range: toRange(bStart, bStart) }
}

function applyChange(content, change) {
  const lines = content.split('\n')
  const { range, text } = change
  const startLine = range.startLineNumber - 1
  const endLine = range.endLineNumber - 1
  const startChar = range.startColumn - 1
  const endChar = range.endColumn - 1
  const before = lines.slice(0, startLine).join('\n')
  const after = lines.slice(endLine + 1).join('\n')
  const startPart = lines[startLine]?.slice(0, startChar) || ''
  const endPart = lines[endLine]?.slice(endChar) || ''
  const middle = startPart + text + endPart
  return [before, middle, after].filter((v, i) => {
    if (i === 0 && before === '') return false
    if (i === 2 && after === '') return false
    return true
  }).join('\n')
}

server.listen(process.env.PORT || 3001, () => {
  console.log('Server running on port 3001')
})