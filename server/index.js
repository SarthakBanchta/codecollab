const express = require('express')
const http = require('http')
const { Server } = require('socket.io')
const cors = require('cors')

const app = express()
app.use(cors())

const server = http.createServer(app)
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
})

let documentContent = '// Start coding...'

io.on('connection', (socket) => {
  console.log('User connected:', socket.id)

  socket.on('request-document', () => {
    socket.emit('init-document', documentContent)
  })

  socket.on('code-change', ({ changes }) => {
    changes.forEach(change => {
      documentContent = applyChange(documentContent, change)
    })
    socket.broadcast.emit('code-change', { changes })
  })

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id)
  })
})


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

server.listen(3001, () => {
  console.log('Server running on port 3001')
})