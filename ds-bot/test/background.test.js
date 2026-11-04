import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { createRuntime } from '../index.js'
import { BACKGROUND_SLOTS, outcomeOf } from '../src/host/background.js'
import { fakeContext } from './fake-ctx.js'

const TEAM = {
  version: 1,
  mainBotIds: ['a'],
  workspaceId: 'w1',
  bots: { a: { id: 'a', name: 'Chief', role: 'Boss', sessionId: 'dm-a', createdAt: 1 } },
  rooms: {},
  agents: {},
  exchanges: [],
  exchangeTurns: {},
  questions: {},
  prefs: {},
  revision: 0,
}

function start(team = TEAM, { background } = {}) {
  const home = mkdtempSync(join(tmpdir(), 'ds-bot-bg-'))
  writeFileSync(join(home, 'state.json'), JSON.stringify(team))
  if (background !== undefined) writeFileSync(join(home, 'background.json'), JSON.stringify(background))
  const ctx = fakeContext()
  const sent = []
  const workspaces = new Map([['w1', { id: 'w1', path: home }]])
  let sessions = 0
  const agents = new Map()
  const agentFor = (sessionId) => {
    let agent = agents.get(sessionId)
    if (agent !== undefined) return agent
    agent = {
      id: sessionId,
      status: 'idle',
      session: { id: sessionId },
      followup: message => sent.push({ sessionId, message, steered: false }),
      steer: message => sent.push({ sessionId, message, steered: true }),
      cancel: () => { agent.cancelled = true },
      ctx: { tools: { register: () => () => {}, restrict: () => () => {} }, on: () => () => {}, effect: execute => execute?.() },
    }
    agents.set(sessionId, agent)
    return agent
  }
  Object.assign(ctx, {
    llm: { listProviders: () => [], listModels: async () => [], resolveModelInfo: async () => ({}) },
    workspaceRegistry: {
      get: id => workspaces.get(id),
      list: () => [...workspaces.values()],
      resolveByPath: async () => undefined,
      create: async path => {
        const workspace = { id: 'w1', path }
        workspaces.set(workspace.id, workspace)
        return workspace
      },
      archiveSession: async () => {},
    },
    sessionController: {
      create: async () => ({ sessionId: `s${++sessions}` }),
      resolveAgent: async sessionId => ({ agent: agentFor(sessionId) }),
      selectModel: async () => {},
      page: async () => ({ records: [], hasMore: false }),
    },
    sessionTitle: { rename: async () => {} },
    sessions: { flush: async () => {} },
  })
  const rt = createRuntime(ctx, { home })
  const tool = (name, args, sessionId = 'dm-a') => rt.teamTools.find(item => item.name === name).execute(args, { agent: { id: sessionId } })
  const emit = async (id, type, data) => {
    for (const listener of ctx.listeners.get('session/event') ?? []) {
      if (listener.length >= 2) await listener({ id }, { type, data })
    }
  }
  const stop = async () => { await ctx.dispose(); rmSync(home, { recursive: true, force: true, maxRetries: 5 }) }
  const saved = () => JSON.parse(readFileSync(join(home, 'background.json'), 'utf8'))
  return { ctx, rt, home, sent, agents, tool, emit, stop, saved, sessions: () => sessions }
}

const assistant = text => ({ type: 'assistant/message', data: { message: { content: [{ type: 'text', text }] } } })
const ended = reason => ({ type: 'turn/end', data: { reason } })

test('outcomeOf keeps the last turn and notices one that is still open', () => {
  const events = [
    { type: 'turn/start' },
    assistant('first'),
    ended({ kind: 'completed' }),
    { type: 'turn/start' },
    assistant('kept'),
    ended({ kind: 'aborted' }),
  ]
  assert.deepEqual(outcomeOf(events), { text: 'kept', reason: { kind: 'aborted' }, open: false })
  assert.equal(outcomeOf([...events, { type: 'turn/start' }, assistant('open')]).open, true)
})

test('a background task is saved before it starts, and its result is queued', async () => {
  const { ctx, home, tool, emit, stop, sent, saved } = start()
  const seen = []
  ctx.sessionController.create = async () => {
    seen.push(JSON.parse(readFileSync(join(home, 'background.json'), 'utf8')))
    return { sessionId: `s${seen.length}` }
  }
  const receipt = await tool('start_background', { title: 'Report', task: 'Write the weekly report.' })
  assert.equal(seen[0].tasks[0].status, 'running')
  assert.equal(seen[0].tasks[0].sessionId, undefined)
  assert.match(receipt, /Started "Report"/)
  const noted = saved().tasks[0]
  assert.equal(noted.status, 'running')
  assert.equal(noted.sessionId, 's1')
  assert.match(noted.task, /weekly report/)
  const prompt = sent.find(item => item.sessionId === 's1')
  assert.equal(prompt.steered, false)
  await emit('s1', 'turn/start', { turn: 1 })
  await emit('s1', 'assistant/message', { message: { content: [{ type: 'text', text: 'The report is ready.' }] } })
  await emit('s1', 'turn/end', { reason: { kind: 'completed' } })
  const back = sent.find(item => item.sessionId === 'dm-a' && item.message?.source?.role === 'background')
  assert.ok(back !== undefined)
  assert.equal(back.steered, false)
  assert.equal(back.message.source.note, 'Report')
  assert.match(back.message.content[0].text, /The report is ready/)
  assert.match(back.message.content[0].text, /Background task "Report"/)
  assert.equal(saved().tasks.length, 0)
  await stop()
})

test('peer traffic does not use a worker slot, and extra tasks wait for one', async () => {
  const { tool, emit, stop, saved } = start()
  const titles = ['One', 'Two', 'Three']
  for (const title of titles) {
    await tool('start_background', { title, task: `Do ${title}` })
  }
  const tasks = saved().tasks
  assert.equal(tasks.filter(task => task.status === 'running').length, BACKGROUND_SLOTS)
  assert.equal(tasks.filter(task => task.status === 'queued').map(task => task.title).join(','), 'Three')
  const running = tasks.find(task => task.title === 'One')
  await emit(running.sessionId, 'turn/start', { turn: 1 })
  await emit(running.sessionId, 'assistant/message', { message: { content: [{ type: 'text', text: 'done' }] } })
  await emit(running.sessionId, 'turn/end', { reason: { kind: 'completed' } })
  const after = saved().tasks
  assert.equal(after.some(task => task.title === 'One'), false)
  assert.equal(after.find(task => task.title === 'Three')?.status, 'running')
  await stop()
})

test('a restart delivers a finished worker result and resumes one that was cut off', async () => {
  const finished = {
    tasks: [
      { id: 'done-1', botId: 'a', title: 'Digest', task: 'Summarize the notes', status: 'running', sessionId: 'worker-done', createdAt: 1, startedAt: 2 },
      { id: 'cut-1', botId: 'a', title: 'Sweep', task: 'Clean the folder', status: 'running', sessionId: 'worker-cut', createdAt: 3, startedAt: 4 },
    ],
  }
  const { rt, sent, stop, saved } = start(TEAM, { background: finished })
  const logs = {
    'worker-done': [
      { type: 'turn/end', data: { reason: { kind: 'completed' } } },
      { type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'Notes summarized.' }] } } },
      { type: 'user/message', data: { content: [{ type: 'text', text: 'go' }] } },
      { type: 'turn/start', data: { turn: 1 } },
    ],
    'worker-cut': [
      { type: 'turn/end', data: { reason: { kind: 'aborted' } } },
      { type: 'user/message', data: { content: [{ type: 'text', text: 'go' }] } },
      { type: 'turn/start', data: { turn: 1 } },
    ],
  }
  rt.eventsBackwards = async function* (sessionId) {
    for (const event of logs[sessionId] ?? []) yield event
  }
  await rt.resumeBackground()
  const back = sent.find(item => item.sessionId === 'dm-a' && item.message?.source?.note === 'Digest')
  assert.ok(back !== undefined)
  assert.equal(back.steered, false)
  assert.match(back.message.content[0].text, /Notes summarized/)
  const again = sent.find(item => item.sessionId === 'worker-cut')
  assert.ok(again !== undefined)
  assert.match(again.message.content[0].text, /restarted while you were doing this task/)
  assert.equal(saved().tasks.find(task => task.id === 'cut-1')?.resumed, true)
  assert.equal(saved().tasks.some(task => task.id === 'done-1'), false)
  await rt.resumeBackground()
  const stopped = sent.find(item => item.sessionId === 'dm-a' && item.message?.source?.note === 'Sweep')
  assert.match(stopped.message.content[0].text, /interrupted again/)
  assert.equal(saved().tasks.some(task => task.id === 'cut-1'), false)
  await stop()
})
