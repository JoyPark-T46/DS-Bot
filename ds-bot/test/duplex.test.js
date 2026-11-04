import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { createRuntime } from '../index.js'
import { SOURCE_KIND, STEER_CUT } from '../src/host/constants.js'
import { BACKGROUND_RESERVED, BACKGROUND_SLOTS } from '../src/host/background.js'
import { fakeContext } from './fake-ctx.js'

// Human messages preempt a busy private chat. Peer, schedule and background
// messages do not. A background task is saved before its worker session exists,
// and its result is queued back without a steer.
const TEAM = {
  version: 1,
  mainBotIds: ['a'],
  workspaceId: 'w1',
  bots: { a: { id: 'a', name: 'Chief', role: 'Boss', sessionId: 'dm-a', createdAt: 1, model: 'mock/chat' } },
  rooms: {},
  agents: {},
  exchanges: [],
  exchangeTurns: {},
  questions: {},
  prefs: {},
  revision: 0,
}

const start = async ({ interrupt } = {}) => {
  const home = mkdtempSync(join(tmpdir(), 'ds-bot-duplex-'))
  const team = structuredClone(TEAM)
  if (interrupt === false) team.bots.a.interrupt = false
  writeFileSync(join(home, 'state.json'), JSON.stringify(team))
  const ctx = fakeContext()
  const agentMap = new Map()
  const created = []
  let workers = 0
  const inboxOf = () => {
    const nextStep = []
    const nextTurn = []
    return {
      nextStep, nextTurn,
      prepend: (target, message) => (target === 'next-turn' ? nextTurn : nextStep).unshift(message),
      append: (target, message) => (target === 'next-turn' ? nextTurn : nextStep).push(message),
      remove: (id) => {
        for (const list of [nextStep, nextTurn]) {
          const at = list.findIndex(message => message.id === id)
          if (at !== -1) { list.splice(at, 1); return true }
        }
        return false
      },
    }
  }
  const calls = []
  const agentOf = (id) => {
    let agent = agentMap.get(id)
    if (agent !== undefined) return agent
    const inbox = inboxOf()
    agent = {
      id,
      session: { id, header: { cwd: '/team' } },
      ctx: { tools: { register: () => () => {}, restrict: () => () => {} }, on: () => () => {}, effect: execute => execute?.() },
      status: 'running',
      followup: message => calls.push(['followup', id, message]),
      steer: message => calls.push(['steer', id, message]),
      inject: message => calls.push(['inject', id, message]),
      send: (message, target) => inbox[target === 'next-turn' ? 'nextTurn' : 'nextStep'].push(message),
      cancel: (cause, options) => calls.push(['cancel', id, cause, options]),
      inbox,
    }
    agentMap.set(id, agent)
    return agent
  }
  const parent = agentOf('dm-a')
  Object.assign(ctx, {
    llm: { listProviders: () => [], listModels: async () => [], resolveModelInfo: async () => ({}) },
    workspaceRegistry: { get: () => ({ id: 'w1', path: join(home, 'workspace') }), list: () => [{ id: 'w1', path: join(home, 'workspace') }], resolveByPath: async () => undefined, create: async path => ({ id: 'w1', path }), archiveSession: async () => {} },
    sessionController: {
      create: async () => { const sessionId = `worker-${++workers}`; created.push(sessionId); return { sessionId } },
      resolveAgent: async sessionId => ({ agent: agentOf(sessionId) }),
      selectModel: async () => {},
      page: async () => ({ records: [], hasMore: false }),
      inspect: async () => ({ events: [] }),
    },
    sessionTitle: { rename: async () => {} },
    sessions: { flush: async () => {} },
  })
  const origGet = ctx.get
  Object.assign(ctx, { get: name => (name === 'agents' ? agentMap : origGet(name)) })
  const rt = createRuntime(ctx, { home })
  await rt.load?.()
  const saved = () => JSON.parse(readFileSync(join(home, 'background.json'), 'utf8'))
  const emit = async (id, type, data) => {
    for (const listener of ctx.listeners.get('session/event') ?? []) {
      if (listener.length >= 2) await listener({ id }, { type, data })
    }
    await new Promise(resolve => setTimeout(resolve, 0))
  }
  const tool = (name, args) => rt.teamTools.find(def => def.name === name).execute(args, { agent: parent })
  const stop = async () => { await ctx.dispose(); rmSync(home, { recursive: true, force: true, maxRetries: 5 }) }
  return { ctx, home, rt, emit, tool, stop, agent: parent, agentOf, calls, created, saved }
}

const userMessage = text => ({ id: `m-${text}`, content: [{ type: 'text', text }], source: { kind: 'user' }, role: 'user' })
const relay = (role, text) => ({ id: `m-${role}-${text}`, content: [{ type: 'text', text }], source: { kind: SOURCE_KIND, role }, role: 'user' })

const cancels = calls => calls.filter(call => call[0] === 'cancel')

test('a queued human message cuts a running model call and is not cut again on the next turn', async () => {
  const { emit, stop, agent, calls } = await start()
  await emit('dm-a', 'turn/start', { turn: 1 })
  await emit('dm-a', 'step/start', { turn: 1, step: 1 })
  const queued = userMessage('还没好吗')
  const older = userMessage('older')
  agent.inbox.nextTurn.push(older, queued)
  await emit('dm-a', 'agent/inbox/spliced', { target: 'next-turn', inserted: [queued] })
  assert.equal(cancels(calls).length, 1)
  assert.deepEqual(cancels(calls)[0][2], { kind: 'hook', reason: STEER_CUT })
  assert.deepEqual(cancels(calls)[0][3], { keepInbox: true })
  assert.deepEqual(agent.inbox.nextTurn.map(message => message.id), [queued.id, older.id])
  await emit('dm-a', 'turn/end', { reason: { kind: 'aborted', reason: { kind: 'hook', reason: STEER_CUT } } })
  await emit('dm-a', 'turn/start', { turn: 2 })
  await emit('dm-a', 'step/start', { turn: 2, step: 1 })
  assert.equal(cancels(calls).length, 1)
  await stop()
})

test('a human message during tools does not cancel until the next model step, and the tool returns an interrupt', async () => {
  const { ctx, emit, stop, agent, calls } = await start()
  await emit('dm-a', 'turn/start', { turn: 1 })
  await emit('dm-a', 'step/start', { turn: 1, step: 1 })
  await emit('dm-a', 'assistant/message', { message: { role: 'assistant', content: [{ type: 'tool-call' }] } })
  const steer = userMessage('stop')
  agent.inbox.nextStep.push(steer)
  await emit('dm-a', 'agent/inbox/spliced', { target: 'next-step', inserted: [steer] })
  assert.equal(cancels(calls).length, 0)
  const hooks = (ctx.listeners.get('tools/execute') ?? []).filter(hook => hook.length >= 2)
  const results = await Promise.all(hooks.map(hook => Promise.race([
    hook({ agent, name: 'read' }, () => new Promise(() => {})),
    new Promise(resolve => setTimeout(() => resolve('still-running'), 30)),
  ])))
  const interrupted = results.find(result => result?.isError === true)
  assert.equal(interrupted.content[0].text.includes('Interrupted'), true)
  await emit('dm-a', 'step/end', {})
  await emit('dm-a', 'step/start', { turn: 1, step: 2 })
  assert.equal(cancels(calls).length, 1)
  assert.deepEqual(cancels(calls)[0][2], { kind: 'hook', reason: STEER_CUT })
  await stop()
})

test('peer, schedule and background messages do not cut a running turn', async () => {
  const { emit, stop, agent, calls } = await start()
  await emit('dm-a', 'turn/start', { turn: 1 })
  await emit('dm-a', 'step/start', { turn: 1, step: 1 })
  for (const message of [relay('request', 'from a bot'), { id: 'sched', content: [{ type: 'text', text: 'tick' }], source: { kind: 'schedule' } }, relay('background', 'task finished')]) {
    agent.inbox.nextStep.push(message)
    await emit('dm-a', 'agent/inbox/spliced', { target: 'next-step', inserted: [message] })
  }
  assert.equal(cancels(calls).length, 0)
  await stop()
})

test('interrupt off leaves a human steer for the step boundary', async () => {
  const { emit, stop, agent, calls } = await start({ interrupt: false })
  await emit('dm-a', 'turn/start', { turn: 1 })
  await emit('dm-a', 'step/start', { turn: 1, step: 1 })
  const steer = userMessage('hello')
  agent.inbox.nextStep.push(steer)
  await emit('dm-a', 'agent/inbox/spliced', { target: 'next-step', inserted: [steer] })
  assert.equal(cancels(calls).length, 0)
  await stop()
})

test('background_task is saved before the worker exists, and the result is queued not steered', async () => {
  const { ctx, emit, stop, tool, calls, created, saved, agentOf } = await start()
  const seen = []
  const create = ctx.sessionController.create
  ctx.sessionController.create = async () => {
    const tasks = Object.values(saved().tasks)
    assert.equal(tasks.length, 1)
    assert.equal(tasks[0].status, 'running')
    assert.equal(tasks[0].task, 'read the logs')
    assert.equal(tasks[0].workerSessionId, undefined)
    seen.push('created')
    return create()
  }
  const receipt = await tool('background_task', { title: 'Logs', task: 'read the logs' })
  assert.match(receipt, /Started "Logs"/)
  assert.deepEqual(created, ['worker-1'])
  assert.deepEqual(seen, ['created'])
  const worker = agentOf('worker-1')
  const prompt = calls.find(call => call[0] === 'followup' && call[1] === 'worker-1')
  assert.match(prompt[2].content[0].text, /read the logs/)
  worker.status = 'idle'
  await emit('worker-1', 'assistant/message', { message: { role: 'assistant', content: [{ type: 'text', text: 'twelve errors' }] } })
  await emit('worker-1', 'turn/end', { reason: { kind: 'completed' } })
  const id = Object.keys(saved().tasks)[0]
  for (let attempt = 0; attempt < 50 && saved().tasks[id].delivered !== true; attempt += 1) {
    await new Promise(resolve => setTimeout(resolve, 10))
  }
  const back = calls.find(call => call[0] === 'followup' && call[1] === 'dm-a')
  assert.ok(back !== undefined)
  assert.equal(calls.some(call => call[0] === 'steer'), false)
  assert.match(back[2].content[0].text, /\[Background task\]/)
  assert.match(back[2].content[0].text, /twelve errors/)
  assert.equal(back[2].source.role, 'background')
  assert.equal(saved().tasks[id].delivered, true)
  await stop()
})

test('background results that arrive mid-turn wait, and do not steer', async () => {
  const { emit, stop, calls } = await start()
  await emit('dm-a', 'turn/start', { turn: 1 })
  await emit('dm-a', 'user/message', { content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })
  await emit('dm-a', 'user/message', {
    content: [{ type: 'text', text: '[Background task]\n"Logs" finished.\ntwelve errors' }],
    source: { kind: SOURCE_KIND, role: 'background', exchangeId: 'bg1' },
  })
  assert.equal(calls.some(call => call[0] === 'followup' || call[0] === 'steer'), false)
  await emit('dm-a', 'assistant/message', { message: { role: 'assistant', content: [{ type: 'text', text: 'on it' }] } })
  await emit('dm-a', 'turn/end', { reason: { kind: 'completed' } })
  const back = calls.find(call => call[0] === 'followup' && call[1] === 'dm-a')
  assert.ok(back !== undefined)
  assert.equal(back[2].source.deferred, true)
  assert.equal(back[2].source.role, 'background')
  assert.match(back[2].content[0].text, /twelve errors/)
  assert.equal(calls.some(call => call[0] === 'steer'), false)
  await stop()
})

test('background slots leave one reserved place for a human task', async () => {
  const { rt, stop, tool, saved } = await start()
  rt.live.set('dm-a', { turn: 1, info: { kind: SOURCE_KIND, role: 'request' }, texts: [] })
  for (let index = 0; index < BACKGROUND_SLOTS; index += 1) {
    const receipt = await tool('background_task', { title: `Peer ${index}`, task: `peer work ${index}` })
    assert.match(receipt, /^Started/)
  }
  const waited = await tool('background_task', { title: 'Peer extra', task: 'must wait' })
  assert.match(waited, /^Queued/)
  rt.live.set('dm-a', { turn: 2, info: undefined, texts: [] })
  const human = await tool('background_task', { title: 'Yours', task: 'the user asked' })
  assert.match(human, /reserved slot/)
  const again = await tool('background_task', { title: 'Yours again', task: 'also the user' })
  assert.match(again, /^Queued/)
  const tasks = Object.values(saved().tasks)
  assert.equal(tasks.filter(task => task.status === 'running').length, BACKGROUND_SLOTS + BACKGROUND_RESERVED)
  assert.equal(tasks.filter(task => task.slot === 'reserved').length, 1)
  assert.equal(tasks.filter(task => task.status === 'queued').length, 2)
  await stop()
})

test('a restart resumes a running worker and delivers a finished result that never arrived', async () => {
  const { ctx, home, tool } = await start()
  await tool('background_task', { title: 'Logs', task: 'read the logs' })
  const first = structuredClone(JSON.parse(readFileSync(join(home, 'background.json'), 'utf8')))
  const runningId = Object.keys(first.tasks)[0]
  first.tasks.done = {
    id: 'done', botId: 'a', parentSessionId: 'dm-a', title: 'Sums', task: 'add',
    priority: 'human', status: 'done', result: 'four', createdAt: 1, updatedAt: 2, delivered: false,
  }
  writeFileSync(join(home, 'background.json'), JSON.stringify(first))
  await ctx.dispose()
  const next = fakeContext()
  const agentMap = new Map()
  const resumed = []
  const agentOfResume = (id) => {
    let agent = agentMap.get(id)
    if (agent !== undefined) return agent
    agent = {
      id, session: { id }, status: 'idle', inbox: { nextStep: [], nextTurn: [] },
      ctx: { tools: { register: () => () => {} } },
      followup: message => resumed.push([id, message]),
      steer() {}, cancel() {},
    }
    agentMap.set(id, agent)
    return agent
  }
  agentOfResume('dm-a')
  agentOfResume('worker-1')
  Object.assign(next, {
    llm: { listProviders: () => [], listModels: async () => [], resolveModelInfo: async () => ({}) },
    workspaceRegistry: { get: () => ({ id: 'w1', path: '/team' }), list: () => [], resolveByPath: async () => undefined, create: async path => ({ id: 'w1', path }), archiveSession: async () => {} },
    sessionController: {
      create: async () => { throw new Error('a resume must not create another session') },
      resolveAgent: async sessionId => ({ agent: agentOfResume(sessionId) }),
      selectModel: async () => {},
      page: async () => ({ records: [], hasMore: false }),
      inspect: async () => ({ events: [] }),
    },
    sessionTitle: { rename: async () => {} },
    sessions: { flush: async () => {} },
  })
  const rt = createRuntime(next, { home })
  await rt.load?.()
  await rt.recoverBackground()
  const note = resumed.find(entry => entry[0] === 'worker-1')
  assert.ok(note !== undefined)
  assert.match(note[1].content[0].text, /DeepSeek Harness restarted/)
  const result = resumed.find(entry => entry[0] === 'dm-a')
  assert.match(result[1].content[0].text, /four/)
  assert.equal(result[1].source.role, 'background')
  const after = JSON.parse(readFileSync(join(home, 'background.json'), 'utf8'))
  assert.equal(after.tasks[runningId].recovered, true)
  assert.equal(after.tasks.done.delivered, true)
  await next.dispose()
  rmSync(home, { recursive: true, force: true, maxRetries: 5 })
})
