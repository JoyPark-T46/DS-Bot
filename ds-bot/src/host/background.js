// Background tasks: long work leaves the Bot's own chat and runs in a separate
// dsh Session. The Bot's own Session stays free for the user (the reserved slot).
// At most BACKGROUND_SLOTS workers run per Bot; the rest wait. A worker's result
// is queued into the Bot's chat and does not cut a running turn. The task is
// written down before it starts, and a restart picks it up from that file.
import { randomUUID } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { HISTORY_TIMEOUT_MS, MESSAGE_INLINE_MAX } from './constants.js'
import { splitRef, textOf } from './text.js'

// The Bot's own chat is not one of these: it stays available for a person.
export const BACKGROUND_SLOTS = 2
const TITLE_MAX = 80
const TASK_MAX = 12_000
const RESUME_DELAY_MS = 3000

const output = { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: String(value) }] }

const cleanTitle = (raw) => String(raw ?? '').replace(/[\r\n"\]]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, TITLE_MAX)

// The last turn of a worker Session: what it said, and whether that turn is still open.
export function outcomeOf(events) {
  let texts = []
  let reason
  let open = false
  for (const event of events) {
    if (event?.type === 'turn/start') {
      open = true
      texts = []
      reason = undefined
      continue
    }
    if (!open) continue
    if (event.type === 'assistant/message') {
      const text = textOf(event.data?.message?.content)
      if (text !== '') texts.push(text)
    }
    if (event.type === 'turn/end') {
      reason = event.data?.reason
      open = false
    }
  }
  return { text: texts.join('\n\n').trim(), reason, open }
}

const readTasks = (file) => {
  try {
    const parsed = JSON.parse(readFileSync(file, 'utf8'))
    return Array.isArray(parsed?.tasks) ? parsed.tasks.filter(task => task !== null && typeof task === 'object' && typeof task.id === 'string') : []
  } catch {
    return []
  }
}

export function install(rt) {
  const { ctx, home } = rt
  const file = join(home, 'background.json')
  let tasks = readTasks(file)
  let writing = Promise.resolve()
  let stopped = false
  // Assistant text of the worker turn still open, by task id. Not saved: a restart
  // reads it back from the Session.
  const partials = new Map()
  const pumps = new Map()

  const stored = (task) => ({
    id: task.id,
    botId: task.botId,
    title: task.title,
    task: task.task,
    status: task.status,
    createdAt: task.createdAt,
    ...(task.sessionId ? { sessionId: task.sessionId } : {}),
    ...(task.startedAt ? { startedAt: task.startedAt } : {}),
    ...(task.resumed ? { resumed: true } : {}),
  })

  const persist = () => {
    if (stopped || rt.readOnly?.() !== undefined) return Promise.resolve()
    const snapshot = JSON.stringify({ tasks: tasks.map(stored) })
    const run = writing.then(async () => {
      await mkdir(home, { recursive: true })
      await writeFile(file, snapshot, 'utf8')
    }).catch(error => rt.warn('ds-bot: background tasks not saved: %s', error))
    writing = run
    return run
  }

  const runningOf = (botId) => tasks.filter(task => task.botId === botId && task.status === 'running').length

  const workerPrompt = (bot, task) => [
    `You are a background worker for ${bot.name}. The user does not see this chat.`,
    'Do the task below with your tools. Do not start a subagent, and do not message anyone.',
    'When you finish, reply with the result and nothing else.',
    '',
    task.title,
    '',
    task.task,
  ].join('\n')

  const recoveryPrompt = (task) => [
    'DeepSeek Harness restarted while you were doing this task, so the last turn stopped before it finished.',
    'Check what is already done, do not repeat finished steps, then finish the task and reply with the result.',
    '',
    task.title,
    '',
    task.task,
  ].join('\n')

  const promptWorker = async (task, text) => {
    const agent = await rt.resolveAgent(task.sessionId)
    agent.followup(createUserMessage({
      content: [{ type: 'text', text }],
      source: { kind: 'user' },
    }))
    await ctx.sessions?.flush?.(agent.session)
  }

  const deliverResult = async (task, text) => {
    let body = text
    let extra = {}
    if (text.length > MESSAGE_INLINE_MAX && typeof rt.attachLong === 'function') {
      const long = await rt.attachLong(text, `Background ${task.title}.txt`)
      if (long !== undefined) {
        body = long.text
        extra = long.length === undefined ? {} : { files: long.files, length: long.length }
      }
    }
    await rt.deliver({ toId: task.botId, role: 'background', note: task.title, steer: false, text: body, ...extra })
  }

  async function finish(task, { text, failed = false }) {
    if (task.settling || task.status !== 'running') return
    if (rt.botOf(task.botId) === undefined) {
      tasks = tasks.filter(item => item.id !== task.id)
      partials.delete(task.id)
      await persist()
      return
    }
    task.settling = true
    const body = text.trim() !== '' ? text : failed
      ? 'The background task stopped before it wrote a result.'
      : 'The background task ended without a result.'
    try {
      await deliverResult(task, text.trim() === '' ? body : text)
      tasks = tasks.filter(item => item.id !== task.id)
      partials.delete(task.id)
      await persist()
    } catch (error) {
      task.settling = false
      rt.warn('ds-bot: background result for %s not delivered: %s', task.title, error)
    }
    await pump(task.botId)
  }

  async function settle(task) {
    if (task.status !== 'running') return
    let agent
    try {
      agent = task.sessionId === undefined ? undefined : await rt.resolveAgent(task.sessionId)
    } catch {
      agent = undefined
    }
    if (agent?.status === 'running') return
    if (task.sessionId === undefined || agent === undefined) {
      task.status = 'queued'
      delete task.sessionId
      task.settling = false
      await persist()
      return
    }
    const events = []
    try {
      for await (const event of rt.eventsBackwards(task.sessionId, AbortSignal.timeout(HISTORY_TIMEOUT_MS))) {
        events.push(event)
      }
    } catch (error) {
      rt.warn('ds-bot: background session %s not read: %s', task.sessionId, error)
    }
    events.reverse()
    const outcome = outcomeOf(events)
    const hadUser = events.some(event => event?.type === 'user/message')
    if (!outcome.open && outcome.reason?.kind === 'completed') {
      await finish(task, { text: outcome.text, failed: outcome.text === '' })
      return
    }
    if (!hadUser) {
      const bot = rt.botOf(task.botId)
      if (bot === undefined) {
        tasks = tasks.filter(item => item.id !== task.id)
        await persist()
        return
      }
      await promptWorker(task, workerPrompt(bot, task)).catch(error => rt.warn('ds-bot: background task %s not started again: %s', task.title, error))
      return
    }
    if (task.resumed === true) {
      await finish(task, { failed: true, text: outcome.text === '' ? 'The background task was interrupted again when DeepSeek Harness restarted, so it stopped.' : `The background task was interrupted again when DeepSeek Harness restarted, so it stopped.\n${outcome.text}` })
      return
    }
    task.resumed = true
    await persist()
    await promptWorker(task, recoveryPrompt(task)).catch(error => rt.warn('ds-bot: background task %s not resumed: %s', task.title, error))
  }

  async function launch(task) {
    const bot = rt.botOf(task.botId)
    if (bot === undefined) {
      tasks = tasks.filter(item => item.id !== task.id)
      await persist()
      return
    }
    // Noted as running before the Session exists, so a restart cannot start it twice
    // and cannot lose it. A Session id is noted before the prompt is sent.
    task.status = 'running'
    task.startedAt = Date.now()
    await persist()
    try {
      const sessionId = await rt.createSession(`Background · ${bot.name}: ${task.title}`.slice(0, TITLE_MAX), { dir: bot.workDir })
      task.sessionId = sessionId
      await persist()
      await rt.applyPermission?.(bot, sessionId)
      const ref = await rt.desiredModel?.(bot).catch(() => undefined)
      const route = ref === undefined ? undefined : splitRef(ref)
      if (route !== undefined) {
        const effort = await rt.desiredEffort?.(bot, ref).catch(() => undefined)
        await rt.selectSessionModel(sessionId, { ...route, ...(effort === undefined ? {} : { reasoningEffort: effort }) })
          .catch(error => rt.warn('ds-bot: background model for %s not applied: %s', bot.name, error))
      }
      await promptWorker(task, workerPrompt(bot, task))
    } catch (error) {
      rt.warn('ds-bot: background task %s did not start: %s', task.title, error)
      tasks = tasks.filter(item => item.id !== task.id)
      await persist()
      task.status = 'failed'
      task.error = error?.message ?? String(error)
    }
  }

  async function fill(botId) {
    for (;;) {
      if (runningOf(botId) >= BACKGROUND_SLOTS) return
      const next = tasks.find(task => task.botId === botId && task.status === 'queued')
      if (next === undefined) return
      await launch(next)
    }
  }

  function pump(botId) {
    const previous = pumps.get(botId) ?? Promise.resolve()
    const run = previous.then(() => fill(botId)).catch(error => rt.warn('ds-bot: background queue for %s stopped: %s', botId, error))
    pumps.set(botId, run)
    return run
  }

  async function resumeBackground() {
    await rt.load()
    if (rt.readOnly?.() !== undefined) return
    for (const task of tasks.slice()) {
      if (task.status === 'running') await settle(task)
    }
    const bots = new Set(tasks.filter(task => task.status === 'queued').map(task => task.botId))
    for (const botId of bots) await pump(botId)
  }

  async function dropBackground(botId) {
    const own = tasks.filter(task => task.botId === botId)
    for (const task of own) {
      if (task.sessionId === undefined) continue
      try {
        const agent = await rt.resolveAgent(task.sessionId)
        agent.cancel?.({ kind: 'user' })
      } catch { /* already gone */ }
    }
    if (own.length === 0) return
    tasks = tasks.filter(task => task.botId !== botId)
    await persist()
  }

  const backgroundView = () => {
    const byBot = {}
    for (const task of tasks) {
      if (task.status !== 'queued' && task.status !== 'running') continue
      const row = { id: task.id, title: task.title, status: task.status, since: task.startedAt ?? task.createdAt }
      byBot[task.botId] = [...(byBot[task.botId] ?? []), row]
    }
    return byBot
  }

  ctx.on('session/event', async (session, event) => {
    const task = tasks.find(item => item.sessionId === session?.id && item.status === 'running')
    if (task === undefined) return
    if (event.type === 'turn/start') partials.set(task.id, [])
    if (event.type === 'assistant/message') {
      const text = textOf(event.data?.message?.content)
      if (text !== '') partials.set(task.id, [...(partials.get(task.id) ?? []), text])
    }
    if (event.type !== 'turn/end') return
    const text = (partials.get(task.id) ?? []).join('\n\n').trim()
    partials.delete(task.id)
    const reason = event.data?.reason
    if (reason?.kind === 'completed') await finish(task, { text, failed: text === '' })
    else await finish(task, { failed: true, text: text === '' ? `The background task stopped (${reason?.kind ?? 'unknown'}).` : text })
  })

  rt.teamTools.push(defineTool({
    name: 'start_background',
    description: 'Hand a long task to a background worker and end your turn. Your own chat stays free for the user. The result arrives later as a "[Background task …]" message and does not interrupt a turn already running. Do not wait or poll. Use this for work that would take many steps; do a short job yourself.',
    parameters: {
      title: { type: 'string', required: true, description: 'Short name the user sees while it runs, at most 80 characters' },
      task: { type: 'string', required: true, description: 'Everything the worker needs. It cannot see this chat.' },
    },
    output,
    async execute(args, exec) {
      await rt.load()
      const bot = rt.botOf(rt.selfOf(exec.agent?.id))
      if (bot === undefined) return 'Only a Bot can start a background task.'
      if (rt.groupOwners.has(exec.agent?.id) || rt.chatOf(bot.id) !== exec.agent?.id) return 'Start a background task from your own chat. In a group, do the work in the group.'
      if (rt.readOnly?.() !== undefined) return rt.readOnly()
      const title = cleanTitle(args?.title)
      const taskText = String(args?.task ?? '').trim()
      if (title === '') return 'Give the background task a short title.'
      if (taskText === '') return 'Say what the background task should do.'
      if (taskText.length > TASK_MAX) return `A background task can hold at most ${TASK_MAX} characters; this one has ${taskText.length}.`
      const task = { id: randomUUID(), botId: bot.id, title, task: taskText, status: 'queued', createdAt: Date.now() }
      tasks.push(task)
      // On disk before anything runs, so a restart still has it.
      await persist()
      await pump(bot.id)
      if (task.status === 'failed') return `Could not start "${title}": ${task.error}`
      if (task.status === 'queued') {
        const ahead = runningOf(bot.id)
        return `Queued "${title}" (${task.id}). ${ahead} worker ${ahead === 1 ? 'slot is' : 'slots are'} already in use, so this one waits. Tell the user it is queued, then end your turn. Do not wait for it.`
      }
      return `Started "${title}" (${task.id}) in the background. Tell the user it is running, then end your turn. Do not wait for it.`
    },
  }))

  rt.teamTools.push(defineTool({
    name: 'stop_background',
    description: 'Stop one background task you started. The worker stops, and whatever it had written comes back as its result. Leave id empty to stop the newest one still running.',
    parameters: {
      id: { type: 'string', description: 'The id start_background returned. Empty stops the newest running task.' },
    },
    output,
    async execute(args, exec) {
      await rt.load()
      const bot = rt.botOf(rt.selfOf(exec.agent?.id))
      if (bot === undefined) return 'Only a Bot can stop a background task.'
      const wanted = String(args?.id ?? '').trim()
      const own = tasks.filter(task => task.botId === bot.id && (task.status === 'running' || task.status === 'queued'))
      const task = wanted === '' ? own.toReversed().find(item => item.status === 'running') ?? own.at(-1) : own.find(item => item.id === wanted)
      if (task === undefined) return wanted === '' ? 'No background task is running.' : `No background task has id ${wanted}.`
      if (task.status === 'queued') {
        tasks = tasks.filter(item => item.id !== task.id)
        await persist()
        return `Cancelled queued background task "${task.title}" before it started.`
      }
      try {
        const agent = await rt.resolveAgent(task.sessionId)
        if (agent?.status === 'running' && typeof agent.cancel === 'function') {
          agent.cancel({ kind: 'user' })
          return `Stopping "${task.title}". What it had written comes back when it stops.`
        }
      } catch { /* the session is already gone; the result path reports that */ }
      await finish(task, { failed: true, text: `Stopped "${task.title}" before it finished.` })
      return `Stopped "${task.title}".`
    },
  }))

  const pending = tasks.some(task => task.status === 'running' || task.status === 'queued')
  ctx.effect(() => {
    const timer = pending ? setTimeout(() => { void resumeBackground().catch(error => rt.warn('ds-bot: background tasks not resumed: %s', error)) }, RESUME_DELAY_MS) : undefined
    timer?.unref?.()
    return () => { stopped = true; clearTimeout(timer); return writing }
  }, 'ds-bot: background tasks')

  Object.assign(rt, { backgroundView, resumeBackground, dropBackground })
}
