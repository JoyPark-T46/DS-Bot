// Background tasks: long work leaves the Bot's own chat and runs in a separate
// Session. The record is written before that Session is created or prompted, so a
// restart can resume it. The result is queued back into the Bot's chat and does
// not cut a running turn. Regular slots are shared; one extra slot is reserved
// for a task the Bot started while answering the user, so background work cannot
// fill every slot.
import { randomUUID } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { mkdir, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { HISTORY_TIMEOUT_MS, SOURCE_KIND } from './constants.js'
import { output } from './tools.js'
import { splitRef, textOf } from './text.js'
import { RESUME_DELAY_MS } from './resume.js'

export const BACKGROUND_SLOTS = 2
export const BACKGROUND_RESERVED = 1

const TITLE_MAX = 80
const TASK_MAX = 12_000

const readTasks = (file) => {
  try {
    const parsed = JSON.parse(readFileSync(file, 'utf8'))
    const tasks = parsed?.tasks
    return tasks !== null && typeof tasks === 'object' && !Array.isArray(tasks) ? tasks : {}
  } catch {
    return {}
  }
}

const cleanTitle = (value) => String(value ?? '').replaceAll('[', '').replaceAll(']', '').replace(/\s+/g, ' ').trim().slice(0, TITLE_MAX)

export function install(rt) {
  const { ctx, home } = rt
  const file = join(home, 'background.json')
  /** @type {Record<string, any>} */
  const tasks = readTasks(file)
  let writing = Promise.resolve()
  const persist = () => {
    const saved = {}
    for (const [id, task] of Object.entries(tasks)) {
      const rest = { ...task }
      delete rest.draft
      saved[id] = rest
    }
    const run = async () => {
      await mkdir(home, { recursive: true })
      // A crash mid-write must not leave an empty file: the next start would
      // forget every task. Rename replaces the previous file in one step.
      const tmp = `${file}.${process.pid}.tmp`
      await writeFile(tmp, JSON.stringify({ version: 1, tasks: saved }), 'utf8')
      await rename(tmp, file)
    }
    writing = writing.then(run, run)
    return writing
  }

  const gates = new Map()
  const lock = (botId, fn) => {
    const prev = gates.get(botId) ?? Promise.resolve()
    const run = prev.then(() => fn())
    gates.set(botId, run.then(() => {}, () => {}))
    return run
  }

  const runningOf = (botId) => Object.values(tasks).filter(task => task.botId === botId && task.status === 'running')
  // 'regular' while a shared slot is free, 'reserved' for a human-priority task
  // once the shared slots are full, or null when the task has to wait.
  const slotFor = (botId, priority) => {
    const running = runningOf(botId)
    const regular = running.filter(task => task.slot !== 'reserved').length
    if (regular < BACKGROUND_SLOTS) return 'regular'
    if (priority === 'human' && !running.some(task => task.slot === 'reserved')) return 'reserved'
    return null
  }
  const queuedOf = (botId) => Object.values(tasks)
    .filter(task => task.botId === botId && task.status === 'queued')
    .sort((a, b) => (a.priority === 'human' ? 0 : 1) - (b.priority === 'human' ? 0 : 1) || a.createdAt - b.createdAt)

  // A turn the user opened (or a turn with no relay trigger yet) may use the
  // reserved slot. A turn opened by another Bot, a schedule or a background
  // result may not.
  const priorityOf = (sessionId) => {
    const info = rt.live.get(sessionId)?.info
    if (info === undefined) return 'human'
    if (info.kind === 'user' || info.role === 'continue' || info.role === 'kickoff' || info.role === 'resume' || info.role === 'update') return 'human'
    if (info.kind === 'schedule' || info.role === 'background') return 'background'
    if (info.kind === SOURCE_KIND) return 'background'
    return 'human'
  }

  const workerPrompt = (bot, task) => [
    `You are doing background work for ${bot.name}. The user is not in this session.`,
    'Do the task below with the tools in this session. Do not start a subagent.',
    'When you finish, reply with the result and nothing else. Reply in the language of the task.',
    'If this session was restarted, check what is already done before you repeat it.',
    '',
    `Task: ${task.title}`,
    task.task,
  ].join('\n')

  const recoveryPrompt = (task) => [
    'DeepSeek Harness restarted while you were working on this, so your last turn stopped before it finished.',
    'Check what is already done before you repeat anything, then finish the task and reply with the result.',
    '',
    `Task: ${task.title}`,
    task.task,
  ].join('\n')

  const workerOf = (sessionId) => Object.values(tasks).find(task => task.workerSessionId === sessionId && task.status === 'running')

  async function eventsOf(sessionId) {
    const inspect = ctx.sessionController?.inspect
    if (typeof inspect !== 'function' || sessionId === undefined) return undefined
    const seen = await inspect(sessionId, AbortSignal.timeout(HISTORY_TIMEOUT_MS)).catch(() => undefined)
    return Array.isArray(seen?.events) ? seen.events : undefined
  }

  async function alreadyDelivered(task) {
    const unread = rt.unreadOf?.(task.parentSessionId) ?? []
    if (unread.some(args => args.exchangeId === task.id)) return true
    const events = await eventsOf(task.parentSessionId)
    return events?.some(event => event.type === 'user/message' && event.data?.source?.exchangeId === task.id) === true
  }

  async function deliverResult(task) {
    if (task.delivered || rt.botOf(task.botId) === undefined) return
    if (await alreadyDelivered(task)) {
      task.delivered = true
      await persist()
      return
    }
    const body = task.status === 'failed'
      ? `"${task.title}" failed.\n${task.error || 'The background session stopped before it finished.'}`
      : `"${task.title}" finished.\n${task.result || '(no reply)'}`
    let text = body
    let files
    let length
    const long = await rt.attachLong?.(body, `Background ${task.title}.txt`)
    if (long !== undefined) { text = long.text; files = long.files; length = long.length }
    await rt.deliver({
      toId: task.botId,
      sessionId: task.parentSessionId,
      text,
      ...(files !== undefined ? { files } : {}),
      ...(length !== undefined ? { length } : {}),
      role: 'background',
      exchangeId: task.id,
      steer: false,
    })
    task.delivered = true
    task.updatedAt = Date.now()
    await persist()
  }

  async function sendToWorker(sessionId, text) {
    const agent = await rt.resolveAgent(sessionId)
    agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } }))
    await ctx.sessions?.flush?.(agent.session)
  }

  // Persist the worker id before the prompt, so a restart resumes this Session
  // instead of creating a second one.
  async function launch(task) {
    const bot = rt.botOf(task.botId)
    if (bot === undefined) {
      task.status = 'failed'
      task.error = 'The Bot is gone.'
      await persist()
      return
    }
    try {
      if (task.workerSessionId === undefined) {
        task.workerSessionId = await rt.createSession(`Background · ${task.title}`, { dir: bot.workDir })
        task.updatedAt = Date.now()
        await persist()
      }
      const route = splitRef(bot.appliedModel ?? bot.model)
      if (route !== undefined && rt.selectSessionModel !== undefined) {
        const effort = bot.appliedEffort ?? bot.effort
        await rt.selectSessionModel(task.workerSessionId, { ...route, ...(effort ? { reasoningEffort: effort } : {}) })
          .catch(error => rt.warn('ds-bot: background model for %s not applied: %s', bot.name, error))
      }
      await rt.applyPermission?.(bot, task.workerSessionId)?.catch(() => {})
      await sendToWorker(task.workerSessionId, task.recovered ? recoveryPrompt(task) : workerPrompt(bot, task))
    } catch (error) {
      task.status = 'failed'
      task.error = error?.message ?? String(error)
      task.updatedAt = Date.now()
      await persist()
      await deliverResult(task)
    }
  }

  // Caller holds the Bot's lock.
  async function pump(botId) {
    for (;;) {
      const next = queuedOf(botId)[0]
      if (next === undefined) return
      const slot = slotFor(botId, next.priority)
      if (slot === null) return
      next.slot = slot
      next.status = 'running'
      next.updatedAt = Date.now()
      await persist()
      await launch(next)
    }
  }

  async function settle(task, reason) {
    if (task.status !== 'running') return
    if (reason?.kind === 'aborted' && reason.reason?.kind === 'disposed') return
    if (reason?.kind === 'completed') {
      task.status = 'done'
      task.result = task.draft || ''
    } else {
      task.status = 'failed'
      task.error = task.draft || reason?.error?.message || 'The background session stopped before it finished.'
    }
    task.updatedAt = Date.now()
    await persist()
    await deliverResult(task)
    await pump(task.botId)
  }

  ctx.on('tools/execute', (exec, next) => rt.preemptedTool?.(exec, next) ?? next())

  ctx.on('session/event', (session, event) => {
    const task = workerOf(session?.id)
    if (task === undefined) return
    if (event.type === 'assistant/message') {
      const text = textOf(event.data?.message?.content)
      if (text !== '') task.draft = text
      return
    }
    if (event.type !== 'turn/end') return
    void lock(task.botId, () => settle(task, event.data?.reason))
      .catch(error => rt.warn('ds-bot: background result for %s not delivered: %s', task.title, error))
  })

  async function recoverOne(task) {
    if (task.status === 'queued') return
    if (task.status === 'done' || task.status === 'failed') {
      if (!task.delivered) await deliverResult(task)
      return
    }
    if (task.status !== 'running') return
    if (task.workerSessionId === undefined) {
      await launch(task)
      return
    }
    const events = await eventsOf(task.workerSessionId)
    const lastEnd = events?.findLast(event => event.type === 'turn/end')
    if (lastEnd?.data?.reason?.kind === 'completed') {
      const text = [...(events ?? [])].reverse().map(event => (event.type === 'assistant/message' ? textOf(event.data?.message?.content) : '')).find(value => value !== '')
      task.status = 'done'
      task.result = text ?? ''
      task.updatedAt = Date.now()
      await persist()
      await deliverResult(task)
      return
    }
    if (task.recovered === true) {
      task.status = 'failed'
      task.error = 'DeepSeek Harness restarted again before this background task finished.'
      task.updatedAt = Date.now()
      await persist()
      await deliverResult(task)
      return
    }
    task.recovered = true
    task.updatedAt = Date.now()
    await persist()
    await sendToWorker(task.workerSessionId, recoveryPrompt(task))
  }

  async function recoverBackground() {
    if (rt.readOnly?.() !== undefined) return
    const byBot = new Map()
    for (const task of Object.values(tasks)) {
      const list = byBot.get(task.botId) ?? []
      list.push(task)
      byBot.set(task.botId, list)
    }
    for (const [botId, list] of byBot) {
      await lock(botId, async () => {
        for (const task of list) {
          try { await recoverOne(task) } catch (error) { rt.warn('ds-bot: background task %s not resumed: %s', task.id, error) }
        }
        await pump(botId)
      })
    }
  }

  const backgroundView = () => {
    const byBot = {}
    for (const task of Object.values(tasks)) {
      if (task.status !== 'running' && task.status !== 'queued') continue
      ;(byBot[task.botId] ??= []).push({ id: task.id, title: task.title, status: task.status, since: task.updatedAt })
    }
    return byBot
  }

  rt.teamTools.push(defineTool({
    name: 'background_task',
    description: 'Hand a long task to a background session and keep this chat free. Use it for work that would take more than a quick step. It starts at once unless the background slots are full, in which case it waits; a task you start while answering the user can still use one reserved slot. The result comes back later as a "[Background task]" message and does not interrupt a running turn. Do not wait or poll.',
    parameters: {
      title: { type: 'string', required: true, description: `Short name, up to ${TITLE_MAX} characters, shown to the user while it runs` },
      task: { type: 'string', required: true, description: 'The whole task, self-contained: the background session cannot see this chat' },
    },
    output,
    async execute(args, exec) {
      await rt.load()
      const bot = rt.botOf(rt.selfOf(exec.agent?.id))
      if (bot === undefined) return 'Only Bots on the team can use background_task.'
      const title = cleanTitle(args.title)
      const taskText = String(args.task ?? '').trim().slice(0, TASK_MAX)
      if (title === '' || taskText === '') return 'Give a title and the task.'
      const priority = priorityOf(exec.agent.id)
      const record = {
        id: randomUUID(),
        botId: bot.id,
        parentSessionId: rt.chatOf(bot.id),
        title,
        task: taskText,
        priority,
        status: 'queued',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        delivered: false,
      }
      await lock(bot.id, async () => {
        tasks[record.id] = record
        await persist()
        await pump(bot.id)
      })
      const where = rt.groupOwners.has(exec.agent?.id) ? ' The result comes back in your own chat, not this group.' : ''
      if (record.status === 'failed') return `Could not start "${title}": ${record.error}`
      if (record.status === 'queued') return `Queued "${title}" (${record.id}). Background slots are full, so it waits. It will not cut into a message from the user.${where}`
      const reserved = record.slot === 'reserved' ? ' It used the reserved slot, so it was not stuck behind other background work.' : ''
      return `Started "${title}" in the background (${record.id}). This chat stays free. The result will come back here as a message; do not wait or poll.${reserved}${where}`
    },
  }))

  if (typeof ctx.effect === 'function') {
    ctx.effect(() => {
      const timer = setTimeout(() => { void recoverBackground().catch(error => rt.warn('ds-bot: background resume failed: %s', error)) }, RESUME_DELAY_MS + 500)
      return () => clearTimeout(timer)
    }, 'ds-bot: background resume')
  }

  Object.assign(rt, { backgroundView, recoverBackground })
}
