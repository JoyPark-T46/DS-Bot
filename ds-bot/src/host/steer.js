// In a Bot's private chat a message from the user must reach the Bot at once,
// not at the next DSH step boundary. While `interrupt` is on (the default):
//  - a queued message cuts the turn the same way a steer does;
//  - a foreground shell call detaches into the jobs registry and returns DSH's
//    promoted result, so the step ends while the command runs on;
//  - any other tool in flight returns an interrupt result so the step can end
//    without dropping the tool call, and the model call that follows is cancelled;
//  - a running model call is cancelled (hook reason) and the user's message is
//    moved to the front of next-turn, so a fresh turn claims it at once.
// Messages from other Bots, groups, schedules and background results do not cut.
// Turning `interrupt` off keeps DSH's own step boundary.
import { isAbsolute, resolve } from 'node:path'
import { boundContextSummary, createUserMessage } from '@deepseek-ai/dsh-llm'
import { STEER_CUT } from './constants.js'

const SHELL_TOOLS = ['pwsh', 'bash']
const STEER_GRACE_MS = 1000
// A tool result the step can record when the user preempts it. The real call may
// still be finishing; the turn must not wait on it, and it must not be left open.
const INTERRUPT_RESULT = {
  isError: true,
  content: [{ type: 'text', text: 'Interrupted because the user sent a new message. Stop this step and answer that message. Do not continue the work you just left unless they still want it.' }],
}

export function install(rt) {
  const ctx = rt.ctx
  let jobs
  const attachJobs = (sub) => {
    const own = sub.jobs
    jobs = own
    sub.effect(() => () => { if (jobs === own) jobs = undefined }, 'ds-bot: jobs service')
  }
  if (typeof ctx.inject === 'function') ctx.inject(['jobs'], attachJobs)
  else if (ctx.get?.('jobs') !== undefined) attachJobs({ jobs: ctx.get('jobs'), effect: (execute, label) => ctx.effect(execute, label) })

  const privateOf = (sessionId) => sessionId !== undefined && rt.chatOwners.has(sessionId) && !rt.groupOwners.has(sessionId)
  const liveAgent = (sessionId) => ctx.get('agents')?.get(sessionId)
  // Missing means on. A group Session is not a private chat, so it never matches.
  const interrupts = (id) => rt.botOf?.(rt.chatOwners.get(id))?.interrupt !== false && privateOf(id)

  // Per-session state kept by the session/event listener below.
  const state = new Map()
  const stateOf = (id) => {
    let entry = state.get(id)
    if (entry === undefined) state.set(id, entry = { phase: 'boundary', jobsByCall: new Map(), jobWaiters: new Map(), steerWaiters: new Set(), preemptWaiters: new Set(), killedJobs: new Set(), cutting: false, preempt: false })
    return entry
  }
  const fromUser = (message) => message?.source?.kind === 'user'
  // A steer cuts into the running step. A message queued for the next turn cuts
  // the same way while interrupt is on: the human does not wait out the turn.
  const isUserPreempt = (event) => (event.data?.inserted ?? []).some(fromUser)
    && (event?.data?.target === 'next-step' || event?.data?.target === 'next-turn')
  const wake = (set, value) => { const waiters = [...set]; set.clear(); for (const fn of waiters) fn(value) }
  const newestUser = (list) => [...(list ?? [])].reverse().find(fromUser)

  // A user message into a private chat must cut a running turn. The check runs
  // off the event that queued it, never inside the append that emitted the event.
  // An open tool call is not cancelled: cancelling it would leave the call with
  // no result. The tool hook returns an interrupt result instead, and the model
  // call that follows this step is the one that is cut.
  const maybeCut = (id) => {
    const entry = rt.live.get(id)
    const agent = liveAgent(id)
    if (!interrupts(id) || rt.readOnly?.() !== undefined || entry === undefined || agent === undefined) return
    if ([...(rt.switchJobs?.() ?? [])].some(job => job?.botId === rt.chatOwners.get(id) && job.roomId === undefined)) return
    const track = stateOf(id)
    if (track.cutting) return
    if (entry.turn === undefined || agent.status !== 'running') return
    const stepped = newestUser(agent.inbox?.nextStep)
    const queued = newestUser(agent.inbox?.nextTurn)
    const last = stepped ?? queued
    if (last === undefined) return
    if (track.phase === 'tools' || (entry.calls?.size ?? 0) > 0) {
      track.preempt = true
      wake(track.preemptWaiters, true)
      return
    }
    if (track.phase !== 'model' && track.phase !== 'boundary') return
    track.cutting = true
    rt.markSteerCut(id)
    agent.cancel({ kind: 'hook', reason: STEER_CUT }, { keepInbox: true })
    if (stepped !== undefined) {
      agent.inbox.remove(last.id)
      agent.send(last, 'next-turn', true)
    }
    if (agent.inbox.nextTurn[0]?.id !== last.id) {
      agent.inbox.remove(last.id)
      agent.inbox.prepend('next-turn', last)
    }
  }
  const scheduleCut = (id) => { queueMicrotask(() => maybeCut(id)) }

  // Renders one jobs.read() result the way tool-jobs' renderModelDelta does:
  // stdout and unlabeled chunks joined, then a [stderr] section, then a
  // dropped-output note when the ring lost bytes.
  const renderJobRead = (read) => {
    const visible = (read.chunks ?? []).filter(chunk => chunk?.channel !== 'log')
    const out = visible.filter(chunk => chunk.channel !== 'stderr').map(chunk => chunk.text ?? '').join('')
    const err = visible.filter(chunk => chunk.channel === 'stderr').map(chunk => chunk.text ?? '').join('')
    const separator = out.length > 0 && !out.endsWith('\n') ? '\n' : ''
    const body = out + (err.length > 0 ? `${separator}[stderr]\n${err}` : '')
    const spillPaths = read.job?.output?.spillPaths ?? []
    if (read.lossy !== true && !visible.some(chunk => chunk.gapBefore === true)) return body
    const notice = `[some output was dropped from memory; full output: ${spillPaths.length > 0 ? spillPaths.join(', ') : '(unavailable)'}]`
    return `${body}${body.length > 0 && !body.endsWith('\n') ? '\n' : ''}${notice}`
  }

  const noticeOf = (name, command, text) => createUserMessage({
    content: [{ type: 'text', text }],
    // Same source as DSH's job-completion notice so every chat renders it the
    // same way, and DS Bot's routing ignores it.
    source: { kind: 'tool-jobs', form: 'notice', summary: boundContextSummary(`${name} ${command} …`) },
  })

  const detachAfter = (jobId, args, exec, def, native, label) => {
    const id = exec.agent.id
    const command = args?.command ?? ''
    native.then(value => {
      const track = stateOf(id)
      if (track.killedJobs.delete(jobId) || liveAgent(id) !== exec.agent) return
      const deliver = (text) => {
        const message = noticeOf(label, command, text)
        if (exec.agent.status === 'idle') return exec.agent.followup(message)
        return exec.agent.inject(message)
      }
      if (value?.kind === 'promoted') {
        // The native timeout passed after the detach; the job stays registered
        // and tool-jobs will notify on settle.
        if (typeof value.output === 'string' && value.output !== '') {
          return deliver(`background job ${jobId} (${label}: ${command}) is still running. Output since your last read:\n${value.output}`)
        }
        return
      }
      const rendered = (def.output?.render?.(args, value) ?? [])
        .map(block => block?.text ?? '')
        .filter(text => text !== '')
        .join('\n')
      return deliver(`background job ${jobId} (${label}: ${command}) finished:\n${rendered}`)
    }, error => {
      const track = stateOf(id)
      if (track.killedJobs.delete(jobId) || liveAgent(id) !== exec.agent) return
      const message = noticeOf(label, command, `background job ${jobId} (${label}: ${command}) failed: ${error?.message ?? String(error)}`)
      if (exec.agent.status === 'idle') return exec.agent.followup(message)
      return exec.agent.inject(message)
    })
  }

  const shadowed = new WeakSet()
  const formatWarned = new Set()
  const shadowShell = (agent, name) => {
    if (typeof ctx.tools?.get !== 'function') return
    const def = ctx.tools.get(name, agent)
    if (def === undefined) return
    const promoted = (def.output?.schema?.oneOf ?? []).find(variant => variant?.properties?.kind?.const === 'promoted')
    if (typeof def.output?.render !== 'function'
      || promoted?.properties?.cwd === undefined || promoted?.properties?.jobId === undefined
      || promoted?.properties?.timeoutMs === undefined || promoted?.properties?.output === undefined) {
      if (!formatWarned.has(name)) {
        formatWarned.add(name)
        rt.warn('ds-bot: %s has no promoted result format; its calls keep the step boundary', name)
      }
      return
    }
    shadowed.add(agent)
    const lift = agent.ctx.tools.register({
      ...def,
      execute: async (args, exec) => {
        const native = def.execute(args, exec)
        const sessionId = exec.agent?.id
        if (args?.run_in_background === true || jobs === undefined || sessionId === undefined
          || !interrupts(sessionId)) return native
        const track = stateOf(sessionId)
        const startedAt = Date.now()
        let steerResolve
        let jobResolve
        let graceTimer
        try {
          // A user steer ends the wait: first for the steer event itself, then a
          // short grace for a fast command, then for the job id the native call
          // registers (it may not exist yet while approval is pending).
          const steerSeen = () => {
            if ((exec.agent.inbox?.nextStep ?? []).some(message => message?.source?.kind === 'user')) return Promise.resolve('steered')
            return new Promise(resolve => { steerResolve = resolve; track.steerWaiters.add(resolve) }).then(() => 'steered')
          }
          const first = await Promise.race([native, steerSeen()])
          if (first !== 'steered') return await native
          const graced = await Promise.race([native, new Promise(resolve => { graceTimer = setTimeout(() => resolve('grace'), STEER_GRACE_MS) })])
          if (graced !== 'grace') return await native
          const known = track.jobsByCall.get(exec.callId)
          const jobWait = known !== undefined ? Promise.resolve(known) : new Promise(resolve => {
            jobResolve = resolve
            const set = track.jobWaiters.get(exec.callId) ?? new Set()
            track.jobWaiters.set(exec.callId, set)
            set.add(resolve)
          })
          const settled = await Promise.race([native.then(() => 'native'), jobWait.then(() => 'job')])
          if (settled !== 'job') return await native
        } catch {
          // A native rejection before or after the steer propagates unchanged.
          return await native
        } finally {
          if (steerResolve !== undefined) track.steerWaiters.delete(steerResolve)
          if (jobResolve !== undefined) {
            const set = track.jobWaiters.get(exec.callId)
            set?.delete(jobResolve)
            if (set?.size === 0) track.jobWaiters.delete(exec.callId)
          }
          clearTimeout(graceTimer)
        }
        const jobId = track.jobsByCall.get(exec.callId)
        let read
        try {
          read = jobs.read(jobId, sessionId)
        } catch {
          return await native
        }
        const status = read?.job?.status
        if (status === 'completed' || status === 'killed' || status === 'failed') return await native
        detachAfter(jobId, args, exec, def, native, name)
        const base = rt.sessionFolder?.(exec.agent.session)
        const cwd = args?.workdir === undefined ? (base ?? '')
          : !isAbsolute(args.workdir) && base !== undefined ? resolve(base, args.workdir) : args.workdir
        return { kind: 'promoted', cwd, jobId, timeoutMs: Date.now() - startedAt, output: renderJobRead(read) }
      },
    })
    // tools.js keeps this with the Session's scope; unmarking lets a later
    // scope register the shadow again.
    return () => { shadowed.delete(agent); lift?.() }
  }
  rt.shadowShells = (agent) => {
    if (agent === undefined || !rt.isBotSession(agent.session?.id)) return
    if (shadowed.has(agent) || typeof agent.ctx?.tools?.register !== 'function') return
    const disposers = []
    for (const name of SHELL_TOOLS) {
      try {
        const lift = shadowShell(agent, name)
        if (typeof lift === 'function') disposers.push(lift)
      } catch (error) {
        rt.warn('ds-bot: shadowing %s failed: %s', name, error)
      }
    }
    return disposers
  }

  ctx.on('session/event', (session, event) => {
    const id = session?.id
    if (id === undefined || !privateOf(id)) return
    const track = stateOf(id)
    switch (event.type) {
      case 'step/start': {
        // A preempt from the tool step cuts this model call. A queued user
        // message does not: the turn that just opened is the one that claims it,
        // and cutting it again would spin.
        track.phase = 'model'
        const wasPreempt = track.preempt
        track.preempt = false
        const agent = liveAgent(id)
        if (wasPreempt || (agent?.inbox?.nextStep ?? []).some(fromUser)) scheduleCut(id)
        break
      }
      case 'assistant/message':
        track.phase = 'tools'
        break
      case 'step/end':
        track.phase = 'boundary'
        track.cutting = false
        break
      case 'turn/start':
      case 'turn/end':
        track.phase = 'boundary'
        track.cutting = false
        track.preempt = false
        break
      case 'job/registered': {
        const callId = event.data?.toolCall?.callId
        if (callId !== undefined) {
          track.jobsByCall.set(callId, event.data.jobId)
          const set = track.jobWaiters.get(callId)
          if (set !== undefined) { track.jobWaiters.delete(callId); wake(set, event.data.jobId) }
        }
        break
      }
      case 'job/settled':
        if (event.data?.toolCall?.callId !== undefined) track.jobsByCall.delete(event.data.toolCall.callId)
        break
      case 'tool/call': {
        if (event.data?.name === 'job_kill') {
          try {
            const parsed = JSON.parse(event.data.arguments ?? '{}')
            if (parsed.job_id !== undefined) track.killedJobs.add(parsed.job_id)
          } catch { /* unparseable args carry no kill target */ }
        }
        break
      }
      case 'agent/inbox/spliced':
        if (interrupts(id) && isUserPreempt(event)) {
          wake(track.steerWaiters, true)
          scheduleCut(id)
        }
        break
    }
  })

  // A tool already inside its call cannot be cancelled without leaving the call
  // unmatched. Returning here lets the step record a result and end; the model
  // call that follows is cut by the listener above. The call itself may still
  // finish in the background. Registered by background.js, after the privacy
  // guard, so that guard stays the first tools/execute listener.
  rt.preemptedTool = async (exec, next) => {
    const id = exec.agent?.id ?? exec.agent?.session?.id
    if (!interrupts(id) || rt.readOnly?.() !== undefined) return next()
    const track = stateOf(id)
    if (track.preempt) return INTERRUPT_RESULT
    let resolve
    const preempted = new Promise(done => { resolve = done; track.preemptWaiters.add(done) })
    const pending = Promise.resolve().then(() => next())
    try {
      const outcome = await Promise.race([
        pending.then(value => ({ kind: 'tool', value }), error => ({ kind: 'error', error })),
        preempted.then(() => ({ kind: 'preempt' })),
      ])
      if (outcome.kind === 'preempt') return INTERRUPT_RESULT
      if (outcome.kind === 'error') throw outcome.error
      return outcome.value
    } finally {
      if (resolve !== undefined) track.preemptWaiters.delete(resolve)
    }
  }
}
