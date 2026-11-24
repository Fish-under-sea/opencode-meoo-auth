// OpenCode & Magpie provider plugin for Meoo 秒悟 (meoo.com).
//
// 由 dsh-meoo-connect（dsh/DeepSeek Harness 插件）迁移而来，同一套 Meoo
// 官方「AI 服务」链路，换到 Magpie 的 provider 插件平面上：
//
//   登录      OAuth Authorization Code + PKCE（public client，本机 loopback 回调）
//   项目 AK   控制面 /open/v1/cli-compat/...（身份 → 项目 → 云服务 environment）
//   模型目录  网关 {gateway}/models（OpenAI 兼容，Bearer 项目服务 AK）
//   转发      不自己做适配器——声明 @ai-sdk/openai-compatible，由宿主 SDK 走网关
//
// 三种登录方式：
//   1. 浏览器授权：点一次「授权」，之后身份/项目/云服务 AK 全自动解析。
//   2. 复用本机登录态：只读 dsh-meoo-connect（$DSH_HOME/meoo/state.json）或
//      meoo-proxy（~/.meoo-proxy/config.json）已保存的登录态，连授权点击都省了。
//   3. 手填项目服务 AK：meoo_project_service_v1.* 开头，适合不想走 OAuth 的人。
//
// 凭证只存放在宿主的账号记录里（client.auth.set），本插件不落地任何文件。
// 调用消耗 Meoo 账号自身额度，不绕过任何计费；与 Meoo 官方无关联。

import { createHash, randomBytes } from "node:crypto"
import { createServer } from "node:http"
import { readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

const PROVIDER_ID = "meoo"
// 外层显示名：Magpie 的插件/供应商列表里露出来的就是这一条短名 "Meoo"；
// 点进去看到的完整包名来自宿主读到的插件 spec —— @fish-under-sea/opencode-meoo-auth。
const PROVIDER_NAME = "Meoo"
// 账号记录里没有用户名时的兜底称呼，保留中文全名，免得两个 Meoo 混淆。
const ACCOUNT_FALLBACK = "Meoo 秒悟"
const NPM = "@ai-sdk/openai-compatible"

const DEFAULT_API_BASE = "https://meoo.com"
const DEFAULT_GATEWAY = "https://api.meoo.host/meoo-ai/compatible-mode/v1"
const PROD_CLIENT_ID = "meoo_app_tdWlRbpvAxN1EjmZ8XICxdTh"
const TEST_CLIENT_ID = "meoo_app_SFNvmHUFNePxyyfGjCd9hyaE"
const OAUTH_SCOPE = "cli.compat"
const CALLBACK_PATH = "/oauth/callback"
// 控制面要求 CLI 客户端标记（与 dsh-meoo-connect 同一套公开契约）。
const CLI_HEADERS = { "x-meoo-client": "cli", "x-meoo-cli-version": "0.5.6" }

const EARLY_MS = 60 * 1000
const LEAD_MS = 5 * 60 * 1000
const KEEP_RENEWED_MS = 60 * 1000
const CATALOG_TTL_MS = 10 * 60 * 1000
// 目录里认不出的新模型先按保守值给，等它也进规格表。
const CONTEXT_WINDOW = 200_000
const DEFAULT_MAX_TOKENS = 64_000

// Meoo 网关对单个请求体的硬上限正好 1 MiB（2026-11 实测：1,048,111 B 放行、
// 1,048,687 B 拒绝）。AI SDK 不知道这个限制，长会话（历史 + 工具输出 + 工具
// schema）序列化后轻易超过 2 MB，于是 100% 413。这里在请求发出前把 body 压到
// 安全水位以下，并把 413 当「还要再压一轮」的信号重试一次。
const MAX_BODY_BYTES = 900 * 1024

/**
 * 模型规格表。网关 /models 只回裸 id，不带上下文/输出上限，所以数字得自己带。
 * 取值来自各家官方文档（2026-11 核对）：
 *   Qwen 3.8/3.7 全家 1M 上下文 / 131072 输出；
 *   DeepSeek V4 全家 1M / 384K；
 *   GLM-5.2 与 GLM-5.2-Fast 1M / 128K；
 *   Qwen VL 262144 / 32768。
 * image 只给官方写明原生多模态的那几个，其余按纯文本，免得让 UI 递图片。
 */
const MODEL_SPECS = [
  { id: "qwen3.8-flash", name: "Qwen3.8-Flash", context: 1_000_000, output: 131_072, image: true },
  { id: "qwen3.8-max", name: "Qwen3.8-Max", context: 1_000_000, output: 131_072, image: true },
  { id: "qwen3.7-max", name: "Qwen3.7-Max", context: 1_000_000, output: 131_072 },
  { id: "qwen3.7-plus", name: "Qwen3.7-Plus", context: 1_000_000, output: 131_072, image: true },
  { id: "deepseek-v4.1-flash-1017", name: "DeepSeek V4.1 Flash", context: 1_000_000, output: 384_000, image: true },
  { id: "deepseek-v4-pro-0813", name: "DeepSeek V4 Pro", context: 1_000_000, output: 384_000 },
  { id: "deepseek-v4-flash-0731", name: "DeepSeek V4 Flash", context: 1_000_000, output: 384_000 },
  { id: "glm-5.2-fast-preview", name: "GLM-5.2-Fast", context: 1_000_000, output: 128_000 },
  { id: "glm-5.2", name: "GLM-5.2", context: 1_000_000, output: 128_000 },
  { id: "qwen3-vl-plus", name: "Qwen VL", context: 262_144, output: 32_768, image: true },
]

/**
 * 候选模型：文档里写了、网关 /models 却不一定列出来的 id。
 *
 * 2026-11 实测：Meoo 文档「支持模型介绍」把 Kimi K3 / K3.1 Pro、GLM-5.3 / 5.3-Flash、
 * MiniMax M3.1 Flash、Qwen3.8-Max-0902 都算作平台模型，但项目的 OpenAI 兼容网关
 * （/meoo-ai/compatible-mode/v1）只收 9 个文本 id：
 *   qwen3.8-flash、qwen3.8-max、qwen3.7-max、qwen3.7-plus、deepseek-v4.1-flash-1017、
 *   deepseek-v4-pro-0813、deepseek-v4-flash-0731、glm-5.2-fast-preview、glm-5.2
 * 其余一律 400「当前仅支持配置模型: …」。Kimi / GLM-5.3 / MiniMax 走的是 Meoo 的
 * Agent 产品（/open/v1/projects/<id>/agent/runs），不是这个网关——那是
 * magpie-plugin-meoo-agent 的活。
 *
 * 所以这里逐个探：网关接了才进目录，不接就当它不存在。Meoo 哪天放开，插件下次
 * 拉目录就自动带上，不需要发版。qwen3-vl-plus 就是这样被探出来的（文档写了、
 * /models 不列、网关可调）。
 */
const SUPPLEMENT_MODELS = [
  "qwen3-vl-plus", // 视觉理解：实测可用
  "qwen3.8-max-0902",
  "glm-5.3",
  "glm-5.3-flash",
  "kimi-k3",
  "kimi-k3.1-pro",
  "minimax-m3.1-flash",
]

/** 用户自己往目录里塞 id：逗号分隔。仍然要过网关那一探，不接的不进目录。 */
const extraModels = () =>
  String(process.env.MEOO_EXTRA_MODELS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)

const MODEL_BY_ID = new Map(MODEL_SPECS.map((m) => [m.id, m]))

/** 目录里出现的未知 id 当纯文本、给保守上限，不把 1M 乱许出去。 */
function specOf(id) {
  return MODEL_BY_ID.get(id) ?? { id, name: id, context: CONTEXT_WINDOW, output: DEFAULT_MAX_TOKENS }
}

const mainOrigin = () => String(process.env.MEOO_MAIN_SITE_URL || DEFAULT_API_BASE).replace(/\/+$/, "")
const gatewayBase = () => String(process.env.MEOO_GATEWAY_BASE_URL || DEFAULT_GATEWAY).replace(/\/+$/, "")
// apiBase 跟着账号记录走（登录用的哪个站点，续期/控制面就打哪个），环境变量只是兜底。
const apiBaseOf = (a) => String(a?.apiBase || mainOrigin()).replace(/\/+$/, "")

function clientIdFor(apiBaseUrl) {
  try {
    return new URL(apiBaseUrl).hostname.toLowerCase() === "meoo.com" ? PROD_CLIENT_ID : TEST_CLIENT_ID
  } catch {
    return TEST_CLIENT_ID
  }
}

const labelOf = (id) => specOf(id).name
const int = (n, fallback = 0) => {
  const v = Math.trunc(Number(n))
  return Number.isFinite(v) && v >= 0 ? v : fallback
}
const vendorError = (json, text) => json?.detail || json?.message || String(text ?? "").slice(0, 160)

/**
 * 最早作废的一笔积分：返回 { day: "YYYY-MM-DD", remaining }，没有则 null。
 * 对应 magpie 的 runsOutAt / runsOut —— 路由在没有重置时间的积分上就按它排先后。
 */
function soonestLapse(grants) {
  let best = null
  for (const g of Array.isArray(grants) ? grants : []) {
    for (const d of g?.grantDetails || []) {
      const remaining = int(d?.remainingAmount)
      if (remaining <= 0) continue
      const day = String(d?.expiresDate || "").slice(0, 10)
      if (!day) continue
      if (!best || day < best.day) best = { day, remaining }
    }
  }
  return best
}

/**
 * 官方「每日登录赠送积分」的只读视图：从套餐权益里取 `daily_login`。
 *
 * 只读取、不代领：按 Meoo 的规则，每日首次登录由服务端发放（pricing.md
 * 「每日赠送」），插件不在后台模拟登录或调领取接口。权益接口偶尔不可用时
 * 返回 null，余额卡照常显示。
 */
function dailyLoginOf(entitlement) {
  const benefits = Array.isArray(entitlement?.benefits) ? entitlement.benefits : []
  const b = benefits.find((x) => x?.typeCode === "daily_login" || x?.typeName === "每日登录赠送积分")
  if (!b) return null
  const raw = Math.trunc(Number(b.quota))
  // -1 在权益里表示「不限」；0 或解析不出来按没有该权益处理。
  if (!Number.isFinite(raw) || raw === 0) return null
  return { quota: raw, unit: String(b.unit || "积分") }
}

// ---- OAuth（Authorization Code + PKCE，public client）----------------------

async function tokenRequest(apiBaseUrl, params) {
  const res = await fetch(`${apiBaseUrl}/oauth/token`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded", accept: "application/json" },
    body: new URLSearchParams(params).toString(),
    signal: AbortSignal.timeout(20_000),
  })
  const text = await res.text()
  let json = {}
  try {
    json = JSON.parse(text)
  } catch {}
  if (!res.ok) {
    const err = new Error(`OAuth 换票失败 HTTP ${res.status}: ${json.error || text.slice(0, 160)}`)
    if (json.error) err.code = json.error
    // invalid_grant 是 OAuth 的终态：refresh token 已被作废，重试也不会好。
    // 宿主读到 signIn:"expired" 会把账号标为需要重新登录并停止退避重试，
    // 否则它会照着一次失败继续按退避重试一个已经死掉的 token。
    if (json.error === "invalid_grant" || (!json.error && res.status === 400 && /invalid_grant/.test(text))) {
      err.signIn = "expired"
      err.code = "invalid_grant"
    }
    throw err
  }
  return json
}

// 单飞续期：同一 refresh token 的并发续期共享一次请求，60 秒内的结果直接复用。
const renewed = new Map()
const renewing = new Map()

async function callRefreshApi(a) {
  const base = apiBaseOf(a)
  const token = await tokenRequest(base, {
    grant_type: "refresh_token",
    client_id: clientIdFor(base),
    refresh_token: a.refresh,
  })
  return {
    access: token.access_token,
    ...(token.refresh_token ? { refresh: token.refresh_token } : {}),
    expires: Date.now() + int(token.expires_in || 3600, 3600) * 1000,
    ...(token.scope ? { scope: token.scope } : {}),
    ...(token.openid ? { openid: token.openid } : {}),
  }
}

async function renewToken(a, save) {
  const key = a.refresh
  const done = renewed.get(key)
  if (done && Date.now() - done.at < KEEP_RENEWED_MS) {
    // 这一代已经在缓存里（换票那趟存过一次），把它存回宿主
    const merged = { ...a, ...done.next }
    await save?.(merged)
    return merged
  }
  let r = renewing.get(key)
  if (!r) {
    r = (async () => {
      const next = await callRefreshApi(a)
      // 存成功之后才放进缓存：Meoo 的 refresh token 每次续期都换新的，旧的
      // 立刻作废，所以这一代 token 只有一份。存回宿主失败时（persistOf 会
      // 抛），若这里已经缓存了它，缓存过期后这一代就彻底丢了——下一次续期会
      // 带着已经作废的旧 token，得到 invalid_grant，看起来和「服务端吊销」一样。
      await save?.({ ...a, ...next })
      renewed.set(key, { next, at: Date.now() })
      return next
    })().finally(() => renewing.delete(key))
    renewing.set(key, r)
  }
  // 换票那一趟已经存过；走到这里的都是等同一个并发续期的调用方，不必再存。
  const next = await r
  return { ...a, ...next }
}

// Meoo 的 access token 过期时间是登录/续期时给的 expires_in，不是 JWT 声明；
// 没有 expires 就当过期（与 dsh 的 ensureAccessToken 语义一致）。
function isStale(a, marginMs = EARLY_MS) {
  const exp = Number(a?.expires) || 0
  return !(exp > Date.now() + marginMs)
}

async function currentAuth(ctx, auth, force = false) {
  let a = { ...auth }
  for (let i = 0; i < 8 && a.refresh; i++) {
    const d = renewed.get(a.refresh)
    if (!d || Date.now() - d.at >= KEEP_RENEWED_MS) break
    a = { ...a, ...d.next }
  }
  const alreadyRenewed = a.access !== auth.access || a.refresh !== auth.refresh
  if ((alreadyRenewed || !force) && !isStale(a, EARLY_MS)) {
    if (alreadyRenewed) await persistOf(ctx)(a)
    return { a, didRenew: alreadyRenewed }
  }
  if (!a.refresh) {
    if (!force && a.access && !isStale(a, 0)) return { a, didRenew: alreadyRenewed }
    const err = new Error("Meoo 登录态已过期且没有 refresh token，请重新登录")
    err.signIn = "expired"
    throw err
  }
  const next = await renewToken(a, persistOf(ctx))
  return { a: next, didRenew: true }
}

// ---- 控制面：账号 / 项目 / 项目服务 AK --------------------------------------

async function controlCall(apiBase, access, method, apiPath, body) {
  const res = await fetch(`${apiBase}${apiPath}`, {
    method,
    headers: {
      accept: "application/json",
      "content-type": "application/json",
      ...CLI_HEADERS,
      authorization: `Bearer ${access}`,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(45_000),
  })
  const text = await res.text()
  let json = null
  try {
    json = text ? JSON.parse(text) : null
  } catch {
    json = null
  }
  // Meoo 的业务错误也走 HTTP 200（{"success":false,"errCode":...}），要在这里拦下来
  if (json && typeof json === "object" && json.success === false) {
    const err = new Error(`${apiPath} 失败: ${json.message || json.errCode || "unknown error"}`)
    err.status = res.status
    err.errCode = json.errCode
    throw err
  }
  if (!res.ok) {
    const err = new Error(
      res.status === 401
        ? `${apiPath} 失败 HTTP 401（登录态已失效，请重新登录）: ${vendorError(json, text)}`
        : `${apiPath} 失败 HTTP ${res.status}: ${vendorError(json, text)}`,
    )
    err.status = res.status
    throw err
  }
  return json?.data !== undefined ? json.data : json
}

/** 运行期控制面调用：先确保 access 新鲜，401 时强制续期重试一次。 */
async function controlRequest(ctx, a, method, apiPath, body) {
  const { a: fresh } = await currentAuth(ctx, a)
  try {
    return await controlCall(apiBaseOf(fresh), fresh.access, method, apiPath, body)
  } catch (e) {
    if (e?.status === 401 && fresh.refresh) {
      const next = await renewToken(fresh, persistOf(ctx))
      return controlCall(apiBaseOf(fresh), next.access, method, apiPath, body)
    }
    throw e
  }
}

async function fetchIdentity(ctx, a) {
  return await controlRequest(ctx, a, "GET", "/open/v1/cli-compat/account/identity")
}

// 账号版本（identity.edition）当套餐名上报：宿主用 plan 给账号分组/折叠。
// 身份接口不必每轮额度都问，缓存 10 分钟；失败就不给 plan，不拖垮额度读取。
const IDENTITY_TTL = 10 * 60 * 1000
const planCache = new Map()
const EDITION_LABELS = { personal: "个人版", team: "团队版", enterprise: "企业版" }
async function planOf(ctx, a) {
  const key = `${apiBaseOf(a)}|${a.openid || a.userId || a.username || ""}`
  const hit = planCache.get(key)
  if (hit && Date.now() - hit.at < IDENTITY_TTL) return hit.plan
  try {
    const id = await fetchIdentity(ctx, a)
    const edition = String(id?.edition || "").trim().toLowerCase()
    const plan = EDITION_LABELS[edition] || edition
    planCache.set(key, { at: Date.now(), plan })
    return plan
  } catch {
    return ""
  }
}

async function listProjects(ctx, a, limit = 50) {
  const data = await controlRequest(ctx, a, "GET", `/open/v1/cli-compat/projects?page_size=${limit}`)
  return data?.list || data?.items || data?.projects || (Array.isArray(data) ? data : [])
}

/**
 * 新账号常常一个项目都没有（项目服务 AK 挂在项目下，没有项目就没有 AK）。
 * 控制面的项目创建接口已验证可用：先试带名字，失败退回空请求体（Meoo 会建
 * 一个 "untitled" 项目）。创建失败返回 null，由调用方决定怎么报错。
 */
async function createProject(ctx, a) {
  try {
    return await controlRequest(ctx, a, "POST", "/open/v1/cli-compat/projects", { name: "AI 服务" })
  } catch {
    try {
      return await controlRequest(ctx, a, "POST", "/open/v1/cli-compat/projects", {})
    } catch {
      return null
    }
  }
}

/**
 * 选中一个项目：显式指定 > 记录里的 projectUrlId > 身份默认项目 > 唯一项目。
 * 多个项目时不替用户做主（计费归属不同），报错并列出可选项。
 */
async function pickProject(ctx, a, explicitUrlId) {
  if (explicitUrlId) return explicitUrlId
  let preferred = a?.projectUrlId || ""
  if (!preferred) {
    try {
      const me = await fetchIdentity(ctx, a)
      preferred = me?.project_url_id || me?.projectUrlId || ""
    } catch {}
  }
  if (preferred) return preferred
  const projects = await listProjects(ctx, a)
  if (!projects.length) {
    const created = await createProject(ctx, a)
    const urlId = created?.url_id || created?.urlId
    if (!urlId) throw new Error("账号下还没有项目，自动创建也失败了：请在 meoo.com 手动建一个项目后重试")
    return urlId
  }
  if (projects.length === 1) {
    const only = projects[0]?.url_id || projects[0]?.urlId
    if (!only) throw new Error("项目列表里没有 url_id，无法定位项目")
    return only
  }
  const options = projects.map((p) => `${p.url_id || p.urlId}（${p.name || "未命名"}）`).join("、")
  const err = new Error(`账号下有多个项目，请改用「项目服务 AK」登录方式并指定项目，或先在 meoo.com 设为默认项目。可选：${options}`)
  err.projects = projects
  throw err
}

/** 取项目服务 AK；项目没开云服务时自动开通一次再取。 */
async function resolveServiceAk(ctx, a, { projectUrlId } = {}) {
  const urlId = await pickProject(ctx, a, projectUrlId)
  if (a?.serviceAk && a.serviceAkProjectId === urlId) return { urlId, ak: a.serviceAk }

  const envPath = `/open/v1/cli-compat/projects/${encodeURIComponent(urlId)}/cloud/environment`
  let ak = null
  try {
    const env = await controlRequest(ctx, a, "GET", envPath)
    ak = env?.meooProjectApiKey || env?.instance?.meooProjectApiKey || null
  } catch {
    ak = null
  }
  let cloudErrMessage = null
  if (!ak) {
    try {
      await controlRequest(ctx, a, "POST", `/open/v1/cli-compat/projects/${encodeURIComponent(urlId)}/cloud`, {
        description: "opencode-meoo-auth: AI 服务",
      })
      const env = await controlRequest(ctx, a, "GET", envPath)
      ak = env?.meooProjectApiKey || env?.instance?.meooProjectApiKey || null
    } catch (e) {
      cloudErrMessage = e?.message ?? String(e)
      ak = null
    }
  }
  if (!ak) {
    throw new Error(
      `项目 ${urlId} 还没开通 Meoo AI 服务：在 meoo.com 打开该项目，在对话框里发一句「帮我开启云服务」（或让 AI 加一个需要后端/AI 的功能，平台会自动开通并签发 AK），开通后重试` +
        (cloudErrMessage ? `（自动开通失败：${cloudErrMessage}）` : ""),
    )
  }
  return { urlId, ak }
}

/** 运行期取 AK：有缓存就用缓存，否则解析并持久化进账号记录。 */
async function serviceAkOf(ctx, a) {
  const urlId = await pickProject(ctx, a)
  if (a?.serviceAk && a.serviceAkProjectId === urlId) return a.serviceAk
  const { urlId: resolved, ak } = await resolveServiceAk(ctx, a, { projectUrlId: urlId })
  // 这里存的是 AK 不是换来的 token：存不住只是下次要再解析一次，这一趟已经
  // 拿到 AK 了，所以不让它把整个操作拖垮。token 那边的存不住是致命的，那边不吞。
  try {
    await persistOf(ctx)({ ...a, projectUrlId: resolved, serviceAk: ak, serviceAkProjectId: resolved })
  } catch {}
  return ak
}

// ---- 模型目录 ---------------------------------------------------------------

let catalogCache = { origin: "", models: null, at: 0 }

const resetCatalogCache = () => {
  catalogCache = { origin: "", models: null, at: 0 }
}

function parseGatewayModels(payload) {
  const list = payload?.models || payload?.data || (Array.isArray(payload) ? payload : [])
  const seen = new Set()
  const out = []
  for (const entry of Array.isArray(list) ? list : []) {
    const id = typeof entry === "string" ? entry.trim() : String(entry?.id ?? "").trim()
    if (!id || seen.has(id)) continue
    seen.add(id)
    const spec = specOf(id)
    out.push({
      id,
      name: (typeof entry === "object" && entry?.name) || spec.name,
      context: int(entry?.context) || spec.context,
      output: int(entry?.max_output || entry?.max_output_tokens || entry?.max_tokens) || spec.output,
      image: !!spec.image,
    })
  }
  return out
}

/**
 * 探一个目录外的模型：1 token 的对话请求，网关接了才算可用。
 * 只用来补 SUPPLEMENT_MODELS，结果随目录一起缓存，10 分钟最多探一次。
 */
async function acceptsModel(ak, id) {
  try {
    const res = await fetch(`${gatewayBase()}/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${ak}`, "content-type": "application/json" },
      body: JSON.stringify({ model: id, messages: [{ role: "user", content: "hi" }], max_tokens: 1, stream: false }),
      signal: AbortSignal.timeout(20_000),
    })
    return res.ok
  } catch {
    return false
  }
}

/** 并发探多个 id，限速 4 条同时，别把网关打满。 */
async function probeModels(ak, ids) {
  const out = []
  const queue = [...ids]
  const worker = async () => {
    for (;;) {
      const id = queue.shift()
      if (!id) return
      if (await acceptsModel(ak, id)) out.push(id)
    }
  }
  await Promise.all(Array.from({ length: Math.min(4, ids.length) }, worker))
  return out
}

async function fetchGatewayModels(ak) {
  const base = gatewayBase()
  if (catalogCache.models && catalogCache.origin === base && Date.now() - catalogCache.at < CATALOG_TTL_MS) {
    return catalogCache.models
  }
  const res = await fetch(`${base}/models`, {
    headers: { authorization: `Bearer ${ak}`, accept: "application/json" },
    signal: AbortSignal.timeout(25_000),
  })
  if (!res.ok) throw new Error(`模型目录 HTTP ${res.status}`)
  const models = parseGatewayModels(await res.json())
  if (!models.length) throw new Error("empty catalog")
  const seen = new Set(models.map((m) => m.id))
  // 文档里有、目录里没有的 id 逐个探；接了就补进目录，不接的不进。
  const missing = [...new Set([...SUPPLEMENT_MODELS, ...extraModels()])].filter((id) => !seen.has(id))
  for (const id of await probeModels(ak, missing)) {
    const spec = specOf(id)
    models.push({ id, name: spec.name, context: spec.context, output: spec.output, image: !!spec.image })
  }
  catalogCache = { origin: base, models, at: Date.now() }
  return models
}

// 思考档位：AI SDK 的 openai-compatible 线把 reasoningEffort 映射成请求体的
// reasoning_effort。网关实测（qwen3.8-flash，2026-11）：low/medium/high 都被
// 接受且真的影响思考长度（reasoning tokens：low≈13、medium/high≈30），不带
// 该参数时走上游默认思考。对应 UI 的 快速/适中/极致 三档。
const REASONING_VARIANTS = {
  low: { reasoningEffort: "low" },
  medium: { reasoningEffort: "medium" },
  high: { reasoningEffort: "high" },
}

function configModel(m) {
  const spec = specOf(m.id)
  const context = m.context ?? spec.context
  const output = m.output ?? spec.output
  const image = m.image ?? spec.image ?? false
  return {
    name: m.name ?? spec.name,
    limit: { context, output },
    tool_call: true,
    reasoning: true,
    attachment: image,
    modalities: { input: image ? ["text", "image"] : ["text"], output: ["text"] },
    variants: REASONING_VARIANTS,
  }
}

function configModels() {
  return Object.fromEntries(MODEL_SPECS.map((m) => [m.id, configModel(m)]))
}

function runtimeModel(provider, m) {
  const spec = specOf(m.id)
  const context = m.context ?? spec.context
  const output = m.output ?? spec.output
  const image = m.image ?? spec.image ?? false
  return {
    id: m.id,
    providerID: provider.id,
    name: m.name ?? spec.name,
    api: { id: m.id, url: gatewayBase(), npm: NPM },
    status: "active",
    headers: {},
    options: {},
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: { context, output },
    capabilities: {
      temperature: true,
      reasoning: true,
      attachment: image,
      toolcall: true,
      input: { text: true, image, audio: false, video: false, pdf: false },
      output: { text: true, image: false, audio: false, video: false, pdf: false },
      interleaved: false,
    },
    release_date: "",
    variants: REASONING_VARIANTS,
  }
}

// ---- 登录 -------------------------------------------------------------------

/**
 * 登录成功后的账号记录：token + 控制面解析出的身份/项目/AK，一起交给宿主保存。
 */
async function buildSignedInAuth(tokens, apiBase) {
  const access = tokens.access_token
  if (!access) throw new Error("OAuth 响应里没有 access_token")
  const a = {
    apiBase: apiBase.replace(/\/+$/, ""),
    access,
    refresh: tokens.refresh_token || "",
    expires: Date.now() + int(tokens.expires_in || 3600, 3600) * 1000,
    ...(tokens.scope ? { scope: tokens.scope } : {}),
    ...(tokens.openid ? { openid: tokens.openid } : {}),
  }
  const me = await controlCall(apiBase, access, "GET", "/open/v1/cli-compat/account/identity")
  if (me?.name) a.username = me.name
  if (me?.user_id) a.userId = me.user_id
  if (me?.project_url_id || me?.projectUrlId) a.projectUrlId = me.project_url_id || me.projectUrlId
  const { urlId, ak } = await resolveServiceAk({ client: null }, a)
  a.projectUrlId = urlId
  a.serviceAk = ak
  a.serviceAkProjectId = urlId
  return {
    type: "success",
    access: a.access,
    refresh: a.refresh,
    expires: a.expires,
    accountId: a.username || a.openid || ACCOUNT_FALLBACK,
    ...(a.username ? { username: a.username } : {}),
    ...(a.userId ? { userId: a.userId } : {}),
    ...(a.openid ? { openid: a.openid } : {}),
    ...(a.scope ? { scope: a.scope } : {}),
    apiBase: a.apiBase,
    projectUrlId: a.projectUrlId,
    serviceAk: a.serviceAk,
    serviceAkProjectId: a.serviceAkProjectId,
  }
}

/** 方式一：浏览器 PKCE 授权，本机 loopback 自动收回调，AK 全自动解析。 */
async function browserSignIn(timeoutMs = 10 * 60 * 1000) {
  const apiBase = mainOrigin()
  const clientId = clientIdFor(apiBase)
  const verifier = randomBytes(32).toString("base64url")
  const challenge = createHash("sha256").update(verifier).digest("base64url")
  const state = randomBytes(32).toString("base64url")

  let complete, timer, origin, received = false, settled = false
  const done = new Promise((resolve) => {
    complete = resolve
  })
  const finish = (result) => {
    if (settled) return
    settled = true
    clearTimeout(timer)
    complete(result)
    server.close()
    server.closeIdleConnections?.()
  }
  const server = createServer(async (req, res) => {
    const reply = (status, text) => {
      res.writeHead(status, {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": "default-src 'none'",
      })
      res.end(text)
    }
    const url = new URL(req.url || "/", "http://127.0.0.1")
    if (req.headers.host !== origin || url.pathname !== CALLBACK_PATH) return reply(404, "Not found")
    if (req.method !== "GET") return reply(405, "Method not allowed")
    const q = url.searchParams
    if (q.getAll("state").length !== 1 || q.get("state") !== state) return reply(400, "回调不属于本次登录，请回到原授权页重试。")
    if (settled || received) return reply(409, "本次回调已经处理，请回到 Magpie 查看结果。")
    if (q.get("error")) {
      reply(400, "登录未获授权，请回到 Magpie 重新登录。")
      finish({ type: "failed", error: `授权被取消：${q.get("error")}${q.get("error_description") ? " " + q.get("error_description") : ""}` })
      return
    }
    const code = q.get("code") || ""
    if (q.getAll("code").length !== 1 || !code || code.length > 4096) return reply(400, "回调缺少有效授权码，请回到原授权页重试。")
    received = true
    try {
      const tokens = await tokenRequest(apiBase, {
        grant_type: "authorization_code",
        client_id: clientId,
        code,
        redirect_uri: `http://${origin}${CALLBACK_PATH}`,
        code_verifier: verifier,
      })
      const success = await buildSignedInAuth(tokens, apiBase)
      if (settled) return reply(408, "登录已超时，请回到 Magpie 重新登录。")
      reply(200, "Meoo 登录成功，可以关闭这个页面回到 Magpie。")
      finish(success)
    } catch (e) {
      // 错误详情直接写进回调页：Magpie 卡片上的报错不一定被看到，浏览器这页一定看得到
      reply(502, `登录未完成：${e?.message ?? String(e)}\n\n请返回 Magpie 处理后重新登录。`)
      finish({ type: "failed", error: e?.message ?? String(e) })
    }
  })
  await new Promise((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", resolve)
  })
  origin = `127.0.0.1:${server.address().port}`
  const authorize = new URL("/oauth/authorize", apiBase)
  authorize.searchParams.set("client_id", clientId)
  authorize.searchParams.set("redirect_uri", `http://${origin}${CALLBACK_PATH}`)
  authorize.searchParams.set("response_type", "code")
  authorize.searchParams.set("scope", OAUTH_SCOPE)
  authorize.searchParams.set("state", state)
  authorize.searchParams.set("code_challenge", challenge)
  authorize.searchParams.set("code_challenge_method", "S256")
  timer = setTimeout(() => finish({ type: "failed", error: "Meoo 登录超时，请重新登录" }), timeoutMs)
  timer.unref?.()
  return {
    url: authorize.toString(),
    instructions: "在浏览器登录 Meoo 并点一次「授权」后，会自动返回此处；项目和 AI 服务 AK 都会自动解析。",
    method: "auto",
    async callback() {
      return await done
    },
  }
}

// ---- 本机已有登录态（只读复用，不写入任何文件）------------------------------

function dshHome() {
  return process.env.DSH_HOME || join(homedir(), ".dsh")
}

/**
 * dsh-meoo-connect（$DSH_HOME/meoo/state.json）与 meoo-proxy
 * （~/.meoo-proxy/config.json）保存的登录态，两处都读、dsh 优先、缺的互填。
 */
function readLegacyState(overrides = {}) {
  const dshFile = overrides.dshState || join(dshHome(), "meoo", "state.json")
  const proxyFile = overrides.proxyConfig || join(overrides.proxyHome || process.env.MEOO_PROXY_HOME || join(homedir(), ".meoo-proxy"), "config.json")
  const readJson = (file) => {
    try {
      return JSON.parse(readFileSync(file, "utf8"))
    } catch {
      return null
    }
  }
  const dsh = readJson(dshFile) || {}
  const proxy = readJson(proxyFile) || {}
  const merged = { ...proxy, ...Object.fromEntries(Object.entries(dsh).filter(([, v]) => v !== null && v !== undefined && v !== 0)) }
  if (!merged.refreshToken && !proxy.refreshToken && !merged.serviceAk && !proxy.serviceAk) return null
  return {
    apiBase: merged.apiBaseUrl || DEFAULT_API_BASE,
    accessToken: merged.accessToken || null,
    refreshToken: merged.refreshToken || proxy.refreshToken || null,
    expires: Number(merged.accessExpiresAt) || 0,
    scope: merged.scope || null,
    openid: merged.openid || null,
    username: merged.username || proxy.username || null,
    userId: merged.userId || proxy.userId || null,
    projectUrlId: merged.projectUrlId || proxy.projectUrlId || null,
    serviceAk: merged.serviceAk || proxy.serviceAk || null,
    serviceAkProjectId: merged.serviceAkProjectId || proxy.serviceAkProjectId || null,
  }
}

/** 方式三：复用本机 dsh-meoo-connect / meoo-proxy 的登录态（严格只读）。 */
function reuseSignIn() {
  return {
    url: "",
    instructions: "只读复用本机已有的 Meoo 登录态（dsh-meoo-connect 的 $DSH_HOME/meoo/state.json 或 meoo-proxy 的 ~/.meoo-proxy/config.json），不会写入这两个文件。",
    method: "auto",
    async callback() {
      const legacy = readLegacyState()
      if (!legacy || (!legacy.refreshToken && !legacy.serviceAk)) {
        return {
          type: "failed",
          error: "未找到本机 Meoo 登录态（已查找 $DSH_HOME/meoo/state.json 与 ~/.meoo-proxy/config.json）；请先用浏览器授权登录一次",
        }
      }
      try {
        const a = {
          apiBase: apiBaseOf(legacy),
          access: legacy.accessToken || "",
          refresh: legacy.refreshToken || "",
          expires: legacy.expires || 0,
          ...(legacy.scope ? { scope: legacy.scope } : {}),
          ...(legacy.openid ? { openid: legacy.openid } : {}),
          ...(legacy.username ? { username: legacy.username } : {}),
          ...(legacy.userId ? { userId: legacy.userId } : {}),
          projectUrlId: legacy.projectUrlId || "",
          serviceAk: legacy.serviceAk || "",
          serviceAkProjectId: legacy.serviceAkProjectId || "",
        }
        if (!a.serviceAk && !a.refresh) {
          return { type: "failed", error: "本机登录态既没有项目服务 AK 也没有 refresh token，无法复用；请用浏览器授权重新登录" }
        }
        if (a.refresh) {
          // 过期就续期（只动宿主自己的账号记录，不碰 dsh/meoo-proxy 的文件）
          const { a: fresh } = await currentAuth({ client: null }, a, isStale(a, 0))
          a.access = fresh.access || a.access
          a.expires = fresh.expires || a.expires
          a.refresh = fresh.refresh || a.refresh
        }
        if (!a.serviceAk) {
          const { urlId, ak } = await resolveServiceAk({ client: null }, a)
          a.projectUrlId = urlId
          a.serviceAk = ak
          a.serviceAkProjectId = urlId
        } else if (!a.projectUrlId) {
          a.projectUrlId = a.serviceAkProjectId
        }
        return {
          type: "success",
          access: a.access,
          refresh: a.refresh,
          expires: a.expires,
          accountId: a.username || a.openid || ACCOUNT_FALLBACK,
          ...(a.username ? { username: a.username } : {}),
          ...(a.openid ? { openid: a.openid } : {}),
          ...(a.scope ? { scope: a.scope } : {}),
          apiBase: a.apiBase,
          projectUrlId: a.projectUrlId,
          serviceAk: a.serviceAk,
          serviceAkProjectId: a.serviceAkProjectId,
        }
      } catch (e) {
        return { type: "failed", error: `复用本机登录态失败：${e?.message ?? String(e)}` }
      }
    },
  }
}

/** 方式四：手填项目服务 AK（meoo_project_service_v1.*，meoo.com 项目 → 云服务里复制）。 */
function manualKeySignIn() {
  return {
    url: "",
    instructions: "在 meoo.com 打开项目 → 云服务，复制项目服务 AK（meoo_project_service_v1. 开头）粘贴到下方；确认前会用网关 /models 校验一次。",
    method: "code",
    async callback(input) {
      try {
        const key = String(input ?? "").trim().replace(/^Bearer\s+/i, "").replace(/^["']|["']$/g, "")
        if (!key) throw new Error("请输入项目服务 AK")
        if (key.startsWith("meoo_ak_")) {
          throw new Error("这是用户 Key（meoo_ak_…），不是项目服务 AK；请填 meoo_project_service_v1. 开头的项目服务 AK")
        }
        const res = await fetch(`${gatewayBase()}/models`, {
          headers: { authorization: `Bearer ${key}`, accept: "application/json" },
          signal: AbortSignal.timeout(15_000),
        })
        if (res.status === 401 || res.status === 403) {
          throw new Error(`AK 被网关拒绝（HTTP ${res.status}）：确认是项目服务 AK，且该项目已开通云服务`)
        }
        if (!res.ok) throw new Error(`网关校验失败 HTTP ${res.status}`)
        return { type: "success", key, accountId: "Meoo 项目服务 AK" }
      } catch (e) {
        return { type: "failed", error: e?.message ?? String(e) }
      }
    },
  }
}

// ---- 插件出口 ---------------------------------------------------------------

/**
 * 把更新后的账号记录存回宿主。
 *
 * 存不回是致命的，不能吞掉：Meoo 的 refresh token 每次续期都换新的，旧的立刻
 * 作废（实测 2026-11）。续期成功但没存回去，下一次就会带着已经作废的 token 去
 * 换票，拿到 HTTP 400 invalid_grant，而这次失败看起来和「token 被服务端吊销」
 * 一模一样，很难分辨。所以这里把失败报出去，让宿主和日志知道凭据没落盘。
 *
 * 宿主没有「这个账号已经登出了」的办法告诉插件（auth.set 在账号被登出时照样
 * 返回成功），所以这里只能把写入本身抛出来的错误报出去：磁盘写不进去时宿主
 * 会抛，那才是真正丢凭据的情况。账号被登出时宿主不抛，插件也就跟着不报错，
 * 那是本来的结果。
 */
function persistOf(ctx) {
  return async (updated) => {
    try {
      await ctx?.client?.auth?.set?.({ path: { id: PROVIDER_ID }, body: updated })
    } catch (e) {
      const err = new Error(
        `Meoo 登录态没能存回宿主（换到的凭据可能已经作废，请重新登录）：${e?.message ?? e}`,
      )
      err.cause = e
      // 存不回凭据和凭据被拒一样，再试一次也没用
      err.signIn = "expired"
      throw err
    }
  }
}

// ---- 请求体瘦身（Meoo 网关 1 MiB 硬上限）----------------------------------

// 和 magpie-wps-lingxi 同一套思路：按字节预算分级裁剪，system 与最后一条保留，
// tool_calls 与 tool 结果成对。比它多两点：历史图片先剥离（base64 最占地方），
// 以及 413 时再压一轮重试。
const MAX_MESSAGE_BYTES = 48 * 1024
const TRUNC_NOTE = "\n…[本条消息过长，已截断（Meoo 网关单请求上限）]…\n"
const IMAGE_OMIT_NOTE = "[图片已省略：Meoo 网关请求体上限]"
const utf8 = new TextEncoder()

const byteLen = (s) => utf8.encode(typeof s === "string" ? s : String(s ?? "")).length

function bytesOf(v) {
  return byteLen(JSON.stringify(v ?? null))
}

// 一个码点的 UTF-8 字节数（不真正编码，省掉逐字符分配）。
const cpBytes = (cp) => (cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4)

/**
 * 按 UTF-8 字节精确切一个字符串的前缀/后缀，不切开码点（emoji 不会被切成半个）。
 * 按字符切会因中文 3 字节、emoji 4 字节而超限；这里按码点线性累加字节。
 */
function sliceBytes(text, maxBytes, fromEnd = false) {
  const s = typeof text === "string" ? text : String(text ?? "")
  if (maxBytes <= 0 || !s) return ""
  if (byteLen(s) <= maxBytes) return s
  const chars = Array.from(s)
  let bytes = 0
  let n = 0
  if (fromEnd) {
    for (let i = chars.length - 1; i >= 0; i--) {
      const b = cpBytes(chars[i].codePointAt(0) ?? 0)
      if (bytes + b > maxBytes) break
      bytes += b
      n++
    }
    return chars.slice(chars.length - n).join("")
  }
  for (let i = 0; i < chars.length; i++) {
    const b = cpBytes(chars[i].codePointAt(0) ?? 0)
    if (bytes + b > maxBytes) break
    bytes += b
    n++
  }
  return chars.slice(0, n).join("")
}

/** 掐头去尾，保留 60% 头 + 40% 尾，中间放省略标记；结果不超过 limit 字节。 */
function headTailBytes(text, limit) {
  const s = typeof text === "string" ? text : String(text ?? "")
  if (byteLen(s) <= limit) return s
  const room = Math.max(0, limit - byteLen(TRUNC_NOTE))
  if (room <= 0) return sliceBytes(s, limit)
  const head = Math.floor(room * 0.6)
  const tail = room - head
  return sliceBytes(s, head) + TRUNC_NOTE + (tail > 0 ? sliceBytes(s, tail, true) : "")
}

const imagePart = (p) => p && typeof p === "object" && (p.type === "image_url" || p.type === "image")

function stripImages(content) {
  if (!Array.isArray(content)) return content
  const parts = content.filter((p) => !imagePart(p))
  if (parts.length === content.length) return content
  if (!parts.length) return [{ type: "text", text: IMAGE_OMIT_NOTE }]
  return parts
}

function elideContent(content, limit) {
  if (typeof content === "string") return headTailBytes(content, limit)
  if (Array.isArray(content)) {
    return content.map((p) => (p && typeof p === "object" && typeof p.text === "string" ? { ...p, text: headTailBytes(p.text, limit) } : p))
  }
  return content
}

/** 单条消息超 MAX_MESSAGE_BYTES 就掐头去尾（字节精确），短消息不动。 */
function elideMessage(message) {
  const content = message?.content
  if (typeof content === "string") {
    return byteLen(content) > MAX_MESSAGE_BYTES ? { ...message, content: headTailBytes(content, MAX_MESSAGE_BYTES) } : message
  }
  if (Array.isArray(content)) {
    let changed = false
    const parts = content.map((p) => {
      if (p && typeof p === "object" && typeof p.text === "string" && byteLen(p.text) > MAX_MESSAGE_BYTES) {
        changed = true
        return { ...p, text: headTailBytes(p.text, MAX_MESSAGE_BYTES) }
      }
      return p
    })
    return changed ? { ...message, content: parts } : message
  }
  return message
}

const blankContent = (content) => {
  if (typeof content === "string") return ""
  if (Array.isArray(content)) return content.map((p) => (p && typeof p === "object" && typeof p.text === "string" ? { ...p, text: "" } : p))
  return content
}

/**
 * 把一个 OpenAI 聊天请求体压到 limit 字节以内，只动 messages、不动 tools 等
 * 其余字段（工具定义必须保留，否则 function calling 失效）。按破坏性从小到大：
 *  1. 去掉除最后一条以外消息里的图片（历史里的 base64 图最占地方、也最不必要）；
 *  2. 单条超 48KB 的消息掐头去尾（字节精确）；
 *  3. 从最旧一端整条丢，保留开头的 system 与最后一条；assistant 带 tool_calls 时
 *     连同它的 tool 结果一起丢，避免留下没有主人的 tool 消息。丢过历史就插一条
 *     system 占位，让模型知道上下文被截了；
 *  4. 连最后一条的图片也去掉（保留文字）；
 *  5. 兜底：按剩余字节预算精确截断所有文本，保证落在 limit 以内。
 * 每一步之后只要达标就返回。
 */
function shrinkChatBody(body, limit = MAX_BODY_BYTES) {
  if (!body || typeof body !== "object") return body
  if (!Array.isArray(body.messages) || !body.messages.length) return body
  if (bytesOf(body) <= limit) return body

  const headCount = body.messages[0]?.role === "system" ? 1 : 0
  let messages = body.messages.slice()
  const size = (list) => bytesOf({ ...body, messages: list })

  // 1) 去掉除最后一条以外消息里的图片
  messages = messages.map((m, i) => (i === messages.length - 1 ? m : { ...m, content: stripImages(m.content) }))
  if (size(messages) <= limit) return { ...body, messages }

  // 2) 单条超长文本掐头去尾
  messages = messages.map((m) => elideMessage(m))
  if (size(messages) <= limit) return { ...body, messages }

  // 3) 从最旧一端整条丢，保留 system + 最后一条
  const dropOldest = () => {
    const dropped = messages[headCount]
    let end = headCount + 1
    if (dropped?.role === "assistant" && Array.isArray(dropped.tool_calls) && dropped.tool_calls.length) {
      const ids = new Set(dropped.tool_calls.map((t) => t?.id).filter(Boolean))
      while (end < messages.length && messages[end]?.role === "tool" && ids.has(messages[end]?.tool_call_id)) end++
    }
    messages = [...messages.slice(0, headCount), ...messages.slice(end)]
  }
  while (messages.length > headCount + 1 && size(messages) > limit) dropOldest()
  // 丢过历史就在 system 后面插一条占位；占位本身也占字节，装不下就再丢一条。
  if (body.messages.length > messages.length) {
    const withNote = () => [
      ...messages.slice(0, headCount),
      { role: "system", content: `[历史已压缩：更早的 ${body.messages.length - messages.length} 条消息被省略（Meoo 网关单请求长度上限）]` },
      ...messages.slice(headCount),
    ]
    while (messages.length > headCount + 1 && size(withNote()) > limit) dropOldest()
    messages = withNote()
  }
  if (size(messages) <= limit) return { ...body, messages }

  // 4) 连最后一条的图片也去掉（保留文字）
  messages = messages.map((m, i) => (i === messages.length - 1 ? { ...m, content: stripImages(m.content) } : m))
  if (size(messages) <= limit) return { ...body, messages }

  // 5) 兜底：按剩余预算精确截断所有文本。UTF-8 一个字符最多 4 字节，按 4 字节/字符
  // 估一个每段上限，并保留消息内容的结构（空文本替代），保证 JSON 序列化后一定
  // 落在 limit 以内。
  const empty = { ...body, messages: messages.map((m) => ({ ...m, content: blankContent(m.content) })) }
  const textBudget = Math.max(0, limit - bytesOf(empty))
  const nParts = messages.reduce((n, m) => n + (Array.isArray(m.content) ? Math.max(1, m.content.length) : 1), 0)
  const perPartBytes = Math.max(1, Math.floor(textBudget / Math.max(nParts, 1)))
  messages = messages.map((m) => ({ ...m, content: elideContent(m.content, perPartBytes) }))
  return { ...body, messages }
}

/**
 * 包一层 fetch：发往网关 /chat/completions 的请求在发出前按字节数压到安全水位，
 * 仍被 413 时再压一轮重试一次。其它请求（/models、换票、控制面等）原样转发。
 */
function gatewayFetch() {
  return async (input, init = {}) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input?.url ?? String(input)
    if (!url.includes("/chat/completions") || typeof init.body !== "string") return fetch(url, init)

    let body = init.body
    try {
      const parsed = JSON.parse(body)
      if (bytesOf(parsed) > MAX_BODY_BYTES) body = JSON.stringify(shrinkChatBody(parsed))
    } catch {}

    const res = await fetch(url, { ...init, body })
    if (res.status !== 413) return res
    // 已经压过还是 413：网关对这份内容仍不收，再压到一半重试一次。
    try {
      const parsed = JSON.parse(body)
      const smaller = shrinkChatBody(parsed, Math.floor(MAX_BODY_BYTES / 2))
      if (bytesOf(smaller) >= bytesOf(parsed)) return res
      try {
        await res.body?.cancel?.()
      } catch {}
      return await fetch(url, { ...init, body: JSON.stringify(smaller) })
    } catch {
      return res
    }
  }
}

export const MeooAuthPlugin = async ({ client }) => {
  const ctx = { client }

  return {
    config: async (config) => {
      config.provider ??= {}
      const was = config.provider[PROVIDER_ID] ?? {}
      config.provider[PROVIDER_ID] = {
        name: PROVIDER_NAME,
        npm: NPM,
        api: gatewayBase(),
        ...was,
        models: { ...configModels(), ...(was.models ?? {}) },
      }
    },
    provider: {
      id: PROVIDER_ID,
      async models(provider, { auth } = {}) {
        try {
          let ak = ""
          if (auth?.type === "api") ak = auth.key || ""
          else if (auth?.type === "oauth" && (auth.access || auth.refresh || auth.serviceAk)) {
            const { a } = await currentAuth(ctx, auth)
            ak = a.serviceAk || (await serviceAkOf(ctx, a))
          }
          if (!ak) return provider.models
          const ms = await fetchGatewayModels(ak)
          if (!ms.length) return provider.models
          return Object.fromEntries(ms.map((m) => [m.id, runtimeModel(provider, m)]))
        } catch {
          return provider.models
        }
      },
    },
    auth: {
      provider: PROVIDER_ID,
      refreshLead: LEAD_MS,
      // Meoo 每次续期都换一个新的 refresh token，旧的立刻作废（实测 2026-11：
      // 登录时的指纹在一次自动续期后就变了）。所以这个登录态不能被复制到另一台
      // 机器或另一个进程去并行续期——两个主人抢一个一次性 token，输的那个拿到
      // HTTP 400 invalid_grant，且无法自愈。宿主读到 rotating: true 就不会把它
      // 带去同步或备份（magpie-community/plugins#884）。
      rotating: true,
      async refresh(auth) {
        if (auth?.type !== "oauth" || !auth.refresh) return undefined
        const { a, didRenew } = await currentAuth(ctx, auth, true)
        if (!didRenew) return undefined
        return {
          access: a.access,
          refresh: a.refresh,
          expires: a.expires,
          ...(a.scope ? { scope: a.scope } : {}),
          ...(a.openid ? { openid: a.openid } : {}),
          ...(a.username ? { username: a.username } : {}),
          ...(a.serviceAk ? { serviceAk: a.serviceAk, serviceAkProjectId: a.serviceAkProjectId, projectUrlId: a.projectUrlId, apiBase: a.apiBase } : {}),
        }
      },
      // 余额：Meoo 的积分账户（控制面 /account/points）。顶层 balance 就是卡片底部那行余额；
      // activeGrants 里是每一批赠送积分（订阅/登录/邀请）的发放量与剩余量，
      // 按来源各给一条 limit/amount/used 的数量窗口——分母是真实发放量，宿主才画进度条。
      // 接口没给批次明细时退回一条聚合窗口（分母 = 当前有效总量 = 剩余 + 已用 + 冻结）。
      async usage(getAuth) {
        const auth = await getAuth()
        if (!auth) throw new Error("还没有连接 Meoo")
        if (auth.type === "api") {
          throw new Error("手填的项目服务 AK 读不到积分账户；要用余额卡，请用浏览器授权登录一次")
        }
        if (auth.type !== "oauth") throw new Error("登录态类型未知，无法读取积分")
        const { a } = await currentAuth(ctx, auth)
        // 积分钱包与套餐权益并行读：权益只用于只读展示，接口不可用时不影响余额卡。
        const [data, entitlement] = await Promise.all([
          controlRequest(ctx, a, "GET", "/open/v1/cli-compat/account/points"),
          controlRequest(ctx, a, "GET", "/open/v1/cli-compat/account/entitlement").catch(() => null),
        ])
        const acc = data?.account || {}
        const left = int(acc.availableBalance)
        const used = int(acc.totalConsumed)
        const frozen = int(acc.frozenBalance)
        const fmt = (n) => new Intl.NumberFormat("en-US").format(n)
        // 不单独给「积分」钱包窗口：卡片的余额行已经用顶层 balance 显示同一个数，
        // 再放一条 balance 窗口只会在顶部重复一遍「剩余 x 积分」。
        const windows = []
        // 每个赠送来源一行：limit=发放量、amount=已消耗（发放 - 剩余），
        // 面板据此算「剩余 = limit - amount」，与钱包余额同源，不会自相矛盾。
        for (const g of Array.isArray(data?.activeGrants) ? data.activeGrants : []) {
          const initial = int(g?.totalInitialAmount)
          if (initial <= 0) continue
          const remain = Math.min(int(g?.totalRemainingAmount), initial)
          const spent = initial - remain
          const row = {
            name: String(g?.sourceTypeName || g?.sourceType || "赠送积分"),
            used: Math.max(0, Math.min(100, (spent / initial) * 100)),
            amount: spent,
            limit: initial,
            unit: "积分",
            display: `剩 ${fmt(remain)} / ${fmt(initial)} 积分`,
            // 不是某条总额的拆分，而是独立的发放批次：不能标 aside，
            // 额度面板会把 aside 窗口整个跳过（不进合计、也不画进度条）。
          }
          // runsOutAt/runsOut 是官方的「剩余额度何时作废、作废多少」：
          // 积分没有重置周期，smart 路由就按最早作废时间给账号排先后。
          const lapse = soonestLapse([g])
          if (lapse) {
            row.runsOutAt = lapse.day
            row.runsOut = lapse.remaining
          }
          windows.push(row)
        }
        // 没有批次明细（或全是 0）时退回一条聚合窗口，免得卡片上只有余额没有条。
        if (windows.length === 0) {
          const total = left + used + frozen
          if (total > 0) {
            const pct = Math.max(0, Math.min(100, (used / total) * 100))
            const agg = {
              name: "积分用量",
              used: pct,
              amount: used,
              limit: total,
              unit: "积分",
              display: `${fmt(used)} / ${fmt(total)} 积分`,
            }
            const lapse = soonestLapse(data?.activeGrants)
            if (lapse) {
              agg.runsOutAt = lapse.day
              agg.runsOut = lapse.remaining
            }
            windows.push(agg)
          }
        }
        if (frozen > 0) windows.push({ name: "冻结", display: `${fmt(frozen)} 积分`, aside: true })
        // 官方「每日登录赠送积分」只做只读展示：额度来自套餐权益，是否到账由服务端在
        // 正常登录时决定；插件不代签、不自动领取。
        const daily = dailyLoginOf(entitlement)
        if (daily) {
          const amount = daily.quota === -1 ? "不限" : `${fmt(daily.quota)} ${daily.unit}/日`
          const granted = entitlement?.dailyLoginGranted === true ? " · 今日已到账" : ""
          windows.push({ name: "每日登录", display: `${amount}${granted}`, aside: true })
        }
        const plan = await planOf(ctx, a)
        return {
          ...(plan ? { plan } : {}),
          user: a.username || "",
          balance: `${fmt(left)} 积分`,
          windows,
        }
      },
      async loader(getAuth) {
        const auth = await getAuth()
        if (auth?.type === "api") return auth.key ? { baseURL: gatewayBase(), apiKey: auth.key, fetch: gatewayFetch() } : {}
        if (auth?.type !== "oauth") return {}
        try {
          const { a } = await currentAuth(ctx, auth)
          const ak = a.serviceAk || (await serviceAkOf(ctx, a))
          if (!ak) return {}
          return { baseURL: gatewayBase(), apiKey: ak, fetch: gatewayFetch() }
        } catch (e) {
          const expired = e?.signIn === "expired"
          return {
            baseURL: gatewayBase(),
            apiKey: "meoo",
            async fetch() {
              return new Response(
                JSON.stringify({ error: { message: e?.message ?? String(e), type: expired ? "authentication_error" : "api_error" } }),
                {
                  status: expired ? 401 : 502,
                  headers: { "Content-Type": "application/json", "X-Magpie-Sign-In": expired ? "expired" : "kept" },
                },
              )
            },
          }
        }
      },
      methods: [
        {
          type: "oauth",
          label: "Meoo 账号（浏览器授权，自动取项目 AK）",
          authorize: async () => browserSignIn(),
        },
        {
          type: "oauth",
          label: "复用本机登录态（dsh-meoo-connect / meoo-proxy）",
          authorize: async () => reuseSignIn(),
        },
        {
          type: "api",
          label: "项目服务 AK（meoo_project_service_v1.…）",
          placeholder: "meoo_project_service_v1.…",
          authorize: async () => manualKeySignIn(),
        },
      ],
    },
  }
}

export const _internal = {
  tokenRequest,
  renewToken,
  currentAuth,
  isStale,
  controlCall,
  controlRequest,
  fetchIdentity,
  listProjects,
  createProject,
  pickProject,
  resolveServiceAk,
  serviceAkOf,
  parseGatewayModels,
  fetchGatewayModels,
  resetCatalogCache,
  configModels,
  runtimeModel,
  buildSignedInAuth,
  browserSignIn,
  reuseSignIn,
  manualKeySignIn,
  readLegacyState,
  clientIdFor,
  apiBaseOf,
  gatewayBase,
  dailyLoginOf,
  shrinkChatBody,
  gatewayFetch,
  bytesOf,
}
