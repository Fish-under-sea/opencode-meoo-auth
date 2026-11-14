// 离线测试：假 fetch + 假 client，不碰真实网络（浏览器登录的 loopback 回调用真实 fetch 打本机端口）。
// 覆盖：模型目录解析、AK 三种登录路径、项目选择、续期、loader/config 契约。
import test from "node:test"
import assert from "node:assert/strict"
import { MeooAuthPlugin, _internal } from "../index.mjs"

const {
  parseGatewayModels,
  manualKeySignIn,
  reuseSignIn,
  readLegacyState,
  pickProject,
  resolveServiceAk,
  currentAuth,
  renewToken: renewTokenInternal,
  fetchGatewayModels,
  resetCatalogCache,
  configModels,
  runtimeModel,
  browserSignIn,
  clientIdFor,
  isStale,
} = _internal

const GATEWAY = "https://gw.test/meoo/v1"

// ---- 工具 -------------------------------------------------------------------

function fakeFetch(routes) {
  const calls = []
  const fn = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url
    calls.push({ url, init })
    for (const [match, reply] of routes) {
      if (typeof match === "string" ? url.includes(match) : match.test(url)) {
        const r = typeof reply === "function" ? await reply(url, init) : reply
        return r
      }
    }
    return new Response("no route: " + url, { status: 599 })
  }
  fn.calls = calls
  return fn
}

const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } })

function fakeClient() {
  const saved = []
  return {
    saved,
    auth: {
      async set(arg) {
        saved.push({ id: arg?.path?.id, body: arg?.body })
        return { data: true }
      },
      async get() {
        return { data: undefined }
      },
    },
  }
}

const OK_TOKEN = { access_token: "acc-1", refresh_token: "ref-1", expires_in: 3600 }

// 控制面全链路的假后端：换票 → 身份 → 项目列表 → 云服务 environment → 网关 /models
function meooBackend({ identity, projects, env, cloudPosted, gatewayModels, gatewayStatus = 200 }) {
  let cloudCalls = 0
  return fakeFetch([
    [/\/oauth\/token/, () => json(OK_TOKEN)],
    [/\/account\/identity/, () => json({ data: identity })],
    [/\/projects\?/, () => json({ data: { list: projects } })],
    [/\/cloud\/environment/, () => json({ data: env ?? {} })],
    [/\/cloud$/, () => {
      cloudCalls++
      return cloudPosted ? json({ data: {} }) : new Response("no cloud", { status: 400 })
    }],
    [/\/models/, () => (gatewayStatus === 200 ? json({ models: gatewayModels }) : new Response("denied", { status: gatewayStatus }))],
  ])
}

/** 换上假 fetch 跑 fn(mock)；fn 拿到的第二个参数是真实 fetch（loopback 回调要用它）。 */
async function withFetch(fn, routes) {
  const real = globalThis.fetch
  const mock = fakeFetch(routes)
  globalThis.fetch = mock
  try {
    return await fn(mock, real)
  } finally {
    globalThis.fetch = real
  }
}

// ---- 模型目录 ---------------------------------------------------------------

test("parseGatewayModels：字符串条目、对象条目、models/data 两种包法、去重", () => {
  assert.deepEqual(parseGatewayModels({ models: ["glm-5.2", { id: "qwen3.8-max", name: "Qwen3.8-Max" }, "glm-5.2"] }).map((m) => [m.id, m.name]), [
    ["glm-5.2", "GLM-5.2"],
    ["qwen3.8-max", "Qwen3.8-Max"],
  ])
  assert.deepEqual(parseGatewayModels({ data: [{ id: "glm-5.2" }] })[0].name, "GLM-5.2")
  assert.deepEqual(parseGatewayModels({ data: [{ id: "whatever-x" }] })[0].name, "whatever-x")
  assert.deepEqual(parseGatewayModels({ data: [{ id: "m1", name: "M1", max_tokens: 8000 }] })[0].output, 8000)
  assert.deepEqual(parseGatewayModels({}), [])
})

test("fetchGatewayModels：带 AK 请求网关并缓存；目录外的候选各探一次", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  resetCatalogCache()
  await withFetch(async (fetchMock) => {
    const ms = await fetchGatewayModels("ak-1")
    assert.equal(ms.length, 2)
    assert.equal(fetchMock.calls[0].url, `${GATEWAY}/models`)
    assert.equal(fetchMock.calls[0].init.headers.authorization, "Bearer ak-1")
    // 目录外的候选（kimi/glm-5.3/minimax/vl…）各探一次；全被网关拒 → 一个都不补
    const probes = fetchMock.calls.filter((c) => c.url.includes("/chat/completions"))
    assert.ok(probes.length >= 6, `候选应该各探一次，实际 ${probes.length}`)
    for (const p of probes) assert.match(String(p.init.body), /"max_tokens":1/)
    // 第二次命中缓存，不再发请求
    const before = fetchMock.calls.length
    await fetchGatewayModels("ak-1")
    assert.equal(fetchMock.calls.length, before)
  }, [
    [/\/models/, () => json({ models: [{ id: "glm-5.2" }, { id: "qwen3.8-max" }] })],
    [/\/chat\/completions/, () => json({ error: { message: "unsupported" } }, 400)],
  ])
})

test("fetchGatewayModels：网关接受的目录外模型补进目录（qwen3-vl-plus），拒绝的不进", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  resetCatalogCache()
  await withFetch(async (fetchMock) => {
    const ms = await fetchGatewayModels("ak-1")
    const vl = ms.find((m) => m.id === "qwen3-vl-plus")
    assert.ok(vl, "视觉模型应该被补进目录")
    assert.equal(vl.context, 262_144)
    assert.equal(vl.output, 32_768)
    assert.equal(vl.image, true)
    // 文档里写了但网关不接的（Kimi/GLM-5.3/MiniMax）不该混进来
    assert.equal(ms.find((m) => m.id === "kimi-k3"), undefined)
    assert.equal(ms.find((m) => m.id === "glm-5.3"), undefined)
    const probe = fetchMock.calls.find((c) => c.url.includes("/chat/completions") && String(c.init.body).includes("qwen3-vl-plus"))
    assert.ok(probe, "应该探过 qwen3-vl-plus")
  }, [
    [/\/models/, () => json({ models: ["glm-5.2"] })],
    [/\/chat\/completions/, (url, init) => (String(init.body).includes("qwen3-vl-plus") ? json({ choices: [{ message: { content: "H" } }] }) : json({ error: { message: "当前仅支持配置模型" } }, 400))],
  ])
})

test("fetchGatewayModels：MEOO_EXTRA_MODELS 里的 id 也要过网关那一探", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  process.env.MEOO_EXTRA_MODELS = "my-private-model, qwen3-vl-plus"
  resetCatalogCache()
  try {
    await withFetch(async (fetchMock) => {
      const ms = await fetchGatewayModels("ak-1")
      assert.ok(ms.find((m) => m.id === "my-private-model"), "自己塞的 id 网关接了就应该出现")
      const probe = fetchMock.calls.find((c) => String(c.init.body).includes("my-private-model"))
      assert.ok(probe)
    }, [
      [/\/models/, () => json({ models: ["glm-5.2"] })],
      [/\/chat\/completions/, (url, init) => (String(init.body).includes("my-private-model") ? json({ choices: [] }) : json({ error: {} }, 400))],
    ])
  } finally {
    delete process.env.MEOO_EXTRA_MODELS
    resetCatalogCache()
  }
})

test("configModels/runtimeModel：规格表给出真实上限，视觉模型按图片能力", () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const cm = configModels()
  assert.equal(cm["glm-5.2"].name, "GLM-5.2")
  // 之前写死 200k，实际 GLM-5.2 是 1M 上下文 / 128K 输出
  assert.equal(cm["glm-5.2"].limit.context, 1_000_000)
  assert.equal(cm["glm-5.2"].limit.output, 128_000)
  assert.equal(cm["deepseek-v4-pro-0813"].limit.context, 1_000_000)
  assert.equal(cm["deepseek-v4-pro-0813"].limit.output, 384_000)
  assert.equal(cm["qwen3.8-max"].limit.output, 131_072)
  assert.equal(cm["glm-5.2"].tool_call, true)
  // 视觉模型：目录里没有，但规格表里有，预登录阶段就该看得见
  assert.equal(cm["qwen3-vl-plus"].limit.context, 262_144)
  assert.equal(cm["qwen3-vl-plus"].attachment, true)
  assert.deepEqual(cm["qwen3-vl-plus"].modalities.input, ["text", "image"])
  assert.equal(cm["glm-5.2"].attachment, false)
  // 思考档位三档：映射成 AI SDK 的 reasoningEffort
  assert.deepEqual(cm["glm-5.2"].variants, {
    low: { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high: { reasoningEffort: "high" },
  })
  const provider = { id: "meoo" }
  const rm = runtimeModel(provider, { id: "glm-5.2", name: "GLM-5.2" })
  assert.equal(rm.providerID, "meoo")
  assert.equal(rm.api.npm, "@ai-sdk/openai-compatible")
  assert.equal(rm.api.url, GATEWAY)
  assert.equal(rm.capabilities.toolcall, true)
  assert.equal(rm.limit.context, 1_000_000)
  // 目录里的未知 id 走保守默认，不把 1M 乱许出去
  const unk = runtimeModel(provider, { id: "brand-new-model" })
  assert.equal(unk.limit.context, 200_000)
  assert.equal(unk.limit.output, 64_000)
  assert.equal(unk.capabilities.input.image, false)
  assert.deepEqual(rm.variants, {
    low: { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high: { reasoningEffort: "high" },
  })
})

// ---- 手填 AK ----------------------------------------------------------------

test("manualKeySignIn：meoo_ak_ 前缀被拒绝，合法 AK 通过网关校验后保存", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const m = manualKeySignIn()
  assert.equal(m.method, "code")

  const empty = await m.callback("  ")
  assert.equal(empty.type, "failed")

  const userKey = await m.callback("meoo_ak_abcdef")
  assert.equal(userKey.type, "failed")
  assert.match(userKey.error, /用户 Key/)

  await withFetch(async (fetchMock) => {
    const ok = await m.callback("meoo_project_service_v1.abc")
    assert.equal(ok.type, "success")
    assert.equal(ok.key, "meoo_project_service_v1.abc")
    assert.match(fetchMock.calls[0].url, /\/models$/)
    assert.equal(fetchMock.calls[0].init.headers.authorization, "Bearer meoo_project_service_v1.abc")
  }, [[/\/models/, () => json({ models: ["glm-5.2"] })]])

  await withFetch(async () => {
    const bad = await m.callback("meoo_project_service_v1.wrong")
    assert.equal(bad.type, "failed")
    assert.match(bad.error, /被网关拒绝/)
  }, [[/\/models/, () => new Response("no", { status: 401 })]])
})

// ---- 本机登录态复用 -----------------------------------------------------------

test("readLegacyState：dsh 优先、proxy 互填、两处都没有返回 null", async () => {
  const { mkdtempSync, writeFileSync, rmSync, mkdirSync } = await import("node:fs")
  const { tmpdir } = await import("node:os")
  const { join } = await import("node:path")
  const dir = mkdtempSync(join(tmpdir(), "meoo-test-"))
  try {
    mkdirSync(join(dir, "proxy"), { recursive: true })
    const stateFile = join(dir, "state.json")
    const proxyFile = join(dir, "proxy", "config.json")
    writeFileSync(stateFile, JSON.stringify({ refreshToken: "ref-dsh", username: "陈同学", projectUrlId: "proj-a", serviceAk: null, accessExpiresAt: 0 }))
    writeFileSync(proxyFile, JSON.stringify({ refreshToken: "ref-proxy", serviceAk: "meoo_project_service_v1.p", serviceAkProjectId: "proj-b" }))

    const onlyDsh = readLegacyState({ dshState: stateFile, proxyConfig: join(dir, "nope.json") })
    assert.equal(onlyDsh.refreshToken, "ref-dsh")
    assert.equal(onlyDsh.username, "陈同学")
    assert.equal(onlyDsh.projectUrlId, "proj-a")
    assert.equal(onlyDsh.serviceAk, null)

    const onlyProxy = readLegacyState({ dshState: join(dir, "nope.json"), proxyConfig: proxyFile })
    assert.equal(onlyProxy.refreshToken, "ref-proxy")
    assert.equal(onlyProxy.serviceAk, "meoo_project_service_v1.p")
    assert.equal(onlyProxy.apiBase, "https://meoo.com")

    const none = readLegacyState({ dshState: join(dir, "nope.json"), proxyConfig: join(dir, "also-nope.json") })
    assert.equal(none, null)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test("reuseSignIn：没有任何本机登录态 → 失败并说明找过哪里", async () => {
  const { mkdtempSync, rmSync } = await import("node:fs")
  const { tmpdir } = await import("node:os")
  const { join } = await import("node:path")
  const dir = mkdtempSync(join(tmpdir(), "meoo-test-"))
  const origDsh = process.env.DSH_HOME
  const origProxy = process.env.MEOO_PROXY_HOME
  process.env.DSH_HOME = join(dir, "empty-dsh")
  process.env.MEOO_PROXY_HOME = join(dir, "empty-proxy")
  try {
    const missing = await reuseSignIn().callback()
    assert.equal(missing.type, "failed")
    assert.match(missing.error, /未找到本机 Meoo 登录态/)
  } finally {
    if (origDsh === undefined) delete process.env.DSH_HOME
    else process.env.DSH_HOME = origDsh
    if (origProxy === undefined) delete process.env.MEOO_PROXY_HOME
    else process.env.MEOO_PROXY_HOME = origProxy
    rmSync(dir, { recursive: true, force: true })
  }
})

test("reuseSignIn：带 refresh token 的旧登录态会续期并解析 AK", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const { mkdtempSync, writeFileSync, rmSync, mkdirSync } = await import("node:fs")
  const { tmpdir } = await import("node:os")
  const { join } = await import("node:path")
  const dir = mkdtempSync(join(tmpdir(), "meoo-test-"))
  const origDshHome = process.env.DSH_HOME
  process.env.DSH_HOME = dir
  try {
    mkdirSync(join(dir, "meoo"), { recursive: true })
    writeFileSync(join(dir, "meoo", "state.json"), JSON.stringify({
      apiBaseUrl: "https://meoo.com",
      refreshToken: "ref-reuse",
      accessToken: "acc-old",
      accessExpiresAt: 0, // 已过期 → 触发续期
      username: "陈同学",
      projectUrlId: "proj-a",
    }))
    const backend = meooBackend({
      identity: { name: "陈同学", user_id: "u1", project_url_id: "proj-a" },
      projects: [{ url_id: "proj-a", name: "默认" }],
      env: { meooProjectApiKey: "meoo_project_service_v1.new" },
      gatewayModels: [{ id: "glm-5.2" }],
    })
    const prev = globalThis.fetch
    globalThis.fetch = backend
    try {
      const result = await reuseSignIn().callback()
      assert.equal(result.type, "success", JSON.stringify(result))
      assert.equal(result.username, "陈同学")
      assert.equal(result.access, "acc-1") // 续期后的 token
      assert.equal(result.serviceAk, "meoo_project_service_v1.new")
      assert.equal(result.projectUrlId, "proj-a")
      // 续期请求打到 OAuth token 端点
      assert.ok(backend.calls.some((c) => c.url.endsWith("/oauth/token")))
    } finally {
      globalThis.fetch = prev
    }
  } finally {
    process.env.DSH_HOME = origDshHome
    rmSync(dir, { recursive: true, force: true })
  }
})

// ---- 项目选择与 AK 解析 --------------------------------------------------------

const AUTH = { apiBase: "https://meoo.test", access: "acc-1", refresh: "ref-cur", expires: Date.now() + 3600_000 }

test("pickProject：显式 > 记录 > 身份默认 > 唯一项目；多项目报错列出可选项", async () => {
  await withFetch(async () => {
    assert.equal(await pickProject({ client: null }, AUTH, "explicit-1"), "explicit-1")
    assert.equal(await pickProject({ client: null }, { ...AUTH, projectUrlId: "saved" }), "saved")
  }, [])

  // 身份有默认项目
  await withFetch(async () => {
    assert.equal(await pickProject({ client: null }, AUTH), "from-identity")
  }, [
    [/\/account\/identity/, () => json({ data: { name: "陈同学", project_url_id: "from-identity" } })],
  ])

  // 多个项目 → 报错并列出
  await withFetch(async () => {
    await assert.rejects(() => pickProject({ client: null }, AUTH), (e) => {
      assert.match(e.message, /多个项目/)
      assert.match(e.message, /p1（A）/)
      assert.match(e.message, /p2（B）/)
      return true
    })
  }, [
    [/\/account\/identity/, () => json({ data: { project_url_id: null } })],
    [/\/projects\?/, () => json({ data: { list: [{ url_id: "p1", name: "A" }, { url_id: "p2", name: "B" }] } })],
  ])

  // 没有项目 → 自动创建（先带名字，失败退回空请求体）
  await withFetch(async (fetchMock) => {
    assert.equal(await pickProject({ client: null }, AUTH), "new-1")
    const posts = fetchMock.calls.filter((c) => c.init.method === "POST" && c.url.endsWith("/projects"))
    assert.equal(posts.length, 1)
    assert.deepEqual(JSON.parse(posts[0].init.body), { name: "AI 服务" })
  }, [
    [/\/account\/identity/, () => json({ data: { project_url_id: null } })],
    [/\/projects\?/, () => json({ data: { list: [] } })],
    [/\/projects$/, () => json({ data: { url_id: "new-1", name: "AI 服务" } })],
  ])

  // 自动创建也失败 → 报错提示手动建项目
  await withFetch(async () => {
    await assert.rejects(() => pickProject({ client: null }, AUTH), /自动创建也失败/)
  }, [
    [/\/account\/identity/, () => json({ data: { project_url_id: null } })],
    [/\/projects\?/, () => json({ data: { list: [] } })],
    [/\/projects$/, () => new Response("denied", { status: 400 })],
  ])

  // 唯一项目 → 直接选它
  await withFetch(async () => {
    assert.equal(await pickProject({ client: null }, AUTH), "only-1")
  }, [
    [/\/account\/identity/, () => json({ data: { project_url_id: null } })],
    [/\/projects\?/, () => json({ data: { list: [{ url_id: "only-1", name: "唯一" }] } })],
  ])
})

test("resolveServiceAk：environment 有 AK 直接用；没有就自动开通云服务再取", async () => {
  await withFetch(async (fetchMock) => {
    const r1 = await resolveServiceAk({ client: null }, AUTH, { projectUrlId: "p1" })
    assert.equal(r1.ak, "meoo_project_service_v1.env")
    assert.equal(r1.urlId, "p1")
    assert.equal(fetchMock.calls.filter((c) => c.url.includes("/cloud/environment")).length, 1)
    assert.equal(fetchMock.calls.filter((c) => c.init.method === "POST" && c.url.endsWith("/cloud")).length, 0)
  }, [
    [/\/cloud\/environment/, () => json({ data: { meooProjectApiKey: "meoo_project_service_v1.env" } })],
  ])

  await withFetch(async (fetchMock) => {
    const r2 = await resolveServiceAk({ client: null }, AUTH, { projectUrlId: "p1" })
    assert.equal(r2.ak, "meoo_project_service_v1.after")
    assert.equal(fetchMock.calls.filter((c) => c.init.method === "POST" && c.url.endsWith("/cloud")).length, 1)
  }, (() => {
    const posts = { n: 0 }
    return [
      [/\/cloud\/environment/, () => json({ data: posts.n ? { meooProjectApiKey: "meoo_project_service_v1.after" } : {} })],
      [/\/cloud$/, () => {
        posts.n++
        return json({ data: {} })
      }],
    ]
  })())

  await withFetch(async () => {
    await assert.rejects(() => resolveServiceAk({ client: null }, AUTH, { projectUrlId: "p1" }), /还没开通 Meoo AI 服务/)
  }, [
    [/\/cloud\/environment/, () => json({ data: {} })],
    [/\/cloud$/, () => new Response("denied", { status: 400 })],
  ])

  // Meoo 的业务错误走 HTTP 200 + success:false（如 QUOTA_EXCEEDED），错误信息要透传出来
  await withFetch(async () => {
    await assert.rejects(
      () => resolveServiceAk({ client: null }, AUTH, { projectUrlId: "p1" }),
      (e) => {
        assert.match(e.message, /还没开通 Meoo AI 服务/)
        assert.match(e.message, /实例数已达上限/)
        return true
      },
    )
  }, [
    [/\/cloud\/environment/, () => json({ data: { enabled: false, instance: null } })],
    [/\/cloud$/, () => json({ success: false, errCode: "QUOTA_EXCEEDED", message: "云服务实例数已达上限，请提醒升级套餐或释放实例后再试。" })],
  ])
})

// ---- 续期 -------------------------------------------------------------------

test("currentAuth：新鲜 token 原样返回；过期走续期并持久化", async () => {
  const fresh = { ...AUTH, expires: Date.now() + 1800_000 }
  const client = fakeClient()
  const r1 = await currentAuth({ client }, fresh)
  assert.equal(r1.didRenew, false)
  assert.equal(r1.a.access, "acc-1")
  assert.equal(client.saved.length, 0)

  const stale = { ...AUTH, expires: Date.now() - 1000, access: "acc-old", refresh: "ref-stale-only" }
  await withFetch(async () => {
    const r2 = await currentAuth({ client }, stale)
    assert.equal(r2.didRenew, true)
    assert.equal(r2.a.access, "acc-1")
    assert.equal(r2.a.refresh, "ref-1")
    assert.equal(client.saved.length, 1)
    assert.equal(client.saved[0].id, "meoo")
    assert.equal(client.saved[0].body.access, "acc-1")
  }, [[/\/oauth\/token/, () => json(OK_TOKEN)]])

  await assert.rejects(() => currentAuth({ client }, { ...AUTH, refresh: "", access: "x", expires: Date.now() - 1000 }), (e) => e.signIn === "expired")
})

test("isStale：没有 expires 视为过期（与 dsh 的 ensureAccessToken 语义一致）", () => {
  assert.equal(isStale({ expires: 0, access: "x" }), true)
  assert.equal(isStale({ expires: Date.now() + 120_000 }), false)
  assert.equal(isStale({ expires: Date.now() + 10_000 }), true) // 60 秒提前量内算临期
})

test("clientIdFor：meoo.com 用生产 client_id，其他用测试", () => {
  assert.equal(clientIdFor("https://meoo.com"), "meoo_app_tdWlRbpvAxN1EjmZ8XICxdTh")
  assert.equal(clientIdFor("https://meoo-dev.example"), "meoo_app_SFNvmHUFNePxyyfGjCd9hyaE")
})

// ---- 插件契约（magpie 1.x host 的入口）----------------------------------------

async function plugin(client = fakeClient()) {
  const hooks = await MeooAuthPlugin({ client })
  return { hooks, client }
}

test("config：注册 meoo provider（name/npm/api/兜底模型）", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const { hooks } = await plugin()
  const config = {}
  await hooks.config(config)
  const p = config.provider.meoo
  // 外层显示名只给短名，完整包名由宿主的插件 spec 呈现（@fish-under-sea/opencode-meoo-auth）。
  assert.equal(p.name, "Meoo")
  assert.equal(p.npm, "@ai-sdk/openai-compatible")
  assert.equal(p.api, GATEWAY)
  assert.ok(p.models["glm-5.2"])
})

test("provider.models：api 账号用 key，失败回落原模型表", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  resetCatalogCache()
  const { hooks } = await plugin()
  const provider = { id: "meoo", models: { "fallback-1": { name: "F1" } } }

  await withFetch(async (fetchMock) => {
    const ms = await hooks.provider.models(provider, { auth: { type: "api", key: "meoo_project_service_v1.k" } })
    assert.ok(ms["glm-5.2"])
    assert.equal(fetchMock.calls[0].init.headers.authorization, "Bearer meoo_project_service_v1.k")
  }, [[/\/models/, () => json({ models: [{ id: "glm-5.2", name: "GLM-5.2" }] })]])

  // 网关挂了（先清缓存强制重发）→ 回落 provider.models
  resetCatalogCache()
  await withFetch(async () => {
    const ms = await hooks.provider.models(provider, { auth: { type: "api", key: "meoo_project_service_v1.k" } })
    assert.deepEqual(ms, provider.models)
  }, [[/\/models/, () => new Response("x", { status: 500 })]])

  // 没有可用的 auth → 原样返回
  const none = await hooks.provider.models(provider, {})
  assert.deepEqual(none, provider.models)
})

test("loader：api 账号给 baseURL+apiKey；oauth 账号续期后用缓存的 AK", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const client = fakeClient()
  const { hooks } = await plugin(client)

  const l1 = await hooks.auth.loader(async () => ({ type: "api", key: "meoo_project_service_v1.k" }))
  assert.equal(l1.baseURL, GATEWAY)
  assert.equal(l1.apiKey, "meoo_project_service_v1.k")
  assert.equal(l1.fetch, undefined)

  const staleAuth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-loader",
    expires: Date.now() - 1000, projectUrlId: "proj-a", serviceAk: "meoo_project_service_v1.cached", serviceAkProjectId: "proj-a",
  }
  const backend = meooBackend({ identity: {}, projects: [], env: {}, gatewayModels: [] })
  const prev = globalThis.fetch
  globalThis.fetch = backend
  try {
    const l2 = await hooks.auth.loader(async () => staleAuth)
    assert.equal(l2.baseURL, GATEWAY)
    assert.equal(l2.apiKey, "meoo_project_service_v1.cached") // serviceAk 与项目匹配 → 用缓存
    assert.equal(client.saved.length, 1) // 续期结果持久化
    assert.equal(client.saved[0].body.access, "acc-1")
  } finally {
    globalThis.fetch = prev
  }
})

test("loader：登录态彻底过期时返回合成 401 fetch，让宿主提示重新登录", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const { hooks } = await plugin()
  const l = await hooks.auth.loader(async () => ({ type: "oauth", access: "acc", refresh: "", expires: Date.now() - 1000 }))
  assert.equal(typeof l.fetch, "function")
  const res = await l.fetch("https://irrelevant")
  assert.equal(res.status, 401)
  const body = await res.json()
  assert.equal(body.error.type, "authentication_error")
})

test("refresh 钩子：oauth 账号续期并保留 AK/项目字段；api 账号不适用", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  const client = fakeClient()
  const { hooks } = await plugin(client)
  assert.equal(await hooks.auth.refresh({ type: "api", key: "k" }), undefined)

  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-refresh",
    expires: Date.now() - 1000, serviceAk: "ak", serviceAkProjectId: "proj-a", projectUrlId: "proj-a",
  }
  await withFetch(async () => {
    const next = await hooks.auth.refresh(auth)
    assert.equal(next.access, "acc-1")
    assert.equal(next.refresh, "ref-1")
    assert.equal(next.serviceAk, "ak")
    assert.equal(next.apiBase, "https://meoo.test")
    assert.equal(client.saved.length, 1)
  }, [[/\/oauth\/token/, () => json(OK_TOKEN)]])
})

test("auth hook：声明 rotating（Meoo 每次续期都换新的 refresh token）", async () => {
  // 实测 2026-11-11：登录时的 refresh token 指纹 c52cd8a49eb0，一次自动续期后
  // 变成 049aaf11569b。旧的立刻作废，所以这个登录态不能被复制出去并行续期。
  const { hooks } = await plugin()
  assert.equal(hooks.auth.rotating, true)
  assert.equal(typeof hooks.auth.refreshLead, "number")
})

test("refresh 钩子：invalid_grant 抛 signIn 'expired'，让宿主标记重新登录而不是无限重试", async () => {
  // 报障场景：Meoo 对已作废的 refresh token 回 HTTP 400 invalid_grant。
  // 宿主只有读到错误的 signIn:"expired" 才会把账号标为失效并停止退避重试。
  const { hooks } = await plugin()
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-dead",
    expires: Date.now() - 1000,
  }
  await withFetch(async () => {
    await assert.rejects(() => hooks.auth.refresh(auth), (e) => {
      assert.equal(e.signIn, "expired")
      assert.equal(e.code, "invalid_grant")
      assert.match(e.message, /invalid_grant/)
      return true
    })
  }, [[/\/oauth\/token/, () => new Response(JSON.stringify({ error: "invalid_grant" }), { status: 400, headers: { "content-type": "application/json" } })]])
})

test("refresh 钩子：瞬时失败（非 invalid_grant）不标 expired，可退避重试", async () => {
  const { hooks } = await plugin()
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-tmp",
    expires: Date.now() - 1000,
  }
  await withFetch(async () => {
    await assert.rejects(() => hooks.auth.refresh(auth), (e) => {
      assert.equal(e.signIn, undefined)
      assert.doesNotMatch(e.message, /invalid_grant/)
      return true
    })
  }, [[/\/oauth\/token/, () => new Response("temporary", { status: 500 })]])
})

// ---- 存回宿主 ---------------------------------------------------------------

/** 一个 auth.set 会失败的 client：模拟宿主写不进 plugin-auth.json。 */
function failingClient(why = "disk full") {
  return {
    saved: [],
    auth: {
      async set() {
        throw new Error(why)
      },
      async get() {
        return { data: undefined }
      },
    },
  }
}

test("续期后存不回宿主：报错并标记 expired，这一代 token 不留在缓存里", async () => {
  // Meoo 的 refresh token 每次续期都换新的，旧的立刻作废。所以存回宿主失败时，
  // 这一代 token 就只剩插件进程里的一份；一旦它过期丢掉，下一次续期会带着已经
  // 作废的旧 token 去换票，得到 invalid_grant，看起来和服务端吊销一模一样。
  const client = failingClient("no space left on device")
  const { hooks } = await plugin(client)
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-nostore",
    expires: Date.now() - 1000,
  }
  await withFetch(async (fetchMock) => {
    await assert.rejects(() => hooks.auth.refresh(auth), (e) => {
      // 存不回凭据要报出去，不能吞：宿主据此知道账号要重新登录
      assert.equal(e.signIn, "expired")
      assert.match(e.message, /没能存回宿主/)
      assert.match(e.message, /no space left on device/)
      return true
    })
    // 换票确实发生过（服务端那边新 token 已经作废了旧的）
    assert.ok(fetchMock.calls.some((c) => c.url.endsWith("/oauth/token")))
  }, [[/\/oauth\/token/, () => json(OK_TOKEN)]])
})

test("续期后存不回宿主：这一代 token 不进缓存，再续一次会重新换票", async () => {
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-nostore2",
    expires: Date.now() - 1000,
  }
  await withFetch(async (fetchMock) => {
    // 第一次：换票成功，但存不回宿主
    await assert.rejects(() => renewTokenInternal(auth, async () => {
      throw new Error("no space left on device")
    }))
    // 第二次（60 秒缓存期内）：如果没落盘的那一代被缓存了，这里会直接拿缓存、
    // 不再换票，并假装成功。正确的行为是不信任它，重新走一次换票。
    const saved = []
    await renewTokenInternal(auth, async (merged) => {
      saved.push(merged)
    })
    const tokenCalls = fetchMock.calls.filter((c) => c.url.endsWith("/oauth/token"))
    assert.equal(tokenCalls.length, 2, "存不回宿主的那一代不能被缓存复用")
    assert.equal(saved.length, 1)
  }, [[/\/oauth\/token/, () => json(OK_TOKEN)]])
})

test("存回宿主成功：换到的 token 被记住并持久化", async () => {
  const client = fakeClient()
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-old", refresh: "ref-ok",
    expires: Date.now() - 1000,
  }
  await withFetch(async () => {
    const next = await renewTokenInternal(auth, async (merged) => {
      client.saved.push({ id: "meoo", body: merged })
    })
    assert.equal(next.access, "acc-1")
    assert.equal(next.refresh, "ref-1")
    assert.equal(client.saved.length, 1)
    assert.equal(client.saved[0].body.refresh, "ref-1")
  }, [[/\/oauth\/token/, () => json(OK_TOKEN)]])
})

test("解析出的 AK 存不回宿主时不拖垮这次操作（AK 已经在手上）", async () => {
  const client = failingClient()
  const { hooks } = await plugin(client)
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-1", refresh: "ref-1",
    expires: Date.now() + 3600_000, projectUrlId: "proj-a",
  }
  await withFetch(async () => {
    // AK 已经解析出来并能用，存不住只是下次要再解析一次，不该报错
    const l = await hooks.auth.loader(async () => auth)
    assert.equal(l.apiKey, "meoo_project_service_v1.env")
  }, [
    [/\/cloud\/environment/, () => json({ data: { meooProjectApiKey: "meoo_project_service_v1.env" } })],
  ])
})

// ---- 余额（usage）-----------------------------------------------------------

test("usage：积分按钱包窗口给出（余额/已用/最近到期）", async () => {
  const { hooks } = await plugin(fakeClient())
  const auth = {
    type: "oauth", apiBase: "https://meoo.test", access: "acc-1", refresh: "ref-1",
    expires: Date.now() + 3600_000, username: "Fishundersea",
  }
  const points = {
    data: {
      account: { availableBalance: 58848, frozenBalance: 0, totalGranted: 65000, totalConsumed: 2152 },
      activeGrants: [
        { grantDetails: [{ remainingAmount: 5000, expiresDate: "2026-12-02T23:59:59" }] },
        {
          grantDetails: [
            { remainingAmount: 1933, expiresDate: "2026-11-10T23:59:59" },
            { remainingAmount: 0, expiresDate: "2026-11-01T23:59:59" },
          ],
        },
      ],
    },
  }
  await withFetch(async (mock) => {
    const u = await hooks.auth.usage(async () => auth)
    assert.equal(u.user, "Fishundersea")
    assert.equal(u.balance, "58,848 积分")
    // 钱包窗口：balance + unit（新宿主），同时带 display（旧宿主当纯文本窗口）
    assert.deepEqual(u.windows[0], { name: "积分", balance: 58848, unit: "积分", display: "58,848 积分", runsOutAt: "2026-11-10" })
    assert.equal(u.windows[1].name, "已用")
    assert.equal(u.windows[1].display, "2,152 积分")
    assert.equal(u.windows[1].aside, true)
    const call = mock.calls.find((c) => c.url.includes("/account/points"))
    assert.equal(call.init.headers.authorization, "Bearer acc-1")
  }, [[/\/account\/points/, () => json(points)]])
})

test("usage：api 账号（手填 AK）读不到积分，明确报错", async () => {
  const { hooks } = await plugin(fakeClient())
  await assert.rejects(
    () => hooks.auth.usage(async () => ({ type: "api", key: "meoo_project_service_v1.x" })),
    /项目服务 AK/,
  )
})

// ---- 浏览器 PKCE 登录（loopback 回调全流程，回调用真实 fetch 打本机端口）----------

test("browserSignIn：授权 URL 带齐 PKCE 参数；回调 state 校验；换票后自动解析身份与 AK", async () => {
  process.env.MEOO_GATEWAY_BASE_URL = GATEWAY
  process.env.MEOO_MAIN_SITE_URL = "https://meoo.test"
  const backend = meooBackend({
    identity: { name: "陈同学", user_id: "u1", project_url_id: "proj-a" },
    projects: [{ url_id: "proj-a", name: "默认" }],
    env: { meooProjectApiKey: "meoo_project_service_v1.auto" },
    gatewayModels: [{ id: "glm-5.2" }],
  })
  const prev = globalThis.fetch
  globalThis.fetch = backend
  try {
    const m = await browserSignIn(60_000)
    assert.equal(m.method, "auto")
    const url = new URL(m.url)
    assert.equal(url.origin + url.pathname, "https://meoo.test/oauth/authorize")
    assert.equal(url.searchParams.get("response_type"), "code")
    assert.equal(url.searchParams.get("scope"), "cli.compat")
    assert.equal(url.searchParams.get("code_challenge_method"), "S256")
    const redirect = url.searchParams.get("redirect_uri")
    assert.match(redirect, /^http:\/\/127\.0\.0\.1:\d+\/oauth\/callback$/)
    const state = url.searchParams.get("state")

    // 错误 state 被拒（真实 fetch 打 loopback）
    const bad = await prev(`${redirect}?code=x&state=wrong`)
    assert.equal(bad.status, 400)

    // 正确回调：换票 + 身份 + AK 全链路成功
    const good = await prev(`${redirect}?code=the-code&state=${encodeURIComponent(state)}`)
    assert.equal(good.status, 200)
    const result = await m.callback()
    assert.equal(result.type, "success", JSON.stringify(result))
    assert.equal(result.access, "acc-1")
    assert.equal(result.accountId, "陈同学")
    assert.equal(result.serviceAk, "meoo_project_service_v1.auto")
    assert.equal(result.projectUrlId, "proj-a")
    const tokenCall = backend.calls.find((c) => c.url.endsWith("/oauth/token"))
    assert.ok(tokenCall)
    const body = new URLSearchParams(tokenCall.init.body)
    assert.equal(body.get("grant_type"), "authorization_code")
    assert.equal(body.get("redirect_uri"), redirect)
    assert.ok(body.get("code_verifier"))
  } finally {
    globalThis.fetch = prev
    delete process.env.MEOO_MAIN_SITE_URL
  }
})

test("browserSignIn：授权被拒 → failed；回调服务器会收尾", async () => {
  process.env.MEOO_MAIN_SITE_URL = "https://meoo.test"
  const prev = globalThis.fetch
  globalThis.fetch = fakeFetch([])
  try {
    const m = await browserSignIn(60_000)
    const url = new URL(m.url)
    const redirect = url.searchParams.get("redirect_uri")
    await prev(`${redirect}?error=access_denied&state=${encodeURIComponent(url.searchParams.get("state"))}`)
    const result = await m.callback()
    assert.equal(result.type, "failed")
    assert.match(result.error, /授权被取消/)
  } finally {
    globalThis.fetch = prev
    delete process.env.MEOO_MAIN_SITE_URL
  }
})
