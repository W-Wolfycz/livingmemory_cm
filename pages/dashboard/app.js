/**
 * LivingMemory Dashboard - 主入口
 * 使用模块化架构，保持主文件简洁清晰
 */

import {
  ApiClient,
  PeekPanel,
  MemoryPage,
  RecallPage,
  SystemPage,
  esc,
  statusPill,
  nodeBadge,
} from "./modules/index.js";

(() => {
  "use strict";

  /* ================================================================
     State
     ================================================================ */
  const state = {
    page: "graph",
    memory: {
      items: [],
      total: 0,
      page: 1,
      pageSize: 20,
      hasMore: false,
      keyword: "",
      session: "",
      status: "all",
      type: "all",
      sort: "created_desc",
      selectedIds: new Set(),
    },
    selectedMemory: null,
    isEditing: false,
    _detailCache: null,
    _nodeDetailCache: null,
    _recallCache: null,
    _systemCache: null,
    pendingSearch: null,
  };

  /* ================================================================
     Initialize Modules
     ================================================================ */
  const api = new ApiClient();
  const peekPanel = new PeekPanel(state, api);
  const memoryPage = new MemoryPage(state, api, peekPanel);
  const recallPage = new RecallPage(state, api, peekPanel);
  const systemPage = new SystemPage(state, api);

  function hydrateIcons() {
    if (!window.lucide || typeof window.lucide.createIcons !== "function") return;
    window.lucide.createIcons({
      attrs: {
        "stroke-width": 1.7,
        "aria-hidden": "true",
      },
    });
  }

  function initMotionField() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let frame = 0;
    document.addEventListener("pointermove", (event) => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        const x = (event.clientX / Math.max(window.innerWidth, 1) - 0.5) * 12;
        const y = (event.clientY / Math.max(window.innerHeight, 1) - 0.5) * 12;
        document.documentElement.style.setProperty("--field-x", x.toFixed(2) + "px");
        document.documentElement.style.setProperty("--field-y", y.toFixed(2) + "px");
        frame = 0;
      });
    }, { passive: true });
  }

  /* ================================================================
     Theme Management
     ================================================================ */
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const darkIcon = document.getElementById("theme-icon-dark");
    const lightIcon = document.getElementById("theme-icon-light");
    if (darkIcon && lightIcon) {
      darkIcon.classList.toggle("hidden", theme === "light");
      lightIcon.classList.toggle("hidden", theme === "dark");
    }
  }

  function getInitialTheme(context) {
    if (context && typeof context.isDark === "boolean") {
      return context.isDark ? "dark" : "light";
    }

    try {
      const saved = localStorage.getItem("lmem_theme");
      if (saved === "dark" || saved === "light") {
        return saved;
      }
    } catch (e) {
      console.warn("[LM] Failed to read theme from localStorage:", e);
    }

    return "light";
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    const next = current === "light" ? "dark" : "light";

    try {
      localStorage.setItem("lmem_theme", next);
    } catch (e) {
      console.warn("[LM] Failed to save theme to localStorage:", e);
    }

    applyTheme(next);
    showToast(window.t(next === "dark" ? "theme.darkToast" : "theme.lightToast"));
  }

  /* ================================================================
     Toast Notification
     ================================================================ */
  let toastTimer;
  function showToast(msg, isError = false) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.remove("visible", "error");
    if (isError) el.classList.add("error");
    void el.offsetWidth;
    el.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.classList.remove("visible");
    }, 2500);
  }

  /* ================================================================
     Sidebar / Routing
     ================================================================ */
  function switchPage(name) {
    state.page = name;

    document.querySelectorAll(".nav-item[data-page]").forEach(item => {
      item.classList.toggle("active", item.dataset.page === name);
    });

    document.querySelectorAll(".page").forEach(p => {
      p.classList.toggle("active", p.id === "page-" + name);
    });

    if (name === "graph") {
      fetchGraphStats();
      if (window.ensureGraphScene) window.ensureGraphScene();
    }
    if (name === "memory") memoryPage.fetch();
    if (name === "recall") { /* 召回页面按需加载 */ }
    if (name === "system") systemPage.fetch();
  }

  function initSidebar() {
    document.querySelectorAll(".nav-item[data-page]").forEach(item => {
      item.addEventListener("click", () => {
        switchPage(item.dataset.page);
      });
    });

    document.getElementById("theme-toggle").addEventListener("click", toggleTheme);

  }

  /* ================================================================
     Persona Selector (CM compatibility)
     ================================================================ */
  function initPersonaSelector() {
    const select = document.getElementById("persona-selector");
    if (!select) return;

    try {
      const saved = localStorage.getItem("lmem_persona_id");
      if (saved) window.lmPersonaId = saved;
    } catch (e) {
      console.warn("[LM] Failed to read persona from localStorage:", e);
    }

    api.get("personas").then(data => {
      const items = data.items || [];
      const current = window.lmPersonaId || "";
      select.querySelectorAll("option:not([value=''])").forEach(option => option.remove());
      items.forEach(id => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        select.appendChild(option);
      });
      select.value = current;
    }).catch(error => console.warn("[LM] Failed to load personas:", error));

    select.addEventListener("change", () => {
      const personaId = select.value;
      window.lmPersonaId = personaId;
      try {
        localStorage.setItem("lmem_persona_id", personaId);
      } catch (e) {
        console.warn("[LM] Failed to save persona to localStorage:", e);
      }
      window.dispatchEvent(new CustomEvent("personachange", { detail: { personaId } }));
    });
  }

  function onPersonaChange() {
    state._systemCache = null;
    state._recallCache = null;
    state._detailCache = null;
    state._nodeDetailCache = null;

    if (state.page === "graph") {
      fetchGraphStats();
      if (window.refreshGraphForPersona) window.refreshGraphForPersona();
    } else if (state.page === "memory") {
      memoryPage.fetch();
    } else if (state.page === "recall") {
      const query = document.getElementById("recall-query");
      if (query && query.value.trim()) recallPage.runRecall();
    } else if (state.page === "system") {
      systemPage.fetch();
    }
  }

  /* ================================================================
     Graph Page (依赖 graph-ui.js)
     ================================================================ */
  async function fetchGraphStats() {
    try {
      const data = await api.get("stats");

      document.getElementById("gs-total").textContent = data.total_memories || 0;
      document.getElementById("gs-nodes").textContent = data.graph_nodes || 0;
      document.getElementById("gs-edges").textContent = data.graph_edges || 0;

      const sessions = data.sessions || {};
      const sessionCount = typeof sessions === "object" ? Object.keys(sessions).length : 0;
      document.getElementById("gs-sessions").textContent = sessionCount;
    } catch (e) {
      showToast(e.message || window.t("misc.statsFail"), true);
    }
  }

  /* ================================================================
     Initialization
     ================================================================ */
  async function init() {
    hydrateIcons();
    initMotionField();
    const context = await api.ready();

    if (api.bridge && typeof api.bridge.onContext === "function") {
      api.bridge.onContext((ctx) => {
        if (ctx && typeof ctx.isDark === "boolean") {
          const newTheme = ctx.isDark ? "dark" : "light";
          const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
          if (newTheme !== currentTheme) {
            applyTheme(newTheme);
          }
        }

      });
    }

    const initialTheme = getInitialTheme(context);
    applyTheme(initialTheme);

    initSidebar();
    initPersonaSelector();

    memoryPage.initEventListeners();
    recallPage.initEventListeners();

    document.getElementById("peek-close").addEventListener("click", () => peekPanel.close());
    document.getElementById("peek-overlay").addEventListener("click", () => peekPanel.close());

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        peekPanel.close();
      }
    });
    window.addEventListener("personachange", onPersonaChange);

    fetchGraphStats();
    switchPage("graph");
  }

  /* ================================================================
     Global Exports (for graph-ui.js and other dependencies)
     ================================================================ */
  window.lmState = state;
  window.lmShowToast = showToast;
  window.lmApiRequest = (path, options) => {
    // 兼容旧 API，转发到 ApiClient
    if (options && options.method === "POST") {
      return api.request(path, options);
    }
    return api.request(path, options || {});
  };
  window.lmOpenPeekNode = (nodeData) => peekPanel.renderNode(nodeData);
  window.lmOpenPeekMemory = (memory) => peekPanel.renderMemory(memory);
  window.lmClosePeek = () => peekPanel.close();
  window.lmFetchGraphStats = fetchGraphStats;
  window.lmRefreshMemories = () => memoryPage.fetch();
  window.lmEsc = esc;
  window.lmStatusPill = statusPill;
  window.lmNodeBadge = nodeBadge;
  window.lmHydrateIcons = hydrateIcons;

  // 图谱小视图绘制函数（如果需要）
  window.lmDrawMiniGraph = (canvas, nodes, edges) => {
    if (!canvas || !nodes || !nodes.length) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // 简单布局算法
    const positions = nodes.map((node, i) => {
      const angle = (i / nodes.length) * Math.PI * 2;
      const r = Math.min(W, H) * 0.3;
      return {
        x: W / 2 + r * Math.cos(angle),
        y: H / 2 + r * Math.sin(angle),
        node
      };
    });

    // 绘制边
    if (edges && edges.length) {
      ctx.strokeStyle = "rgba(100, 100, 100, 0.3)";
      ctx.lineWidth = 1;
      edges.forEach(edge => {
        const source = positions.find(p => p.node.id === edge.source || p.node.id === edge.from);
        const target = positions.find(p => p.node.id === edge.target || p.node.id === edge.to);
        if (source && target) {
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }
      });
    }

    // 绘制节点
    positions.forEach(pos => {
      ctx.fillStyle = "#4a90e2";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  // 启动应用
  init();
})();
