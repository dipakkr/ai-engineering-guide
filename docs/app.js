// ─── Environment ─────────────────────────────────────────────────────────────

const IS_LOCAL = ['localhost', '127.0.0.1'].includes(window.location.hostname);
const CONTENT_BASE = IS_LOCAL
  ? '/content'
  : 'https://raw.githubusercontent.com/dipakkr/ai-engineering-guide/main';

// ─── Course Structure ─────────────────────────────────────────────────────────

const SECTIONS = [
  {
    id: 1, title: "LLM Foundations", dir: "01-llm-foundations",
    lessons: [
      { file: "what-is-an-llm",              title: "What is an LLM?" },
      { file: "01-transformer-intuition",   title: "Transformer Intuition" },
      { file: "02-tokenization",             title: "Tokenization" },
      { file: "03-attention-mechanisms",     title: "Attention Mechanisms" },
      { file: "04-context-windows",          title: "Context Windows" },
      { file: "05-training-pipeline",        title: "Training Pipeline" },
      { file: "06-model-landscape",          title: "Model Landscape" },
      { file: "07-small-language-models",    title: "Small Language Models" },
      { file: "08-quantization",             title: "Quantization" },
      { file: "09-fine-tuning",              title: "Fine-Tuning" },
      { file: "10-distillation-and-pruning", title: "Distillation and Pruning" },
    ],
  },
  {
    id: 2, title: "Prompt Engineering", dir: "02-prompt-engineering",
    lessons: [
      { file: "what-is-prompt-engineering",    title: "What is Prompt Engineering?" },
      { file: "01-prompting-patterns",         title: "Prompting Patterns" },
      { file: "02-context-engineering",  title: "Context Engineering" },
      { file: "03-structured-generation",title: "Structured Generation" },
      { file: "04-prompt-optimization",  title: "Prompt Optimization" },
      { file: "05-prompt-security",      title: "Prompt Security" },
    ],
  },
  {
    id: 3, title: "Retrieval and RAG", dir: "03-retrieval-and-rag",
    lessons: [
      { file: "what-is-rag",              title: "What is RAG?" },
      { file: "01-rag-fundamentals",      title: "RAG Fundamentals" },
      { file: "02-embedding-models",      title: "Embedding Models" },
      { file: "03-vector-indexing",       title: "Vector Indexing" },
      { file: "04-vector-databases",      title: "Vector Databases" },
      { file: "05-chunking-strategies",   title: "Chunking Strategies" },
      { file: "06-hybrid-search",         title: "Hybrid Search" },
      { file: "07-reranking",             title: "Reranking" },
      { file: "08-query-transformation",  title: "Query Transformation" },
      { file: "09-advanced-rag-patterns", title: "Advanced RAG Patterns" },
      { file: "10-multimodal-rag",        title: "Multimodal RAG" },
      { file: "11-rag-evaluation",        title: "RAG Evaluation" },
    ],
  },
  {
    id: 4, title: "Agents and Orchestration", dir: "04-agents-and-orchestration",
    lessons: [
      { file: "01-agent-fundamentals",          title: "Agent Fundamentals" },
      { file: "02-tool-use-and-function-calling",title: "Tool Use and Function Calling" },
      { file: "03-mcp-protocol",                title: "MCP Protocol" },
      { file: "04-langchain-overview",          title: "LangChain Overview" },
      { file: "05-langgraph-deep-dive",         title: "LangGraph Deep Dive" },
      { file: "06-dspy-framework",              title: "DSPy Framework" },
      { file: "07-crewai-and-autogen",          title: "CrewAI and AutoGen" },
      { file: "08-llamaindex-haystack",         title: "LlamaIndex and Haystack" },
      { file: "09-multi-agent-systems",         title: "Multi-Agent Systems" },
      { file: "10-memory-and-state",            title: "Memory and State" },
      { file: "11-agentic-patterns",            title: "Agentic Patterns" },
      { file: "12-browser-and-computer-use",    title: "Browser and Computer Use" },
    ],
  },
  {
    id: 5, title: "Evaluation", dir: "05-evaluation",
    lessons: [
      { file: "01-eval-fundamentals",       title: "Eval Fundamentals" },
      { file: "02-retrieval-and-rag-eval",  title: "Retrieval and RAG Eval" },
      { file: "03-llm-as-judge",            title: "LLM as Judge" },
      { file: "04-agent-and-e2e-eval",      title: "Agent and E2E Eval" },
    ],
  },
  {
    id: 6, title: "Production and Ops", dir: "06-production-and-ops",
    lessons: [
      { file: "01-observability-and-tracing", title: "Observability and Tracing" },
      { file: "02-guardrails-and-safety",     title: "Guardrails and Safety" },
      { file: "03-caching-strategies",        title: "Caching Strategies" },
      { file: "04-inference-infrastructure",  title: "Inference Infrastructure" },
      { file: "05-drift-and-monitoring",      title: "Drift and Monitoring" },
      { file: "06-mlops-for-llms",            title: "MLOps for LLMs" },
      { file: "07-cost-optimization",         title: "Cost Optimization" },
    ],
  },
  {
    id: 7, title: "System Design Interview", dir: "07-system-design-interview",
    lessons: [
      { file: "01-interview-framework",      title: "Interview Framework" },
      { file: "02-design-patterns-catalog",  title: "Design Patterns Catalog" },
      { file: "03-architecture-templates",   title: "Architecture Templates" },
      { file: "04-case-enterprise-rag",      title: "Case: Enterprise RAG" },
      { file: "05-case-code-assistant",      title: "Case: Code Assistant" },
      { file: "06-case-customer-support",    title: "Case: Customer Support" },
      { file: "07-case-doc-intelligence",    title: "Case: Doc Intelligence" },
      { file: "08-case-search-engine",       title: "Case: Search Engine" },
      { file: "09-practice-problems",        title: "Practice Problems" },
      { file: "10-conceptual-questions",     title: "Conceptual Questions" },
    ],
  },
  {
    id: 8, title: "Appendices", dir: "appendices",
    lessons: [
      { file: "model-pricing-reference",   title: "Model Pricing Reference" },
      { file: "glossary",                  title: "Glossary" },
      { file: "cost-estimation-formulas",  title: "Cost Estimation Formulas" },
      { file: "essential-papers",          title: "Essential Papers" },
    ],
  },
];

// Flat list of all lessons for sequential navigation
const ALL_LESSONS = [];
SECTIONS.forEach(section => {
  section.lessons.forEach((lesson, i) => {
    ALL_LESSONS.push({
      path: `${section.dir}/${lesson.file}.md`,
      title: lesson.title,
      sectionId: section.id,
      num: `${section.id}.${i + 1}`,
    });
  });
});

const TOTAL = ALL_LESSONS.length;

// ─── State ────────────────────────────────────────────────────────────────────

const state = {
  currentPath: null,
  completed: new Set(JSON.parse(localStorage.getItem('asg_completed') || '[]')),
  expanded:  new Set(JSON.parse(localStorage.getItem('asg_expanded')  || '[1]')),
  dark: localStorage.getItem('asg_dark') === 'true',
};

// ─── DOM helpers ──────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

// ─── Theme ────────────────────────────────────────────────────────────────────

function applyTheme() {
  document.documentElement.dataset.theme = state.dark ? 'dark' : 'light';
  $('theme-toggle').textContent = state.dark ? '☀️' : '🌙';
  $('code-theme').href = state.dark
    ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css'
    : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
  mermaid.initialize({ startOnLoad: false, theme: state.dark ? 'dark' : 'default' });
}

function toggleTheme() {
  state.dark = !state.dark;
  localStorage.setItem('asg_dark', state.dark);
  applyTheme();
  // Re-render current lesson so mermaid picks up the new theme
  if (state.currentPath) loadLesson(state.currentPath, true);
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

function renderSidebar() {
  const done = state.completed.size;
  $('progress-count').textContent = `${done} / ${TOTAL} lessons`;

  $('course-nav').innerHTML = SECTIONS.map(section => {
    const expanded = state.expanded.has(section.id);
    return `
      <div class="section">
        <div class="section-header" onclick="toggleSection(${section.id})">
          <span class="section-title">${section.id}. ${section.title}</span>
          <span class="section-chevron">${expanded ? '▾' : '›'}</span>
        </div>
        <div class="section-lessons ${expanded ? 'expanded' : ''}">
          ${section.lessons.map((lesson, i) => {
            const path = `${section.dir}/${lesson.file}.md`;
            const isActive = path === state.currentPath;
            const isDone   = state.completed.has(path);
            return `
              <div class="lesson-item ${isActive ? 'active' : ''}"
                   onclick="loadLesson('${path}')">
                <span class="lesson-num">${section.id}.${i + 1}</span>
                <span class="lesson-title">${lesson.title}</span>
                ${isDone ? '<span class="lesson-check">✓</span>' : ''}
              </div>`;
          }).join('')}
        </div>
      </div>`;
  }).join('');
}

function toggleSection(id) {
  state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id);
  localStorage.setItem('asg_expanded', JSON.stringify([...state.expanded]));
  renderSidebar();
}

// ─── Markdown rendering ───────────────────────────────────────────────────────

function setupMarked() {
  // Custom renderer: handle mermaid blocks separately from code blocks
  const renderer = new marked.Renderer();

  renderer.code = function(code, lang) {
    if (lang === 'mermaid') {
      // Wrap source in a div for mermaid.run() to find
      const escaped = code
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      return `<div class="mermaid-container"><div class="mermaid">${escaped}</div></div>`;
    }
    const language = (lang && hljs.getLanguage(lang)) ? lang : 'plaintext';
    const highlighted = hljs.highlight(code, { language, ignoreIllegals: true }).value;
    return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`;
  };

  // Open external links in new tab
  const originalLink = renderer.link.bind(renderer);
  renderer.link = function(href, title, text) {
    const html = originalLink(href, title, text);
    if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
      return html.replace('<a ', '<a target="_blank" rel="noopener" ');
    }
    return html;
  };

  marked.use({ renderer, gfm: true, breaks: false });
}

// ─── Load Lesson ──────────────────────────────────────────────────────────────

// ─── URL Routing ──────────────────────────────────────────────────────────────

function pathToHash(path) {
  // "01-llm-foundations/01-transformer-intuition.md" → "#01-llm-foundations/01-transformer-intuition"
  return '#' + path.replace(/\.md$/, '');
}

function hashToPath(hash) {
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  return raw ? raw + '.md' : null;
}

function updateURL(path) {
  const hash = pathToHash(path);
  if (window.location.hash !== hash) {
    history.pushState(null, '', hash);
  }
}

// ─── Load Lesson ──────────────────────────────────────────────────────────────

async function loadLesson(path, silent = false) {
  state.currentPath = path;
  if (!silent) {
    updateURL(path);
    localStorage.setItem('asg_current', path);
  }

  // Expand the section that contains this lesson
  const lesson = ALL_LESSONS.find(l => l.path === path);
  if (lesson && !state.expanded.has(lesson.sectionId)) {
    state.expanded.add(lesson.sectionId);
    localStorage.setItem('asg_expanded', JSON.stringify([...state.expanded]));
  }

  renderSidebar();

  if (!silent) {
    $('content').innerHTML = '<div class="loading">Loading…</div>';
    $('lesson-title').textContent = lesson?.title || '';
    $('read-time').textContent = '⏱ — min read';
    $('toc').innerHTML = '';
  }

  try {
    // Fetch with 10s timeout
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10_000);
    const res = await fetch(`${CONTENT_BASE}/${path}`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${path}`);
    const md = await res.text();

    // Parse markdown to HTML
    const html = marked.parse(md);
    const content = $('content');
    content.innerHTML = html;

    // Hoist the first H1 into the header (avoid duplicating it in body)
    const h1 = content.querySelector('h1');
    if (h1) {
      $('lesson-title').textContent = h1.textContent;
      h1.remove();
    } else {
      $('lesson-title').textContent = lesson?.title || '';
    }

    // Read time estimate
    const words = content.textContent.trim().split(/\s+/).length;
    $('read-time').textContent = `⏱ ${Math.max(1, Math.round(words / 200))} min read`;

    // Render mermaid diagrams with 5s timeout per batch
    const mermaidEls = Array.from(content.querySelectorAll('.mermaid'));
    if (mermaidEls.length > 0) {
      try {
        await Promise.race([
          mermaid.run({ nodes: mermaidEls }),
          new Promise((_, reject) => setTimeout(() => reject(new Error('mermaid timeout')), 5_000))
        ]);
      } catch (e) {
        console.warn('Mermaid skipped:', e.message);
        // Replace failed diagrams with a plain code block fallback
        mermaidEls.forEach(el => {
          if (!el.querySelector('svg')) {
            el.style.cssText = 'background:#f8fafc;padding:12px;border-radius:6px;font-size:12px;overflow-x:auto;white-space:pre';
          }
        });
      }
    }

    // Build TOC from H2 / H3 headings
    renderTOC(content);

    // Intercept internal .md links and navigate within the app
    content.querySelectorAll('a[href]').forEach(a => {
      const href = a.getAttribute('href');
      if (!href || href.startsWith('http') || href.startsWith('#')) return;
      if (href.endsWith('.md')) {
        a.addEventListener('click', e => {
          e.preventDefault();
          const filename = href.split('/').pop().replace('.md', '');
          const found = ALL_LESSONS.find(l => l.path.includes(filename));
          if (found) loadLesson(found.path);
        });
      }
    });

    updateNavButtons();
    updateCompleteButton();
    if (!silent) $('main').scrollTop = 0;

  } catch (err) {
    const isTimeout = err.name === 'AbortError';
    $('content').innerHTML = `
      <div class="error">
        <p>Could not load <strong>${path}</strong></p>
        <p style="margin-top:12px;font-size:13px;">
          ${isTimeout ? 'Request timed out. Check your connection and try again.' : err.message}
        </p>
        ${IS_LOCAL ? `<p style="margin-top:12px;font-size:13px;">
          Make sure the server is running:<br>
          <code>cd docs &amp;&amp; python3 server.py</code>
        </p>` : ''}
      </div>`;
  }
}

// ─── Table of Contents ────────────────────────────────────────────────────────

function renderTOC(content) {
  const headings = Array.from(content.querySelectorAll('h2, h3'));
  const toc = $('toc');

  if (headings.length === 0) {
    toc.innerHTML = '<div class="toc-empty">No sections</div>';
    return;
  }

  headings.forEach((h, i) => {
    if (!h.id) {
      h.id = `h-${i}-` + h.textContent
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/(^-|-$)/g, '');
    }
  });

  toc.innerHTML = headings.map(h => `
    <a class="toc-item toc-${h.tagName.toLowerCase()}" href="#${h.id}">${h.textContent}</a>
  `).join('') + '<a class="toc-back" href="#">Back to top</a>';

  toc.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const href = a.getAttribute('href');
      if (href === '#') {
        $('main').scrollTop = 0;
      } else {
        const el = document.querySelector(href);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// ─── Navigation ───────────────────────────────────────────────────────────────

function getCurrentIndex() {
  return ALL_LESSONS.findIndex(l => l.path === state.currentPath);
}

function updateNavButtons() {
  const idx = getCurrentIndex();
  const prev = ALL_LESSONS[idx - 1];
  const next = ALL_LESSONS[idx + 1];

  const prevBtn = $('prev-btn');
  const nextBtn = $('next-btn');

  prevBtn.textContent = prev ? `← ${prev.title}` : '← Previous';
  prevBtn.disabled = !prev;
  prevBtn.onclick = prev ? () => loadLesson(prev.path) : null;

  nextBtn.textContent = next ? `${next.title} →` : 'Next →';
  nextBtn.disabled = !next;
  nextBtn.onclick = next ? () => loadLesson(next.path) : null;
}

function updateCompleteButton() {
  const btn = $('complete-btn');
  const done = state.completed.has(state.currentPath);
  btn.textContent = done ? '✓ Completed' : '✓ Mark Complete';
  btn.classList.toggle('completed', done);
}

function markComplete() {
  if (!state.currentPath) return;
  if (state.completed.has(state.currentPath)) {
    state.completed.delete(state.currentPath);
  } else {
    state.completed.add(state.currentPath);
    // Auto-advance to next lesson
    const next = ALL_LESSONS[getCurrentIndex() + 1];
    if (next) setTimeout(() => loadLesson(next.path), 400);
  }
  localStorage.setItem('asg_completed', JSON.stringify([...state.completed]));
  renderSidebar();
  updateCompleteButton();
}

// ─── Init ─────────────────────────────────────────────────────────────────────

function init() {
  mermaid.initialize({ startOnLoad: false, theme: state.dark ? 'dark' : 'default' });
  setupMarked();
  applyTheme();
  renderSidebar();

  $('theme-toggle').onclick = toggleTheme;
  $('complete-btn').onclick = markComplete;

  // Keyboard navigation: left/right arrow keys
  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === 'ArrowRight') {
      const next = ALL_LESSONS[getCurrentIndex() + 1];
      if (next) loadLesson(next.path);
    }
    if (e.key === 'ArrowLeft') {
      const prev = ALL_LESSONS[getCurrentIndex() - 1];
      if (prev) loadLesson(prev.path);
    }
  });

  // Browser back/forward navigation
  window.addEventListener('popstate', () => {
    const path = hashToPath(window.location.hash);
    const found = path && ALL_LESSONS.find(l => l.path === path);
    if (found) loadLesson(found.path);
  });

  // Resolve starting lesson: URL hash → localStorage → first lesson
  const hashPath = hashToPath(window.location.hash);
  const fromHash = hashPath && ALL_LESSONS.find(l => l.path === hashPath);
  const last = localStorage.getItem('asg_current');
  const fromStorage = ALL_LESSONS.find(l => l.path === last);
  const start = fromHash || fromStorage || ALL_LESSONS[0];
  loadLesson(start.path);
}

// ─── GitHub Star Count ────────────────────────────────────────────────────────

async function loadStarCount() {
  try {
    const res = await fetch('https://api.github.com/repos/dipakkr/ai-engineering-guide');
    if (!res.ok) return;
    const data = await res.json();
    const count = data.stargazers_count;
    const label = count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
    document.getElementById('star-count').textContent = label;
  } catch {
    // Silently fail — button still works as a link
  }
}

document.addEventListener('DOMContentLoaded', () => { init(); loadStarCount(); });
