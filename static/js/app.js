/**
 * Stock Deep Diagnostic Report - Frontend
 */
(function () {
  'use strict';

  // ========== State ==========
  let currentReport = null;
  let chatOpen = false;
  let settingsOpen = false;
  let deepAnalyzing = {};  // track which dims are currently analyzing

  // ========== Query History ==========
  const HISTORY_KEY = 'stock_analyzer_history';
  const MAX_HISTORY = 10;

  function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
    catch (e) { return []; }
  }
  function addHistory(code, name) {
    let h = getHistory();
    h = h.filter(item => item.code !== code);
    h.unshift({ code, name: name || code, time: new Date().toLocaleTimeString() });
    if (h.length > MAX_HISTORY) h = h.slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
    renderHistoryBar();
  }

  function renderHistoryBar() {
    const h = getHistory();
    const bar = $('#historyBar');
    if (!bar) return;
    if (h.length === 0) {
      bar.style.display = 'none';
      return;
    }
    bar.style.display = 'flex';
    bar.innerHTML = `<span style="font-size:0.8rem;color:var(--primary);font-weight:600;white-space:nowrap">📋 最近查询：</span>` +
      h.slice(0, 7).map(item =>
        `<span class="history-chip" onclick="document.getElementById('searchCode').value='${item.code}';document.getElementById('searchBtn').click()" title="${item.time}">${item.name}</span>`
      ).join('');
  }

  // ========== DOM Refs ==========
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);
  const searchInput = $('#searchCode');
  const searchBtn = $('#searchBtn');
  const loadingOverlay = $('#loadingOverlay');
  const loadingText = $('#loadingText');
  const reportArea = $('#reportArea');
  const emptyState = $('#emptyState');
  const chatToggle = $('#chatToggle');
  const chatPanel = $('#chatPanel');
  const chatMessages = $('#chatMessages');
  const chatInput = $('#chatInput');
  const chatSend = $('#chatSend');
  const chatClose = $('#chatClose');
  const settingsBtn = $('#settingsBtn');
  const settingsOverlay = $('#settingsOverlay');
  const settingsModal = $('#settingsModal');

  // ========== Loading ==========
  function showLoading(msg) {
    loadingText.textContent = msg || '正在分析中...';
    loadingOverlay.classList.add('active');
  }
  function hideLoading() {
    loadingOverlay.classList.remove('active');
  }

  // ========== Search ==========
  async function analyzeStock(code) {
    if (!code) return;
    showLoading(`正在获取 ${code} 的数据...`);

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      const data = await resp.json();
      _logFallbackInfo('报告分析', data, resp);
      if (data.error) {
        alert(data.error);
        return;
      }
      currentReport = data;
      renderReport(data);
      emptyState.classList.add('hidden');
      reportArea.classList.remove('hidden');
      addHistory(data.code, data.name);
      loadAlternatives(code);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      alert('请求失败: ' + err.message);
    } finally {
      hideLoading();
    }
  }

  searchBtn.addEventListener('click', () => {
    const code = searchInput.value.trim();
    analyzeStock(code);
  });
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') analyzeStock(searchInput.value.trim());
  });

  // ========== Render Report ==========
  function getBoardEmoji(boardName) {
    const map = {
      '白酒': '🍶', '啤酒': '🍺', '饮料': '🥤', '食品': '🍔',
      '银行': '🏦', '证券': '📈', '保险': '🛡️', '房地产': '🏠',
      '医药': '💊', '医疗': '🏥', '生物': '🧬', '制药': '💉',
      '汽车': '🚗', '新能源': '⚡', '电池': '🔋', '光伏': '☀️', '电力': '⚡',
      '半导体': '💻', '芯片': '🔲', '电子': '📱', '计算机': '🖥️', '软件': '⌨️',
      '通信': '📡', '互联网': '🌐', '传媒': '📺', '游戏': '🎮',
      '军工': '🛩️', '航空航天': '🚀', '船舶': '🚢',
      '钢铁': '🔩', '有色': '🪙', '煤炭': '⛏️', '石油': '🛢️', '化工': '⚗️',
      '建材': '🧱', '建筑': '🏗️', '机械': '⚙️', '电气': '🔌',
      '纺织': '🧵', '服装': '👔', '家电': '📺',
      '农林': '🌾', '牧渔': '🐟', '养殖': '🐷',
      '交通运输': '🚄', '物流': '📦', '仓储': '🏭',
      '商贸': '🛒', '零售': '🏪', '旅游': '✈️', '酒店': '🏨',
      '环保': '♻️', '公用事业': '🏭',
      '教育': '📚', '出版': '📖',
    };
    for (const [key, emoji] of Object.entries(map)) {
      if (boardName.includes(key)) return emoji;
    }
    return '📊';
  }

  function renderReport(r) {
    const name = r.name || r.code;
    const price = r.price || 0;
    const pe = r.pe || 0;
    const pb = r.pb || 0;
    const totalMv = r.total_mv ? (r.total_mv).toFixed(2) : '--';
    const circMv = r.circ_mv ? (r.circ_mv).toFixed(2) : '--';

    const totalScore = r.total_score || 0;
    const maxScore = r.max_score || 100;
    const ratio = totalScore / maxScore;
    let scoreClass = '';
    if (ratio >= 0.6) scoreClass = 'good';
    else if (ratio >= 0.35) scoreClass = 'mid';

    const recd = r.recommendation || '--';

    // ========== Stock Title Banner ==========
    const boardName = (r.scores?.industry?.detail?.board_name) || (r.raw?.industry?.board) || '';
    const industryName = (r.scores?.industry?.detail?.industry_name) || (r.raw?.industry?.name) || '';
    const boardLabel = boardName || industryName || '';
    const boardEmoji = getBoardEmoji(boardLabel);
    const stockTitleHtml = `
      <div class="stock-title-banner">
        <div class="stock-title-bg-emoji">${boardEmoji}</div>
        <div class="stock-title-glass">
          <div class="stock-title-left">
            <span class="stock-title-emoji">${boardEmoji}</span>
          </div>
          <div class="stock-title-center">
            <h1 class="stock-title-name">${name}</h1>
            <div class="stock-title-code">${r.code || code}</div>
            ${boardLabel ? `<div class="stock-title-board">${boardLabel}</div>` : ''}
            <div class="stock-title-time">📅 分析时间：${r.report_time || ''}</div>
          </div>
          <div class="stock-title-right">
            <div class="stock-title-price">¥ ${price.toFixed(2)}</div>
            <div class="stock-title-change ${r.change_pct >= 0 ? 'trend-up' : 'trend-down'}">${r.change_pct > 0 ? '+' : ''}${(r.change_pct||0).toFixed(2)}%</div>
          </div>
        </div>
      </div>
    `;

    // Build warnings banner
    const warnings = r.warnings || [];
    let warningsHtml = '';
    if (warnings.length > 0) {
      warningsHtml = `
        <div class="warnings-banner">
          <div class="warnings-title">⚠️ 部分数据获取失败，以下维度评分可能不准确：</div>
          ${warnings.map(w => `<div class="warning-item">• <b>${w.dim}</b>: ${w.msg}</div>`).join('')}
        </div>`;
    }

    // Score overview
    const scores = r.scores || {};
    let scoreCards = '';
    let totalCard = '';

    // Total score card first
    totalCard = `
      <div class="score-card score-total ${scoreClass}">
        <div class="label">综合评分</div>
        <div class="value">${totalScore}</div>
        <div class="sub">/ ${maxScore} | ${recd}</div>
      </div>
    `;

    // Dimension score cards
    const dimNames = {
      fundamental: { label: '基本面', cls: 'score-fund' },
      technical: { label: '技术面', cls: 'score-tech' },
      capital: { label: '资金面', cls: 'score-flow' },
      events: { label: '事件催化', cls: 'score-basic' },
      industry: { label: '同业对标', cls: 'score-basic' },
      value: { label: '投资性价比', cls: 'score-fund' },
    };

    for (const [key, dim] of Object.entries(dimNames)) {
      const s = scores[key] || { score: '--', max: '--' };
      scoreCards += `
        <div class="score-card ${dim.cls}">
          <div class="label">${dim.label}</div>
          <div class="value">${s.score}/${s.max}</div>
          <div class="sub">${s.summary || ''}</div>
        </div>
      `;
    }

    reportArea.innerHTML = `
      ${stockTitleHtml}
      ${warningsHtml}

      <!-- Query History Bar -->
      <div id="historyBar" class="history-bar" style="display:flex"></div>

      <!-- Scoring Methodology -->
      <div class="scoring-methodology">
        <h4>📐 评分体系说明（公开透明）</h4>
        <div class="scoring-method-grid">
          <div class="scoring-method-item">📋 <b>基本面 (25分)</b> — ROE、营收增速、EPS趋势、资产负债率、PE估值合理性</div>
          <div class="scoring-method-item">📈 <b>技术面 (20分)</b> — 均线多空排列、MACD金叉死叉、KDJ超买超卖、布林带位置、半年涨跌幅</div>
          <div class="scoring-method-item">💰 <b>资金面 (15分)</b> — 近5日主力净流入、超大单/大单动向、量价背离检测、主力vs散户结构化分析</div>
          <div class="scoring-method-item">📅 <b>事件催化 (10分)</b> — 加权关键词分析：增持/回购/中标(+3~5)、减持/处罚/违约(-3~5)等，评分更加细化。配置 LLM 后可网络搜索+深度分析</div>
          <div class="scoring-method-item">🏭 <b>同业对标 (15分)</b> — 行业PE/PB合理区间对标、行业ROE对比、申万行业分类估值评估</div>
          <div class="scoring-method-item">🎯 <b>投资性价比 (15分)</b> — 股息率、PE估值分位、PEG成长性、ROE盈利能力综合评估</div>
        </div>
        <p style="font-size:0.72rem;color:var(--text-secondary);margin-top:8px">
          💡 总分 ≥ 60 → 持有/增持 | 40-59 → 谨慎持有 | 25-39 → 减仓观望 | &lt;25 → 不推荐。评分基于公开数据自动计算，仅供参考。
        </p>
      </div>

      <!-- Score Overview -->
      <div class="score-overview">${totalCard}${scoreCards}</div>

      <!-- Basic Info -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>📊 基本信息</h3>
          <div style="display:flex;gap:8px">
            <button class="btn-deep-analyze btn-analyze-all" onclick="event.stopPropagation();handleAnalyzeAll()" title="一键生成所有维度的深度分析">🚀 一键深度分析</button>
          </div>
        </div>
        <div class="section-body">
          <div class="stat-grid">
            <div class="stat-item"><div class="stat-label">最新股价</div><div class="stat-value">${price.toFixed(2)} 元</div></div>
            <div class="stat-item"><div class="stat-label">涨跌幅</div><div class="stat-value ${r.change_pct >= 0 ? 'trend-up' : 'trend-down'}">${r.change_pct > 0 ? '+' : ''}${r.change_pct.toFixed(2)}%</div></div>
            <div class="stat-item"><div class="stat-label">PE(TTM)</div><div class="stat-value">${pe > 0 ? pe.toFixed(2) : '亏损'}</div></div>
            <div class="stat-item"><div class="stat-label">PB</div><div class="stat-value">${pb.toFixed(2)}</div></div>
            <div class="stat-item"><div class="stat-label">总市值</div><div class="stat-value">${totalMv} 亿</div></div>
            <div class="stat-item"><div class="stat-label">流通市值</div><div class="stat-value">${circMv} 亿</div></div>
          </div>
        </div>
      </div>

      <!-- K-line Chart -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>🕯️ K线走势图</h3>
          <span style="font-size:0.75rem;color:var(--text-secondary)">拖拽滑块调整时间范围 · 7条均线 · 共${(r.prices_data||[]).length}根K线</span>
        </div>
        <div class="section-body">
          <div id="klineChart" style="width:100%;height:420px"></div>
          <div class="kline-ma-legend">
            <span class="ma-dot ma5">MA5</span>
            <span class="ma-dot ma10">MA10</span>
            <span class="ma-dot ma20">MA20</span>
            <span class="ma-dot ma30">MA30</span>
            <span class="ma-dot ma60">MA60</span>
            <span class="ma-dot ma120">MA120</span>
            <span class="ma-dot ma250">MA250</span>
          </div>
        </div>
      </div>

      <!-- Section 1: Fundamental -->
        <div class="section-body">
          ${buildFundamentalSection(r)}
          <div class="deep-analyze-panel" id="deep-fundamental" style="display:none"></div>
        </div>
      </div>

      <!-- Section 2: Technical -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>📈 二、技术面扫描 <span class="score-badge ${(scores.technical||{}).score >= 14 ? 'good' : 'warning'}">${(scores.technical||{}).score||0}/${(scores.technical||{}).max||20}</span></h3>
          <button class="btn-deep-analyze" onclick="event.stopPropagation();handleDeepAnalyze('technical')" title="AI 深度分析 + 多空辩论">🔬 深度分析</button>
        </div>
        <div class="section-body">
          ${buildTechnicalSection(r)}
          <div class="deep-analyze-panel" id="deep-technical" style="display:none"></div>
        </div>
      </div>

      <!-- Section 3: Capital Flow -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>💰 三、资金面透视 <span class="score-badge ${(scores.capital||{}).score >= 10 ? 'good' : 'warning'}">${(scores.capital||{}).score||0}/${(scores.capital||{}).max||15}</span></h3>
          <button class="btn-deep-analyze" onclick="event.stopPropagation();handleDeepAnalyze('capital')" title="AI 深度分析 + 多空辩论">🔬 深度分析</button>
        </div>
        <div class="section-body">
          ${buildCapitalSection(r)}
          <div class="deep-analyze-panel" id="deep-capital" style="display:none"></div>
        </div>
      </div>

      <!-- Section 4: Events -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>📅 四、事件催化 <span class="score-badge warning">${(scores.events||{}).score||0}/${(scores.events||{}).max||10}</span></h3>
          <button class="btn-deep-analyze" onclick="event.stopPropagation();handleDeepAnalyze('events')" title="AI 深度分析 + 多空辩论">🔬 深度分析</button>
        </div>
        <div class="section-body">
          ${buildEventsSection(r)}
          <div class="deep-analyze-panel" id="deep-events" style="display:none"></div>
        </div>
      </div>

      <!-- Section 5: Industry -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>🏭 五、同业对标 <span class="score-badge warning">${(scores.industry||{}).score||0}/${(scores.industry||{}).max||15}</span></h3>
          <button class="btn-deep-analyze" onclick="event.stopPropagation();handleDeepAnalyze('industry')" title="AI 深度分析 + 多空辩论">🔬 深度分析</button>
        </div>
        <div class="section-body">
          ${buildIndustrySection(r)}
          <div class="deep-analyze-panel" id="deep-industry" style="display:none"></div>
        </div>
      </div>

      <!-- Section 6: Value -->
      <div class="report-section">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>🎯 六、投资性价比 <span class="score-badge ${(scores.value||{}).score >= 10 ? 'good' : 'danger'}">${(scores.value||{}).score||0}/${(scores.value||{}).max||15}</span></h3>
          <button class="btn-deep-analyze" onclick="event.stopPropagation();handleDeepAnalyze('value')" title="AI 深度分析 + 多空辩论">🔬 深度分析</button>
        </div>
        <div class="section-body">
          ${buildValueSection(r)}
          <div class="deep-analyze-panel" id="deep-value" style="display:none"></div>
        </div>
      </div>

      <!-- Buy Timing (conditional) -->
      ${totalScore >= 40 ? `
      <div class="report-section" id="timingSection">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>⏰ 购入时机分析</h3>
          <button class="btn-deep-analyze" onclick="event.stopPropagation();handleTimingAnalysis()" title="分析当前买入时机">📊 开始分析</button>
        </div>
        <div class="section-body">
          <div id="timingContent">
            <p style="color:var(--text-secondary)">点击「开始分析」按钮，AI 将基于当前数据评估买入时机...</p>
          </div>
        </div>
      </div>
      ` : `
      <div class="report-section" style="opacity:0.6">
        <div class="section-header">
          <h3>⏰ 购入时机分析 <span class="score-badge danger">暂不推荐</span></h3>
        </div>
        <div class="section-body">
          <div class="risk-alert danger">综合评分 ${totalScore}/${maxScore}，低于 40 分阈值，暂不具备购入条件，不推荐进行时机分析。</div>
        </div>
      </div>
      `}

      <!-- Alternatives -->
      <div class="report-section" id="altSection">
        <div class="section-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
          <h3>🔄 替代标的推荐</h3>
        </div>
        <div class="section-body">
          <div class="alt-tabs">
            <button class="alt-tab active" data-alt-mode="industry">🏭 同板块</button>
            <button class="alt-tab" data-alt-mode="price_similar">💰 相似股价</button>
            <button class="alt-tab" data-alt-mode="recommended">⭐ 综合推荐</button>
          </div>
          <div id="altContent"><p style="color:var(--text-secondary)">正在加载替代标的...</p></div>
        </div>
      </div>

      <div style="text-align:center;padding:20px;color:var(--text-secondary);font-size:0.8rem">
        ⚠️ 免责声明：以上内容由AI基于公开数据自动生成，仅供参考，不构成任何投资建议。投资有风险，决策需谨慎。
        <br>数据截止：${r.report_time || ''}
      </div>
    `;

    // Render history bar after content is in DOM
    setTimeout(renderHistoryBar, 50);

    // Render K-line chart
    setTimeout(function() { renderKlineChart(r); }, 100);

    // Debug: show scoring breakdown for all dimensions
    if (isDebug()) {
      const dims = ['fundamental', 'technical', 'capital', 'events', 'industry', 'value'];
      const dimNames = { fundamental: '基本面体检', technical: '技术面分析', capital: '资金面分析', events: '事件催化', industry: '同业对标', value: '投资性价比' };
      const totalScore = r.total_score || 0;
      const maxScore = r.max_score || 100;

      let breakdownText = `📊 综合得分: ${totalScore}/${maxScore} | 评级: ${r.recommendation || '--'}\n\n`;
      breakdownText += `═`.repeat(50) + `\n\n`;

      dims.forEach(dimKey => {
        const dimScore = r.scores[dimKey] || {};
        const detail = dimScore.detail || {};
        const breakdown = detail.score_breakdown || [];
        const dimName = dimNames[dimKey] || dimKey;

        breakdownText += `📌 ${dimName} — ${dimScore.score || 0}/${dimScore.max || '--'} ${dimScore.summary ? '(' + dimScore.summary + ')' : ''}\n`;
        breakdownText += `─`.repeat(40) + `\n`;

        if (breakdown.length > 0) {
          breakdown.forEach(b => {
            const sign = b.change > 0 ? '+' : '';
            const arrow = b.change > 0 ? '⬆' : b.change < 0 ? '⬇' : ' ';
            breakdownText += `  ${arrow} ${b.item}: ${sign}${b.change} 分 → ${b.score_after}分\n`;
            breakdownText += `    └ ${b.detail}\n`;
          });
        } else {
          breakdownText += `  (无详细打分明细)\n`;
        }
        breakdownText += `\n`;
      });

      addDebugLog('📊 评分算法明细', breakdownText);
    }
  }

  function buildFundamentalSection(r) {
    const fd = (r.scores.fundamental || {}).detail || {};
    const roe = fd.latest_roe || fd.roe || 0;
    const dY = fd.dividend_yield || 0;
    const revGrowth = fd.revenue_growth;
    const trends = fd.revenue_trend || [];
    const debtRatio = fd.debt_ratio;
    const epsTrend = fd.eps_trend;
    const latestRev = fd.latest_revenue || 0;
    const latestNp = fd.latest_net_profit || 0;
    const latestEps = fd.latest_eps || 0;
    const note = fd.note || '';
    const growthNote = fd.growth_note || '';

    let trendRows = '';
    if (trends.length > 0) {
      trendRows = trends.map(t => {
        const yoy = t.yoy_growth != null ? `(${t.yoy_growth > 0 ? '+' : ''}${Number(t.yoy_growth).toFixed(1)}%)` : '';
        return `<tr>
          <td>${t.period || '--'}</td>
          <td>${Number(t.revenue).toFixed(2)} 亿</td>
          <td>${Number(t.net_profit).toFixed(2)} 亿</td>
          <td>${Number(t.roe).toFixed(2)}%</td>
          <td>${Number(t.eps).toFixed(2)}</td>
          <td>${yoy}</td>
        </tr>`;
      }).join('');
    }

    return `
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">ROE</div><div class="stat-value ${roe < 0 ? 'trend-down' : 'trend-up'}">${Number(roe).toFixed(2)}%</div></div>
        <div class="stat-item"><div class="stat-label">营收(最新年报)</div><div class="stat-value">${Number(latestRev).toFixed(1)}亿</div></div>
        <div class="stat-item"><div class="stat-label">净利润(最新年报)</div><div class="stat-value">${Number(latestNp).toFixed(1)}亿</div></div>
        <div class="stat-item"><div class="stat-label">EPS</div><div class="stat-value">${Number(latestEps).toFixed(2)}</div></div>
        <div class="stat-item"><div class="stat-label">资产负债率</div><div class="stat-value">${debtRatio != null ? Number(debtRatio).toFixed(2) + '%' : '--'}</div></div>
        <div class="stat-item"><div class="stat-label">股息率</div><div class="stat-value">${Number(dY).toFixed(2)}%</div></div>
      </div>
      ${revGrowth !== undefined ? `<div class="risk-alert ${revGrowth < -10 ? 'danger' : ''}">📊 营收增速：${revGrowth > 0 ? '+' : ''}${revGrowth}% ${growthNote ? '— ' + growthNote : ''}</div>` : ''}
      ${epsTrend ? `<div class="risk-alert ${epsTrend === '连续增长' ? 'good' : ''}">📈 EPS趋势：${epsTrend}</div>` : ''}
      ${roe < 0 ? '<div class="risk-alert danger">⚡ 核心风险：ROE为负，盈利能力堪忧</div>' : ''}
      ${note ? `<p style="color:var(--text-secondary);font-size:0.85rem">${note}</p>` : ''}
      ${trends.length > 0 ? `
        <h4 style="margin:12px 0 8px;font-size:0.9rem">年度财务趋势</h4>
        <table class="data-table">
          <thead><tr><th>报告期</th><th>营业收入</th><th>净利润</th><th>ROE</th><th>EPS</th><th>营收同比</th></tr></thead>
          <tbody>${trendRows}</tbody>
        </table>
      ` : '<p style="color:var(--text-secondary)">暂无详细财务数据</p>'}
    `;
  }

  function buildTechnicalSection(r) {
    const td = (r.scores.technical || {}).detail || {};
    const mas = td.mas || {};
    const macd = td.macd || {};
    const kdj = td.kdj || {};
    const boll = td.bollinger || {};
    const hyReturn = td.half_year_return;
    const turnover = td.turnover || 0;

    let maRows = '';
    for (const [name, val] of Object.entries(mas)) {
      const vs = val ? (r.price - val > 0 ? '上方' : '下方') : '--';
      maRows += `<tr><td>${name}</td><td>${val ? val.toFixed(2) : '--'}</td><td>${val ? ((r.price - val) / val * 100).toFixed(1) + '%' : '--'}</td><td class="${r.price > val ? 'trend-up' : 'trend-down'}">${val ? vs : '--'}</td></tr>`;
    }

    return `
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">换手率</div><div class="stat-value">${turnover.toFixed(2)}%</div></div>
        <div class="stat-item"><div class="stat-label">半年涨跌</div><div class="stat-value ${hyReturn > 0 ? 'trend-up' : 'trend-down'}">${hyReturn ? hyReturn.toFixed(2) + '%' : '--'}</div></div>
        <div class="stat-item"><div class="stat-label">MACD DIF</div><div class="stat-value">${macd.DIF || '--'}</div></div>
        <div class="stat-item"><div class="stat-label">KDJ-K</div><div class="stat-value">${kdj.K || '--'}</div></div>
      </div>
      <h4 style="margin:12px 0 8px;font-size:0.9rem">均线系统</h4>
      <table class="data-table">
        <thead><tr><th>均线</th><th>价格</th><th>偏离</th><th>位置</th></tr></thead>
        <tbody>${maRows}</tbody>
      </table>
      ${Object.keys(mas).length === 0 ? '<p style="color:var(--text-secondary)">暂无均线数据</p>' : ''}
    `;
  }

  function buildCapitalSection(r) {
    const cd = (r.scores.capital || {}).detail || {};
    if (!cd.data_ok) {
      return `<div class="risk-alert danger">⚠️ ${cd.error || '资金流向数据获取失败，无法进行资金面分析'}</div>
        <p style="color:var(--text-secondary);font-size:0.85rem">数据源: push2his.eastmoney.com（无需配置）</p>`;
    }

    const main5d = cd.main_5d_net || 0;
    const super5d = cd.super_large_5d_net || 0;
    const large5d = cd.large_5d_net || 0;
    const retail5d = cd.retail_5d_net || 0;
    const inflowDays = cd.main_inflow_days || 0;
    const trend = cd.trend || '--';
    const structure = cd.structure || '';
    const divergence = cd.divergence_msg || '';

    // Build mini flow chart from records
    const records = cd.records || [];
    let flowBars = '';
    if (records.length > 0) {
      const maxVal = Math.max(...records.map(r => Math.abs(r.main_net / 1e4)), 1);
      flowBars = records.map(r => {
        const val = (r.main_net / 1e4);
        const w = Math.min(100, (Math.abs(val) / maxVal) * 100);
        const cls = val >= 0 ? 'flow-bar-in' : 'flow-bar-out';
        return `<div class="flow-bar-row"><span class="flow-date">${(r.date||'').slice(5)}</span><div class="flow-bar-wrap"><div class="flow-bar ${cls}" style="width:${w}%"></div></div><span class="flow-val ${val>=0?'trend-up':'trend-down'}">${val>0?'+':''}${val.toFixed(0)}万</span></div>`;
      }).join('');
    }

    return `
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">近5日主力净流入</div><div class="stat-value ${main5d >= 0 ? 'trend-up' : 'trend-down'}">${main5d > 0 ? '+' : ''}${main5d.toFixed(0)} 万</div></div>
        <div class="stat-item"><div class="stat-label">超大单净流入</div><div class="stat-value ${super5d >= 0 ? 'trend-up' : 'trend-down'}">${super5d > 0 ? '+' : ''}${super5d.toFixed(0)} 万</div></div>
        <div class="stat-item"><div class="stat-label">大单净流入</div><div class="stat-value ${large5d >= 0 ? 'trend-up' : 'trend-down'}">${large5d > 0 ? '+' : ''}${large5d.toFixed(0)} 万</div></div>
        <div class="stat-item"><div class="stat-label">近5日散户净流入</div><div class="stat-value ${retail5d >= 0 ? 'trend-up' : 'trend-down'}">${retail5d > 0 ? '+' : ''}${retail5d.toFixed(0)} 万</div></div>
        <div class="stat-item"><div class="stat-label">主力流入天数(近5日)</div><div class="stat-value">${inflowDays}/5</div></div>
        <div class="stat-item"><div class="stat-label">近10日趋势</div><div class="stat-value" style="font-size:0.9rem">${trend}</div></div>
      </div>
      ${structure ? `<div class="risk-alert ${structure.includes('偏多') ? 'good' : 'danger'}">📊 ${structure}</div>` : ''}
      ${divergence ? `<div class="risk-alert danger">⚠️ ${divergence}</div>` : ''}
      ${flowBars ? `
        <h4 style="margin:12px 0 8px;font-size:0.9rem">近10日主力资金流向</h4>
        <div class="flow-chart">${flowBars}</div>
        <div style="display:flex;gap:12px;font-size:0.75rem;margin-top:4px;color:var(--text-secondary)">
          <span><span class="flow-legend-in"></span> 流入</span>
          <span><span class="flow-legend-out"></span> 流出</span>
        </div>
      ` : ''}
    `;
  }

  function buildEventsSection(r) {
    const ed = (r.scores.events || {}).detail || {};
    const events = ed.events || [];
    const positive = ed.positive_count || 0;
    const negative = ed.negative_count || 0;
    const total = ed.total_count || events.length;
    const posWeight = ed.positive_weight || 0;
    const negWeight = ed.negative_weight || 0;
    const keyEvents = ed.key_events || [];
    const methodNote = ed.method_note || '';

    let eventItems = '';
    if (events.length > 0) {
      eventItems = events.map(e => {
        const sScore = e.sentiment_score || 0;
        let icon = '⚪';
        if (e.sentiment === 'positive') icon = sScore >= 4 ? '🟢🟢' : '🟢';
        else if (e.sentiment === 'negative') icon = sScore >= 4 ? '🔴🔴' : '🔴';
        const evtUrl = e.url || '';
        const titleHtml = evtUrl
          ? `<a href="${evtUrl}" target="_blank" rel="noopener" class="event-title-link" title="点击查看公告原文">${e.title || ''}</a>`
          : `<span class="event-title">${e.title || ''}</span>`;
        return `<div class="event-item ${e.sentiment}">
          <span class="event-icon">${icon}</span>
          <span class="event-date">${e.date || ''}</span>
          ${titleHtml}
          ${sScore > 0 ? `<span class="event-weight">权重:${sScore}</span>` : ''}
          ${evtUrl ? `<a href="${evtUrl}" target="_blank" rel="noopener" class="event-ext-link" title="查看原文">🔗</a>` : ''}
        </div>`;
      }).join('');
    }

    return `
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">近期公告</div><div class="stat-value">${total} 条</div></div>
        <div class="stat-item"><div class="stat-label">偏多事件 (加权)</div><div class="stat-value trend-up">${positive}条 / +${posWeight}</div></div>
        <div class="stat-item"><div class="stat-label">偏空事件 (加权)</div><div class="stat-value trend-down">${negative}条 / -${negWeight}</div></div>
      </div>
      <div class="event-list">
        ${eventItems || '<p style="color:var(--text-secondary)">近30日无重大公告</p>'}
      </div>
      ${keyEvents.length > 0 ? `
        <h4 style="margin:12px 0 8px;font-size:0.9rem">⚠️ 重点关注事件</h4>
        <div class="event-list">
          ${keyEvents.map(e => {
            const keUrl = e.url || '';
            const keTitleHtml = keUrl
              ? `<a href="${keUrl}" target="_blank" rel="noopener" class="event-title-link" title="点击查看公告原文"><b>${e.title || ''}</b></a>`
              : `<span class="event-title"><b>${e.title || ''}</b></span>`;
            return `<div class="event-item ${e.sentiment}" style="background:#f8f9fa;border-radius:6px;padding:6px 10px;margin:4px 0">
            <span class="event-icon">${e.sentiment === 'positive' ? '🟢' : '🔴'}</span>
            <span class="event-date">${e.date || ''}</span>
            ${keTitleHtml}
            ${keUrl ? `<a href="${keUrl}" target="_blank" rel="noopener" class="event-ext-link" title="查看原文">🔗</a>` : ''}
          </div>`;
          }).join('')}
        </div>
      ` : ''}
      <div class="risk-alert" style="background:#f0f4ff;border-color:#74b9ff;color:#0984e3;margin-top:8px;font-size:0.78rem">
        💡 ${methodNote || '当前使用关键词匹配引擎分析事件。配置 LLM API 后可使用「🔬 深度分析」进行事件驱动的多维评估。'}
      </div>
    `;
  }

  function buildIndustrySection(r) {
    const id = (r.scores.industry || {}).detail || {};
    if (!id.data_ok) {
      return `<div class="risk-alert danger">⚠️ 行业分类数据获取失败，无法进行同业对标分析</div>
        <p style="color:var(--text-secondary);font-size:0.85rem">使用默认PE/PB基准进行简易评估</p>
        <div class="stat-grid">
          <div class="stat-item"><div class="stat-label">PE评估</div><div class="stat-value" style="font-size:0.9rem">${id.pe_assessment || '--'}</div></div>
        </div>`;
    }

    const indName = id.industry_name || '--';
    const boardName = id.board_name || '--';
    const benchmark = id.pe_benchmark || {};
    const peAssess = id.pe_assessment || '';
    const pbAssess = id.pb_assessment || '';
    const roeAssess = id.roe_assessment || '';
    const coRoe = id.company_roe || 0;

    return `
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">所属行业(CSRC)</div><div class="stat-value" style="font-size:0.9rem">${indName}</div></div>
        <div class="stat-item"><div class="stat-label">所属板块</div><div class="stat-value" style="font-size:0.9rem">${boardName}</div></div>
        <div class="stat-item"><div class="stat-label">行业PE合理区间</div><div class="stat-value">${benchmark.low || '--'} ~ ${benchmark.high || '--'}</div></div>
        <div class="stat-item"><div class="stat-label">行业PB合理区间</div><div class="stat-value">${benchmark.pb_low || '--'} ~ ${benchmark.pb_high || '--'}</div></div>
      </div>
      <div class="peer-comparison" style="margin-top:12px">
        <table class="data-table">
          <thead><tr><th>维度</th><th>本公司</th><th>行业标准</th><th>评估</th></tr></thead>
          <tbody>
            <tr><td>PE</td><td>${r.pe > 0 ? r.pe.toFixed(1) : '亏损'}</td><td>${benchmark.low}~${benchmark.high}</td><td>${peAssess}</td></tr>
            <tr><td>PB</td><td>${r.pb.toFixed(2)}</td><td>${benchmark.pb_low}~${benchmark.pb_high}</td><td>${pbAssess || '--'}</td></tr>
            <tr><td>ROE</td><td>${coRoe ? coRoe.toFixed(2) + '%' : '--'}</td><td>>${benchmark.roe_avg || 8}%</td><td>${roeAssess || '--'}</td></tr>
          </tbody>
        </table>
      </div>
    `;
  }

  function buildValueSection(r) {
    const vd = (r.scores.value || {}).detail || {};
    const dY = vd.dividend_yield || 0;
    const dCash = vd.dividend_cash_per_share || 0;
    const dNote = vd.dividend_note || '';
    const peAssess = vd.pe_assessment || '';
    const peg = vd.peg;
    const pegAssess = vd.peg_assessment || '';
    const roe = vd.roe || 0;
    const divHistory = vd.dividend_history || [];

    let divRows = '';
    if (divHistory.length > 0) {
      divRows = divHistory.map(d => {
        const yieldPct = r.price > 0 ? ((d.cash_per_share / r.price) * 100).toFixed(2) : '--';
        return `<tr><td>${d.ex_date || '--'}</td><td>${d.cash_per_share.toFixed(2)} 元/股</td><td>${yieldPct}%</td><td>${(d.dividend_ratio||0).toFixed(1)}%</td></tr>`;
      }).join('');
    }

    return `
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">股息率(最新)</div><div class="stat-value ${dY > 3 ? 'trend-up' : ''}">${dY > 0 ? dY.toFixed(2) + '%' : '--'}</div></div>
        <div class="stat-item"><div class="stat-label">每股分红</div><div class="stat-value">${dCash > 0 ? dCash.toFixed(2) + ' 元' : '--'}</div></div>
        <div class="stat-item"><div class="stat-label">ROE</div><div class="stat-value">${Number(roe).toFixed(2)}%</div></div>
        <div class="stat-item"><div class="stat-label">毛利率</div><div class="stat-value">${vd.gross_margin != null ? vd.gross_margin.toFixed(2) + '%' : '--'}</div></div>
        <div class="stat-item"><div class="stat-label">净利率</div><div class="stat-value">${vd.net_margin != null ? vd.net_margin.toFixed(2) + '%' : '--'}</div></div>
        <div class="stat-item"><div class="stat-label">PE估值</div><div class="stat-value" style="font-size:0.9rem">${r.pe > 0 ? r.pe.toFixed(1) : '亏损'}</div></div>
        ${peg !== undefined ? `<div class="stat-item"><div class="stat-label">PEG</div><div class="stat-value ${peg < 1 ? 'trend-up' : 'trend-down'}">${peg.toFixed(2)}</div></div>` : ''}
        ${pegAssess ? `<div class="stat-item"><div class="stat-label">PEG评估</div><div class="stat-value" style="font-size:0.8rem">${pegAssess}</div></div>` : ''}
      </div>
      ${peAssess ? `<div class="risk-alert ${r.pe < 0 ? 'danger' : ''}">📊 ${peAssess}</div>` : ''}
      ${dNote ? `<p style="color:var(--text-secondary);font-size:0.85rem">${dNote}</p>` : ''}
      ${divHistory.length > 0 ? `
        <h4 style="margin:12px 0 8px;font-size:0.9rem">分红历史</h4>
        <table class="data-table">
          <thead><tr><th>除权日</th><th>每股分红</th><th>股息率</th><th>分红率</th></tr></thead>
          <tbody>${divRows}</tbody>
        </table>
      ` : ''}
      ${r.pe <= 0 ? '<div class="risk-alert danger">当前处于亏损状态，无法用PE估值法评估性价比</div>' : ''}
    `;
  }


  // ========== K-line Chart ==========
  /**
   * Calculate moving average from price data.
   */
  function _calcMA(data, period) {
    var result = [];
    for (var i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.push('-');
        continue;
      }
      var sum = 0;
      for (var j = 0; j < period; j++) {
        sum += data[i - j][1]; // data[i] = [date, close]
      }
      result.push((sum / period).toFixed(2));
    }
    return result;
  }

  /**
   * Render interactive K-line chart with ECharts.
   * Supports candlestick, MA lines (5/10/20/30/60/120/250), and dataZoom slider.
   */
  function renderKlineChart(report) {
    var chartDom = document.getElementById('klineChart');
    if (!chartDom) return;
    if (typeof echarts === 'undefined') {
      chartDom.innerHTML = '<p style="color:#999;text-align:center;padding:40px">ECharts 加载中...</p>';
      return;
    }

    var prices = report.prices_data || [];
    if (prices.length < 10) {
      chartDom.innerHTML = '<p style="color:#999;text-align:center;padding:40px">K线数据不足（' + prices.length + '条）</p>';
      return;
    }

    // ---- Data diagnostic logging ----
    console.log('%c[K线] 数据诊断', 'font-weight:bold;color:#3b82f6',
      '共', prices.length, '条K线 |',
      '首条:', prices[0]['日期'], '开' + prices[0]['开盘'], '收' + prices[0]['收盘'],
      '| 末条:', prices[prices.length-1]['日期'], '开' + prices[prices.length-1]['开盘'], '收' + prices[prices.length-1]['收盘']);

    // Check for any anomalous bars
    var badBars = [];
    for (var bi = 0; bi < prices.length; bi++) {
      var bar = prices[bi];
      var o = parseFloat(bar['开盘']) || 0;
      var c = parseFloat(bar['收盘']) || 0;
      var h = parseFloat(bar['最高']) || 0;
      var l = parseFloat(bar['最低']) || 0;
      var minVal = Math.min(o, c, h, l);
      var maxVal = Math.max(o, c, h, l);
      var avgVal = (o + c + h + l) / 4;
      // Check relative consistency
      if (h < l || h < Math.max(o, c) - 0.001 || l > Math.min(o, c) + 0.001) {
        badBars.push({i: bi, date: bar['日期'], o: o, c: c, h: h, l: l, reason: 'OHLC逻辑错误'});
      }
      // Check absolute consistency: no value should be 50x away from the average
      else if (maxVal > 0 && (maxVal / Math.max(minVal, 0.01) > 50)) {
        badBars.push({i: bi, date: bar['日期'], o: o, c: c, h: h, l: l, reason: '数值异常(50x偏离)'});
      }
      // Check open specifically: should not be 10x away from close
      else if (c > 0 && Math.abs(o - c) / c > 10) {
        badBars.push({i: bi, date: bar['日期'], o: o, c: c, h: h, l: l, reason: 'open/close偏离10x+'});
      }
    }
    if (badBars.length > 0) {
      console.warn('%c[K线] ⚠️ ' + badBars.length + ' 条异常OHLC数据:', 'color:#ff4444');
      badBars.slice(0, 10).forEach(function(b) {
        console.warn('  [' + b.i + '] ' + b.date + ' 开=' + b.o + ' 收=' + b.c + ' 高=' + b.h + ' 低=' + b.l + ' (' + b.reason + ')');
      });
    } else {
      console.log('%c[K线] ✅ 全部' + prices.length + '条OHLC数据合法', 'color:#44bb44');
    }

    // Ensure chart instance is disposed before creating new one
    var existingInstance = echarts.getInstanceByDom(chartDom);
    if (existingInstance) existingInstance.dispose();

    var chart = echarts.init(chartDom);

    // Prepare data arrays
    var dates = [];
    var ohlc = [];
    var volumes = [];
    var closePrices = [];

    for (var i = 0; i < prices.length; i++) {
      var d = prices[i];
      dates.push(d['日期'] || '');
      ohlc.push([
        parseFloat(d['开盘']) || 0,
        parseFloat(d['收盘']) || 0,
        parseFloat(d['最低']) || 0,
        parseFloat(d['最高']) || 0
      ]);
      volumes.push([i, parseFloat(d['成交量']) || 0, (parseFloat(d['收盘']) >= parseFloat(d['开盘'])) ? 1 : -1]);
      closePrices.push([dates[i], parseFloat(d['收盘']) || 0]);
    }

    // Calculate MAs
    var maPeriods = [5, 10, 20, 30, 60, 120, 250];
    var maColors = ['#ff6b6b', '#ffa502', '#7bed9f', '#70a1ff', '#a29bfe', '#fd79a8', '#00b894'];
    var maSeries = [];
    for (var m = 0; m < maPeriods.length; m++) {
      var p = maPeriods[m];
      var maData = _calcMA(closePrices, p);
      // Convert to [date, value] for category axis
      var mappedData = [];
      for (var i2 = 0; i2 < dates.length; i2++) {
        mappedData.push(maData[i2] === '-' ? '-' : parseFloat(maData[i2]));
      }
      maSeries.push({
        name: 'MA' + p,
        type: 'line',
        data: mappedData,
        smooth: true,
        lineStyle: { width: 1, color: maColors[m], opacity: 0.7 },
        itemStyle: { color: maColors[m] },
        symbol: 'none',
        connectNulls: false,
      });
    }

    // Build option
    // Default to showing ~200 bars, user can zoom out for full history
    var defaultEnd = 100;
    var defaultStart = Math.max(0, 100 - (200 / prices.length) * 100);

    var option = {
      animation: false,
      backgroundColor: '#fff',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: function(params) {
          if (!params || params.length === 0) return '';
          var idx = params[0].dataIndex;
          if (idx < 0 || idx >= prices.length) return '';

          // Read directly from raw data — never trust ECharts internal state
          var d = prices[idx];
          var o = parseFloat(d['开盘']) || 0;
          var c = parseFloat(d['收盘']) || 0;
          var h = parseFloat(d['最高']) || 0;
          var l = parseFloat(d['最低']) || 0;
          var date = d['日期'] || '';

          var html = '<div style="font-size:13px"><b>' + date + '</b><br/>';

          // Detect anomalies in this bar
          var avg = (o + c + h + l) / 4;
          var isAnomaly = false;
          var anomalyReason = '';
          if (h < l) {
            isAnomaly = true; anomalyReason = '高<低';
          } else if (avg > 0 && (Math.max(o,c,h,l) / Math.max(Math.min(o,c,h,l), 0.01) > 50)) {
            isAnomaly = true; anomalyReason = '数值偏离';
          } else if (c > 0 && Math.abs(o - c) / c > 10) {
            isAnomaly = true; anomalyReason = '开收偏离';
          }

          if (isAnomaly) {
            html += '<span style="color:#ff4444;font-weight:bold">⚠️ 数据异常(' + anomalyReason + ')</span><br/>';
          }

          html += '<span style="color:#ef4444">● K线</span> ' +
            '开:' + o.toFixed(2) + ' 收:' + c.toFixed(2) + ' 低:' + l.toFixed(2) + ' 高:' + h.toFixed(2) + '<br/>';

          // Add MA lines
          for (var k = 0; k < params.length; k++) {
            var p = params[k];
            if (p.seriesName === 'Volume' || p.seriesName === 'K线') continue;
            if (typeof p.value === 'number' && p.value !== '-') {
              html += '<span style="color:' + (p.color || '#333') + '">● ' + p.seriesName + '</span> ' + p.value.toFixed(2) + '<br/>';
            }
          }

          html += '</div>';
          return html;
        }
      },
      axisPointer: {
        link: [{ xAxisIndex: [0, 1] }]
      },
      grid: [
        { left: '8%', right: '3%', top: 10, height: '60%' },
        { left: '8%', right: '3%', top: '75%', height: '15%' }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          scale: true,
          boundaryGap: true,
          axisLine: { onZero: false },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax',
          axisLabel: { show: false },
          axisPointer: { label: { show: false } }
        },
        {
          type: 'category',
          gridIndex: 1,
          data: dates,
          scale: true,
          boundaryGap: true,
          axisLine: { onZero: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          min: 'dataMin',
          max: 'dataMax'
        }
      ],
      yAxis: [
        {
          scale: true,
          splitArea: { show: true },
          splitLine: { lineStyle: { color: '#f0f0f0' } },
          axisLabel: { fontSize: 10 }
        },
        {
          scale: true,
          gridIndex: 1,
          splitNumber: 2,
          axisLabel: { show: false },
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { show: false }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: defaultStart,
          end: defaultEnd
        },
        {
          show: true,
          xAxisIndex: [0, 1],
          type: 'slider',
          bottom: 5,
          start: defaultStart,
          end: defaultEnd,
          height: 25,
          borderColor: '#e0e0e0',
          fillerColor: 'rgba(59,130,246,0.1)',
          handleStyle: { color: '#3b82f6' },
          textStyle: { fontSize: 10 }
        }
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: ohlc,
          itemStyle: {
            color: '#ef4444',        // up = red (Chinese convention)
            color0: '#22c55e',       // down = green
            borderColor: '#ef4444',
            borderColor0: '#22c55e'
          }
        },
        {
          name: 'Volume',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes.map(function(v) {
            return { value: v[1], itemStyle: { color: v[2] === 1 ? '#ef4444' : '#22c55e', opacity: 0.4 } };
          })
        }
      ].concat(maSeries)
    };

    // Verify ECharts OHLC format: [open, close, lowest, highest]
    console.log('%c[K线] ECharts OHLC样本 (最新5条):', 'color:#3b82f6');
    for (var si = Math.max(0, ohlc.length - 5); si < ohlc.length; si++) {
      var s = ohlc[si];
      console.log('  [' + dates[si] + '] open=' + s[0] + ' close=' + s[1] + ' low=' + s[2] + ' high=' + s[3] +
        ' | volume=' + (volumes[si] ? volumes[si][1] : '?'));
    }

    chart.setOption(option);

    // Handle window resize
    window.addEventListener('resize', function() {
      if (chart && !chart.isDisposed()) chart.resize();
    });
  }

  // ========== Alternatives ==========
  // Store alternatives data globally for deep analysis
  let _altDataCache = [];
  let _altCache = { industry: [], price_similar: [], recommended: [], _loaded: false };
  let _altActiveMode = 'industry';

  /**
   * Log API fallback info to browser console with color coding.
   * Called after every /api/analyze and /api/alternatives response.
   */
  function _logFallbackInfo(label, data, resp) {
    const fbEvents = (data && data._meta && data._meta.fb) || [];
    const fbHeader = resp && resp.headers.get('X-API-Fallback');

    if (!fbHeader && fbEvents.length === 0) return;

    const hasFailures = fbHeader && fbHeader.includes('fail=');
    const bgColor = hasFailures ? '#ff6b35' : '#4ecdc4';
    const textColor = hasFailures ? '#fff' : '#000';

    console.groupCollapsed(
      `%c🔧 ${label} API Fallback %c${fbHeader || ''}`,
      `background:${bgColor};color:${textColor};padding:2px 6px;border-radius:3px;font-weight:bold`,
      'color:#888;font-size:0.85em'
    );

    // Show failed APIs
    const failedSources = [];
    fbEvents.forEach(function(e) {
      if (!e.ok) {
        failedSources.push(e.source + ':' + e.func);
      }
    });
    if (failedSources.length > 0) {
      console.log('%c❌ 失败的 API 源:', 'color:#ff4444;font-weight:bold', failedSources.join(', '));
    }

    // Show successful fallbacks
    const fallbackSources = fbEvents.filter(function(e) { return e.ok && e.source !== 'primary'; });
    if (fallbackSources.length > 0) {
      console.log('%c✅ 使用的回退方案:', 'color:#44bb44;font-weight:bold');
      fallbackSources.forEach(function(e) {
        console.log('  →', e.source, '|', e.func, '|', e.detail || '');
      });
    }

    if (fbEvents.length === 0 && fbHeader) {
      console.log('%cℹ️ 可能触发回退 (来自响应头)', 'color:#ffaa00');
    }

    console.groupEnd();
  }

  // ========== Alternatives: 3-Mode System (Progressive) ==========

  // Score source tracking for display
  let _altFullScores = {};  // code_full -> {total_score, recommendation, scores_breakdown, source}
  let _altScoreLoadState = 'idle';  // 'idle' | 'loading' | 'done'

  /**
   * Progressive loading: base preview → full scoring
   */
  async function loadAllAlternatives(code) {
    var container = $('#altContent');
    if (!container) return;

    // Reset
    _altActiveMode = 'industry';
    _altFullScores = {};
    _altScoreLoadState = 'idle';
    updateAltTabUI();

    // Clear old content immediately — show loading
    container.innerHTML = '<p style="color:var(--text-secondary);padding:16px 0">⏳ 正在加载替代标的...</p>';

    // Also clear cache info bar
    var cacheInfo = $('#altCacheInfo');
    if (cacheInfo) cacheInfo.innerHTML = '';

    try {
      // Step 1: Get base preview (fast, < 3s)
      var baseResp = await fetch('/api/alternatives/base', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      var baseData = await baseResp.json();

      _altCache.industry = baseData.industry || [];
      _altCache.price_similar = baseData.price_similar || [];
      _altCache.recommended = baseData.recommended || [];
      _altCache._loaded = true;
      _altCache._cache_meta = baseData._cache_meta;

      _logFallbackInfo('替代标的(基础)', baseData, baseResp);

      // ---- Mode completeness check ----
      // [EXPAND] If adding more modes in the future, add them to this list
      var ALL_MODES = ['industry', 'price_similar', 'recommended'];
      var emptyModes = [];
      ALL_MODES.forEach(function(m) {
        if ((_altCache[m] || []).length === 0) {
          emptyModes.push(m);
        }
      });
      if (emptyModes.length > 0) {
        console.warn(
          '%c[Alt] ⚠️ ' + emptyModes.length + ' 个模式无数据: ' + emptyModes.join(', '),
          'color:#ffa500;font-weight:bold'
        );
        // If price_similar is empty, try to get it from the legacy /all endpoint
        if (emptyModes.indexOf('price_similar') >= 0 || emptyModes.indexOf('recommended') >= 0) {
          console.log('%c[Alt] 尝试回退加载缺失的模式...', 'color:#3b82f6');
          try {
            var fbResp = await fetch('/api/alternatives/all', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ code: code }),
            });
            var fbData = await fbResp.json();
            emptyModes.forEach(function(m) {
              if (fbData[m] && fbData[m].length > 0) {
                _altCache[m] = fbData[m];
                console.log('%c[Alt] ✅ 回退成功加载 ' + m + ': ' + fbData[m].length + ' 只', 'color:#44bb44');
              }
            });
          } catch (fbErr) {
            console.warn('[Alt] 回退加载失败:', fbErr.message);
          }
        }
      }

      // Render cards with lightweight scores + "计算中" badge
      renderAltContent();

      // Show cache info
      showCacheInfo();

      // Step 2: Collect all codes for full scoring
      // [EXPAND] If adding more modes, update ALL_MODES above
      var allCodes = [];
      var seen = {};
      ALL_MODES.forEach(function(mode) {
        (_altCache[mode] || []).forEach(function(a) {
          var fc = a.code_full || (a.code && a.code.startsWith('6') ? a.code + '.SH' : a.code + '.SZ');
          if (fc && !seen[fc]) {
            seen[fc] = true;
            allCodes.push(fc);
          }
        });
      });

      if (allCodes.length === 0) return;

      // Step 3: Score in batches of 4
      _altScoreLoadState = 'loading';
      var BATCH = 4;
      var totalBatches = Math.ceil(allCodes.length / BATCH);
      var completedBatches = 0;
      var totalErrors = [];

      for (var b = 0; b < allCodes.length; b += BATCH) {
        var batch = allCodes.slice(b, b + BATCH);
        var batchNum = Math.floor(b / BATCH) + 1;

        try {
          var scoreResp = await fetchWithTimeout('/api/alternatives/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ codes: batch }),
          }, 12000);

          var scoreData = await scoreResp.json();
          var sc = scoreData.scores || [];

          // Store full scores
          sc.forEach(function(s) {
            if (s.source === 'full' && s.total_score > 0) {
              _altFullScores[s.code] = s;
            }
          });

          if (scoreData.errors && scoreData.errors.length > 0) {
            totalErrors = totalErrors.concat(scoreData.errors);
            console.warn(
              '%c[Alt Score] 批次 ' + batchNum + '/' + totalBatches + ' 部分失败:',
              'color:#ffa500',
              scoreData.errors
            );
          }

          completedBatches++;
          console.log(
            '%c[Alt Score] 批次 ' + batchNum + '/' + totalBatches +
            ' 完成 (' + sc.filter(function(s) { return s.source === 'full'; }).length + '/' + batch.length + ' 成功)' +
            (scoreData._cache_meta && scoreData._cache_meta.from_cache ? ' [缓存]' : ''),
            'color:#44bb44'
          );

        } catch (err) {
          totalErrors.push({ batch: batchNum, error: err.message || 'timeout' });
          console.error(
            '%c[Alt Score] 批次 ' + batchNum + '/' + totalBatches + ' 超时/失败:',
            'color:#ff4444',
            err.message || err
          );
        }

        // Update cards after each batch so user sees progress
        renderAltContent();
        showCacheInfo();
      }

      _altScoreLoadState = 'done';

      // Final render
      renderAltContent();
      showCacheInfo();

      if (totalErrors.length > 0) {
        console.warn(
          '%c[Alt Score] 完成，' + totalErrors.length + ' 个错误',
          'color:#ffa500',
          totalErrors
        );
      }

    } catch (err) {
      console.error('[Alt] Load all error:', err);
      if (container) container.innerHTML = '<p style="color:var(--text-secondary)">替代标的数据加载失败，请刷新重试</p>';
    }
  }

  /**
   * Helper: fetch with timeout
   */
  function fetchWithTimeout(url, options, timeoutMs) {
    return new Promise(function(resolve, reject) {
      var timer = setTimeout(function() {
        reject(new Error('fetch timeout ' + timeoutMs + 'ms'));
      }, timeoutMs);
      fetch(url, options).then(function(resp) {
        clearTimeout(timer);
        resolve(resp);
      }).catch(function(err) {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  // Backward compat
  async function loadAlternatives(code) {
    return loadAllAlternatives(code);
  }

  function switchAltTab(mode) {
    if (!_altCache._loaded) return;
    _altActiveMode = mode;
    updateAltTabUI();
    renderAltContent();
  }

  function updateAltTabUI() {
    var tabs = document.querySelectorAll('.alt-tab');
    for (var i = 0; i < tabs.length; i++) {
      var t = tabs[i];
      var mode = t.getAttribute('data-alt-mode');
      if (mode === _altActiveMode) {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    }
  }

  /**
   * Show cache info below the alt tabs
   */
  function showCacheInfo() {
    var el = $('#altCacheInfo');
    if (!el) {
      // Create cache info bar
      var container = $('#altContent');
      if (!container) return;
      var existing = $('#altCacheInfo');
      if (existing) existing.remove();
      el = document.createElement('div');
      el.id = 'altCacheInfo';
      el.style.cssText = 'font-size:0.7rem;color:#94a3b8;margin-top:8px;display:flex;align-items:center;gap:8px;justify-content:flex-end';
      container.parentNode.appendChild(el);
    }

    var meta = _altCache._cache_meta || {};
    var timeStr = meta.time || '';
    var fromCache = meta.from_cache;
    var scoreState = _altScoreLoadState;

    var html = '';
    if (timeStr) {
      html += '📦 数据' + (fromCache ? '缓存于 ' : '获取于 ') + timeStr + ' (10分钟有效)';
    }
    if (scoreState === 'loading') {
      html += ' · ⏳ 深度评分计算中...';
    } else if (scoreState === 'done') {
      var fullCount = Object.keys(_altFullScores).length;
      html += ' · ✅ ' + fullCount + ' 只已深度计算';
    }
    html += ' <button class="btn-alt-cache-clear" style="font-size:0.65rem;padding:2px 8px;background:#fee;border:1px solid #fcc;border-radius:10px;cursor:pointer;color:#c33">🔄 刷新缓存</button>';

    el.innerHTML = html;
  }

  /**
   * Manual cache clear — reloads alternatives
   */
  async function clearAltCache() {
    var codeInput = document.getElementById('searchCode');
    var code = codeInput ? codeInput.value : '';
    if (!code) return;

    // Clear display immediately
    var container = $('#altContent');
    if (container) container.innerHTML = '<p style="color:var(--text-secondary);padding:16px 0">⏳ 正在刷新替代标的...</p>';
    var cacheInfo = $('#altCacheInfo');
    if (cacheInfo) cacheInfo.innerHTML = '';

    try {
      await fetch('/api/alternatives/cache/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code }),
      });
      console.log('%c[Cache] 已清除替代标的缓存，重新加载中...', 'color:#44bb44');
    } catch (e) {
      console.warn('[Cache] Clear failed:', e);
    }

    // Reload
    loadAllAlternatives(code);
  }

  function renderAltContent() {
    var container = $('#altContent');
    if (!container) return;

    var alts = _altCache[_altActiveMode] || [];
    var modeLabels = {
      industry: '🏭 同板块低PE标的（来自申万行业分类）',
      price_similar: '💰 相似价格区间标的（±30%）',
      recommended: '⭐ 综合推荐标的（跨行业优质筛选）',
    };
    var emptyLabels = {
      industry: '未找到同行业替代标的',
      price_similar: '未找到同价位优质标的',
      recommended: '暂无综合推荐标的',
    };

    if (alts.length === 0) {
      container.innerHTML = '<p style="color:var(--text-secondary)">' + (emptyLabels[_altActiveMode] || '暂无数据') + '</p>';
      return;
    }

    // Build cards
    var cardsHtml = '';
    for (var i = 0; i < alts.length; i++) {
      var a = alts[i];
      var fullCode = a.code_full || (a.code && a.code.startsWith('6') ? a.code + '.SH' : a.code + '.SZ');

      // Check for full score
      var fs = _altFullScores[fullCode];
      var realScore, scoreCls, recd, scoreBadge;
      if (fs && fs.source === 'full') {
        realScore = fs.total_score || 0;
        recd = fs.recommendation || '';
        scoreCls = realScore >= 60 ? 'good' : realScore >= 40 ? 'mid' : 'low';
        scoreBadge = '<span style="font-size:0.6rem;color:#22c55e;margin-left:4px">✅ 深度</span>';
      } else if (_altScoreLoadState === 'loading') {
        realScore = a.total_score || 0;
        recd = a.recommendation || '';
        scoreCls = realScore >= 60 ? 'good' : realScore >= 40 ? 'mid' : 'low';
        scoreBadge = '<span style="font-size:0.6rem;color:#f59e0b;margin-left:4px">⏳ 计算中</span>';
      } else if (_altScoreLoadState === 'done') {
        // Done loading but no full score → lightweight only
        realScore = a.total_score || 0;
        recd = a.recommendation || '';
        scoreCls = realScore >= 60 ? 'good' : realScore >= 40 ? 'mid' : 'low';
        scoreBadge = '<span style="font-size:0.6rem;color:#94a3b8;margin-left:4px">⚠️ 轻量估算</span>';
      } else {
        realScore = a.total_score || 0;
        recd = a.recommendation || '';
        scoreCls = realScore >= 60 ? 'good' : realScore >= 40 ? 'mid' : 'low';
        scoreBadge = '';
      }

      var mcapDisplay = '';
      if (a.market_cap) {
        mcapDisplay = '<div class="alt-mcap">市值 ' + (a.market_cap / 1e8).toFixed(0) + '亿</div>';
      }

      cardsHtml += '<div class="alt-card-wrap">' +
        '<div class="alt-card" data-alt-fullcode="' + fullCode + '" style="cursor:pointer">' +
          '<div class="alt-name"><span class="alt-rank">#' + (i+1) + '</span> ' + (a.name || '') + '</div>' +
          '<div class="alt-code">' + fullCode + '</div>' +
          '<div class="alt-stats">' +
            '<div class="alt-stat"><div class="alt-stat-label">价格</div><span>¥' + ((a.price||0)).toFixed(2) + '</span></div>' +
            '<div class="alt-stat"><div class="alt-stat-label">PE</div><span>' + ((a.pe||0) > 0 ? (a.pe||0).toFixed(1) : '亏损') + '</span></div>' +
            '<div class="alt-stat"><div class="alt-stat-label">PB</div><span>' + ((a.pb||0)).toFixed(2) + '</span></div>' +
            '<div class="alt-stat"><div class="alt-stat-label">涨跌</div><span class="' + ((a.change||0) >= 0 ? 'trend-up' : 'trend-down') + '">' + ((a.change||0) > 0 ? '+' : '') + (a.change||0).toFixed(2) + '%</span></div>' +
          '</div>' +
          mcapDisplay +
          (realScore > 0
            ? '<div class="alt-score-bar"><div class="alt-score-label">综合评分 ' + scoreBadge + (recd ? ' · ' + recd : '') + '</div><div class="alt-score-value ' + scoreCls + '">' + realScore + '分</div></div>'
            : '<div class="alt-score-bar"><div class="alt-score-label" style="color:#999">评分计算中...</div></div>') +
        '</div>' +
        '<button class="btn-alt-deep" data-alt-deep="' + i + '" title="展开对比分析">' +
          '📊 深度对比' +
          '<span class="alt-deep-arrow" id="altDeepArrow' + i + '">▾</span>' +
        '</button>' +
        '<div class="alt-deep-panel" id="altDeepPanel' + i + '" style="display:none"></div>' +
        '</div>';
    }

    container.innerHTML = '<div class="alt-grid">' + cardsHtml + '</div>' +
      '<p style="font-size:0.75rem;color:var(--text-secondary);margin-top:8px">' +
        (modeLabels[_altActiveMode] || '') + ' | 点击卡片可切换分析该股票' +
      '</p>';

    _altDataCache = alts;
  }

  // ---- Event delegation: alt-tab clicks ----
  document.addEventListener('click', function(e) {
    var tab = e.target.closest('.alt-tab');
    if (tab) {
      var mode = tab.getAttribute('data-alt-mode');
      if (mode) switchAltTab(mode);
      return;
    }
    // Cache clear button
    var cacheBtn = e.target.closest('.btn-alt-cache-clear');
    if (cacheBtn) {
      e.stopPropagation();
      clearAltCache();
      return;
    }
    // Alt card click: switch to that stock
    var card = e.target.closest('.alt-card');
    if (card) {
      var fc = card.getAttribute('data-alt-fullcode');
      if (fc) {
        var searchInput = document.getElementById('searchCode');
        var searchBtn = document.getElementById('searchBtn');
        if (searchInput) searchInput.value = fc;
        if (searchBtn) searchBtn.click();
      }
    }
    // Deep analysis button click
    var deepBtn = e.target.closest('.btn-alt-deep');
    if (deepBtn) {
      e.stopPropagation();
      var idx = parseInt(deepBtn.getAttribute('data-alt-deep'));
      if (!isNaN(idx)) toggleAltDeepAnalysis(idx);
    }
  });

  // ========== Alternative Deep Analysis ==========
  function toggleAltDeepAnalysis(index) {
    const panel = document.getElementById('altDeepPanel' + index);
    const arrow = document.getElementById('altDeepArrow' + index);
    if (!panel) return;

    const isOpen = panel.style.display !== 'none';
    if (isOpen) {
      panel.style.display = 'none';
      if (arrow) arrow.textContent = '▾';
      // Stop any ongoing streaming
      if (panel._streamAbort) { panel._streamAbort.abort(); panel._streamAbort = null; }
      return;
    }

    // Build deep analysis
    const alt = _altDataCache[index];
    if (!alt || !currentReport) return;

    panel.style.display = 'block';
    if (arrow) arrow.textContent = '▴';

    // Build rule-based HTML with placeholder index
    let html = buildAltDeepAnalysis(alt, currentReport);
    // Replace INDEX_PLACEHOLDER with actual index so IDs are unique
    html = html.replace(/INDEX_PLACEHOLDER/g, index);

    panel.innerHTML = html;

    // Start AI streaming comparison
    fetchAltDeepCompareStream(index, alt, currentReport);
  }

  function buildAltDeepAnalysis(alt, cur) {
    const altName = alt.name || '替代标的';
    const curName = cur.name || '当前股票';

    // Get dimension scores
    const altScores = alt.scores_breakdown || {};
    const curScores = cur.scores || {};

    const dimKeys = ['fundamental', 'technical', 'capital', 'events', 'industry', 'value'];
    const dimLabels = {
      fundamental: '基本面', technical: '技术面', capital: '资金面',
      events: '事件催化', industry: '同业对标', value: '投资性价比'
    };
    const dimMax = { fundamental: 25, technical: 20, capital: 15, events: 10, industry: 15, value: 15 };

    // Build score comparison rows
    let scoreCompRows = '';
    let altTotalWins = 0, curTotalWins = 0;
    for (const dim of dimKeys) {
      const altS = (altScores[dim]?.score || 0);
      const curS = (curScores[dim]?.score || 0);
      const maxS = dimMax[dim];
      const altPct = maxS > 0 ? (altS / maxS * 100) : 0;
      const curPct = maxS > 0 ? (curS / maxS * 100) : 0;
      const winner = altPct > curPct + 2 ? 'alt' : curPct > altPct + 2 ? 'cur' : 'tie';
      if (winner === 'alt') altTotalWins++;
      else if (winner === 'cur') curTotalWins++;

      scoreCompRows += `
        <tr class="alt-comp-row">
          <td class="alt-comp-dim">${dimLabels[dim]}</td>
          <td class="alt-comp-val ${winner === 'alt' ? 'alt-winner' : ''}">${altS}/${maxS}</td>
          <td class="alt-comp-val ${winner === 'cur' ? 'alt-winner' : ''}">${curS}/${maxS}</td>
          <td class="alt-comp-diff">
            ${winner === 'alt' ? '🏆 ' + altName : winner === 'cur' ? '🏆 ' + curName : '持平'}
          </td>
        </tr>`;
    }

    // Build financial comparison rows
    const altPE = alt.pe || 0, curPE = cur.pe || 0;
    const altPB = alt.pb || 0, curPB = cur.pb || 0;
    const altROE = alt.roe || 0, curROE = (cur.scores?.fundamental?.detail?.latest_roe) || 0;
    const altDiv = alt.dividend_yield || 0, curDiv = (cur.scores?.fundamental?.detail?.dividend_yield) || 0;
    const altMcap = alt.market_cap || 0, curMcap = cur.total_mv || 0;
    const altPEG = alt.peg || 0, curPEG = (cur.scores?.value?.detail?.peg) || 0;
    const altPrice = alt.price || 0, curPrice = cur.price || 0;

    let finCompRows = '';
    const finCompare = [
      { label: 'PE (TTM)', altVal: altPE > 0 ? altPE.toFixed(1) : '亏损', curVal: curPE > 0 ? curPE.toFixed(1) : '亏损', unit: '', lowerBetter: true },
      { label: 'PB', altVal: altPB.toFixed(2), curVal: curPB.toFixed(2), unit: '', lowerBetter: true },
      { label: 'ROE', altVal: altROE > 0 ? altROE.toFixed(1) + '%' : '--', curVal: curROE > 0 ? curROE.toFixed(1) + '%' : '--', unit: '', lowerBetter: false },
      { label: '股息率', altVal: altDiv > 0 ? altDiv.toFixed(2) + '%' : '--', curVal: curDiv > 0 ? curDiv.toFixed(2) + '%' : '--', unit: '', lowerBetter: false },
      { label: '市值', altVal: altMcap > 0 ? (altMcap/1e8).toFixed(0) + '亿' : '--', curVal: curMcap > 0 ? curMcap.toFixed(0) + '亿' : '--', unit: '', lowerBetter: true },
      { label: 'PEG', altVal: altPEG > 0 ? altPEG.toFixed(2) : '--', curVal: curPEG > 0 ? curPEG.toFixed(2) : '--', unit: '', lowerBetter: true },
      { label: '股价', altVal: '¥' + altPrice.toFixed(2), curVal: '¥' + curPrice.toFixed(2), unit: '', lowerBetter: null },
    ];

    for (const f of finCompare) {
      const altIs = typeof f.altVal === 'number' ? f.altVal : f.altVal;
      const curIs = typeof f.curVal === 'number' ? f.curVal : f.curVal;
      let winner = '';
      if (f.lowerBetter === true) {
        winner = (typeof f.altVal === 'number' && typeof f.curVal === 'number' && f.altVal < f.curVal) ? 'alt' :
                 (typeof f.altVal === 'number' && typeof f.curVal === 'number' && f.curVal < f.altVal) ? 'cur' : '';
      } else if (f.lowerBetter === false) {
        winner = (typeof f.altVal === 'number' && typeof f.curVal === 'number' && f.altVal > f.curVal) ? 'alt' :
                 (typeof f.altVal === 'number' && typeof f.curVal === 'number' && f.curVal > f.altVal) ? 'cur' : '';
      }

      finCompRows += `
        <tr class="alt-comp-row">
          <td class="alt-comp-dim">${f.label}</td>
          <td class="alt-comp-val ${winner === 'alt' ? 'alt-winner' : ''}">${altIs}</td>
          <td class="alt-comp-val ${winner === 'cur' ? 'alt-winner' : ''}">${curIs}</td>
          <td class="alt-comp-diff">${winner === 'alt' ? '✅ 更优' : winner === 'cur' ? '— 更优' : ''}</td>
        </tr>`;
    }

    // Debate-style pros/cons
    let pros = [], cons = [];
    if (altPE > 0 && curPE > 0 && altPE < curPE * 0.8) { pros.push('PE估值显著低于当前标的，存在估值优势'); }
    else if (altPE > 0 && curPE > 0 && altPE > curPE * 1.3) { cons.push('PE估值偏高，当前价格安全边际不足'); }

    if (altROE > 15) { pros.push('ROE > 15% 展现出优秀的盈利能力'); }
    else if (altROE < 5 && altROE >= 0) { cons.push('ROE偏低，资本利用效率有待提升'); }

    if (altDiv > 3) { pros.push('股息率 > 3%，现金回报对投资者友好'); }
    else if (altDiv > 0 && altDiv < 1) { cons.push('股息率偏低，现金回报能力有限'); }

    if ((alt.total_score || 0) >= 60) { pros.push('综合评分 ≥ 60 分，多维度表现稳健'); }
    else if ((alt.total_score || 0) < 40 && (alt.total_score || 0) > 0) { cons.push('综合评分偏低，多维度存在短板'); }

    if ((alt.total_score || 0) > (cur.total_score || 0) + 5) { pros.push('综合评分显著高于当前标的，整体质量更优'); }

    if (altPEG > 0 && altPEG < 0.8) { pros.push('PEG < 0.8，成长性被当前估值低估'); }
    else if (altPEG > 2) { cons.push('PEG偏高(>2)，当前估值已透支成长预期'); }

    if (altMcap / 1e8 > 500) { pros.push('市值较大，流动性好，适合稳健配置'); }

    if (cons.length === 0) cons.push('暂无明显风险点，建议进一步查看最新公告和研报');

    const prosHtml = pros.map(p => `<div class="alt-debate-item pros">✅ ${p}</div>`).join('');
    const consHtml = cons.map(c => `<div class="alt-debate-item cons">⚠️ ${c}</div>`).join('');

    // Overall verdict
    const altTotal = alt.total_score || 0, curTotal = cur.total_score || 0;
    let verdict = '';
    if (altTotal > curTotal + 10) {
      verdict = `<span class="alt-verdict strong-buy">🟢 综合评价优于当前标的</span> — ${altName} 在多个维度上表现更佳，可考虑作为替代/补充配置。`;
    } else if (altTotal >= curTotal - 5 && altTotal <= curTotal + 5) {
      verdict = `<span class="alt-verdict neutral">🟡 综合评价与当前标的相当</span> — 两者各有优劣，${altName} 可作为分散风险的备选标的。`;
    } else {
      verdict = `<span class="alt-verdict caution">🔴 综合评价弱于当前标的</span> — 当前标的整体表现更优，${altName} 仅适合深度价值投资者关注。`;
    }

    return `
      <div class="alt-deep-body">
        <!-- Score Comparison Table -->
        <div class="alt-deep-section">
          <h5 class="alt-deep-title">📈 六维度评分对比</h5>
          <table class="alt-comp-table">
            <thead>
              <tr><th>维度</th><th>${altName}</th><th>${curName}</th><th>优势方</th></tr>
            </thead>
            <tbody>${scoreCompRows}</tbody>
          </table>
          <div class="alt-ai-section" id="altAiScore_INDEX_PLACEHOLDER">
            <div class="alt-ai-loading"><span class="alt-ai-dot"></span> AI 正在深入分析评分差异...</div>
          </div>
        </div>

        <!-- Financial Comparison Table -->
        <div class="alt-deep-section">
          <h5 class="alt-deep-title">💰 关键财务指标对比</h5>
          <table class="alt-comp-table">
            <thead>
              <tr><th>指标</th><th>${altName}</th><th>${curName}</th><th>优劣</th></tr>
            </thead>
            <tbody>${finCompRows}</tbody>
          </table>
          <div class="alt-ai-section" id="altAiFinance_INDEX_PLACEHOLDER">
            <div class="alt-ai-loading"><span class="alt-ai-dot"></span> AI 正在分析财务差异...</div>
          </div>
        </div>

        <!-- Debate-style analysis -->
        <div class="alt-deep-section">
          <h5 class="alt-deep-title">⚖️ 辩论式分析 — ${altName}</h5>
          <div class="alt-debate-grid">
            <div class="alt-debate-col">
              <div class="alt-debate-header pros-header">📈 优势 / 看多理由</div>
              ${prosHtml}
            </div>
            <div class="alt-debate-col">
              <div class="alt-debate-header cons-header">📉 劣势 / 看空理由</div>
              ${consHtml}
            </div>
          </div>
          <div class="alt-ai-section" id="altAiDebate_INDEX_PLACEHOLDER">
            <div class="alt-ai-loading"><span class="alt-ai-dot"></span> AI 正在生成辩论式分析...</div>
          </div>
        </div>

        <!-- Verdict -->
        <div class="alt-deep-section">
          <h5 class="alt-deep-title">🎯 综合研判</h5>
          <div class="alt-verdict-box">${verdict}</div>
          <div class="alt-ai-section" id="altAiVerdict_INDEX_PLACEHOLDER">
            <div class="alt-ai-loading"><span class="alt-ai-dot"></span> AI 正在生成综合评价...</div>
          </div>
        </div>

        <div class="alt-deep-footer">
          ⚠️ 以上分析基于公开数据自动计算，仅供参考，不构成投资建议。建议结合最新财报、行业研报综合判断。
        </div>
      </div>
    `;
  }

  // ========== SSE Streaming for Alt Deep Compare ==========
  const sectionIdMap = {
    score_analysis: 'altAiScore_',
    financial_analysis: 'altAiFinance_',
    debate_analysis: 'altAiDebate_',
    verdict: 'altAiVerdict_',
  };

  async function fetchAltDeepCompareStream(index, alt, cur) {
    const panel = document.getElementById('altDeepPanel' + index);
    if (!panel) return;

    // Abort any previous stream for this panel
    if (panel._streamAbort) { panel._streamAbort.abort(); }
    const controller = new AbortController();
    panel._streamAbort = controller;

    const completedSections = new Set();

    try {
      const curSimple = {
        name: cur.name, code: cur.code, price: cur.price, pe: cur.pe, pb: cur.pb,
        total_score: cur.total_score, change_pct: cur.change_pct, total_mv: cur.total_mv,
        scores: cur.scores,
      };

      console.log('[AltDeep] Starting SSE stream for index', index);

      const resp = await fetch('/api/alt_deep_compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alt_stock: alt, current_stock: curSimple, ai_chat: getAiConfig() }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;
          if (!data) continue;

          try {
            const parsed = JSON.parse(data);
            const { section, content } = parsed;

            // Handle progress updates
            if (section === 'progress') {
              continue; // silently ignore, loading animation already shown
            }

            if (section === 'error') {
              console.error('[AltDeep] Error from server:', content);
              for (const [sKey, elId] of Object.entries(sectionIdMap)) {
                const el = document.getElementById(elId + index);
                if (el && !completedSections.has(sKey)) {
                  el.innerHTML = `<div class="alt-ai-content alt-ai-error">❌ ${escapeHtml(content)}</div>`;
                  completedSections.add(sKey);
                }
              }
              continue;
            }

            const elId = sectionIdMap[section];
            if (!elId) continue;

            const el = document.getElementById(elId + index);
            if (!el) {
              console.warn('[AltDeep] Element not found:', elId + index);
              continue;
            }

            // Replace loading state with actual content
            if (!completedSections.has(section)) {
              completedSections.add(section);
              console.log('[AltDeep] Rendering section:', section);
              el.innerHTML = `<div class="alt-ai-content">${renderSimpleMarkdown(content)}</div>`;
            } else {
              // Append additional content
              const contentEl = el.querySelector('.alt-ai-content');
              if (contentEl) {
                contentEl.innerHTML += renderSimpleMarkdown(content);
              }
            }

            el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          } catch (e) {
            console.warn('[AltDeep] JSON parse error:', e.message, 'data:', data.substring(0, 100));
          }
        }
      }

      // Mark remaining sections
      for (const [sKey, elId] of Object.entries(sectionIdMap)) {
        if (!completedSections.has(sKey)) {
          const el = document.getElementById(elId + index);
          if (el) {
            el.innerHTML = '<div class="alt-ai-content alt-ai-empty">（AI 未生成该部分内容）</div>';
          }
        }
      }
      console.log('[AltDeep] Stream complete for index', index);
    } catch (err) {
      if (err.name === 'AbortError') { console.log('[AltDeep] Stream aborted for index', index); return; }
      console.error('[AltDeep] Stream error:', err);
      for (const [sKey, elId] of Object.entries(sectionIdMap)) {
        if (!completedSections.has(sKey)) {
          const el = document.getElementById(elId + index);
          if (el) {
            el.innerHTML = `<div class="alt-ai-content alt-ai-error">⚠️ AI 分析请求失败: ${escapeHtml(err.message)}</div>`;
          }
        }
      }
    }
  }

  // ========== Deep Analysis ==========
  async function handleDeepAnalyze(dimKey, forceReanalyze = false) {
    if (!currentReport) return;
    if (deepAnalyzing[dimKey]) return; // Already loading

    const panel = document.getElementById(`deep-${dimKey}`);
    if (!panel) return;

    // If already loaded and not forcing re-analyze, just toggle visibility
    if (panel.dataset.loaded === 'true' && !forceReanalyze) {
      panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
      return;
    }

    // Start loading
    deepAnalyzing[dimKey] = true;
    panel.style.display = 'block';
    panel.innerHTML = '<div class="deep-loading"><div class="deep-spinner"></div><span>AI 正在生成深度分析，请稍候...</span></div>';

    try {
      const payload = {
        dim: dimKey,
        stock_data: currentReport,
        force: forceReanalyze,
        debug: isDebug(),
        ai_chat: getAiConfig(),
      };
      addDebugLog(`🔬 深度分析请求 [${dimKey}]`, {
        api: '/api/deep_analyze',
        dim: dimKey,
        force: forceReanalyze,
        stock_code: currentReport?.code,
        stock_name: currentReport?.name,
      });

      const resp = await fetch('/api/deep_analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();

      addDebugLog(`🔬 深度分析响应 [${dimKey}]`, {
        from_cache: data.from_cache,
        cached_at: data.cached_at,
        need_config: data.need_config,
        reply_length: data.reply?.length || 0,
        reply_preview: data.reply?.slice(0, 300) || '(empty)',
        debug: data.debug || null,
      });

      if (data.need_config) {
        panel.innerHTML = `<div class="deep-need-config">
          <div class="deep-config-msg">${data.reply}</div>
        </div>`;
      } else if (data.error) {
        panel.innerHTML = `<div class="deep-error">⚠️ ${data.reply}</div>`;
      } else {
        panel.dataset.loaded = 'true';
        panel.innerHTML = renderDeepAnalysis(data.reply, dimKey, data.from_cache, data.cached_at);
      }
    } catch (err) {
      panel.innerHTML = `<div class="deep-error">⚠️ 分析请求失败: ${err.message}</div>`;
    } finally {
      deepAnalyzing[dimKey] = false;
    }
  }

  function renderDeepAnalysis(reply, dimKey, fromCache, cachedAt) {
    // Simple markdown-like rendering for the debate format
    let html = reply
      // Headers
      .replace(/^### (.+)$/gm, '<h4 class="deep-h4">$1</h4>')
      .replace(/^## (.+)$/gm, '<h4 class="deep-h3">$1</h4>')
      // Bold
      .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
      // Bull/bear markers
      .replace(/🔴/g, '<span class="deep-icon">🔴</span>')
      .replace(/🟢/g, '<span class="deep-icon">🟢</span>')
      .replace(/🟡/g, '<span class="deep-icon">🟡</span>')
      // Bullet lists
      .replace(/^- /gm, '<span class="deep-bullet">•</span> ')
      // Newlines
      .replace(/\n\n/g, '</p><p class="deep-p">')
      .replace(/\n/g, '<br>');

    html = '<p class="deep-p">' + html + '</p>';

    const cacheTag = fromCache
      ? `<span class="cache-indicator" title="缓存时间: ${cachedAt || ''}">💾 缓存内容 · ${cachedAt || ''} · <a href="javascript:void(0)" onclick="handleDeepAnalyze('${dimKey}',true)" style="color:#0984e3;text-decoration:underline">🔄 重新分析</a></span>`
      : '';

    return `
      <div class="deep-content">
        <div class="deep-header">
          <span class="deep-badge">🔬 AI 深度分析</span>
          <div style="display:flex;gap:8px;align-items:center">
            ${cacheTag}
            <button class="deep-close" onclick="this.closest('.deep-analyze-panel').style.display='none'">✕</button>
          </div>
        </div>
        <div class="deep-body">${html}</div>
        <div class="deep-disclaimer">⚠️ AI 生成内容仅供参考，不构成投资建议</div>
      </div>
    `;
  }

  // ========== Timing Analysis ==========
  async function handleTimingAnalysis() {
    if (!currentReport) return;
    const content = $('#timingContent');
    if (!content) return;

    if (content.dataset.loaded === 'true') return;

    content.innerHTML = '<div class="deep-loading"><div class="deep-spinner"></div><span>AI 正在分析购入时机...</span></div>';

    try {
      const resp = await fetch('/api/timing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock_data: currentReport, ai_chat: getAiConfig() }),
      });
      const data = await resp.json();

      if (data.skip) {
        content.innerHTML = `<div class="risk-alert danger">${data.reply}</div>`;
        return;
      }

      content.dataset.loaded = 'true';

      // Render timing analysis with improved markdown
      let html = renderSimpleMarkdown(data.reply);

      content.innerHTML = `
        <div class="deep-body" style="border-left: 4px solid var(--primary);padding-left: 16px;">${html}</div>
        ${data.rule_based ? '<div class="deep-disclaimer" style="margin-top:12px">⚠️ 当前使用规则引擎生成（未配置 LLM API），建议配置 AI 获取更精准分析</div>' : '<div class="deep-disclaimer" style="margin-top:12px">⚠️ AI 生成内容仅供参考，不构成投资建议。投资有风险，决策需谨慎。</div>'}
      `;
    } catch (err) {
      content.innerHTML = `<div class="deep-error">⚠️ 时机分析请求失败: ${err.message}</div>`;
    }
  }

  // ========== One-click Deep Analyze All ==========
  async function handleAnalyzeAll() {
    const dims = ['fundamental', 'technical', 'capital', 'events', 'industry', 'value'];
    const dimNames = {
      fundamental: '基本面', technical: '技术面', capital: '资金面',
      events: '事件催化', industry: '同业对标', value: '投资性价比'
    };

    // Unfold all sections
    document.querySelectorAll('.section-body').forEach(b => b.classList.remove('collapsed'));

    // Trigger all in parallel
    for (const dim of dims) {
      if (!deepAnalyzing[dim]) {
        await handleDeepAnalyze(dim);
      }
    }
  }

  // Expose to global scope for onclick handlers
  window.handleDeepAnalyze = handleDeepAnalyze;
  window.handleTimingAnalysis = handleTimingAnalysis;
  window.handleAnalyzeAll = handleAnalyzeAll;
  window.toggleAltDeepAnalysis = toggleAltDeepAnalysis;
  chatToggle.addEventListener('click', () => {
    chatOpen = !chatOpen;
    chatPanel.classList.toggle('open', chatOpen);
    if (chatOpen) chatInput.focus();
  });
  chatClose.addEventListener('click', () => {
    chatOpen = false;
    chatPanel.classList.remove('open');
  });

  function addChatMessage(text, type) {
    const div = document.createElement('div');
    div.className = `chat-msg ${type}`;
    div.innerHTML = renderSimpleMarkdown(text);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
  }

  function renderSimpleMarkdown(text) {
    // Escape HTML first to prevent XSS, then apply markdown-like formatting
    let html = escapeHtml(text);

    // ---- Tables (must be processed BEFORE line-level transforms) ----
    // Match markdown tables: header row, separator row, body rows
    html = html.replace(/(\|[^\n]+\|\n\|[\s\-:|]+\|\n((?:\|[^\n]+\|\n?)*))/g, function(match) {
      const lines = match.trim().split('\n');
      if (lines.length < 2) return match; // not a valid table

      // Parse header
      const headers = lines[0].split('|').map(h => h.trim()).filter(h => h);
      // Skip separator line (index 1)
      const bodyLines = lines.slice(2);

      let tableHtml = '<table class="md-table"><thead><tr>';
      for (const h of headers) {
        tableHtml += `<th>${h}</th>`;
      }
      tableHtml += '</tr></thead><tbody>';

      for (const row of bodyLines) {
        const cells = row.split('|').map(c => c.trim()).filter(c => c);
        if (cells.length === 0) continue;
        tableHtml += '<tr>';
        for (let i = 0; i < headers.length; i++) {
          tableHtml += `<td>${cells[i] || ''}</td>`;
        }
        tableHtml += '</tr>';
      }
      tableHtml += '</tbody></table>';
      return tableHtml;
    });

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    html = html.replace(/__(.+?)__/g, '<b>$1</b>');

    // Italic: *text* or _text_ (but not inside **)
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<i>$1</i>');
    html = html.replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, '<i>$1</i>');

    // Inline code: `code`
    html = html.replace(/`(.+?)`/g, '<code style="background:#e8e8e8;padding:1px 4px;border-radius:3px;font-family:monospace;font-size:0.85em">$1</code>');

    // Code blocks: ```...``` (multi-line)
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(m, lang, code) {
      return '<pre style="background:#2d2d2d;color:#f8f8f2;padding:10px 14px;border-radius:6px;overflow-x:auto;font-size:0.82em;margin:8px 0;white-space:pre-wrap">' +
        '<code>' + escapeHtml(code.trim()) + '</code></pre>';
    });

    // Headers: ### text
    html = html.replace(/^### (.+)$/gm, '<h5 style="font-size:0.9rem;margin:10px 0 6px;color:#6c5ce7">$1</h5>');
    html = html.replace(/^## (.+)$/gm, '<h4 style="font-size:0.95rem;margin:12px 0 8px;color:#0984e3">$1</h4>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr style="border:none;border-top:1px solid #dfe6e9;margin:10px 0">');

    // Ordered lists: 1. text
    html = html.replace(/^(\d+)\. (.+)$/gm, '<div style="margin:2px 0"><span style="color:#0984e3;font-weight:600">$1.</span> $2</div>');

    // Unordered lists: - text or * text
    html = html.replace(/^[\-\*] (.+)$/gm, '<div style="margin:2px 0;padding-left:8px">• $1</div>');

    // Convert double newlines to paragraph breaks
    html = html.replace(/\n\n/g, '<br><br>');
    // Convert remaining single newlines
    html = html.replace(/\n/g, '<br>');

    return html;
  }

  async function sendChatMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    chatInput.value = '';
    addChatMessage(msg, 'user');

    const thinkingDiv = addChatMessage('思考中...', 'bot');

    // Build stock context
    let stockCtx = '';
    if (currentReport) {
      stockCtx = `当前股票: ${currentReport.name}(${currentReport.code}), 价格: ${currentReport.price}元, PE: ${currentReport.pe}, 综合评分: ${currentReport.total_score}/${currentReport.max_score}`;
    }

    try {
      const aiCfg = getAiConfig();
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, stock_context: stockCtx, ai_chat: aiCfg }),
      });
      const data = await resp.json();
      thinkingDiv.remove();
      addChatMessage(data.reply, data.error ? 'error' : 'bot');
      if (data.need_config) {
        // Prompt to open settings
      }
    } catch (err) {
      thinkingDiv.remove();
      addChatMessage('网络错误: ' + err.message, 'error');
    }
  }

  chatSend.addEventListener('click', sendChatMessage);
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });

  // ========== Settings Modal ==========
  settingsBtn.addEventListener('click', () => openSettings());
  settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) closeSettings();
  });

  async function openSettings() {
    settingsOverlay.classList.add('open');
    // Load from localStorage first (always works), then try server
    const localChat = {
      provider: localStorage.getItem('ai_provider') || 'openai',
      api_key: localStorage.getItem('ai_api_key') || '',
      api_base: localStorage.getItem('ai_api_base') || 'https://api.openai.com/v1',
      model: localStorage.getItem('ai_model') || 'gpt-4o-mini',
      system_prompt: localStorage.getItem('ai_system_prompt') || '',
    };
    $('#apiProvider').value = localChat.provider;
    $('#apiKey').value = '';
    $('#apiKey').placeholder = localChat.api_key ? localChat.api_key.slice(0, 8) + '****' : '输入你的 API Key';
    $('#apiBase').value = localChat.api_base;
    $('#apiModel').value = localChat.model;
    $('#systemPrompt').value = localChat.system_prompt;

    // Also try loading from server (may fail on Vercel)
    try {
      const resp = await fetch('/api/config');
      const cfg = await resp.json();
      const chat = cfg.ai_chat || {};
      if (chat.provider) $('#apiProvider').value = chat.provider;
      if (chat.api_base) $('#apiBase').value = chat.api_base;
      if (chat.model) $('#apiModel').value = chat.model;
      if (chat.system_prompt) $('#systemPrompt').value = chat.system_prompt;
    } catch (err) {
      // Server load failed, localStorage values already set
    }
  }

  function closeSettings() {
    settingsOverlay.classList.remove('open');
  }

  $('#saveSettings').addEventListener('click', async () => {
    const apiKey = $('#apiKey').value.trim();
    const apiBase = $('#apiBase').value.trim();
    const apiModel = $('#apiModel').value.trim();
    const systemPrompt = $('#systemPrompt').value.trim();

    const chatCfg = {
      provider: $('#apiProvider').value,
      api_key: apiKey && !apiKey.includes('****') ? apiKey : (localStorage.getItem('ai_api_key') || ''),
      api_base: apiBase || 'https://api.openai.com/v1',
      model: apiModel || 'gpt-4o-mini',
      system_prompt: systemPrompt || '你是一位专业的股票投资分析师。',
    };

    // Always save to localStorage (works on Vercel)
    localStorage.setItem('ai_provider', chatCfg.provider);
    localStorage.setItem('ai_api_key', chatCfg.api_key);
    localStorage.setItem('ai_api_base', chatCfg.api_base);
    localStorage.setItem('ai_model', chatCfg.model);
    localStorage.setItem('ai_system_prompt', chatCfg.system_prompt);

    // Also try to save to server (may fail on Vercel)
    try {
      const payload = { ai_chat: chatCfg };
      delete payload.ai_chat.api_key;  // don't send empty key
      if (apiKey && !apiKey.includes('****')) {
        payload.ai_chat.api_key = apiKey;
      }
      const resp = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (data.status === 'ok') {
        alert('设置已保存！');
        closeSettings();
        return;
      }
    } catch (err) {
      // Server save failed, but localStorage is already saved
    }
    alert('设置已保存到本地浏览器！（服务器端保存不可用）');
    closeSettings();
  });

  $('#testChat').addEventListener('click', async () => {
    const apiKey = $('#apiKey').value.trim();
    const apiBase = $('#apiBase').value.trim();
    const apiModel = $('#apiModel').value.trim();
    const resultDiv = $('#testResult');

    // First save current settings temporarily
    if (apiKey && !apiKey.includes('****')) {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ai_chat: {
            provider: $('#apiProvider').value,
            api_key: apiKey,
            api_base: apiBase,
            model: apiModel,
          }
        }),
      });
    }

    resultDiv.style.display = 'block';
    resultDiv.className = 'test-result';
    resultDiv.textContent = '测试中...';

    try {
      const aiCfg = getAiConfig();
      const resp = await fetch('/api/test_chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ai_chat: aiCfg }),
      });
      const data = await resp.json();
      if (data.success) {
        resultDiv.className = 'test-result success';
        resultDiv.textContent = '✅ ' + data.message;
      } else {
        resultDiv.className = 'test-result fail';
        resultDiv.textContent = '❌ ' + data.message;
      }
    } catch (err) {
      resultDiv.className = 'test-result fail';
      resultDiv.textContent = '❌ 网络错误: ' + err.message;
    }
  });

  // ========== Debug Mode ==========
  const DEBUG_STORAGE_KEY = 'stock_analyzer_debug';
  const DEBUG_PASSWORD = 'Ciallo~';
  let debugMode = localStorage.getItem(DEBUG_STORAGE_KEY) === 'true';

  function isDebug() { return debugMode; }

  function addDebugLog(label, content) {
    if (!debugMode) return;
    const panel = $('#debugMessages');
    if (!panel) return;
    const entry = document.createElement('div');
    entry.className = 'debug-entry';
    const ts = new Date().toLocaleTimeString();
    entry.innerHTML = `
      <div class="debug-entry-header" onclick="this.nextElementSibling.classList.toggle('collapsed')">
        <span class="debug-ts">${ts}</span> ${label}
      </div>
      <div class="debug-entry-body">
        <pre class="debug-pre">${escapeHtml(typeof content === 'string' ? content : JSON.stringify(content, null, 2))}</pre>
      </div>
    `;
    panel.appendChild(entry);
    panel.scrollTop = panel.scrollHeight;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function getAiConfig() {
    return {
      api_key: localStorage.getItem('ai_api_key') || '',
      api_base: localStorage.getItem('ai_api_base') || 'https://api.openai.com/v1',
      model: localStorage.getItem('ai_model') || 'gpt-4o-mini',
      provider: localStorage.getItem('ai_provider') || 'openai',
      system_prompt: localStorage.getItem('ai_system_prompt') || '',
    };
  }

  // Debug unlock
  const debugUnlockBtn = $('#debugUnlock');
  const debugPasswordInput = $('#debugPassword');
  const debugStatusDiv = $('#debugStatus');
  const debugPanel = $('#debugPanel');
  const debugCloseBtn = $('#debugClose');
  const debugClearBtn = $('#debugClear');

  if (debugUnlockBtn) {
    debugUnlockBtn.addEventListener('click', () => {
      const pw = (debugPasswordInput?.value || '').trim();
      if (pw === DEBUG_PASSWORD) {
        debugMode = true;
        localStorage.setItem(DEBUG_STORAGE_KEY, 'true');
        if (debugStatusDiv) debugStatusDiv.innerHTML = '<span style="color:#00b894">✅ Debug 模式已解锁！所有 LLM API 调用将显示在下方面板。</span>';
        if (debugPanel) { debugPanel.style.display = 'flex'; debugPanel.classList.add('open'); }
        addDebugLog('🔓 Debug 模式已激活', '密码验证通过。所有 LLM API Prompt、参数和原始响应将记录于此。');
      } else if (pw) {
        if (debugStatusDiv) debugStatusDiv.innerHTML = '<span style="color:#d63031">❌ 密码错误</span>';
      }
    });
  }

  if (debugCloseBtn) {
    debugCloseBtn.addEventListener('click', () => {
      if (debugPanel) debugPanel.style.display = 'none';
    });
  }

  if (debugClearBtn) {
    debugClearBtn.addEventListener('click', () => {
      const msgs = $('#debugMessages');
      if (msgs) msgs.innerHTML = '';
    });
  }

  // Check if already in debug mode on load
  if (debugMode) {
    setTimeout(() => {
      if (debugPanel) { debugPanel.style.display = 'flex'; debugPanel.classList.add('open'); }
      if (debugStatusDiv) debugStatusDiv.innerHTML = '<span style="color:#00b894">✅ Debug 模式已激活</span>';
    }, 500);
  }
  // Auto-analyze default stock on load if none in URL
  const urlParams = new URLSearchParams(window.location.search);
  const initCode = urlParams.get('code');
  if (initCode) {
    searchInput.value = initCode;
    analyzeStock(initCode);
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (chatOpen) { chatOpen = false; chatPanel.classList.remove('open'); }
      if (settingsOverlay.classList.contains('open')) closeSettings();
    }
    // Ctrl+K for search focus
    if (e.ctrlKey && e.key === 'k') {
      e.preventDefault();
      searchInput.focus();
    }
  });

})();
