/**
 * TRADEGUARD - Modern Light Financial Analytics & Anomaly Intelligence
 * Client-side Controller & Chart Engine
 * Course: Pattern Recognition and Anomaly Detection (Semester 5)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Chart.js instances cache
  const charts = {};

  // Application State
  const state = {
    currentPage: 1,
    pageSize: 20,
    totalPages: 1,
    currentFilter: {
      search: "",
      symbol: "ALL",
      status: "ALL",
      reason: "ALL",
      sortBy: "mahalanobis_distance",
      sortOrder: "desc"
    },
    scatterDataCache: null,
    scatterMode: "all",
    dashboardDataCache: null
  };

  // Light Mode Chart Theme Tokens
  const LIGHT_THEME = {
    textMuted: "#64748B",
    textDark: "#0F172A",
    gridColor: "#F1F5F9",
    borderColor: "#E2E8F0",
    primaryBlue: "#2563EB",
    indigo: "#4F46E5",
    normalGreen: "#16A34A",
    anomalyRed: "#DC2626",
    warningAmber: "#D97706",
    scatterNormalFill: "rgba(147, 197, 253, 0.45)",
    scatterNormalStroke: "rgba(59, 130, 246, 0.7)",
    scatterAnomalyFill: "rgba(220, 38, 38, 0.95)",
    scatterAnomalyStroke: "#FFFFFF"
  };

  // Configure Chart.js global defaults for clean light theme
  if (typeof Chart !== "undefined") {
    Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif";
    Chart.defaults.color = LIGHT_THEME.textMuted;
    Chart.defaults.plugins.tooltip.backgroundColor = "#0F172A";
    Chart.defaults.plugins.tooltip.titleColor = "#FFFFFF";
    Chart.defaults.plugins.tooltip.bodyColor = "#E2E8F0";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 6;
  }

  // =========================================================
  // TOAST NOTIFICATIONS (LIGHT MODE)
  // =========================================================
  function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      toast.style.transition = "all 0.25s ease";
      setTimeout(() => toast.remove(), 250);
    }, 3800);
  }

  // =========================================================
  // TAB NAVIGATION
  // =========================================================
  function switchPage(pageId) {
    document.querySelectorAll(".nav-tab-item, .nav-pill-item, .nav-item").forEach(item => {
      if (item.dataset.target === pageId) {
        item.classList.add("active");
      } else {
        item.classList.remove("active");
      }
    });

    document.querySelectorAll(".page-section").forEach(sec => {
      if (sec.id === pageId) {
        sec.classList.add("active-section");
      } else {
        sec.classList.remove("active-section");
      }
    });

    if (pageId === "dashboard-page") {
      fetchDashboardData();
    } else if (pageId === "market-viz-page") {
      renderMarketVisualizations();
    } else if (pageId === "results-page") {
      fetchTradesPage();
    }
  }

  document.querySelectorAll(".nav-tab-item, .nav-pill-item, .nav-item").forEach(item => {
    item.addEventListener("click", () => {
      const targetId = item.dataset.target;
      if (targetId) switchPage(targetId);
    });
  });

  // =========================================================
  // DASHBOARD DATA FETCH & METRIC POPULATION
  // =========================================================
  async function fetchDashboardData() {
    try {
      const res = await fetch("/api/dashboard");
      const json = await res.json();
      if (json.status !== "success") throw new Error("Failed to load dashboard data");

      const stats = json.statistics;
      state.dashboardDataCache = stats;

      // Update Top KPI Metric Tiles
      document.getElementById("card-total-trades").textContent = stats.total_trades.toLocaleString();
      document.getElementById("card-normal-trades").textContent = stats.normal_trades.toLocaleString();
      document.getElementById("card-anomalous-trades").textContent = stats.anomalous_trades.toLocaleString();
      document.getElementById("card-anomaly-pct").textContent = `${stats.anomaly_percentage}% of total trades`;
      document.getElementById("card-avg-volume").textContent = `$${stats.average_volume.toLocaleString()}`;
      document.getElementById("card-max-volume").textContent = `$${stats.maximum_volume.toLocaleString()}`;
      document.getElementById("card-avg-md").textContent = stats.average_mahalanobis.toFixed(2);
      document.getElementById("card-threshold-display").textContent = `Cutoff \u03c4_M = ${stats.mahalanobis_threshold.toFixed(2)}`;
      document.getElementById("card-dataset-name").textContent = `Dataset: ${json.dataset_name}`;
      document.getElementById("active-dataset-lbl").textContent = json.dataset_name;

      // Sliders & Values
      document.getElementById("slider-md-thresh").value = stats.mahalanobis_threshold;
      document.getElementById("val-md-thresh").textContent = stats.mahalanobis_threshold.toFixed(2);
      document.getElementById("slider-vol-thresh").value = stats.volume_z_threshold;
      document.getElementById("val-vol-thresh").textContent = stats.volume_z_threshold.toFixed(2);

      // Model Validation Box
      if (stats.evaluation && stats.evaluation.has_ground_truth) {
        document.getElementById("dashboard-eval-box").style.display = "block";
        document.getElementById("eval-precision").textContent = (stats.evaluation.precision * 100).toFixed(1) + "%";
        document.getElementById("eval-recall").textContent = (stats.evaluation.recall * 100).toFixed(1) + "%";
        document.getElementById("eval-f1").textContent = (stats.evaluation.f1_score * 100).toFixed(1) + "%";
        document.getElementById("eval-confusion").textContent = `${stats.evaluation.tp} / ${stats.evaluation.fp} / ${stats.evaluation.tn} / ${stats.evaluation.fn}`;
      } else {
        document.getElementById("dashboard-eval-box").style.display = "none";
      }

      // Render Charts
      await renderDashboardCharts();

      // Populate Quick-Select Anomaly List in Trade Analysis
      populateQuickAnomalySelect();

    } catch (err) {
      console.error(err);
      showToast("Error loading dashboard data", "error");
    }
  }

  // =========================================================
  // CHART RENDERING (Chart.js in Light Theme)
  // =========================================================
  async function renderDashboardCharts() {
    try {
      const res = await fetch("/api/market-data");
      const data = await res.json();
      if (data.status !== "success") return;

      state.scatterDataCache = data.scatter;

      // 1. Normal vs Anomalous Doughnut Chart
      const ctxDoughnut = document.getElementById("chart-doughnut")?.getContext("2d");
      if (ctxDoughnut) {
        if (charts.doughnut) charts.doughnut.destroy();
        charts.doughnut = new Chart(ctxDoughnut, {
          type: "doughnut",
          data: {
            labels: data.doughnut.labels,
            datasets: [{
              data: data.doughnut.data,
              backgroundColor: [LIGHT_THEME.normalGreen, LIGHT_THEME.anomalyRed],
              borderColor: "#FFFFFF",
              borderWidth: 3,
              hoverOffset: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: "bottom",
                labels: { color: LIGHT_THEME.textMuted, font: { size: 12 }, padding: 14 }
              }
            },
            cutout: "68%"
          }
        });
      }

      // 2. Mahalanobis Distance Distribution Histogram
      const ctxMdHist = document.getElementById("chart-md-hist")?.getContext("2d");
      if (ctxMdHist) {
        if (charts.mdHist) charts.mdHist.destroy();
        charts.mdHist = new Chart(ctxMdHist, {
          type: "bar",
          data: {
            labels: data.md_histogram.labels,
            datasets: [{
              label: "Observations",
              data: data.md_histogram.counts,
              backgroundColor: data.md_histogram.labels.map(l => {
                const upper = parseFloat(l.split("-")[1] || 0);
                return upper > data.md_histogram.threshold ? LIGHT_THEME.anomalyRed : LIGHT_THEME.indigo;
              }),
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: (ctx) => `Trades Count: ${ctx.parsed.y}`
                }
              }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 }, maxRotation: 45 }
              },
              y: {
                grid: { color: LIGHT_THEME.gridColor },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 } }
              }
            }
          }
        });
      }

      // 3. Trade Volume Distribution Histogram
      const ctxVolHist = document.getElementById("chart-vol-hist")?.getContext("2d");
      if (ctxVolHist) {
        if (charts.volHist) charts.volHist.destroy();
        charts.volHist = new Chart(ctxVolHist, {
          type: "bar",
          data: {
            labels: data.volume_histogram.labels,
            datasets: [{
              label: "Trades Count",
              data: data.volume_histogram.counts,
              backgroundColor: LIGHT_THEME.primaryBlue,
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 }, maxRotation: 45 }
              },
              y: {
                grid: { color: LIGHT_THEME.gridColor },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 } }
              }
            }
          }
        });
      }

      // 4. Simulated Multi-Asset Price Movement Line Chart
      const ctxPriceLine = document.getElementById("chart-price-line")?.getContext("2d");
      if (ctxPriceLine) {
        if (charts.priceLine) charts.priceLine.destroy();
        const symbolColors = {
          TECH: "#4F46E5",
          BANK: "#2563EB",
          AUTO: "#059669",
          ENERGY: "#D97706",
          PHARMA: "#7C3AED"
        };
        const datasets = Object.keys(data.price_movement.series).map(sym => ({
          label: sym,
          data: data.price_movement.series[sym],
          borderColor: symbolColors[sym] || "#64748B",
          backgroundColor: "transparent",
          borderWidth: 1.8,
          pointRadius: 0,
          spanGaps: true,
          tension: 0.2
        }));

        charts.priceLine = new Chart(ctxPriceLine, {
          type: "line",
          data: {
            labels: data.price_movement.labels,
            datasets: datasets
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            plugins: {
              legend: {
                position: "top",
                labels: { color: LIGHT_THEME.textMuted, boxWidth: 12, font: { size: 11 } }
              }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: LIGHT_THEME.textMuted, maxTicksLimit: 8, font: { size: 10 } }
              },
              y: {
                grid: { color: LIGHT_THEME.gridColor },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 } }
              }
            }
          }
        });
      }

      // 5. Intraday Trading Activity Timeline
      const ctxActivity = document.getElementById("chart-activity-timeline")?.getContext("2d");
      if (ctxActivity) {
        if (charts.activityTimeline) charts.activityTimeline.destroy();
        charts.activityTimeline = new Chart(ctxActivity, {
          type: "bar",
          data: {
            labels: data.activity_time.labels,
            datasets: [
              {
                label: "Normal Flow",
                data: data.activity_time.total_trades,
                backgroundColor: "rgba(147, 197, 253, 0.6)",
                stack: "flow",
                borderRadius: 2
              },
              {
                label: "Potential Anomalies",
                data: data.activity_time.anomaly_trades,
                backgroundColor: LIGHT_THEME.anomalyRed,
                stack: "flow",
                borderRadius: 2
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: "top",
                labels: { color: LIGHT_THEME.textMuted, boxWidth: 12, font: { size: 11 } }
              }
            },
            scales: {
              x: {
                grid: { display: false },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 } }
              },
              y: {
                grid: { color: LIGHT_THEME.gridColor },
                ticks: { color: LIGHT_THEME.textMuted, font: { size: 10 } }
              }
            }
          }
        });
      }

    } catch (err) {
      console.error("Error rendering charts:", err);
    }
  }

  // =========================================================
  // 2D MARKET VISUALIZATION (SCATTER PLOTS)
  // =========================================================
  function renderMarketVisualizations() {
    if (!state.scatterDataCache) return;

    const data = state.scatterDataCache;

    const makeScatterDataset = (normalPts, anomalyPts) => {
      const ds = [];
      if (state.scatterMode !== "anomalies_only") {
        ds.push({
          label: "Normal Trades",
          data: normalPts,
          backgroundColor: LIGHT_THEME.scatterNormalFill,
          borderColor: LIGHT_THEME.scatterNormalStroke,
          borderWidth: 1,
          pointRadius: 3.5,
          pointHoverRadius: 6
        });
      }
      ds.push({
        label: "Potential Anomalies",
        data: anomalyPts,
        backgroundColor: LIGHT_THEME.scatterAnomalyFill,
        borderColor: LIGHT_THEME.scatterAnomalyStroke,
        borderWidth: 1.5,
        pointRadius: 5.5,
        pointHoverRadius: 8
      });
      return ds;
    };

    // 1. Price vs Volume
    const ctxPV = document.getElementById("chart-scatter-price-vol")?.getContext("2d");
    if (ctxPV) {
      if (charts.scatterPV) charts.scatterPV.destroy();
      charts.scatterPV = new Chart(ctxPV, {
        type: "scatter",
        data: {
          datasets: makeScatterDataset(data.price_volume.normal, data.price_volume.anomaly)
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "top",
              labels: { color: LIGHT_THEME.textMuted, font: { size: 12 } }
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const pt = ctx.raw;
                  return `${pt.id} [${pt.sym}]: Price $${pt.x}, Vol $${pt.y.toLocaleString()}${pt.md ? ` | D_M: ${pt.md}` : ""}`;
                }
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: "Executed Price ($)", color: LIGHT_THEME.textMuted },
              grid: { color: LIGHT_THEME.gridColor },
              ticks: { color: LIGHT_THEME.textMuted }
            },
            y: {
              title: { display: true, text: "Trade Volume ($)", color: LIGHT_THEME.textMuted },
              grid: { color: LIGHT_THEME.gridColor },
              ticks: { color: LIGHT_THEME.textMuted }
            }
          }
        }
      });
    }

    // 2. Price vs Quantity
    const ctxPQ = document.getElementById("chart-scatter-price-qty")?.getContext("2d");
    if (ctxPQ) {
      if (charts.scatterPQ) charts.scatterPQ.destroy();
      charts.scatterPQ = new Chart(ctxPQ, {
        type: "scatter",
        data: {
          datasets: makeScatterDataset(data.price_quantity.normal, data.price_quantity.anomaly)
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: LIGHT_THEME.textMuted } },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.raw.id} [${ctx.raw.sym}]: Price $${ctx.raw.x}, Qty ${ctx.raw.y}`
              }
            }
          },
          scales: {
            x: { title: { display: true, text: "Executed Price ($)", color: LIGHT_THEME.textMuted }, grid: { color: LIGHT_THEME.gridColor }, ticks: { color: LIGHT_THEME.textMuted } },
            y: { title: { display: true, text: "Quantity (Contracts)", color: LIGHT_THEME.textMuted }, grid: { color: LIGHT_THEME.gridColor }, ticks: { color: LIGHT_THEME.textMuted } }
          }
        }
      });
    }

    // 3. Trade Volume vs Price Change
    const ctxVP = document.getElementById("chart-scatter-vol-pricechange")?.getContext("2d");
    if (ctxVP) {
      if (charts.scatterVP) charts.scatterVP.destroy();
      charts.scatterVP = new Chart(ctxVP, {
        type: "scatter",
        data: {
          datasets: makeScatterDataset(data.volume_price_change.normal, data.volume_price_change.anomaly)
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: LIGHT_THEME.textMuted } },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.raw.id}: Δ $${ctx.raw.x}, Vol $${ctx.raw.y.toLocaleString()}`
              }
            }
          },
          scales: {
            x: { title: { display: true, text: "Price Change Δ ($)", color: LIGHT_THEME.textMuted }, grid: { color: LIGHT_THEME.gridColor }, ticks: { color: LIGHT_THEME.textMuted } },
            y: { title: { display: true, text: "Volume ($)", color: LIGHT_THEME.textMuted }, grid: { color: LIGHT_THEME.gridColor }, ticks: { color: LIGHT_THEME.textMuted } }
          }
        }
      });
    }
  }

  // Market Visualization Controls
  document.getElementById("btn-filter-anomalies-only")?.addEventListener("click", () => {
    state.scatterMode = "anomalies_only";
    renderMarketVisualizations();
    showToast("Displaying anomalous trades only");
  });

  document.getElementById("btn-filter-all")?.addEventListener("click", () => {
    state.scatterMode = "all";
    renderMarketVisualizations();
    showToast("Displaying all market observations");
  });

  document.getElementById("btn-reset-view")?.addEventListener("click", () => {
    state.scatterMode = "all";
    renderMarketVisualizations();
    showToast("Visualizations reset to default view");
  });

  // =========================================================
  // TRADE ANALYSIS & GAUGE DEVIATION METER
  // =========================================================
  async function inspectTrade(tradeId) {
    if (!tradeId) return;
    try {
      const res = await fetch(`/api/trade/${encodeURIComponent(tradeId.trim())}`);
      const json = await res.json();
      if (json.status !== "success") {
        showToast(json.message || "Trade not found", "error");
        return;
      }

      const t = json.trade;
      const md = parseFloat(t.mahalanobis_distance);
      const vz = parseFloat(t.volume_zscore);

      document.getElementById("td-trade-id").textContent = t.trade_id;
      document.getElementById("td-timestamp").textContent = t.timestamp;
      document.getElementById("td-symbol").textContent = t.symbol;
      document.getElementById("td-price").textContent = `$${parseFloat(t.price).toFixed(2)}`;
      document.getElementById("td-quantity").textContent = parseInt(t.quantity).toLocaleString();
      document.getElementById("td-volume").textContent = `$${parseFloat(t.volume).toLocaleString()}`;
      document.getElementById("td-price-change").textContent = (t.price_change >= 0 ? "+" : "") + parseFloat(t.price_change).toFixed(3);
      document.getElementById("td-reason").textContent = t.reason;
      document.getElementById("td-md-dist").textContent = md.toFixed(2);
      document.getElementById("td-vol-z").textContent = vz.toFixed(2);

      // Position Needle Pointer on Light Scale (0.0 to 6.0 range, threshold at 3.0 = 50%)
      const needlePercent = Math.min(98, Math.max(2, (md / 6.0) * 100));
      const needleEl = document.getElementById("meter-needle");
      if (needleEl) {
        needleEl.style.left = `${needlePercent}%`;
      }
      const meterValEl = document.getElementById("meter-display-val");
      if (meterValEl) {
        meterValEl.innerHTML = `D<sub>M</sub>: ${md.toFixed(2)}`;
      }

      const isAnomaly = t.status === "POTENTIAL ANOMALY";
      const statusBadge = document.getElementById("td-status-badge");
      statusBadge.innerHTML = isAnomaly
        ? `<span class="status-chip anomaly">POTENTIAL ANOMALY</span>`
        : `<span class="status-chip normal">NORMAL</span>`;

      const interpBox = document.getElementById("td-interp-box");
      interpBox.className = `narrative-card ${isAnomaly ? "anomaly" : "normal"}`;
      document.getElementById("td-interp-text").textContent = t.interpretation;

      // Update search input to match
      document.getElementById("trade-search-input").value = t.trade_id;

    } catch (err) {
      console.error(err);
      showToast("Error inspecting trade", "error");
    }
  }

  async function populateQuickAnomalySelect() {
    try {
      const res = await fetch("/api/results");
      const json = await res.json();
      if (json.status !== "success") return;

      const select = document.getElementById("quick-anomaly-select");
      if (!select) return;

      select.innerHTML = `<option value="">-- Choose an anomalous trade --</option>`;
      json.top_by_mahalanobis.forEach(t => {
        const opt = document.createElement("option");
        opt.value = t.trade_id;
        opt.textContent = `${t.trade_id} [${t.symbol}] - D_M: ${parseFloat(t.mahalanobis_distance).toFixed(2)} (${t.reason})`;
        select.appendChild(opt);
      });
    } catch (e) {
      console.error("Could not populate anomaly select:", e);
    }
  }

  document.getElementById("btn-fetch-trade")?.addEventListener("click", () => {
    const tradeId = document.getElementById("trade-search-input").value;
    inspectTrade(tradeId);
  });

  document.getElementById("trade-search-input")?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      inspectTrade(e.target.value);
    }
  });

  document.getElementById("quick-anomaly-select")?.addEventListener("change", (e) => {
    if (e.target.value) {
      inspectTrade(e.target.value);
    }
  });

  // =========================================================
  // RUN DETECTION & CALIBRATION
  // =========================================================
  const sliderMd = document.getElementById("slider-md-thresh");
  const valMd = document.getElementById("val-md-thresh");
  sliderMd?.addEventListener("input", (e) => {
    valMd.textContent = parseFloat(e.target.value).toFixed(2);
  });

  const sliderVol = document.getElementById("slider-vol-thresh");
  const valVol = document.getElementById("val-vol-thresh");
  sliderVol?.addEventListener("input", (e) => {
    valVol.textContent = parseFloat(e.target.value).toFixed(2);
  });

  document.getElementById("btn-run-detection")?.addEventListener("click", async () => {
    const mdVal = parseFloat(sliderMd.value);
    const volVal = parseFloat(sliderVol.value);
    const btn = document.getElementById("btn-run-detection");

    btn.disabled = true;
    btn.textContent = "Processing Mahalanobis distances...";

    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mahalanobis_threshold: mdVal,
          volume_z_threshold: volVal
        })
      });

      const json = await res.json();
      if (json.status !== "success") throw new Error(json.message);

      showToast("Detection completed successfully!", "success");

      const preview = document.getElementById("detection-run-preview");
      preview.style.display = "block";
      document.getElementById("detection-preview-stats").textContent =
        `Scored ${json.statistics.total_trades.toLocaleString()} records: flagged ${json.statistics.anomalous_trades.toLocaleString()} potential anomalies (${json.statistics.anomaly_percentage}%).`;

      await fetchDashboardData();

    } catch (err) {
      console.error(err);
      showToast(err.message || "Detection failed", "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
        Run Anomaly Detection
      `;
    }
  });

  // Drag and Drop CSV Upload
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone?.addEventListener("click", () => fileInput?.click());

  dropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone?.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput?.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file.name.endsWith(".csv")) {
      showToast("Please select a valid .csv file", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    showToast(`Uploading and validating ${file.name}...`, "info");

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      const json = await res.json();
      if (json.status !== "success") throw new Error(json.message);

      showToast(`Analyzed ${file.name} successfully!`, "success");
      await fetchDashboardData();

    } catch (err) {
      console.error(err);
      showToast(err.message || "Failed to upload file", "error");
    }
  }

  // Restore Default Dataset
  document.getElementById("btn-reset-default-data")?.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/reset-default-dataset", { method: "POST" });
      const json = await res.json();
      if (json.status === "success") {
        showToast("Restored default synthetic dataset", "success");
        await fetchDashboardData();
      }
    } catch (err) {
      showToast("Failed to reset dataset", "error");
    }
  });

  // Quick Refresh Data button
  document.getElementById("btn-refresh-data")?.addEventListener("click", () => {
    fetchDashboardData();
    showToast("Dashboard data refreshed", "info");
  });

  // =========================================================
  // RESULTS & TRADES TABLE
  // =========================================================
  async function fetchTradesPage() {
    const tbody = document.getElementById("trades-table-body");
    tbody.innerHTML = `
      <tr>
        <td colspan="12" style="text-align: center; padding: 25px; color: var(--text-muted);">
          Loading records...
        </td>
      </tr>
    `;

    const params = new URLSearchParams({
      page: state.currentPage,
      page_size: state.pageSize,
      search: state.currentFilter.search,
      symbol: state.currentFilter.symbol,
      status: state.currentFilter.status,
      reason: state.currentFilter.reason,
      sort_by: state.currentFilter.sortBy,
      sort_order: state.currentFilter.sortOrder
    });

    try {
      const res = await fetch(`/api/trades?${params.toString()}`);
      const data = await res.json();
      if (data.status !== "success") throw new Error("Failed to load trades");

      state.totalPages = data.total_pages;

      const startRecord = (data.page - 1) * data.page_size + 1;
      const endRecord = Math.min(data.page * data.page_size, data.total_records);
      document.getElementById("pagination-info").textContent =
        `Showing ${data.total_records > 0 ? startRecord : 0} to ${endRecord} of ${data.total_records.toLocaleString()} records`;
      document.getElementById("current-page-lbl").textContent = `${data.page} / ${data.total_pages}`;

      document.getElementById("btn-prev-page").disabled = (data.page <= 1);
      document.getElementById("btn-next-page").disabled = (data.page >= data.total_pages);

      if (data.trades.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="12" style="text-align: center; padding: 30px; color: var(--text-muted);">
              No records match filter criteria.
            </td>
          </tr>
        `;
        return;
      }

      tbody.innerHTML = data.trades.map(t => {
        const isAnomaly = t.status === "POTENTIAL ANOMALY";
        const statusChip = isAnomaly
          ? `<span class="status-chip anomaly">POTENTIAL ANOMALY</span>`
          : `<span class="status-chip normal">NORMAL</span>`;

        let reasonBadge = `<span style="color: var(--text-muted);">&mdash;</span>`;
        if (t.reason === "Both") {
          reasonBadge = `<span class="status-chip anomaly" style="font-size: 0.68rem;">Both</span>`;
        } else if (t.reason === "Mahalanobis Distance") {
          reasonBadge = `<span class="status-chip warning" style="font-size: 0.68rem;">Mahalanobis</span>`;
        } else if (t.reason === "Volume Spike") {
          reasonBadge = `<span class="status-chip warning" style="font-size: 0.68rem;">Volume Spike</span>`;
        }

        const priceChangeColor = t.price_change > 0 ? "var(--color-success-text)" : (t.price_change < 0 ? "var(--color-danger-text)" : "inherit");

        return `
          <tr>
            <td class="mono" style="font-weight: 600; color: var(--text-primary);">${t.trade_id}</td>
            <td style="font-size: 0.76rem; color: var(--text-muted); font-family: var(--font-mono);">${t.timestamp}</td>
            <td><span class="symbol-pill ${t.symbol}">${t.symbol}</span></td>
            <td class="mono">$${parseFloat(t.price).toFixed(2)}</td>
            <td class="mono">${parseInt(t.quantity).toLocaleString()}</td>
            <td class="mono">$${parseFloat(t.volume).toLocaleString()}</td>
            <td class="mono" style="color: ${priceChangeColor}; font-weight: 600;">${t.price_change >= 0 ? "+" : ""}${parseFloat(t.price_change).toFixed(3)}</td>
            <td class="mono" style="font-weight: 700; color: ${isAnomaly ? 'var(--color-danger-text)' : 'inherit'};">${parseFloat(t.mahalanobis_distance).toFixed(2)}</td>
            <td class="mono" style="color: ${parseFloat(t.volume_zscore) > 3 ? 'var(--color-danger-text)' : 'inherit'};">${parseFloat(t.volume_zscore).toFixed(2)}</td>
            <td>${reasonBadge}</td>
            <td>${statusChip}</td>
            <td>
              <button class="btn-clean secondary sm btn-inspect-row" data-id="${t.trade_id}">Inspect</button>
            </td>
          </tr>
        `;
      }).join("");

      document.querySelectorAll(".btn-inspect-row").forEach(btn => {
        btn.addEventListener("click", () => {
          const id = btn.dataset.id;
          switchPage("trade-analysis-page");
          inspectTrade(id);
        });
      });

    } catch (err) {
      console.error(err);
      tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; color: var(--color-danger); padding: 20px;">Failed to load records.</td></tr>`;
    }
  }

  // Filter Event Listeners
  document.getElementById("btn-apply-filters")?.addEventListener("click", () => {
    state.currentFilter.search = document.getElementById("filter-search").value.trim();
    state.currentFilter.symbol = document.getElementById("filter-symbol").value;
    state.currentFilter.status = document.getElementById("filter-status").value;
    state.currentFilter.reason = document.getElementById("filter-reason").value;
    state.currentFilter.sortBy = document.getElementById("filter-sort").value;
    state.currentPage = 1;
    fetchTradesPage();
  });

  document.getElementById("btn-reset-filters")?.addEventListener("click", () => {
    document.getElementById("filter-search").value = "";
    document.getElementById("filter-symbol").value = "ALL";
    document.getElementById("filter-status").value = "ALL";
    document.getElementById("filter-reason").value = "ALL";
    document.getElementById("filter-sort").value = "mahalanobis_distance";
    state.currentFilter = {
      search: "",
      symbol: "ALL",
      status: "ALL",
      reason: "ALL",
      sortBy: "mahalanobis_distance",
      sortOrder: "desc"
    };
    state.currentPage = 1;
    fetchTradesPage();
  });

  document.getElementById("filter-search")?.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      document.getElementById("btn-apply-filters").click();
    }
  });

  document.getElementById("btn-prev-page")?.addEventListener("click", () => {
    if (state.currentPage > 1) {
      state.currentPage--;
      fetchTradesPage();
    }
  });

  document.getElementById("btn-next-page")?.addEventListener("click", () => {
    if (state.currentPage < state.totalPages) {
      state.currentPage++;
      fetchTradesPage();
    }
  });

  // =========================================================
  // INITIALIZATION ON LOAD
  // =========================================================
  fetchDashboardData().then(() => {
    inspectTrade("TRD-100063");
  });
});
