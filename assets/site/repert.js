"use strict";

// RePert website: progressive enhancement. Content is readable without JavaScript.
(() => {
  const menuButton = document.querySelector(".menu-toggle");
  const nav = document.getElementById("site-nav");

  function closeMenu() {
    if (!menuButton || !nav) return;
    nav.classList.remove("is-open");
    menuButton.setAttribute("aria-expanded", "false");
  }

  if (menuButton && nav) {
    menuButton.addEventListener("click", () => {
      const open = nav.classList.toggle("is-open");
      menuButton.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeMenu);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeMenu();
    });
    document.addEventListener("click", (event) => {
      if (!nav.contains(event.target) && !menuButton.contains(event.target)) closeMenu();
    });
  }

  const budgetButtons = Array.from(document.querySelectorAll(".budget-button"));
  const rawBar = document.getElementById("raw-bar");
  const recaBar = document.getElementById("reca-bar");
  const rawScore = document.getElementById("raw-score");
  const recaScore = document.getElementById("reca-score");
  const gainScore = document.getElementById("gain-score");
  const compoundCount = document.getElementById("compound-count");

  // The E endpoint is a Fisher-z difference, not a probability.
  // Keep the same visual scale (0 to 0.30) at every support budget.
  function renderBudget(button) {
    const raw = Number(button.dataset.raw);
    const reca = Number(button.dataset.reca);
    const count = Number(button.dataset.count);
    if (!(raw >= 0 && reca >= 0 && raw <= 0.30 && reca <= 0.30 && count > 0)) return;
    if (![rawBar, recaBar, rawScore, recaScore, gainScore, compoundCount].every(Boolean)) return;

    budgetButtons.forEach((b) => {
      const active = b === button;
      b.classList.toggle("is-active", active);
      b.setAttribute("aria-pressed", String(active));
    });
    rawBar.style.width = (100 * raw / 0.30).toFixed(2) + "%";
    recaBar.style.width = (100 * reca / 0.30).toFixed(2) + "%";
    rawScore.textContent = raw.toFixed(4);
    recaScore.textContent = reca.toFixed(4);
    const delta = reca - raw;
    gainScore.textContent = (delta >= 0 ? "+" : "") + delta.toFixed(4);
    compoundCount.textContent = count.toLocaleString("en-US");
  }

  budgetButtons.forEach((button) => {
    button.addEventListener("click", () => renderBudget(button));
  });
  const defaultBudget = budgetButtons.find((b) => b.getAttribute("aria-pressed") === "true");
  if (defaultBudget) renderBudget(defaultBudget);

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.getAttribute("data-copy");
      const target = id ? document.getElementById(id) : null;
      if (!target) return;
      const label = button.textContent;
      const selection = window.getSelection();
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
        await navigator.clipboard.writeText(target.textContent.trim());
        button.textContent = "Copied ✓";
      } catch (_error) {
        const range = document.createRange();
        range.selectNodeContents(target);
        if (selection) {
          selection.removeAllRanges();
          selection.addRange(range);
        }
        button.textContent = "Select & copy";
      }
      window.setTimeout(() => { button.textContent = label; }, 1800);
    });
  });
})();
