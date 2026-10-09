"use strict";
const menuButton = document.querySelector(".menu-toggle");
const nav = document.querySelector("#site-nav");
if (menuButton && nav) {
  menuButton.addEventListener("click", () => {
    const isOpen = nav.classList.toggle("is-open");
    menuButton.setAttribute("aria-expanded", String(isOpen));
  });
  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      nav.classList.remove("is-open");
      menuButton.setAttribute("aria-expanded", "false");
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      nav.classList.remove("is-open");
      menuButton.setAttribute("aria-expanded", "false");
    }
  });
}
document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.getAttribute("data-copy"));
    if (!target) return;
    const previous = button.textContent;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = "Copied ✓";
      window.setTimeout(() => { button.textContent = previous; }, 1700);
    } catch (error) {
      button.textContent = "Select & copy";
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      if (selection) { selection.removeAllRanges(); selection.addRange(range); }
    }
  });
});
