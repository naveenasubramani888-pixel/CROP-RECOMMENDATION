/* AgriSense AI - Common UI Application Script */

const API_BASE = window.location.origin;

document.addEventListener("DOMContentLoaded", () => {
  // Highlight active navigation link
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll(".nav-links a");
  
  navLinks.forEach(link => {
    const href = link.getAttribute("href");
    if (href === currentPath || (currentPath === "/" && href === "index.html")) {
      link.classList.add("active");
    }
  });
});
