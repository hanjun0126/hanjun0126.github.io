// header-title.js
document.addEventListener("DOMContentLoaded", function () {
  const headerTitle = document.getElementById('header-post-title');
  const mainTitle = document.querySelector('.post-title');

  if (!headerTitle || !mainTitle) return;

  // Set the text from the main title
  headerTitle.textContent = mainTitle.textContent.trim();
  headerTitle.style.cursor = 'pointer';

  headerTitle.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  function updateHeaderVisibility() {
    const mainTitleRect = mainTitle.getBoundingClientRect();
    if (mainTitleRect.bottom < 56) {
      document.body.classList.add('title-visible');
    } else {
      document.body.classList.remove('title-visible');
    }
  }

  window.addEventListener('scroll', updateHeaderVisibility, { passive: true });
  // Initial check
  updateHeaderVisibility();
});
