// toc-scrollspy.js
document.addEventListener("DOMContentLoaded", function () {
  const toc = document.querySelector('.post-toc');
  if (!toc) return;

  const links = toc.querySelectorAll('a');
  if (links.length === 0) return;

  // Track the sections corresponding to the links
  let sections = [];
  links.forEach(link => {
    const targetId = link.getAttribute('href').substring(1);
    const targetEl = document.getElementById(targetId);
    if (targetEl) {
      sections.push({ linkEl: link, targetEl: targetEl });
    }
  });

  if (sections.length === 0) return;

  // Intercept clicks to avoid adding history stack entries
  links.forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').substring(1);
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        // Scroll to element manually (considering a top margin)
        const topOffset = targetEl.getBoundingClientRect().top + window.scrollY - 80;
        window.scrollTo({
          top: topOffset,
          behavior: 'smooth'
        });
        
        // Update URL hash without adding to browser history
        history.replaceState(null, null, `#${targetId}`);
      }
    });
  });

  // Function to determine which section is active
  function updateTocSpy() {
    let currentSection = sections[0]; // fallback to first 

    // Find the last section whose top is somewhat above or near the viewport top
    const scrollPosition = window.scrollY + 100; // offset a bit for fixed headers.

    sections.forEach(sec => {
      // getBoundingClientRect().top is relative to viewport,
      // adding window.scrollY gets absolute document position.
      const topOffset = sec.targetEl.getBoundingClientRect().top + window.scrollY;
      if (scrollPosition >= topOffset - 50) { 
        currentSection = sec;
      }
    });

    // Check if we are at the bottom of the page
    if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 10) {
      currentSection = sections[sections.length - 1];
    }

    // Clear active classes from all links & parents
    links.forEach(link => {
      const parentLi = link.closest('li');
      if (parentLi) {
        parentLi.classList.remove('active');
      }
    });

    // Add active class to the current list item and its tree up to toc-list
    if (currentSection) {
      let currentLi = currentSection.linkEl.closest('li');
      while (currentLi && currentLi.closest('.toc-list')) {
        currentLi.classList.add('active');
        // move up to a potential parent list item if this is a sub-list
        const parentUl = currentLi.closest('ul');
        if (parentUl && parentUl.classList.contains('toc-list')) break; // stopped at root.
        
        if (parentUl) {
           const parentLi = parentUl.closest('li');
           if (parentLi) {
               currentLi = parentLi;
           } else {
               break;
           }
        } else {
           break;
        }
      }
    }
  }

  // Bind scroll event
  window.addEventListener('scroll', updateTocSpy, { passive: true });
  // Call it once on load
  updateTocSpy();
});
