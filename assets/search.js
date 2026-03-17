// search.js
document.addEventListener("DOMContentLoaded", function() {
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  const postListContainer = document.querySelector('.post-list').parentNode; // To hide/show default list
  const defaultPostList = document.querySelector('.post-list');

  if (!searchInput || !searchResults) return;

  let postsData = [];

  // Fetch the search index
  fetch('/search.json')
    .then(response => response.json())
    .then(data => {
      postsData = data;
      searchInput.disabled = false;
      searchInput.placeholder = "Search " + postsData.length + " posts...";
    })
    .catch(error => {
      console.error("Error fetching search.json:", error);
      searchInput.placeholder = "Search is currently unavailable.";
    });

  // Listen for input changes
  searchInput.addEventListener('input', function() {
    const query = this.value.toLowerCase().trim();
    
    // Clear results
    searchResults.innerHTML = '';
    
    if (query === '') {
      searchResults.style.display = 'none';
      if(defaultPostList) defaultPostList.style.display = 'block';
      return;
    }

    // Hide default post list while searching
    if(defaultPostList) defaultPostList.style.display = 'none';
    searchResults.style.display = 'block';

    // Filter posts
    const matches = postsData.filter(post => {
      return post.title.toLowerCase().includes(query) || post.excerpt.toLowerCase().includes(query);
    });

    if (matches.length === 0) {
      searchResults.innerHTML = '<li><p class="no-results">No results found for "' + query + '".</p></li>';
      return;
    }

    // Render results
    matches.forEach(post => {
      const li = document.createElement('li');
      li.innerHTML = `
        <span class="post-meta">${post.date}</span>
        <h3>
          <a class="post-link" href="${post.url}">${post.title}</a>
        </h3>
        <p>${post.excerpt}</p>
      `;
      searchResults.appendChild(li);
    });
  });
});
