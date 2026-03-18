```md
## Reading List (how to update)

The reading list is **static** and fully controlled via a YAML file.  
There is no scraping, no JavaScript state, and no automatic updates.

### Where it lives
The list is defined in:

```

_data/reading.yml

````

Each entry corresponds to **one paper**.

---

### Entry structure

A minimal entry looks like this:

```yaml
- title: "Paper title"
  authors: "Author A, Author B"
  venue: "Journal name"
  year: 2025
  url: "https://doi.org/..."
  read: true
  recommended: false
````

Fields:

* `title`, `authors`, `venue`, `year`: shown on the website
* `url`: optional, makes the title clickable
* `read`: `true` or `false` (checkbox)
* `recommended`: `true` or `false` (★ shown on the right)

---

### Changing the set of papers

To generate a **new list based on different journals or keywords**:

1. Collect papers externally
   (e.g. Google Scholar, Scopus, journal TOCs, alerts).

2. Filter manually by:

   * journal (e.g. *JEDC*, *JEBO*, *AEJ Macro*)
   * keywords (e.g. heterogeneous agents, learning, networks)

3. Replace or edit entries in `_data/reading.yml`.

The website updates automatically on rebuild.

There is **no automatic querying** from journals by design.
This keeps the list:

* transparent
* curated
* version-controlled
* academically honest

---

### Ordering

Papers appear **in the order written in the YAML file**.

Reorder entries manually to:

* put recent papers first
* group by topic
* separate core vs background reading

---

### Symbols on the website

* ☑ / ☐ : read / not read
* ★ : explicitly recommended for careful reading

No interaction: the page is static and public.

---

### Why YAML (and not BibTeX)

* Works on GitHub Pages without plugins
* Easy to diff and review
* Explicit editorial control
* No hidden automation

If you prefer BibTeX, you can still keep a `.bib` file separately and
copy selected entries into the YAML file.

---

That’s all that is required to maintain the reading list.

```
