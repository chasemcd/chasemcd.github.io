# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal academic website for Chase McDonald, built with the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme. Deployed to GitHub Pages at chasemcd.github.io.

## Build & Development Commands

```bash
bundle install          # Install Ruby dependencies
bundle exec jekyll serve  # Local dev server (default: localhost:4000)
bundle exec jekyll build  # Build static site to _site/
```

## Deployment

Pushes to `master` trigger the GitHub Actions workflow (`.github/workflows/deploy.yml`) which builds and deploys to the `gh-pages` branch. Manual deploy: `./bin/deploy`.

## Architecture

- **Jekyll site** using kramdown markdown, Rouge syntax highlighting, and SCSS for styles
- **`_config.yml`**: Central configuration — site metadata, social links, plugin settings, feature flags (dark mode, math, masonry layout, etc.)
- **`_pages/about.md`**: Homepage content (layout: `about`, permalink: `/`). Profile image, bio, and news/selected papers toggles are set in YAML front matter
- **`_bibliography/papers.bib`**: BibTeX file driving the publications page via jekyll-scholar. Author highlighting configured in `_config.yml` under `scholar:` (last_name/first_name)
- **`_data/coauthors.yml`**: Co-author metadata for auto-linking in publications
- **`_news/`**: News items displayed on the homepage (limited by `news_limit` in config)
- **`_projects/`**: Project pages rendered on a masonry grid
- **`_layouts/`**: `default.html` → base; `about.html` → homepage; `bib.html` → bibliography entries; `post.html` → blog posts; `distill.html` → distill.pub-style posts
- **`_sass/`**: `_themes.scss` for theme color (`$theme-color`), `_variables.scss`, `_base.scss`, `_layout.scss`
- **`_site/`**: Generated output (gitignored effectively, but present locally)

## Key Patterns

- Pages use YAML front matter with `nav: true` to appear in the navbar
- Publications page (`_pages/publications.md`) loops over `years` array in front matter to query BibTeX entries by year
- Blog posts go in `_posts/` with `YYYY-MM-DD-title.md` naming convention
- Images go in `assets/img/`; profile image is referenced by filename in `_pages/about.md` front matter
