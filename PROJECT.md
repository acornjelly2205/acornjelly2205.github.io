Project Acorn
Overview

Project Acorn is a personal technical knowledge base focused on CUDA, GPU architecture, sparse computing, performance optimization, and AI systems.

The project combines daily study notes, research explanations, implementation records, and long-term project documentation in a structured Astro-based website.

Goals
Build a sustainable system for publishing technical study notes.
Present GPU and HPC projects as an organized engineering portfolio.
Improve technical English through regular writing.
Provide Korean translations for selected important content.
Connect related Study and Project entries through shared tags.
Build the foundation for a future interactive knowledge graph.
Content Types
Study

Study contains focused learning notes and technical explanations.

Current topic groups include:

CUDA
GPU Architecture
Sparse Matrix
AI Systems
Papers

Study entries are generally ordered by publication date.

Projects

Projects contains structured records of research and implementation work.

Current project groups include:

TC-BELL
Instruction Roofline
CUDA Experiments
SpMM Attribute Selector

Project entries may use an explicit display order.

Internationalization Strategy

English is the default language.

/
/study/
/projects/
/tags/

Korean pages use the /ko/ prefix.

/ko/
/ko/study/
/ko/projects/
/ko/tags/

Translations are optional.

Each translated pair may share a translationKey.

lang: en
translationKey: warp-shuffle-reduction
lang: ko
translationKey: warp-shuffle-reduction

When a translation exists, the interface should provide a direct language switch. When no translation exists, the page should clearly indicate that the entry is available in only one language.

Tag Architecture

Tags connect related entries across the Study and Projects collections.

Canonical tag IDs are language-independent.

tags:
  - cuda
  - performance-optimization
  - nsight-compute

Localized labels are managed separately.

{
  "performance-optimization": {
    labels: {
      en: "Performance Optimization",
      ko: "성능 최적화"
    }
  }
}

The tag system must support:

Aggregating Study and Project entries
Excluding drafts
Counting entries by content type
Sorting tags by entry count
Filtering entries by tag
Language-specific results
Pagination when required
Links from entry pages to tag pages

Blog posts from the original AstroPaper collection are not included in Project Acorn tag statistics unless this decision is changed later.

v0.2 Scope
Documentation checkpoint
Update README
Maintain PROJECT.md
Maintain CHANGELOG.md
Record architecture decisions under docs/
Bilingual foundation
Add en and ko locales
Keep English routes unprefixed
Add /ko/ routes
Add localized UI strings
Add lang metadata
Add optional translationKey
Add language switching
Add canonical and hreflang metadata
Unified Study and Project tags
Create a canonical tag registry
Aggregate tags from Study and Projects
Build a tag statistics page
Build tag-specific entry pages
Display Study and Project counts
Add clickable tag badges to content pages
Metadata centralization
Centralize Study category labels
Centralize Project labels and descriptions
Support localized labels
Remove duplicated metadata from page components
Non-Goals for v0.2

The following features are intentionally deferred:

Automatic machine translation
Mandatory translation of every entry
Interactive graph visualization
User accounts
Comments
Server-side database
Dynamic content editing
Planned Versions
v0.2 — Bilingual Knowledge Navigation

Internationalization, shared tags, localized taxonomy, and structured content discovery.

v0.3 — Knowledge Relationships

Related entries, tag relationships, and knowledge graph data generation.

v1.0 — Public Portfolio Release

Stable public deployment, polished representative content, SEO validation, and portfolio-ready project pages.